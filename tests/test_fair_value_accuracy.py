"""Deterministic checks for accounting alignment and valuation evidence."""
import copy
import datetime as dt
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import sma_router
from fv_router import _row_for_fy
from core import fair_value


def model(pred=120.0, count=8, as_of="2026-03-31"):
    return {"pred_price": pred, "r2": 0.8, "gain_pct": 20.0, "eq": "Price = 1 + 2 × OP",
            "series": [{"year": 2018 + i, "profit": 100 + i * 10,
                        "price": 50 + i * 5} for i in range(count)],
            "historical": {"2025": {"fair": pred, "actual": 100.0, "profit": 170.0}},
            "sample_count": count, "slope": 0.5, "positive_fundamentals": True,
            "fundamental_as_of": as_of, "validation_mae_pct": 12.0,
            "validation_observations": count - 4}


def base_result():
    return {"ticker": "TEST", "current_price": 100.0, "op": model(),
            "has_sales": True, "sales": model(125.0), "has_ttm": False, "ttm": {},
            "composite_fair_price": 122.0, "model_count": 2,
            "price_as_of": "2026-11-09", "price_history": []}


class AccountingAlignmentTests(unittest.TestCase):
    def test_march_end_year_matches_fy_start_and_keeps_first_column(self):
        table = pd.DataFrame({"Mar 2024": [100], "Mar 2025": [200], "TTM": [999]},
                             index=["Operating Profit"])
        actual = sma_router._extract_metric("Operating Profit", table)
        self.assertEqual(actual.to_dict(), {2023: 100, 2024: 200})

    def test_missing_is_not_zero_and_real_zero_is_preserved(self):
        table = pd.DataFrame({"Mar 2023": ["0"], "Mar 2024": ["—"], "Mar 2025": ["100"]},
                             index=["Operating Profit"])
        actual = sma_router._extract_metric("Operating Profit", table)
        self.assertEqual(actual.to_dict(), {2022: 0.0, 2024: 100.0})

    def test_accounting_parentheses_and_unicode_minus_are_losses(self):
        values = sma_router._clean_numeric(pd.Series(["(1,234)", "−45", "1e+3", "—"]))
        self.assertEqual(values.iloc[:3].tolist(), [-1234., -45., 1000.])
        self.assertTrue(pd.isna(values.iloc[-1]))

    def test_ttm_does_not_bridge_a_missing_quarter(self):
        quarters = pd.to_datetime(["2024-03-31", "2024-06-30", "2024-12-31", "2025-03-31"])
        self.assertTrue(sma_router._continuous_ttm(pd.Series([10., 20., 30., 40.], index=quarters)).empty)

    def test_ttm_includes_real_zero_quarter(self):
        quarters = pd.date_range("2024-03-31", periods=4, freq="QE")
        result = sma_router._continuous_ttm(pd.Series([10., 0., 30., 40.], index=quarters))
        self.assertEqual(result.iloc[-1], 80.0)

    def test_full_engine_uses_aligned_prices_and_all_annual_columns(self):
        years = list(range(2017, 2027))
        profits = np.arange(1, 11) * 100.
        table = pd.DataFrame({f"Mar {y}": [x, x * 5] for y, x in zip(years, profits)},
                             index=["Operating Profit", "Sales+"])
        dates = [pd.bdate_range(f"{y}-03-01", f"{y}-03-31")[-1] for y in years]
        prices = profits * 0.2 + 100
        bars = pd.DataFrame({"Close": prices, "Open": prices, "High": prices + 1,
                             "Low": prices - 1, "Volume": 1000}, index=dates)
        with patch.object(sma_router, "_get_cached", side_effect=lambda t, k, f: table if k == "pl" else pd.DataFrame()), \
             patch.object(sma_router, "is_bank_or_finance", return_value=False), \
             patch.object(sma_router.yf, "Ticker", return_value=SimpleNamespace(history=lambda **kwargs: bars)):
            result = sma_router._analyze_ticker("TEST", 2014, False, True)
        self.assertNotIn("error", result)
        self.assertEqual(result["years_count"], 10)
        self.assertEqual(result["op"]["series"][0]["year"], 2016)
        self.assertAlmostEqual(result["op"]["pred_price"], 300.0)
        self.assertAlmostEqual(result["op"]["r2"], 1.0)
        self.assertFalse(result["historical_fair_values_point_in_time_safe"])

    def test_historical_bucket_uses_historical_close(self):
        result = base_result()
        result["current_price"] = 9999.0
        row = _row_for_fy(result, 2025)
        self.assertGreater(row["comp_gain"], 20.0)
        self.assertEqual(row["bucket"], "UNDERVALUED")
        self.assertEqual(row["actual_price"], 100.0)
        self.assertFalse(row["point_in_time_safe"])


class ValuationEvidenceTests(unittest.TestCase):
    def setUp(self):
        fair_value._MEM_CACHE.clear()
        fair_value._CACHE_TIMES.clear()

    def evaluate(self, result, **kwargs):
        with patch.object(fair_value, "_HAS_ENGINES", True), \
             patch.object(fair_value, "_analyze_ticker", return_value=copy.deepcopy(result)):
            return fair_value.compute_fair_value("TEST", **kwargs)

    def test_bad_predictions_and_two_point_fit_are_excluded(self):
        base = base_result()
        base["op"] = model(-20.0)
        base["sales"] = model(10000.0, count=2)
        base["sales"]["r2"] = 1.0
        output = self.evaluate(base)
        self.assertIsNone(output["composite_fair_price"])
        self.assertEqual(output["model_count"], 0)
        self.assertEqual(output["valuation_confidence"], "unavailable")

    def test_actual_screened_price_overrides_cached_quote(self):
        base = base_result()
        with patch.object(fair_value, "_HAS_ENGINES", True), \
             patch.object(fair_value, "_analyze_ticker", return_value=base) as analyze:
            initial = fair_value.compute_fair_value("TEST", current_price=100)
            later = fair_value.compute_fair_value("TEST", current_price=200)
        self.assertEqual(analyze.call_count, 1)
        self.assertEqual(initial["composite_fair_price"], later["composite_fair_price"])
        self.assertAlmostEqual(later["composite_gain_pct"], -38.89, places=2)
        self.assertEqual(later["valuation_bucket"], "OVERVALUED")
        self.assertEqual(later["models"]["op"]["gain_pct"], -40.0)

    def test_cache_includes_accounting_options_and_does_not_leak_mutations(self):
        with patch.object(fair_value, "_HAS_ENGINES", True), \
             patch.object(fair_value, "_analyze_ticker", return_value=base_result()) as analyze:
            first = fair_value.compute_fair_value("TEST", include_other_income=True)
            first["models"]["op"]["pred_price"] = 99999
            second = fair_value.compute_fair_value("TEST", include_other_income=True)
            fair_value.compute_fair_value("TEST", include_other_income=False)
        self.assertEqual(second["models"]["op"]["pred_price"], 120.0)
        self.assertEqual(analyze.call_count, 2)

    def test_stale_and_negative_slope_models_cannot_claim_undervaluation(self):
        base = base_result()
        base["op"]["fundamental_as_of"] = "2020-03-31"
        base["sales"]["slope"] = -1
        output = self.evaluate(base)
        self.assertEqual(output["valuation_bucket"], "N/A")
        self.assertIsNone(output["composite_fair_price"])

    def test_quarterly_sales_survives_missing_quarterly_profit(self):
        base = base_result()
        quarterly = model(130.0, count=12, as_of="2026-06-30")
        quarterly["historical"] = {"2026-06-30": {"fair": 130.0, "actual": 100.0, "ttm_sales": 1000.0}}
        base["has_ttm"] = True
        base["ttm"] = {"sales": quarterly}
        output = self.evaluate(base)
        self.assertIn("ttm_sales", output["eligible_model_names"])
        self.assertEqual(output["quarterly_rows"][0]["ttm_sales"], 1000.0)
        self.assertEqual(output["quarterly_rows"][0]["price"], 100.0)

    def test_nonfinite_values_are_json_safe(self):
        base = base_result()
        base["op"]["pred_price"] = float("nan")
        base["op"]["series"][0]["ratio"] = float("inf")
        output = self.evaluate(base)
        json.dumps(output, allow_nan=False)
        self.assertIsNone(output["models"]["op"]["pred_price"])

    def test_missing_data_failure_is_not_cached_forever(self):
        with patch.object(fair_value, "_HAS_ENGINES", True), \
             patch.object(fair_value, "_analyze_ticker", side_effect=[{"error": "temporary failure"}, base_result()]) as analyze:
            first = fair_value.compute_fair_value("TEST")
            second = fair_value.compute_fair_value("TEST")
        self.assertEqual(first["status"], "NO_DATA")
        self.assertEqual(second["status"], "SUCCESS")
        self.assertEqual(analyze.call_count, 2)


if __name__ == "__main__":
    unittest.main()
