"""Deterministic market scenarios; no downloads or saved portfolio writes."""
import datetime as dt
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from core.swing_reversal import (
    _find_pivot_lows, _independent_touches,
    analyze_stock_swing_reversal, scan_swing_reversals,
)


def base_frame(anchors=None, width=1.0):
    anchors = anchors or {0: 140, 20: 135, 40: 128, 60: 100, 80: 124, 100: 101, 119: 103.4}
    close = np.interp(np.arange(120), list(anchors), list(anchors.values()))
    return pd.DataFrame({"Open": close - .2, "High": close + width,
                         "Low": close - width, "Close": close,
                         "Volume": np.full(120, 2_000_000)},
                        index=pd.bdate_range("2025-01-01", periods=120))


def replace_close(frame, positions, values, width=1.0):
    for position, close in zip(positions, values):
        frame.iloc[position, frame.columns.get_indexer(["Open", "High", "Low", "Close"])] = [
            close - .2, close + width, close - width, close]


def confirmed_frame():
    frame = base_frame(width=.5)
    replace_close(frame, range(114, 120), [103, 102.7, 102.4, 102, 103.2, 104.4], width=.5)
    return frame


class SwingAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.fv = patch("core.swing_reversal.compute_fair_value", return_value={})
        self.fv_mock = self.fv.start()
        self.addCleanup(self.fv.stop)
        reference = dt.datetime(2025, 6, 17, 16, 0, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))
        for target in ("core.swing_reversal.market_now", "core.data_fetcher.market_now"):
            clock = patch(target, return_value=reference)
            clock.start()
            self.addCleanup(clock.stop)

    def test_requested_distance_and_upside_are_enforced(self):
        frame = base_frame()
        self.assertIsNotNone(analyze_stock_swing_reversal("TEST", frame, min_upside_pct=15))
        self.assertIsNone(analyze_stock_swing_reversal("TEST", frame, max_dist_to_floor_pct=3))
        self.assertIsNone(analyze_stock_swing_reversal("TEST", frame, min_upside_pct=25))

    def test_nearest_resistance_cannot_be_bypassed_by_fair_value(self):
        frame = base_frame({0: 140, 20: 135, 40: 128, 60: 100, 80: 124,
                            100: 101, 108: 111, 114: 102, 119: 103.4})
        self.fv_mock.return_value = {"composite_fair_price": 500, "valuation_confidence": "high"}
        self.assertIsNone(analyze_stock_swing_reversal("TEST", frame, min_upside_pct=15))
        result = analyze_stock_swing_reversal("TEST", frame, min_upside_pct=5)
        self.assertIsNotNone(result)
        self.assertEqual(result["target_price"], 112)
        self.assertEqual(result["target_source"], "nearest_confirmed_resistance")
        self.assertEqual(result["secondary_targets"], [125])
        self.assertAlmostEqual(result["fv_gain_pct"], (500 / 103.4 - 1) * 100, places=2)

    def test_current_close_below_floor_is_not_a_buy(self):
        frame = base_frame()
        replace_close(frame, [119], [98])
        self.assertIsNone(analyze_stock_swing_reversal("TEST", frame))

    def test_recovered_broken_shelf_is_not_recycled(self):
        frame = base_frame()
        replace_close(frame, [90], [96])
        result = analyze_stock_swing_reversal("TEST", frame)
        if result is not None:
            self.assertLess(result["floor_price"], 99)
            self.assertFalse(result["is_actionable"])

    def test_above_ema_alone_does_not_confirm_reversal_or_double_bottom(self):
        result = analyze_stock_swing_reversal("TEST", base_frame())
        self.assertTrue(result["is_above_ema9"])
        self.assertFalse(result["reversal_confirmed"])
        self.assertFalse(result["is_actionable"])
        self.assertNotEqual(result["alert_status"], "BREAKOUT_CONFIRMED")
        self.assertTrue(result["is_double_bottom"])
        self.assertFalse(result["double_bottom_confirmed"])

    def test_confirmed_reversal_requires_breakout_high_and_liquidity(self):
        frame = confirmed_frame()
        result = analyze_stock_swing_reversal("TEST", frame)
        self.assertTrue(result["reversal_confirmed"])
        self.assertTrue(result["is_actionable"])
        self.assertGreaterEqual(result["shelf_touches"], 2)
        self.assertGreaterEqual(result["risk_reward_ratio"], 2)
        self.assertFalse(result["score_is_probability"])
        # A higher close still below the breakout high is insufficient.
        replace_close(frame, [119], [103.4], width=.5)
        result = analyze_stock_swing_reversal("TEST", frame)
        self.assertFalse(result["reversal_confirmed"])
        self.assertFalse(result["is_actionable"])

    def test_falling_price_near_floor_remains_watch_only(self):
        frame = base_frame()
        replace_close(frame, [117, 118, 119], [102, 101.5, 101])
        result = analyze_stock_swing_reversal("TEST", frame, min_upside_pct=.5)
        self.assertIsNotNone(result)
        self.assertLess(result["dist_to_floor_pct"], 3.5)
        self.assertEqual(result["alert_status"], "APPROACHING_SUPPORT")
        self.assertFalse(result["is_actionable"])
        self.assertTrue(any("still falling" in warning for warning in result["warnings"]))

    def test_low_volume_blocks_actionability(self):
        frame = confirmed_frame()
        frame["Volume"] = 0
        result = analyze_stock_swing_reversal("TEST", frame)
        self.assertTrue(result["reversal_confirmed"])
        self.assertFalse(result["is_actionable"])
        self.assertTrue(any("liquidity" in warning for warning in result["warnings"]))

    def test_prior_volume_average_excludes_signal_bar_and_sparse_data_blocks(self):
        frame = confirmed_frame()
        frame.iloc[-1, frame.columns.get_loc("Volume")] = 4_000_000
        result = analyze_stock_swing_reversal("TEST", frame)
        self.assertEqual(result["volume_ratio"], 2)
        self.assertEqual(result["confirmation_volume_ratio"], 2)
        self.assertTrue(result["is_actionable"])
        frame.iloc[-21:-15, frame.columns.get_loc("Volume")] = 0
        result = analyze_stock_swing_reversal("TEST", frame)
        self.assertFalse(result["is_actionable"])

    def test_stale_or_invalid_data_blocks_actionability(self):
        frame = confirmed_frame()
        now = dt.datetime(2025, 6, 25, 16, 0, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))
        with patch("core.swing_reversal.market_now", return_value=now):
            result = analyze_stock_swing_reversal("TEST", frame)
        self.assertFalse(result["is_actionable"])
        self.assertGreater(result["data_age_days"], 5)
        self.assertTrue(any("stale" in message for message in result["actionable_blockers"]))
        duplicates = pd.concat([frame, frame.tail(1)])
        result = analyze_stock_swing_reversal("TEST", duplicates)
        self.assertFalse(result["is_actionable"])
        self.assertTrue(any("duplicate" in message for message in result["actionable_blockers"]))

    def test_low_confidence_valuation_is_context_not_an_actionability_blocker(self):
        self.fv_mock.return_value = {"composite_fair_price": 500, "valuation_confidence": "low"}
        result = analyze_stock_swing_reversal("TEST", confirmed_frame())
        self.assertTrue(result["is_actionable"])
        self.assertFalse(result["fv_confluence"])
        self.assertTrue(result["warnings"])
        self.assertEqual(result["actionable_blockers"], [])
        self.fv_mock.assert_called_once_with("TEST", current_price=104.4)

    def test_flat_adjacent_lows_are_one_touch_and_unconfirmed_tail_is_excluded(self):
        values = np.array([110, 108, 105, 100, 100, 100, 105, 106, 103, 99], dtype=float)
        dates = pd.bdate_range("2025-01-01", periods=len(values))
        pivots = _find_pivot_lows(values, dates)
        self.assertEqual([p[0] for p in pivots], [5])
        noisy = [(3, "a", 100), (5, "b", 100), (11, "c", 100)]
        self.assertEqual(len(_independent_touches(noisy, np.full(15, 101), 2)), 1)

    def test_atr_buffered_stop_and_risk_use_actual_prices(self):
        result = analyze_stock_swing_reversal("TEST", base_frame())
        expected = result["floor_price"] - max(result["floor_price"] * .01, result["atr14"] * .75)
        self.assertAlmostEqual(result["stop_loss_price"], expected, delta=.01)
        self.assertAlmostEqual(result["risk_pct"],
                               (result["current_price"] - result["stop_loss_price"]) / result["current_price"] * 100,
                               delta=.02)

    def test_incomplete_session_is_ignored_before_nse_close(self):
        frame = base_frame()
        frame.loc[pd.Timestamp("2026-11-10")] = [195, 201, 190, 200, 2_000_000]
        now = dt.datetime(2026, 11, 10, 11, 0, tzinfo=dt.timezone(dt.timedelta(hours=5, minutes=30)))
        with patch("core.data_fetcher.market_now", return_value=now):
            result = analyze_stock_swing_reversal("TEST", frame)
        self.assertEqual(result["current_price"], 103.4)
        self.assertEqual(result["history_bars"], 120)
        self.assertTrue(result["completed_candles_only"])

    def test_registry_batch_scope_deduplicates_and_skips_missing_downloads(self):
        registry = pd.DataFrame({"symbol": ["TEST", "TEST.NS", "MISSING"]})
        with patch("core.swing_reversal.load_ticker_registry", return_value=registry) as load, \
             patch("core.swing_reversal.fetch_batch_ohlcv", return_value={"TEST": base_frame()}) as fetch:
            result = scan_swing_reversals(source="all")
        load.assert_called_once_with("all")
        fetch.assert_called_once_with(["TEST", "MISSING"], interval="1d", lookback_days=400)
        self.assertEqual([row["ticker"] for row in result], ["TEST"])

    def test_insufficient_history_cannot_establish_major_support(self):
        self.assertIsNone(analyze_stock_swing_reversal("TEST", base_frame().tail(50)))


if __name__ == "__main__":
    unittest.main()
