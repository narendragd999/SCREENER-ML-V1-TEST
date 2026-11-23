"""
Core Fair Value Valuation Engine — Comprehensive Edition
=========================================================
- Delivers deep fundamental valuation with complete multi-model breakdowns:
    1. Annual Operating Profit (OP) Model + R² + Equation + Historical Table
    2. Annual Sales Model + R² + Equation + Historical Table
    3. Quarterly TTM Operating Profit Model + R² + Historical Table
    4. Quarterly TTM Sales Model + R² + Historical Table
    5. Composite Fair Value (R²-weighted blend)
    6. Historical Actual Stock Price vs Fair Value Trajectory
    7. Full Year-by-Year (FY) Summary Table
"""

import os
import asyncio
import datetime as dt
import copy
import math
import time
from numbers import Real
from typing import Dict, List, Optional, Any

try:
    from sma_router import _analyze_ticker
    from fv_router import fv_pine_json
    _HAS_ENGINES = True
except ImportError:
    _HAS_ENGINES = False

_MEM_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TIMES: Dict[str, float] = {}
CACHE_TTL_SECONDS = 600


def _positive_number(value) -> Optional[float]:
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError):
        return None


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Real) and not isinstance(value, bool):
        return value if math.isfinite(value) else None
    return value


def _valuation_summary(payload: Dict[str, Any], base: Dict[str, Any], price=None) -> Dict[str, Any]:
    """Gate descriptive regressions by evidence, then use the supplied quote."""
    models = payload.get("models", {})
    blocks = [("op", models.get("op"), base.get("op", {}), 1.0, 5),
              ("sales", models.get("sales"), base.get("sales", {}), 0.8, 5),
              ("ttm", models.get("ttm"), base.get("ttm", {}), 1.2, 8),
              ("ttm_sales", (models.get("ttm") or {}).get("sales"),
               base.get("ttm", {}).get("sales", {}), 0.8, 8)]
    eligible = []
    warnings = ["R² describes regression fit and is not a probability of trading success.",
                "Historical fair-value curves use retrospective fits; accounting publication dates are unavailable.",
                "Annual and TTM profit/sales models are correlated; model count is not independent evidence."]
    today = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).date()
    for name, model, raw, multiplier, min_samples in blocks:
        if not model:
            continue
        for key in ("sample_count", "slope", "fundamental_as_of", "validation_mae_pct",
                    "validation_observations", "positive_fundamentals", "fit_scope"):
            model[key] = raw.get(key)
        if model["sample_count"] is None:
            model["sample_count"] = len(raw.get("series", []))
        model["validation_observations"] = raw.get("validation_observations") or 0
        if name == "ttm_sales":
            model["equation"] = raw.get("eq")
        reasons = []
        pred = _positive_number(model.get("pred_price"))
        try:
            r2 = float(model.get("r2", 0))
        except (TypeError, ValueError):
            r2 = 0.0
        if pred is None:
            reasons.append("Fair price is not finite and positive")
        if not math.isfinite(r2) or r2 <= 0.25 or r2 > 1:
            reasons.append("Regression fit is too weak or invalid")
        if model["sample_count"] < min_samples:
            reasons.append(f"Requires at least {min_samples} aligned observations")
        if _positive_number(model.get("slope")) is None:
            reasons.append("Fundamental/price slope is nonpositive or unavailable")
        if model.get("positive_fundamentals") is not True:
            reasons.append("Profit/revenue observations contain losses, zeroes or invalid values")
        try:
            as_of = dt.date.fromisoformat(model.get("fundamental_as_of", ""))
            age = (today - as_of).days
        except (TypeError, ValueError):
            age = None
        model["age_days"] = age
        if age is None or age < 0:
            reasons.append("Fundamental observation date is unavailable or in the future")
        elif age > (240 if name.startswith("ttm") else 550):
            reasons.append("Fundamental observations are stale")
        model["eligible"] = not reasons
        model["exclusion_reasons"] = reasons
        if reasons:
            warnings.append(f"{name}: " + "; ".join(reasons))
        else:
            eligible.append((name, pred, r2 * multiplier, model))
    quote = _positive_number(price) if price is not None else _positive_number(payload.get("current_price"))
    payload["current_price"] = quote
    payload["model_count"] = len(eligible)
    payload["eligible_model_names"] = [m[0] for m in eligible]
    total_weight = sum(m[2] for m in eligible)
    composite = sum(m[1] * m[2] for m in eligible) / total_weight if total_weight else None
    gap = (composite / quote - 1) * 100 if composite and quote else None
    payload["composite_fair_price"] = round(composite, 2) if composite else None
    payload["composite_gain_pct"] = round(gap, 2) if gap is not None else None
    payload["valuation_bucket"] = ("N/A" if gap is None else "UNDERVALUED" if gap > 15 else
                                   "OVERVALUED" if gap < -15 else "FAIR")
    dispersion = (max(m[1] for m in eligible) - min(m[1] for m in eligible)) / composite * 100 if eligible else None
    payload["model_dispersion_pct"] = round(dispersion, 2) if dispersion is not None else None
    payload["fair_value_range"] = {"low": round(min(m[1] for m in eligible), 2),
                                   "high": round(max(m[1] for m in eligible), 2)} if eligible else None
    reliable = len(eligible) >= 2 and dispersion <= 25 and all(
        m[3].get("validation_observations", 0) >= 2 and
        m[3].get("validation_mae_pct") is not None and m[3]["validation_mae_pct"] <= 30
        for m in eligible)
    payload["valuation_confidence"] = "medium" if reliable else "low" if eligible else "unavailable"
    payload["confidence_basis"] = "Descriptive valuation evidence; confidence capped at medium without publication-timed validation"
    dates = [m[3]["fundamental_as_of"] for m in eligible]
    payload["valuation_as_of"] = max(dates) if dates else None
    payload["valuation_age_days"] = min(m[3]["age_days"] for m in eligible) if dates else None
    payload["price_as_of"] = base.get("price_as_of")
    payload["historical_fair_values_point_in_time_safe"] = False
    if dispersion is not None and dispersion > 25:
        warnings.append("Eligible models disagree by more than 25%; use the model range instead of treating the composite as precise.")
    payload["valuation_warnings"] = warnings
    for _, model, _, _, _ in blocks:
        if model:
            prediction = _positive_number(model.get("pred_price"))
            model["gain_pct"] = round((prediction / quote - 1) * 100, 2) if prediction and quote else None
    for row in payload.get("fy_rows", []):
        if row.get("is_current_prediction"):
            row.update(composite=payload["composite_fair_price"], bucket=payload["valuation_bucket"],
                       comp_gain=payload["composite_gain_pct"], actual_price=quote)
    payload["status"] = "SUCCESS" if composite else "NO_RELIABLE_MODELS"
    return _json_safe(payload)


def compute_fair_value(
    ticker: str,
    fy_start: int = 2014,
    force_refresh: bool = False,
    include_other_income: bool = True,
    current_price: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive multi-model Fair Value analysis for a stock.
    Returns complete annual, quarterly, and historical trend data.
    """
    sym = ticker.strip().upper().removeprefix("$")
    if sym.endswith((".NS", ".BO")):
        sym = sym[:-3]
    cache_key = f"{sym}_{fy_start}_{bool(include_other_income)}"
    if (not force_refresh and cache_key in _MEM_CACHE and
            time.time() - _CACHE_TIMES.get(cache_key, 0) < CACHE_TTL_SECONDS):
        cached = copy.deepcopy(_MEM_CACHE[cache_key])
        return _valuation_summary(cached, cached.pop("_base_metadata"), current_price)

    if _HAS_ENGINES:
        try:
            # 1. Base analysis
            base = _analyze_ticker(
                sym,
                fy_start=fy_start,
                force=force_refresh,
                include_other_income=include_other_income,
            )

            if "error" not in base and base.get("op"):
                # 2. Extract FY rows
                # Run the fv_pine_json logic synchronously for fy_rows
                from fv_router import _current_fy, _row_for_fy
                cur_fy = _current_fy()

                fy_years = set()
                for yr_str in base.get("op", {}).get("historical", {}):
                    try:
                        fy_years.add(int(yr_str))
                    except ValueError:
                        pass
                if base.get("has_sales"):
                    for yr_str in base.get("sales", {}).get("historical", {}):
                        try:
                            fy_years.add(int(yr_str))
                        except ValueError:
                            pass

                fy_rows = []
                for fy in sorted(fy_years):
                    if fy < fy_start:
                        continue
                    r = _row_for_fy(base, fy)
                    op_h = base.get("op", {}).get("historical", {}).get(str(fy), {})
                    sales_h = base.get("sales", {}).get("historical", {}).get(str(fy), {})
                    r["profit"] = op_h.get("profit")
                    r["sales"] = sales_h.get("sales")
                    r["actual_price"] = op_h.get("actual") or sales_h.get("actual")
                    fy_rows.append(r)

                if cur_fy not in fy_years:
                    cur_row = _row_for_fy(base, cur_fy)
                    cur_row["fair_op"] = base.get("op", {}).get("pred_price", 0.0)
                    cur_row["fair_s"] = base.get("sales", {}).get("pred_price", 0.0) if base.get("has_sales") else 0.0
                    cur_row["fair_t"] = base.get("ttm", {}).get("pred_price", 0.0) if base.get("has_ttm") else 0.0
                    cur_row["composite"] = base.get("composite_fair_price", 0.0) or 0.0
                    cur_row["bucket"] = base.get("valuation_bucket", "FAIR")
                    cur_row["comp_gain"] = base.get("composite_gain_pct", 0.0) or 0.0
                    cur_row["actual_price"] = base.get("current_price")
                    cur_row["is_current_prediction"] = True
                    fy_rows.append(cur_row)

                # 3. Format Quarterly TTM Table
                quarterly_rows = []
                ttm_hist = base.get("ttm", {}).get("historical", {})
                ttm_sales_hist = base.get("ttm", {}).get("sales", {}).get("historical", {})
                for date_key in sorted(set(ttm_hist) | set(ttm_sales_hist)):
                    q_data = ttm_hist.get(date_key, {})
                    qs_data = ttm_sales_hist.get(date_key, {})
                    quarterly_rows.append({
                        "quarter": date_key,
                        "price": q_data.get("actual") or q_data.get("price") or qs_data.get("actual"),
                        "ttm_profit": q_data.get("ttm"),
                        "fair_value": q_data.get("fair"),
                        "misprice_pct": q_data.get("misprice_pct"),
                        "status": q_data.get("status"),
                        "ttm_sales": qs_data.get("ttm_sales"),
                        "fair_sales": qs_data.get("fair"),
                        "sales_misprice_pct": qs_data.get("misprice_pct"),
                    })

                # Assemble complete rich object
                full_payload = {
                    "ticker": sym,
                    "current_price": base.get("current_price"),
                    "composite_fair_price": base.get("composite_fair_price"),
                    "composite_gain_pct": base.get("composite_gain_pct"),
                    "valuation_bucket": base.get("valuation_bucket", "N/A"),
                    "model_count": base.get("model_count", 0),
                    "is_bank": base.get("is_bank", False),
                    "profit_label": base.get("profit_label", "Operating Profit"),
                    "models": {
                        "op": {
                            "label": base.get("profit_label", "Operating Profit"),
                            "pred_price": base.get("op", {}).get("pred_price"),
                            "r2": base.get("op", {}).get("r2"),
                            "gain_pct": base.get("op", {}).get("gain_pct"),
                            "equation": base.get("op", {}).get("eq"),
                            "historical": base.get("op", {}).get("historical", {}),
                            "series": base.get("op", {}).get("series", []),
                        },
                        "sales": {
                            "label": base.get("sales", {}).get("label", "Sales"),
                            "pred_price": base.get("sales", {}).get("pred_price") if base.get("has_sales") else None,
                            "r2": base.get("sales", {}).get("r2") if base.get("has_sales") else None,
                            "gain_pct": base.get("sales", {}).get("gain_pct") if base.get("has_sales") else None,
                            "equation": base.get("sales", {}).get("eq") if base.get("has_sales") else None,
                            "historical": base.get("sales", {}).get("historical", {}) if base.get("has_sales") else {},
                            "series": base.get("sales", {}).get("series", []) if base.get("has_sales") else [],
                        } if base.get("has_sales") else None,
                        "ttm": {
                            "label": base.get("ttm", {}).get("label", "TTM Model"),
                            "pred_price": base.get("ttm", {}).get("pred_price") if base.get("has_ttm") else None,
                            "r2": base.get("ttm", {}).get("r2") if base.get("has_ttm") else None,
                            "gain_pct": base.get("ttm", {}).get("gain_pct") if base.get("has_ttm") else None,
                            "equation": base.get("ttm", {}).get("eq") if base.get("has_ttm") else None,
                            "historical": base.get("ttm", {}).get("historical", {}) if base.get("has_ttm") else {},
                            "series": base.get("ttm", {}).get("series", []) if base.get("has_ttm") else [],
                            "sales": base.get("ttm", {}).get("sales") if base.get("has_ttm") else None,
                        } if base.get("has_ttm") else None,
                    },
                    "fy_rows": fy_rows,
                    "quarterly_rows": quarterly_rows,
                    "price_history": base.get("price_history", []),
                    "status": "SUCCESS",
                }

                full_payload = _valuation_summary(full_payload, base)
                _MEM_CACHE[cache_key] = copy.deepcopy(full_payload)
                _MEM_CACHE[cache_key]["_base_metadata"] = copy.deepcopy(base)
                _CACHE_TIMES[cache_key] = time.time()
                return _valuation_summary(copy.deepcopy(full_payload), base, current_price)

        except Exception as e:
            print(f"[FairValue] Exception computing fair value for {sym}: {e}")

    # Fallback
    fallback = {
        "ticker": sym,
        "current_price": None,
        "composite_fair_price": None,
        "composite_gain_pct": None,
        "valuation_bucket": "N/A",
        "model_count": 0,
        "models": {},
        "fy_rows": [],
        "quarterly_rows": [],
        "price_history": [],
        "status": "NO_DATA",
        "valuation_confidence": "unavailable",
        "valuation_as_of": None,
        "valuation_age_days": None,
        "valuation_warnings": ["No usable aligned fundamental and price observations are available."],
        "historical_fair_values_point_in_time_safe": False,
    }
    return fallback


def get_cached_fair_value(ticker: str, current_price: Optional[float] = None,
                          fy_start: int = 2014, include_other_income: bool = True) -> Optional[Dict[str, Any]]:
    """Read existing valuation without starting network work for a price chart."""
    sym = ticker.strip().upper().removeprefix("$").removesuffix(".NS").removesuffix(".BO")
    key = f"{sym}_{fy_start}_{bool(include_other_income)}"
    if key not in _MEM_CACHE or time.time() - _CACHE_TIMES.get(key, 0) >= CACHE_TTL_SECONDS:
        return None
    cached = copy.deepcopy(_MEM_CACHE[key])
    base = cached.pop("_base_metadata", None)
    return _valuation_summary(cached, base, current_price) if base else None


def batch_enrich_fair_value(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """Batch compute composite fair values."""
    res = {}
    for t in tickers:
        res[t.strip().upper()] = compute_fair_value(t)
    return res
