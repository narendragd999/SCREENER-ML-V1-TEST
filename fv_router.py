"""
fv_router.py — Fair Value Bridge for TradingView Pine Script
============================================================
Adds two endpoints:

  GET /api/fv/pine?ticker=RELIANCE&fy=2022
      → Returns a plain CSV line Pine Script can parse:
        "FY,fair_op,r2_op,fair_s,r2_s,fair_t,r2_t,composite,bucket,misprice"
        e.g. "2022,1842.50,0.923,1790.00,0.851,0.00,0.000,1820.00,UNDERVALUED,+12.3"

  GET /api/fv/pine/json?ticker=RELIANCE
      → Full JSON payload: all FY rows for every model
        Used by the debug overlay + the HTML test harness below

Mount this in main.py:
    from fv_router import router as fv_router
    app.include_router(fv_router)
"""

import asyncio
import datetime as dt_module
import math
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

# Re-use the exact same analysis function from sma_router
# Adjust import path to match your project structure
from sma_router import _analyze_ticker

router = APIRouter()

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _current_fy() -> int:
    """Return the current Indian financial year (Apr–Mar).
       e.g. April 2024 → FY 2024,  January 2024 → FY 2023
    """
    today = dt_module.datetime.now(dt_module.timezone(dt_module.timedelta(hours=5, minutes=30))).date()
    return today.year if today.month >= 4 else today.year - 1


def _row_for_fy(result: dict, fy: int) -> dict:
    """
    Extract per-FY fair values from all three models (OP / Sales / TTM).
    TTM history keys are quarterly dates (YYYY-MM-DD); we pick the last
    quarter that falls within the requested FY (Apr–Mar window).
    Returns a dict ready for CSV serialisation.
    """
    fy_str = str(fy)

    # ── OP model ──────────────────────────────────────────────
    hist_op = result.get("op", {}).get("historical", {})
    op_row  = hist_op.get(fy_str, {})
    fair_op = op_row.get("fair", 0.0)
    r2_op   = result.get("op", {}).get("r2", 0.0)
    mp_op   = op_row.get("misprice_pct", 0.0)

    # ── Sales model ───────────────────────────────────────────
    fair_s = r2_s = mp_s = 0.0
    if result.get("has_sales"):
        hist_s = result.get("sales", {}).get("historical", {})
        s_row  = hist_s.get(fy_str, {})
        fair_s = s_row.get("fair", 0.0)
        r2_s   = result.get("sales", {}).get("r2", 0.0)
        mp_s   = s_row.get("misprice_pct", 0.0)

    # ── TTM model — find last quarter inside this FY ──────────
    fair_t = r2_t = mp_t = 0.0
    ttm_reference = None
    if result.get("has_ttm"):
        hist_t = result.get("ttm", {}).get("historical", {})
        # FY window: Apr 1 of `fy` → Mar 31 of `fy+1`
        fy_start = dt_module.date(fy, 4, 1)
        fy_end   = dt_module.date(fy + 1, 3, 31)
        best_date = None
        for date_str, row in hist_t.items():
            try:
                d = dt_module.date.fromisoformat(date_str)
            except ValueError:
                continue
            if fy_start <= d <= fy_end:
                if best_date is None or d > best_date:
                    best_date = d
                    fair_t    = row.get("fair", 0.0)
                    r2_t      = result.get("ttm", {}).get("r2", 0.0)
                    mp_t      = row.get("misprice_pct", 0.0)
                    ttm_reference = row.get("actual")

    # ── Composite (weighted by R²) ────────────────────────────
    entries = []
    def valid(fair, r2, block, minimum):
        return (isinstance(fair, (int, float)) and isinstance(r2, (int, float)) and
                math.isfinite(fair) and fair > 0 and math.isfinite(r2) and 0.25 < r2 <= 1 and
                block.get("sample_count", 0) >= minimum and block.get("slope", 0) > 0 and
                block.get("positive_fundamentals") is True)
    if valid(fair_op, r2_op, result.get("op", {}), 5): entries.append((fair_op, r2_op, 1.0))
    if valid(fair_s, r2_s, result.get("sales", {}), 5): entries.append((fair_s, r2_s, 0.8))
    if valid(fair_t, r2_t, result.get("ttm", {}), 8): entries.append((fair_t, r2_t, 1.2))

    composite = 0.0
    if entries:
        total_w   = sum(w * mw for _, w, mw in entries)
        composite = sum(f * w * mw for f, w, mw in entries) / total_w

    # A historical FY row must be compared with that FY's actual close.
    # Using today's close makes the historical bucket change every day.
    reference = op_row.get("actual")
    if not reference and result.get("has_sales"):
        reference = result.get("sales", {}).get("historical", {}).get(fy_str, {}).get("actual")
    if not reference:
        reference = ttm_reference
    comp_gain = round((composite - reference) / reference * 100, 1) if composite and reference else 0.0

    # Pick misprice relative to current price (Pine wants the gap, not historical)
    # For historical bars we use the OP misprice as representative
    misprice = round((reference - composite) / composite * 100, 1) if composite and reference else 0.0

    bucket = (
        "NODATA" if not composite or not reference else
        "UNDERVALUED" if comp_gain > 15 else
        "OVERVALUED"  if comp_gain < -15 else
        "FAIR"
    )

    return {
        "fy":        fy,
        "fair_op":   round(fair_op,   2),
        "r2_op":     round(r2_op,     3),
        "fair_s":    round(fair_s,    2),
        "r2_s":      round(r2_s,      3),
        "fair_t":    round(fair_t,    2),
        "r2_t":      round(r2_t,      3),
        "composite": round(composite, 2),
        "bucket":    bucket,
        "misprice":  round(misprice,  1),
        "comp_gain": comp_gain,
        "actual_price": reference,
        "fit_scope": "RETROSPECTIVE_IN_SAMPLE",
        "point_in_time_safe": False,
    }


# ─────────────────────────────────────────────────────────────
#  PINE-FRIENDLY PLAIN-TEXT ENDPOINT
# ─────────────────────────────────────────────────────────────

@router.get(
    "/api/fv/pine",
    response_class=PlainTextResponse,
    summary="Single-line CSV for Pine Script request.security()",
)
async def fv_pine_csv(
    ticker: str = Query(..., description="NSE symbol e.g. RELIANCE"),
    fy:     int = Query(0,   description="Indian FY year (2022 = FY22). 0 = current FY"),
):
    """
    Returns exactly one CSV line (no header) that Pine Script can split():
        FY,fair_op,r2_op,fair_s,r2_s,fair_t,r2_t,composite,bucket,misprice
        2022,1842.50,0.923,1790.00,0.851,0.00,0.000,1820.00,UNDERVALUED,+12.3

    Pine usage (see fair_value.pine):
        url  = "http://localhost:8002/api/fv/pine?ticker=" + syminfo.ticker + "&fy=" + str.tostring(fy_year)
        raw  = request.security(url, "D", close)
        cols = str.split(raw, ",")
        fv   = str.tonumber(array.get(cols, 7))   // composite
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise HTTPException(400, "ticker required")

    if fy == 0:
        fy = _current_fy()

    try:
        result = await asyncio.to_thread(
            _analyze_ticker, ticker, 2014, False, True
        )
    except Exception as exc:
        # Return a safe "no data" row so Pine doesn't crash
        return PlainTextResponse(
            f"{fy},0,0,0,0,0,0,0,NODATA,0",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    if "error" in result:
        return PlainTextResponse(
            f"{fy},0,0,0,0,0,0,0,NODATA,0",
            headers={"Access-Control-Allow-Origin": "*"},
        )

    row = _row_for_fy(result, fy)

    misprice_str = (
        f"+{row['misprice']}" if row["misprice"] >= 0 else str(row["misprice"])
    )

    csv_line = (
        f"{row['fy']},"
        f"{row['fair_op']},{row['r2_op']},"
        f"{row['fair_s']},{row['r2_s']},"
        f"{row['fair_t']},{row['r2_t']},"
        f"{row['composite']},"
        f"{row['bucket']},"
        f"{misprice_str}"
    )

    return PlainTextResponse(
        csv_line,
        headers={"Access-Control-Allow-Origin": "*"},
    )


# ─────────────────────────────────────────────────────────────
#  JSON ENDPOINT — full history for all FYs
# ─────────────────────────────────────────────────────────────

@router.get(
    "/api/fv/pine/json",
    summary="Full per-FY fair value history (JSON) — for debug overlay & test harness",
)
async def fv_pine_json(
    ticker:   str = Query(..., description="NSE symbol"),
    fy_start: int = Query(2014, description="First FY to include"),
):
    """
    Returns ALL yearly fair values so the test harness can
    draw the stepped FV line over price history.
    """
    ticker = ticker.strip().upper()

    try:
        result = await asyncio.to_thread(
            _analyze_ticker, ticker, fy_start, False, True
        )
    except Exception as exc:
        raise HTTPException(500, str(exc))

    if "error" in result:
        raise HTTPException(404, result["error"])

    current_fy  = _current_fy()
    # Collect all FY years that appear in any model's historical dict
    fy_years = set()
    for yr_str in result.get("op", {}).get("historical", {}):
        try: fy_years.add(int(yr_str))
        except ValueError: pass
    if result.get("has_sales"):
        for yr_str in result.get("sales", {}).get("historical", {}):
            try: fy_years.add(int(yr_str))
            except ValueError: pass

    rows = []
    for fy in sorted(fy_years):
        if fy < fy_start:
            continue
        rows.append(_row_for_fy(result, fy))

    # Also append current FY (uses latest model prediction)
    if current_fy not in fy_years:
        cur_row = _row_for_fy(result, current_fy)
        # For current FY use the model's forward prediction
        cur_row["fair_op"]   = result.get("op",    {}).get("pred_price", 0.0)
        cur_row["fair_s"]    = result.get("sales",  {}).get("pred_price", 0.0) if result.get("has_sales") else 0.0
        cur_row["fair_t"]    = result.get("ttm",    {}).get("pred_price", 0.0) if result.get("has_ttm")   else 0.0
        cur_row["composite"] = result.get("composite_fair_price", 0.0) or 0.0
        cur_row["bucket"]    = result.get("valuation_bucket", "FAIR")
        cur_row["comp_gain"] = result.get("composite_gain_pct", 0.0) or 0.0
        rows.append(cur_row)

    return {
        "ticker":        ticker,
        "current_price": result.get("current_price"),
        "composite_fair_price": result.get("composite_fair_price"),
        "valuation_bucket":     result.get("valuation_bucket"),
        "composite_gain_pct":   result.get("composite_gain_pct"),
        "price_history": result.get("price_history", []),
        "fy_rows":       rows,
    }
