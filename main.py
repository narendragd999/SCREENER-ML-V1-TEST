"""
Equity 9-EMA Screener & Fair Value 3% Strategy Platform
======================================================
100% Pure Equity Architecture:
  - 9-EMA Breakout & Higher-High Confirmation Screener
  - Aligned Fundamental Fair Value Valuation Engine (R²-weighted regressions)
  - 3% Target Strategy (+3.0% Gain Target & 9-EMA Close Invalidation)
  - Institutional Walk-Forward Historical Backtesting Engine
  - Real-Time Signals & Portfolio Tracker
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Literal
from pathlib import Path
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.data_fetcher import (
    load_ticker_registry,
    get_tickers_list,
    fetch_ticker_ohlcv,
    normalize_ticker,
    clean_ohlcv,
    HistoryUnavailableError,
)
from core.fair_value import compute_fair_value, batch_enrich_fair_value, get_cached_fair_value
from core.screener import run_ema9_screener, analyze_ticker_ema9
from core.strategy import compute_trend, indicator_frame
from core.backtest import run_equity_backtest
from core.duration_backtest import run_duration_backtest
from core.swing_reversal import analyze_stock_swing_reversal, scan_swing_reversals
from core.signals import (
    load_signals,
    save_signals,
    add_or_update_signal,
    delete_signal,
    clear_signals,
    refresh_active_signals,
    get_signals_summary,
)
from core.scan_jobs import router as scan_jobs_router, register_job_handler

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CONFIG_FILE = BASE_DIR / "config.json"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="Equity 9-EMA & Fair Value 3% Screener",
    description="High-precision equity momentum & valuation trading system",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(scan_jobs_router)


def _read_config() -> Dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "strategy": "3_PERCENT_EQUITY_EMA9_FAIR_VALUE",
        "target_pct": 3.0,
        "ema_period": 9,
        "trend_sma_period": 50,
        "require_uptrend": True,
        "allow_sideways": False,
        "max_candles_ago": 10,
        "fair_value_enabled": True,
        "default_ticker_source": "fno",
    }


def _write_config(cfg: Dict[str, Any]) -> bool:
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
#  REQUEST SCHEMAS
# ─────────────────────────────────────────────────────────────
class ScreenerScanRequest(BaseModel):
    tickers: Optional[List[str]] = None
    source: Literal["fno", "all", "custom"] = "fno"
    lookback_days: int = Field(365, ge=60, le=3650)
    max_candles_ago: int = Field(10, ge=1, le=60)
    target_pct: float = Field(3.0, gt=0, le=30, allow_inf_nan=False)
    require_uptrend: bool = True
    allow_sideways: bool = False
    require_fair_value_undervalued: bool = False
    actionable_only: bool = False
    min_setup_score: int = Field(0, ge=0, le=100)
    require_market_uptrend: bool = False
    require_relative_strength: bool = False
    enrich_watchlist: bool = False


class BacktestRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1)
    years: int = Field(2, ge=1, le=5)
    target_pct: float = Field(3.0, ge=0.5, le=20.0)
    max_hold_candles: int = Field(20, ge=5, le=60)
    compare_fair_value: bool = True
    slippage_bps: float = Field(5, ge=0, le=200, allow_inf_nan=False)
    commission_bps: float = Field(10, ge=0, le=200, allow_inf_nan=False)
    require_uptrend: bool = True


class DurationBacktestRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1)
    years: int = Field(5, ge=1, le=10)
    target_pct: float = Field(3.0, ge=0.5, le=20, allow_inf_nan=False)
    max_hold_sessions: int = Field(90, ge=1, le=250)
    mode: Literal["legacy_target_study", "duration_next_open"] = "legacy_target_study"
    require_uptrend: bool = False
    slippage_bps: float = Field(5, ge=0, le=200, allow_inf_nan=False)
    commission_bps: float = Field(10, ge=0, le=200, allow_inf_nan=False)
    research_filters: bool = False
    avoid_near_resistance: bool = False
    require_pullback: bool = False
    include_current_fair_value: bool = False


class SignalSaveRequest(BaseModel):
    ticker: str
    entry_price: float = Field(..., gt=0, allow_inf_nan=False)
    target_3pct: Optional[float] = Field(None, gt=0, allow_inf_nan=False)
    target_pct: float = Field(3.0, gt=0, le=100, allow_inf_nan=False)
    stop_loss_price: Optional[float] = Field(None, gt=0, allow_inf_nan=False)
    entry_date: Optional[str] = None
    entry_at_open: bool = False
    stop_mode: Literal["EMA9_CLOSE", "PRICE"] = "EMA9_CLOSE"
    max_hold_candles: int = Field(20, ge=1, le=250)
    slippage_bps: float = Field(5, ge=0, le=200, allow_inf_nan=False)
    commission_bps: float = Field(10, ge=0, le=200, allow_inf_nan=False)
    breakout_date: Optional[str] = None
    confirm_date: Optional[str] = None
    composite_fair_price: Optional[float] = None
    valuation_bucket: Optional[str] = None
    notes: Optional[str] = ""


class PositionSizeRequest(BaseModel):
    capital: float = Field(..., gt=0, allow_inf_nan=False)
    risk_pct: float = Field(1, gt=0, le=10, allow_inf_nan=False)
    entry_price: float = Field(..., gt=0, allow_inf_nan=False)
    stop_price: float = Field(..., gt=0, allow_inf_nan=False)
    target_price: float = Field(..., gt=0, allow_inf_nan=False)
    round_trip_cost_bps: float = Field(30, ge=0, le=500, allow_inf_nan=False)


class RecommendedJobRequest(BaseModel):
    source: Literal["fno", "all"] = "fno"
    limit: int = Field(10, ge=1, le=100)


class SwingJobRequest(BaseModel):
    source: Literal["fno", "all"] = "fno"
    min_upside: float = Field(15, ge=0, le=100, allow_inf_nan=False)
    max_dist_floor: float = Field(14, gt=0, le=50, allow_inf_nan=False)


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the redesigned modern single-page equity dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h2>Dashboard is loading...</h2>")


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Equity 9-EMA 3% Screener",
        "version": "2.0.0",
        "strategy": "3% Target + Aligned Fair Value",
    }


@app.get("/api/config")
async def get_config():
    return _read_config()


@app.post("/api/config")
async def update_config(cfg: Dict[str, Any] = Body(...)):
    current = _read_config()
    current.update(cfg)
    if _write_config(current):
        return {"status": "success", "config": current}
    raise HTTPException(500, "Failed to save configuration")


@app.get("/api/screener/tickers")
async def api_tickers(source: str = "fno", q: str = "", limit: int = 50):
    """Return autocomplete ticker suggestions with company names."""
    results = get_tickers_list(source, q, limit)
    total_reg = len(load_ticker_registry(source))
    return {"total": total_reg, "tickers": results}


@app.post("/api/screener/scan")
async def api_screener_scan(req: ScreenerScanRequest):
    """
    Run the 9-EMA equity screener across selected tickers or universe.
    Returns prime targets (confluence), technical signals, and full stats.
    """
    try:
        results = await run_ema9_screener(
            tickers=req.tickers,
            source=req.source,
            lookback_days=req.lookback_days,
            max_candles_ago=req.max_candles_ago,
            target_pct=req.target_pct,
            require_uptrend=req.require_uptrend,
            allow_sideways=req.allow_sideways,
            require_fair_value_undervalued=req.require_fair_value_undervalued,
            actionable_only=req.actionable_only,
            min_setup_score=req.min_setup_score,
            require_market_uptrend=req.require_market_uptrend,
            require_relative_strength=req.require_relative_strength,
            enrich_watchlist=req.enrich_watchlist,
        )
        return results
    except Exception as e:
        raise HTTPException(500, f"Screener execution error: {e}")


@app.get("/api/screener/quick-recommended")
async def api_quick_recommended(source: Literal["fno", "all"] = "fno", limit: int = Query(10, ge=1, le=100)):
    return await _recommended_scan(source, limit)


async def _recommended_scan(source: str, limit: int, progress=None):
    """
    Fast query for currently recommended 3% picks:
    Uptrend + Confirmed 9-EMA Breakout + Undervalued vs Fair Value.
    """
    try:
        # Use top liquid leaders for instant initial response
        sample_tickers = load_ticker_registry(source)["symbol"].head(60).tolist()
        scan_res = await run_ema9_screener(
            tickers=sample_tickers,
            source=source,
            lookback_days=365,
            max_candles_ago=10,
            target_pct=3.0,
            require_uptrend=True,
            progress=progress,
        )
        prime = scan_res.get("prime_targets", [])
        return {
            "recommended": prime[:limit],
            "total_prime": len(prime),
            "scanned_count": len(sample_tickers),
            "universe_size": len(load_ticker_registry(source)),
            "scan_scope": "First 60 symbols in the selected registry; use full screener for all symbols",
            "market_context": scan_res.get("market_context"),
            "technical_only": scan_res.get("technical_only", [])[:limit],
            "watchlist": scan_res.get("watchlist", [])[:limit],
            "data_failures": scan_res.get("data_failures", []),
        }
    except Exception as e:
        raise HTTPException(500, f"Quick recommended scan error: {e}")


@app.get("/api/fair-value/{ticker}")
async def api_fair_value(ticker: str, fy_start: int = 2014, force: bool = False):
    """Get complete Fair Value valuation model breakdown for a ticker."""
    try:
        fv = await asyncio.to_thread(compute_fair_value, ticker, fy_start, force)
        return fv
    except Exception as e:
        raise HTTPException(500, f"Fair Value calculation error: {e}")


@app.post("/api/position-size")
async def api_position_size(req: PositionSizeRequest):
    """Size whole-share, unlevered entries using a user-selected loss budget."""
    if req.stop_price >= req.entry_price or req.target_price <= req.entry_price:
        raise HTTPException(400, "Long positions require stop < entry < target")
    cost = req.entry_price * req.round_trip_cost_bps / 10000
    unit_risk = req.entry_price - req.stop_price + cost
    unit_reward = req.target_price - req.entry_price - cost
    risk_budget = req.capital * req.risk_pct / 100
    risk_quantity = int(risk_budget // unit_risk)
    # Reserve the estimated transaction cost instead of exceeding available cash.
    cash_quantity = int(req.capital // (req.entry_price + cost))
    quantity = max(0, min(risk_quantity, cash_quantity))
    return {"quantity": quantity, "capital_required": round(quantity * (req.entry_price + cost), 2),
            "position_value": round(quantity * req.entry_price, 2),
            "risk_budget": round(risk_budget, 2), "max_loss": round(quantity * unit_risk, 2),
            "potential_net_gain": round(quantity * unit_reward, 2),
            "net_reward_risk": round(unit_reward / unit_risk, 2),
            "estimated_cost": round(quantity * cost, 2),
            "limiting_factor": "risk_budget" if risk_quantity <= cash_quantity else "available_cash",
            "assumption": "Estimated stop fill; overnight gaps and EMA close exits can exceed the loss budget"}


@app.get("/api/analysis/{ticker}")
async def api_stock_analysis(ticker: str, include_valuation: bool = False):
    """Useful stock diagnostics even when no recent breakout is present."""
    sym = normalize_ticker(ticker)
    df = await _load_completed_history(sym, 400)
    indicators = indicator_frame(df)
    row = indicators.iloc[-1]
    numeric = lambda x: round(float(x), 4) if pd.notna(x) and np.isfinite(x) else None
    price = float(row.Close)
    setup = analyze_ticker_ema9(sym, df, 30)
    fv = (await asyncio.to_thread(compute_fair_value, sym, current_price=price)
          if include_valuation else get_cached_fair_value(sym, current_price=price)) or {}
    reversal = await asyncio.to_thread(analyze_stock_swing_reversal, sym, df,
                                       min_upside_pct=0, max_dist_to_floor_pct=25,
                                       include_fair_value=False)
    return {"ticker": sym, "as_of": str(df.index[-1].date()), "current_price": price,
            "trend": compute_trend(df),
            "indicators": {key: numeric(row[key]) for key in
                           ("ema9", "sma50", "sma200", "atr14", "rsi14", "volume_ratio", "turnover20")},
            "range_52week": {"high": numeric(df.High.iloc[-252:].max()),
                             "low": numeric(df.Low.iloc[-252:].min()),
                             "sessions_available": min(len(df), 252)},
            "returns_pct": {str(n): numeric((price / df.Close.iloc[-n-1] - 1) * 100)
                            if len(df) > n else None for n in (5, 20, 60, 120)},
            "ema_setup": {k: v for k, v in (setup or {}).items() if k != "candles"} or None,
            "swing_setup": reversal, "fair_value": {k: fv.get(k) for k in
                ("composite_fair_price", "composite_gain_pct", "valuation_confidence", "valuation_as_of",
                 "valuation_warnings", "model_count", "model_dispersion_pct")},
            "completed_candles_only": True, "history_bars": len(df),
            "data_source": df.attrs.get("data_source"),
            "history_warnings": (df.attrs.get("history_warnings", []) +
                ([f"Only {len(df)} completed daily bars are available; EMA breakout confirmation needs 31 and a verified long-term trend needs 200."] if len(df) < 200 else [])),
            "valuation_status": "available" if fv else "deferred",
            "valuation_note": "Open Valuation for full fundamental analysis; price research does not wait for valuation downloads." if not fv else None}


async def _load_completed_history(ticker: str, lookback_days: int):
    try:
        df = await asyncio.to_thread(fetch_ticker_ohlcv, ticker, "1d", lookback_days,
                                      min_bars=1, allow_cached_shorter=True, raise_on_error=True)
    except HistoryUnavailableError as exc:
        raise HTTPException(503, f"Price history is temporarily unavailable for {ticker}. {exc} Retry the chart or check your internet connection.")
    df = clean_ohlcv(df)
    if df.empty:
        raise HTTPException(503, f"Price data is temporarily unavailable for {ticker}; no completed valid candles were returned. Retry or check your internet connection.")
    return df


@app.post("/api/backtest/run")
async def api_backtest_run(req: BacktestRequest):
    """
    Execute walk-forward backtest across historical daily bars.
    Returns full stats, equity curve, monthly breakdown, and trade log.
    """
    try:
        res = await run_equity_backtest(
            tickers=req.tickers,
            years=req.years,
            target_pct=req.target_pct,
            max_hold_candles=req.max_hold_candles,
            compare_fair_value=req.compare_fair_value,
            slippage_bps=req.slippage_bps,
            commission_bps=req.commission_bps,
            require_uptrend=req.require_uptrend,
        )
        return res
    except Exception as e:
        raise HTTPException(500, f"Backtest execution error: {e}")


@app.post("/api/backtest/duration")
async def api_duration_backtest(req: DurationBacktestRequest):
    """Separate no-stop target study and next-open duration simulation."""
    try:
        return await run_duration_backtest(**req.model_dump())
    except Exception as e:
        raise HTTPException(500, f"Duration study error: {e}")


@app.get("/api/backtest/presets")
async def api_backtest_presets():
    """Return standard ticker presets for easy one-click backtesting."""
    fno_df = load_ticker_registry("fno")
    fno_tickers = fno_df["symbol"].tolist() if not fno_df.empty else []

    top_20 = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "BHARTIARTL",
        "SBIN", "ITC", "HINDUNILVR", "LT", "BAJFINANCE", "TATAMOTORS",
        "SUNPHARMA", "TITAN", "MARUTI", "AXISBANK", "ONGC", "NTPC",
        "KALYANKJIL", "DIXON"
    ]

    return {
        "presets": [
            {"name": "Top 20 Liquid Leaders", "tickers": top_20},
            {"name": "Full F&O Universe (~200)", "tickers": fno_tickers[:200]},
            {"name": "Banking & Financials", "tickers": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "BAJFINANCE"]},
            {"name": "IT & Tech Growth", "tickers": ["TCS", "INFY", "WIPRO", "HCLTECH", "PERSISTENT", "COFORGE"]},
        ]
    }


@app.get("/api/signals")
async def api_get_signals():
    """Get all saved signals and current portfolio summary."""
    signals = load_signals()
    summary = get_signals_summary()
    return {"signals": signals, "summary": summary}


@app.post("/api/signals")
async def api_save_signal(sig: SignalSaveRequest):
    """Save or update a tracked signal."""
    data = sig.model_dump(exclude_none=True)
    try:
        updated_signals = add_or_update_signal(data)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"status": "saved", "signals": updated_signals}


@app.delete("/api/signals/{ticker}")
async def api_delete_signal(ticker: str):
    """Remove a tracked signal."""
    updated = delete_signal(ticker)
    return {"status": "deleted", "signals": updated}


@app.delete("/api/signals")
async def api_clear_signals():
    """Clear all saved signals."""
    clear_signals()
    return {"status": "cleared"}


@app.post("/api/signals/refresh")
async def api_refresh_signals():
    """Refresh live quotes and mark-to-market status for active signals."""
    refreshed = await refresh_active_signals()
    summary = get_signals_summary()
    return {"status": "refreshed", "signals": refreshed, "summary": summary}


@app.get("/api/chart/{ticker}")
async def api_chart_data(ticker: str, lookback_days: int = Query(365, ge=30, le=3650), trade_id: Optional[str] = None):
    """
    Get daily candlestick bars with 9-EMA, 50-SMA, 3% Target line,
    and Stop Loss line for lightweight charting.
    """
    sym = normalize_ticker(ticker)
    df = indicator_frame(await _load_completed_history(sym, max(365, lookback_days)))

    current_price = round(float(df["Close"].iloc[-1]), 2)
    current_ema9 = round(float(df["ema9"].iloc[-1]), 2)
    target_price = round(current_price * 1.03, 2)
    stop_price = current_ema9
    setup = analyze_ticker_ema9(sym, df)
    target_basis = "new_entry_estimate_from_latest_completed_close"
    if setup:
        target_price = setup["target_3pct"]
        target_basis = setup["entry_basis"]
    if trade_id:
        saved = next((s for s in load_signals() if s.get("trade_id") == trade_id and normalize_ticker(s.get("ticker")) == sym), None)
        if saved is None:
            raise HTTPException(404, "Tracked entry does not exist for this ticker")
    else:
        saved = next((s for s in load_signals() if normalize_ticker(s.get("ticker")) == sym and s.get("status") in ("ACTIVE", "STOP_PENDING", "TIME_EXIT_PENDING")), None)
    if saved:
        target_price = saved.get("target_3pct") or target_price
        stop_price = (saved.get("stop_loss_price") or stop_price) if saved.get("stop_mode") == "PRICE" else current_ema9
        target_basis = "saved_trade_entry"

    candles = []
    for idx, row in df.iterrows():
        d_str = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
        candles.append({
            "time": d_str,
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row.get("Volume", 0)),
            "ema9": round(float(row["ema9"]), 2) if not pd.isna(row["ema9"]) else None,
            "sma50": round(float(row["sma50"]), 2) if not pd.isna(row["sma50"]) else None,
        })

    fv = get_cached_fair_value(sym, current_price=current_price) or {}

    return {
        "ticker": sym,
        "current_price": current_price,
        "ema9": current_ema9,
        "target_3pct": target_price,
        "stop_loss": stop_price,
        "fair_value": fv.get("composite_fair_price"),
        "valuation_bucket": fv.get("valuation_bucket", "N/A"),
        "valuation_confidence": fv.get("valuation_confidence", "unavailable"),
        "valuation_status": "cached" if fv else "deferred",
        "target_basis": target_basis,
        "tracked_entry": {k: saved.get(k) for k in ("trade_id", "entry_date", "entry_price", "target_pct", "status", "stop_mode", "exit_date", "exit_price")} if saved else None,
        "setup": {k: v for k, v in (setup or {}).items() if k != "candles"},
        "as_of": str(df.index[-1].date()),
        "candles": candles,
    }


# ─────────────────────────────────────────────────────────────
#  15%+ SWING REVERSAL & DEEP SUPPORT ENGINE ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/api/swing-reversal/scan")
async def api_swing_reversal_scan(
    min_upside: float = Query(15.0, ge=0, le=100),
    max_dist_floor: float = Query(14.0, gt=0, le=50),
    source: Literal["fno", "all"] = "fno",
):
    return await _swing_scan(min_upside, max_dist_floor, source)


async def _swing_scan(min_upside, max_dist_floor, source, progress=None):
    """
    Scan universe for 15%+ major bottom reversal setups matching
    patterns like TCS (~2000), HDFCBANK (~680), and INFY (~980).
    """
    results = await asyncio.to_thread(
        scan_swing_reversals,
        min_upside_pct=min_upside,
        max_dist_to_floor_pct=max_dist_floor,
        source=source,
        progress=progress,
    )

    accumulation_count = sum(1 for r in results if r["alert_status"] == "ACCUMULATION_ZONE")
    breakout_count = sum(1 for r in results if r["alert_status"] == "BREAKOUT_CONFIRMED")
    approaching_count = sum(1 for r in results if r["alert_status"] == "APPROACHING_SUPPORT")

    return {
        "status": "SUCCESS",
        "total_candidates": len(results),
        "accumulation_count": accumulation_count,
        "breakout_count": breakout_count,
        "approaching_count": approaching_count,
        "candidates": results,
    }


@app.get("/api/swing-reversal/chart/{ticker}")
async def api_swing_reversal_chart(ticker: str, lookback_days: int = Query(250, ge=80, le=3650)):
    """
    Get detailed candlestick and pattern layout data for plotting
    the exact 15%+ bottom reversal chart pattern.
    """
    sym = normalize_ticker(ticker)
    df = indicator_frame(await _load_completed_history(sym, max(365, lookback_days)))

    # Pattern analysis
    pattern_data = await asyncio.to_thread(
        analyze_stock_swing_reversal,
        sym,
        df,
        min_upside_pct=0.0,
        max_dist_to_floor_pct=25.0,
        lookback_days=lookback_days,
        include_fair_value=False,
    )

    candles = []
    for idx, row in df.iterrows():
        d_str = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
        candles.append({
            "time": d_str,
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row.get("Volume", 0)),
            "ema9": round(float(row["ema9"]), 2) if not pd.isna(row["ema9"]) else None,
            "sma50": round(float(row["sma50"]), 2) if not pd.isna(row["sma50"]) else None,
        })

    return {
        "ticker": sym,
        "candles": candles,
        "pattern": pattern_data,
        "current_price": float(df.Close.iloc[-1]),
        "ema9": float(df.ema9.iloc[-1]),
        "as_of": str(df.index[-1].date()),
        "history_bars": len(df),
        "data_source": df.attrs.get("data_source"),
    }


@app.get("/api/swing-reversal/alerts")
async def api_swing_reversal_alerts():
    """
    Get live suggestions and alerts for stocks in the accumulation zone
    or triggering a 9-EMA reversal off the major support floor.
    """
    results = await asyncio.to_thread(scan_swing_reversals, min_upside_pct=15.0)
    alerts = []
    for r in results:
        if r["alert_status"] in ("ACCUMULATION_ZONE", "BREAKOUT_CONFIRMED"):
            alerts.append({
                "ticker": r["ticker"],
                "alert_status": r["alert_status"],
                "alert_label": r["alert_label"],
                "floor_price": r["floor_price"],
                "current_price": r["current_price"],
                "dist_to_floor_pct": r["dist_to_floor_pct"],
                "target_price": r["target_price"],
                "target_gain_pct": r["target_gain_pct"],
                "stop_loss_price": r["stop_loss_price"],
                "pattern_type": r["pattern_type"],
                "action": r["action_recommendation"],
            })
    return {"status": "SUCCESS", "alerts_count": len(alerts), "alerts": alerts}


async def _job_screener(parameters, progress):
    return await run_ema9_screener(**parameters, progress=progress)


async def _job_recommended(parameters, progress):
    return await _recommended_scan(**parameters, progress=progress)


async def _job_swing(parameters, progress):
    return await _swing_scan(**parameters, progress=progress)


async def _job_backtest(parameters, progress):
    return await run_equity_backtest(**parameters, progress=progress)


async def _job_duration(parameters, progress):
    return await run_duration_backtest(**parameters, progress=progress)


register_job_handler("screener", ScreenerScanRequest, _job_screener)
register_job_handler("recommended", RecommendedJobRequest, _job_recommended)
register_job_handler("swing", SwingJobRequest, _job_swing)
register_job_handler("backtest", BacktestRequest, _job_backtest)
register_job_handler("duration", DurationBacktestRequest, _job_duration)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
