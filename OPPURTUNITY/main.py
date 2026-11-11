"""
main.py — FastAPI Application for Opportunity Screener, Fair Value & No-Stop Backtest
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.data_fetcher import (
    load_ticker_registry,
    fetch_ticker_ohlcv,
    fetch_ticker_ohlcv_async,
    normalize_ticker,
    bare_symbol,
    get_market_context,
)
from core.strategy import compute_trend, indicator_frame
from core.fair_value import compute_stock_fair_value
from core.support_detector import (
    detect_horizontal_support,
    get_all_support_floors,
    detect_structural_support,
    get_all_structural_low_floors,
    detect_powerful_macro_floor,
)
from core.no_stop_engine import run_no_stop_backtest
from core.screener import scan_opportunities, analyze_stock_opportunity
from core.tracker import (
    get_tracked_positions,
    add_tracked_position,
    remove_tracked_position,
    refresh_tracked_positions,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CONFIG_FILE = BASE_DIR / "config.json"

app = FastAPI(
    title="Opportunity: No-Stop Equity Momentum & Fair Value Engine",
    description="High-Conviction Indian Equity Research: Horizontal Support Floors, 9-EMA Momentum, Intrinsic Fair Value & No-Stop Duration Strategy",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Pydantic Schemas
class BacktestRequest(BaseModel):
    mode: Optional[str] = "single"  # "single" or "basket"
    ticker: Optional[str] = "SBIN"
    scope: Optional[int] = 60       # 60, 150, 0 for all 307
    target_pct: float = Field(default=3.0, ge=1.0, le=25.0)
    require_uptrend: bool = False
    period: str = "2y"


class TrackTradeRequest(BaseModel):
    symbol: str
    entry_price: float
    entry_date: Optional[str] = None
    target_pct: float = Field(default=3.0, ge=0.5, le=50.0)
    capital: float = Field(default=100000.0, ge=1000.0)
    notes: Optional[str] = ""


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Opportunity Engine UI Loading...</h1>")


@app.get("/api/health")
async def health_check():
    registry = load_ticker_registry()
    market = get_market_context()
    return {
        "status": "online",
        "app_name": "Opportunity Engine",
        "vetted_tickers_count": len(registry),
        "market_regime": market.get("trend_regime", "UNKNOWN"),
        "market_as_of": market.get("as_of"),
    }


@app.get("/api/market")
async def market_status():
    return get_market_context()


@app.get("/api/tickers")
async def list_tickers():
    return load_ticker_registry()


@app.get("/api/scan")
async def run_scan(
    limit: int = Query(default=60, ge=0, le=350, description="0 for full 307 registry"),
    target_pct: float = Query(default=3.0, ge=0.5, le=20.0),
    max_candles_ago: int = Query(default=15, ge=1, le=30),
    uptrend_only: bool = Query(default=False),
    support_only: bool = Query(default=False, description="Filter for stocks at verified support floors"),
    structural_only: bool = Query(default=False, description="Filter for stocks sitting on Major Structural Low Support Prices"),
    macro_floor_only: bool = Query(default=False, description="Filter for stocks at Rock-Bottom Cycle Floors (High Gain Probability)"),
    pending_only: bool = Query(default=False, description="Hide already-completed TARGET_HIT setups"),
    min_score: int = Query(default=35, ge=0, le=100),
):
    """Scan the 307 vetted stocks with primary focus on Structural Low Support Price."""
    return await scan_opportunities(
        limit=limit,
        target_pct=target_pct,
        max_candles_ago=max_candles_ago,
        uptrend_only=uptrend_only,
        support_only=support_only,
        structural_only=structural_only,
        macro_floor_only=macro_floor_only,
        pending_only=pending_only,
        min_score=min_score,
    )


@app.get("/api/stock/{ticker}")
async def inspect_stock(ticker: str, target_pct: float = 3.0):
    """Detailed multi-factor analysis: Structural Lows, Horizontal Floors, Trend & Fair Value."""
    sym = bare_symbol(ticker).upper()
    df = await fetch_ticker_ohlcv_async(sym, period="3y")
    if df.empty or len(df) < 30:
        raise HTTPException(status_code=404, detail=f"Data unavailable for {ticker}")

    opp = analyze_stock_opportunity(sym, df, target_pct=target_pct, max_candles_ago=20)
    current_price = float(df["Close"].iloc[-1])

    # Fair value detailed models
    fv = compute_stock_fair_value(sym, current_price)

    # Powerful Macro Structural Low (Rock-Bottom Cycle Floor)
    macro_floor = detect_powerful_macro_floor(df, lookback_bars=750)

    # Primary Focus: Structural Low Support
    structural_support = detect_structural_support(df, lookback_bars=750, max_proximity_pct=25.0)
    all_structural_floors = get_all_structural_low_floors(df, lookback_bars=750, min_rally_pct=8.0)

    # General Horizontal Support detection
    support = detect_horizontal_support(df, lookback_bars=600, min_touches=2, cluster_pct=2.5, max_proximity_pct=15.0)
    all_floors = get_all_support_floors(df, lookback_bars=600, min_touches=2, cluster_pct=2.5)

    # Recent completed candles (last 60 bars)
    recent_bars = df.iloc[-60:].copy()
    candles = []
    for idx, row in recent_bars.iterrows():
        candles.append({
            "date": str(idx.date()),
            "open": round(float(row.Open), 2),
            "high": round(float(row.High), 2),
            "low": round(float(row.Low), 2),
            "close": round(float(row.Close), 2),
            "volume": int(row.Volume) if pd.notna(row.Volume) else 0,
        })

    return {
        "symbol": sym,
        "opportunity": opp,
        "fair_value": fv,
        "macro_floor": macro_floor,
        "structural_support": structural_support,
        "all_structural_floors": all_structural_floors,
        "support": support,
        "all_floors": all_floors,
        "candles": candles,
        "as_of": str(df.index[-1].date()),
    }


@app.post("/api/backtest/run")
async def execute_backtest(req: BacktestRequest):
    """Walk-forward No-Stop strategy backtesting (single ticker or entire 307 basket)."""
    if req.mode == "basket":
        registry = load_ticker_registry()
        if not registry:
            raise HTTPException(status_code=400, detail="Ticker registry is empty")
        target_tickers = registry if req.scope == 0 else registry[:req.scope]
        from core.no_stop_engine import run_basket_no_stop_backtest
        res = await run_basket_no_stop_backtest(
            target_tickers,
            period=req.period,
            target_pct=req.target_pct,
            require_uptrend=req.require_uptrend,
        )
        res["period"] = req.period
        res["target_pct"] = req.target_pct
        return res

    # Single ticker mode
    sym = bare_symbol(req.ticker or "SBIN").upper()
    df = await fetch_ticker_ohlcv_async(sym, period=req.period)
    if df.empty or len(df) < 50:
        raise HTTPException(status_code=400, detail=f"Insufficient history bars for {sym}")

    res = run_no_stop_backtest(
        df,
        target_pct=req.target_pct,
        require_uptrend=req.require_uptrend,
    )
    res["ticker"] = sym
    res["period"] = req.period
    res["target_pct"] = req.target_pct
    res["is_basket"] = False
    return res


@app.get("/api/tracker/positions")
async def list_positions():
    return get_tracked_positions()


@app.post("/api/tracker/add")
async def track_trade(req: TrackTradeRequest):
    return add_tracked_position(
        symbol=req.symbol,
        entry_price=req.entry_price,
        entry_date=req.entry_date,
        target_pct=req.target_pct,
        capital=req.capital,
        notes=req.notes or "",
    )


@app.post("/api/tracker/refresh")
async def refresh_positions():
    return refresh_tracked_positions()


@app.delete("/api/tracker/{trade_id}")
async def delete_position(trade_id: str):
    success = remove_tracked_position(trade_id)
    if not success:
        raise HTTPException(status_code=404, detail="Trade ID not found")
    return {"status": "deleted", "id": trade_id}


if __name__ == "__main__":
    import uvicorn
    # Default port 8500 to avoid conflicting with existing port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8500, reload=True)
