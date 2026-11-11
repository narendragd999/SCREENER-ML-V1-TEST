"""
no_stop_engine.py — "Hold Till Target Achieved" (No Stop-Loss) Simulation & Analytics
"""
import asyncio
import math
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from core.strategy import indicator_frame, confirmed_breakout_mask, trend_regime_mask


def simulate_no_stop_trade(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    target_pct: float = 3.0,
    cost_pct: float = 0.30,
) -> Dict[str, Any]:
    """
    Simulates a trade that remains open indefinitely until target is achieved.
    Accurately records:
    - Target hit status and exit date
    - Sessions required to hit target
    - Maximum Adverse Excursion (MAE): the deepest paper loss endured before target was hit
    """
    n = len(df)
    if entry_idx >= n or entry_price <= 0:
        return {}

    target_price = entry_price * (1.0 + target_pct / 100.0)
    entry_date = str(df.index[entry_idx].date())
    
    lowest_low = float(df.Low.iloc[entry_idx])
    
    # Iterate session by session from entry onwards
    for j in range(entry_idx, n):
        high = float(df.High.iloc[j])
        low = float(df.Low.iloc[j])
        
        lowest_low = min(lowest_low, low)
        mae_pct = ((lowest_low / entry_price) - 1.0) * 100.0
        
        # Check if target is achieved on this session's High
        if high >= target_price:
            hold_sessions = j - entry_idx + 1
            return {
                "status": "TARGET_HIT",
                "entry_idx": entry_idx,
                "entry_date": entry_date,
                "entry_price": round(entry_price, 2),
                "target_price": round(target_price, 2),
                "target_pct": target_pct,
                "exit_idx": j,
                "exit_date": str(df.index[j].date()),
                "exit_price": round(target_price, 2),
                "hold_sessions": hold_sessions,
                "gross_pnl_pct": round(target_pct, 2),
                "net_pnl_pct": round(target_pct - cost_pct, 2),
                "mae_pct": round(mae_pct, 2),
                "lowest_price": round(lowest_low, 2),
                "is_open": False,
            }

    # If target has not been reached by the final available session
    hold_sessions = n - entry_idx
    latest_close = float(df.Close.iloc[-1])
    unrealized_pnl = ((latest_close / entry_price) - 1.0) * 100.0
    mae_pct = ((lowest_low / entry_price) - 1.0) * 100.0

    return {
        "status": "OPEN",
        "entry_idx": entry_idx,
        "entry_date": entry_date,
        "entry_price": round(entry_price, 2),
        "target_price": round(target_price, 2),
        "target_pct": target_pct,
        "exit_idx": None,
        "exit_date": None,
        "exit_price": None,
        "current_price": round(latest_close, 2),
        "hold_sessions": hold_sessions,
        "gross_pnl_pct": round(unrealized_pnl, 2),
        "net_pnl_pct": round(unrealized_pnl - cost_pct, 2),
        "unrealized_pnl_pct": round(unrealized_pnl, 2),
        "mae_pct": round(mae_pct, 2),
        "lowest_price": round(lowest_low, 2),
        "is_open": True,
    }


def run_no_stop_backtest(
    df: pd.DataFrame,
    target_pct: float = 3.0,
    require_uptrend: bool = True,
    cost_pct: float = 0.30,
) -> Dict[str, Any]:
    """
    Executes a walk-forward historical simulation of the No-Stop Strategy.
    Rule: Only one position allowed per stock at a time (sequential execution).
    """
    bars = indicator_frame(df)
    n = len(bars)
    if n < 50:
        return {"total_trades": 0, "error": "Insufficient history bars"}

    confirm_mask = confirmed_breakout_mask(bars).to_numpy()
    regimes = trend_regime_mask(bars).to_numpy() if require_uptrend else np.array(["UPTREND"] * n)

    trades = []
    i = 30
    while i < n - 1:
        # Check for confirmed breakout setup
        if confirm_mask[i]:
            if not require_uptrend or regimes[i] == "UPTREND":
                entry_idx = i + 1
                if entry_idx < n:
                    entry_price = float(bars.Open.iloc[entry_idx])
                    trade = simulate_no_stop_trade(bars, entry_idx, entry_price, target_pct, cost_pct)
                    if trade:
                        trades.append(trade)
                        # Advance index past the trade exit so we don't open overlapping positions
                        if trade["status"] == "TARGET_HIT":
                            i = trade["exit_idx"]
                        else:
                            break  # Trade is still open till end of dataset
        i += 1

    if not trades:
        return {
            "total_trades": 0,
            "target_hit_count": 0,
            "open_trades_count": 0,
            "target_hit_rate_pct": 0.0,
            "avg_sessions": 0.0,
            "median_sessions": 0.0,
            "max_sessions": 0,
            "avg_mae_pct": 0.0,
            "worst_mae_pct": 0.0,
            "trades": [],
            "distribution": {},
        }

    closed_trades = [t for t in trades if t["status"] == "TARGET_HIT"]
    open_trades = [t for t in trades if t["status"] == "OPEN"]

    # Target hit rate based on completed trades
    total_trades = len(trades)
    target_hit_count = len(closed_trades)
    target_hit_rate = (target_hit_count / total_trades * 100.0) if total_trades > 0 else 0.0

    # Holding sessions analytics
    sessions_list = [t["hold_sessions"] for t in closed_trades]
    avg_sessions = float(np.mean(sessions_list)) if sessions_list else 0.0
    median_sessions = float(np.median(sessions_list)) if sessions_list else 0.0
    max_sessions = int(np.max(sessions_list)) if sessions_list else 0

    # MAE (Maximum Adverse Excursion / Drawdown) analytics
    mae_list = [t["mae_pct"] for t in trades]
    avg_mae = float(np.mean(mae_list)) if mae_list else 0.0
    worst_mae = float(np.min(mae_list)) if mae_list else 0.0

    # Duration Distribution
    dist = {
        "1_to_5_days": sum(1 for s in sessions_list if 1 <= s <= 5),
        "6_to_15_days": sum(1 for s in sessions_list if 6 <= s <= 15),
        "16_to_30_days": sum(1 for s in sessions_list if 16 <= s <= 30),
        "31_to_60_days": sum(1 for s in sessions_list if 31 <= s <= 60),
        "over_60_days": sum(1 for s in sessions_list if s > 60),
    }

    # MAE Drawdown Distribution
    mae_dist = {
        "shallow_0_to_3pct": sum(1 for m in mae_list if 0 >= m >= -3.0),
        "moderate_3_to_7pct": sum(1 for m in mae_list if -3.0 > m >= -7.0),
        "deep_7_to_15pct": sum(1 for m in mae_list if -7.0 > m >= -15.0),
        "severe_over_15pct": sum(1 for m in mae_list if m < -15.0),
    }

    return {
        "total_trades": total_trades,
        "target_hit_count": target_hit_count,
        "open_trades_count": len(open_trades),
        "target_hit_rate_pct": round(target_hit_rate, 2),
        "avg_sessions": round(avg_sessions, 1),
        "median_sessions": round(median_sessions, 1),
        "max_sessions": max_sessions,
        "avg_mae_pct": round(avg_mae, 2),
        "worst_mae_pct": round(worst_mae, 2),
        "distribution": dist,
        "mae_distribution": mae_dist,
        "trades": trades[::-1],  # newest first
    }


async def run_basket_no_stop_backtest(
    registry_tickers: List[Dict[str, Any]],
    period: str = "2y",
    target_pct: float = 3.0,
    require_uptrend: bool = False,
    cost_pct: float = 0.30,
    max_concurrency: int = 10,
) -> Dict[str, Any]:
    """
    Executes a walk-forward No-Stop backtest across multiple/all 307 tickers.
    Aggregates entire universe portfolio statistics, duration breakdown, and trade logs.
    """
    from core.data_fetcher import fetch_ticker_ohlcv_async, bare_symbol
    sem = asyncio.Semaphore(max_concurrency)
    all_trades = []
    ticker_summaries = []

    async def _process_ticker(rec):
        ticker = rec["ticker"]
        sym = bare_symbol(ticker)
        async with sem:
            df = await fetch_ticker_ohlcv_async(ticker, period=period)
            if df.empty or len(df) < 50:
                return
            res = run_no_stop_backtest(df, target_pct=target_pct, require_uptrend=require_uptrend, cost_pct=cost_pct)
            if res.get("total_trades", 0) > 0:
                for t in res.get("trades", []):
                    t_copy = dict(t)
                    t_copy["symbol"] = sym
                    t_copy["security"] = rec.get("security", sym)
                    all_trades.append(t_copy)
                ticker_summaries.append({
                    "symbol": sym,
                    "security": rec.get("security", sym),
                    "total_trades": res["total_trades"],
                    "target_hit_count": res["target_hit_count"],
                    "open_trades_count": res["open_trades_count"],
                    "hit_rate_pct": res["target_hit_rate_pct"],
                    "avg_sessions": res["avg_sessions"],
                    "worst_mae_pct": res["worst_mae_pct"],
                })

    tasks = [_process_ticker(r) for r in registry_tickers]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Sort all trades by entry date descending
    all_trades.sort(key=lambda t: t.get("entry_date", ""), reverse=True)
    ticker_summaries.sort(key=lambda x: (x["hit_rate_pct"], x["total_trades"]), reverse=True)

    closed_trades = [t for t in all_trades if t["status"] == "TARGET_HIT"]
    open_trades = [t for t in all_trades if t["status"] == "OPEN"]
    total_trades = len(all_trades)
    target_hit_count = len(closed_trades)
    target_hit_rate = (target_hit_count / total_trades * 100.0) if total_trades > 0 else 0.0

    sessions_list = [t["hold_sessions"] for t in closed_trades]
    avg_sessions = float(np.mean(sessions_list)) if sessions_list else 0.0
    median_sessions = float(np.median(sessions_list)) if sessions_list else 0.0
    max_sessions = int(np.max(sessions_list)) if sessions_list else 0

    mae_list = [t["mae_pct"] for t in all_trades]
    avg_mae = float(np.mean(mae_list)) if mae_list else 0.0
    worst_mae = float(np.min(mae_list)) if mae_list else 0.0

    dist = {
        "1_to_5_days": sum(1 for s in sessions_list if 1 <= s <= 5),
        "6_to_15_days": sum(1 for s in sessions_list if 6 <= s <= 15),
        "16_to_30_days": sum(1 for s in sessions_list if 16 <= s <= 30),
        "31_to_60_days": sum(1 for s in sessions_list if 31 <= s <= 60),
        "over_60_days": sum(1 for s in sessions_list if s > 60),
    }

    mae_dist = {
        "shallow_0_to_3pct": sum(1 for m in mae_list if 0 >= m >= -3.0),
        "moderate_3_to_7pct": sum(1 for m in mae_list if -3.0 > m >= -7.0),
        "deep_7_to_15pct": sum(1 for m in mae_list if -7.0 > m >= -15.0),
        "severe_over_15pct": sum(1 for m in mae_list if m < -15.0),
    }

    return {
        "is_basket": True,
        "ticker": f"BASKET ({len(ticker_summaries)} Stocks)",
        "tickers_tested": len(ticker_summaries),
        "total_tickers_requested": len(registry_tickers),
        "total_trades": total_trades,
        "target_hit_count": target_hit_count,
        "open_trades_count": len(open_trades),
        "target_hit_rate_pct": round(target_hit_rate, 2),
        "avg_sessions": round(avg_sessions, 1),
        "median_sessions": round(median_sessions, 1),
        "max_sessions": max_sessions,
        "avg_mae_pct": round(avg_mae, 2),
        "worst_mae_pct": round(worst_mae, 2),
        "distribution": dist,
        "mae_distribution": mae_dist,
        "ticker_summaries": ticker_summaries,
        "trades": all_trades,
    }
