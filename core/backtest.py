"""Causal daily-bar execution and net performance for the equity strategy.

Confirmation is observed at a completed close; entry and EMA-close exits fill
at a subsequent session open. Current valuations never gate historical trades.
"""
import asyncio
import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from core.data_fetcher import clean_ohlcv, fetch_batch_ohlcv, normalize_ticker
from core.strategy import confirmed_breakout_mask, compute_trend


def _calc_ema9(closes: List[float]) -> List[float]:
    return pd.Series(closes, dtype=float).ewm(span=9, adjust=False).mean().tolist()


def _date_str(index: Any) -> str:
    return str(index.date()) if hasattr(index, "date") else str(index)[:10]


def simulate_long_trade(
    df: pd.DataFrame, entry_idx: int, entry_price: float,
    target_pct: float = 3.0, max_hold_candles: int = 20,
    slippage_bps: float = 5.0, commission_bps: float = 10.0,
    stop_mode: str = "EMA9_CLOSE", stop_loss_price: Optional[float] = None,
    include_entry_bar: bool = True, entry_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Replay actual buy fill against completed OHLCV bars.

    PRICE stops win same-bar intraday barrier ties because OHLC data cannot
    reveal their order. EMA stops are close decisions, executed next open;
    therefore a same-day intraday target precedes an EMA-close decision.
    Final-close decisions remain pending instead of receiving fictitious fills.
    """
    if not math.isfinite(entry_price) or entry_price <= 0:
        raise ValueError("entry_price must be a finite positive number")
    if not math.isfinite(target_pct) or target_pct <= 0:
        raise ValueError("target_pct must be a finite positive number")
    if max_hold_candles < 1 or not 0 <= slippage_bps < 10000 or not 0 <= commission_bps < 10000:
        raise ValueError("Holding period or execution costs are invalid")
    if stop_mode not in ("EMA9_CLOSE", "PRICE"):
        raise ValueError("stop_mode must be EMA9_CLOSE or PRICE")
    if stop_mode == "PRICE" and (stop_loss_price is None or not math.isfinite(stop_loss_price) or stop_loss_price <= 0):
        raise ValueError("PRICE stops require a finite positive stop_loss_price")
    if not 0 <= entry_idx < len(df):
        raise ValueError("entry_idx must identify an available bar")
    ema = df["Close"].ewm(span=9, adjust=False).mean()
    target = entry_price * (1.0 + target_pct / 100.0)
    slip, fee = slippage_bps / 10000.0, commission_bps / 10000.0
    start = entry_idx if include_entry_bar else entry_idx + 1
    pending_reason = None
    result = {
        "entry_date": entry_date or _date_str(df.index[entry_idx]),
        "entry_price": round(entry_price, 4), "target_price": round(target, 4),
        "exit_date": None, "exit_price": None, "exit_idx": None,
        "status": "ACTIVE", "pending_reason": None, "status_reason": None,
        "stop_reason": None, "gross_pnl_pct": None, "net_pnl_pct": None,
        "pnl_pct": None, "hold_candles": 0,
    }

    def finish(j: int, price: float, status: str, reason: str) -> Dict[str, Any]:
        gross = (price / entry_price - 1.0) * 100.0
        net = (price * (1.0-fee) / (entry_price * (1.0+fee)) - 1.0) * 100.0
        result.update({
            "status": status, "status_reason": reason, "stop_reason": reason,
            "exit_date": _date_str(df.index[j]), "exit_price": round(price, 4),
            "exit_idx": j, "hold_candles": j-start+1,
            "gross_pnl_pct": round(gross, 4), "net_pnl_pct": round(net, 4),
            "pnl_pct": round(net, 4), "pending_reason": None,
        })
        return result

    for j in range(start, len(df)):
        row = df.iloc[j]
        opening, high, low, closing = (float(row[c]) for c in ("Open", "High", "Low", "Close"))
        if pending_reason:
            return finish(j, opening * (1.0-slip), "STOPPED_OUT" if pending_reason == "EMA9_BREACH" else "TIME_EXIT", pending_reason)
        if stop_mode == "PRICE" and opening <= stop_loss_price:
            return finish(j, opening * (1.0-slip), "STOPPED_OUT", "PRICE_STOP_GAP")
        if opening >= target:
            return finish(j, max(target, opening * (1.0-slip)), "TARGET_HIT", "TARGET_GAP")
        if stop_mode == "PRICE" and low <= stop_loss_price:
            return finish(j, stop_loss_price * (1.0-slip), "STOPPED_OUT", "AMBIGUOUS_BAR_STOP_FIRST" if high >= target else "PRICE_STOP")
        if high >= target:
            return finish(j, target, "TARGET_HIT", "TARGET_HIT")
        if stop_mode == "EMA9_CLOSE" and closing < float(ema.iloc[j]):
            pending_reason = "EMA9_BREACH"
        elif j-start+1 >= max_hold_candles:
            pending_reason = "MAX_HOLD_EXPIRED"
    result.update({
        "status": "STOP_PENDING" if pending_reason == "EMA9_BREACH" else ("TIME_EXIT_PENDING" if pending_reason else "ACTIVE"),
        "pending_reason": pending_reason, "status_reason": pending_reason or "AWAITING_OUTCOME",
        "hold_candles": max(0, len(df)-start),
        "unrealized_pnl_pct": round((float(df["Close"].iloc[-1])/entry_price-1.0)*100.0, 4),
    })
    return result


def backtest_single_ticker(
    ticker: str, df: pd.DataFrame, fair_price: Optional[float] = None,
    target_pct: float = 3.0, max_hold_candles: int = 20,
    require_fair_value: bool = False, slippage_bps: float = 5.0,
    commission_bps: float = 10.0, require_uptrend: bool = True,
    fair_value_history: Optional[pd.Series] = None,
    start_date: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """One position per ticker, using information known at confirmation close.

    ``fair_price`` is accepted for old callers but ignored: a current scalar has
    no historical availability date. A historical gate requires a Series indexed
    by dates each valuation became available (publication dates, not FY dates).
    """
    df = clean_ohlcv(df, completed_only=True)
    if len(df) < 32:
        return []
    confirmation = confirmed_breakout_mask(df)
    historical_fv = None
    if fair_value_history is not None:
        historical_fv = pd.to_numeric(fair_value_history, errors="coerce").dropna()
        historical_fv.index = pd.to_datetime(historical_fv.index)
        if historical_fv.index.tz is not None:
            historical_fv.index = historical_fv.index.tz_convert("Asia/Kolkata").tz_localize(None)
        historical_fv = historical_fv.sort_index()
    trades, occupied_until = [], -1
    for confirm_idx in range(29, len(df)-1):
        if not bool(confirmation.iloc[confirm_idx]) or confirm_idx <= occupied_until:
            continue
        entry_idx = confirm_idx+1
        if start_date is not None and pd.Timestamp(df.index[entry_idx]) < pd.Timestamp(start_date):
            continue
        if require_uptrend and compute_trend(df.iloc[:confirm_idx+1]).get("trend_regime") != "UPTREND":
            continue
        entry_price = float(df["Open"].iloc[entry_idx]) * (1.0+slippage_bps/10000.0)
        historical_price = None
        if historical_fv is not None:
            decision_close = pd.Timestamp(df.index[confirm_idx]).normalize()+pd.Timedelta(hours=15, minutes=30)
            known = historical_fv[historical_fv.index <= decision_close]
            if len(known) and float(known.iloc[-1]) > 0:
                historical_price = float(known.iloc[-1])
        if require_fair_value and (historical_price is None or entry_price >= historical_price):
            trades.append({
                "ticker": ticker, "entry_date": _date_str(df.index[entry_idx]), "entry_price": round(entry_price, 4),
                "target_price": round(entry_price*(1+target_pct/100), 4), "fair_price": historical_price, "fair_gap_pct": None,
                "exit_date": None, "exit_price": None, "hold_candles": 0, "pnl_pct": None,
                "outcome": "SKIPPED_FAIR_VALUE_UNAVAILABLE" if historical_price is None else "SKIPPED_OVERVALUED", "stop_reason": "FAIR_VALUE_FILTER",
            })
            continue
        replay = simulate_long_trade(df, entry_idx, entry_price, target_pct, max_hold_candles, slippage_bps, commission_bps)
        occupied_until = replay["exit_idx"] if replay["exit_idx"] is not None else len(df)
        pnl = replay["net_pnl_pct"]
        trades.append({
            **replay, "ticker": ticker, "breakout_date": _date_str(df.index[confirm_idx-1]), "confirm_date": _date_str(df.index[confirm_idx]),
            "fair_price": historical_price, "fair_gap_pct": round((historical_price/entry_price-1)*100, 2) if historical_price else None,
            "outcome": "OPEN" if pnl is None else ("WIN" if pnl > 0 else ("LOSS" if pnl < 0 else "BREAKEVEN")),
        })
    return trades


def compute_performance_stats(trades: List[Dict[str, Any]], capital_tickers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Net closed-trade statistics; unresolved trades do not count as losses."""
    closed = [t for t in trades if t.get("outcome") in ("WIN", "LOSS", "BREAKEVEN") and t.get("pnl_pct") is not None]
    wins, losses = [t for t in closed if t["pnl_pct"] > 0], [t for t in closed if t["pnl_pct"] < 0]
    count = len(closed)
    profit, loss = sum(t["pnl_pct"] for t in wins), -sum(t["pnl_pct"] for t in losses)
    target_hits = sum(t.get("status") == "TARGET_HIT" or t.get("stop_reason") in ("TARGET_HIT", "TARGET_GAP") for t in closed)
    curve, monthly = [], {}
    equity = peak = 100.0
    max_dd = consecutive = max_consecutive = 0
    tickers = set(capital_tickers) if capital_tickers is not None else {t.get("ticker", "UNKNOWN") for t in trades if not str(t.get("outcome", "")).startswith("SKIPPED")}
    allocation = 1.0/max(1, len(tickers))
    ordered = sorted(closed, key=lambda t: t.get("exit_date") or "")
    if ordered:
        curve.append({"date": min(t["entry_date"] for t in ordered), "equity": 100.0, "return_pct": 0.0})
    for trade in ordered:
        pnl = trade["pnl_pct"]
        equity += pnl*allocation
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak-equity)/peak*100.0)
        consecutive = consecutive+1 if pnl < 0 else 0
        max_consecutive = max(max_consecutive, consecutive)
        curve.append({"date": trade["exit_date"], "ticker": trade.get("ticker"), "equity": round(equity, 4), "return_pct": round(equity-100, 4), "trade_pnl": pnl, "outcome": trade["outcome"]})
        month = (trade.get("exit_date") or "Unknown")[:7]
        row = monthly.setdefault(month, {"month": month, "trades": 0, "wins": 0, "losses": 0, "pnl": 0.0})
        row["trades"] += 1
        row["wins"] += int(pnl > 0)
        row["losses"] += int(pnl < 0)
        row["pnl"] += pnl*allocation
    for row in monthly.values():
        row["win_rate"] = round(row["wins"]/row["trades"]*100, 2)
        row["pnl"] = round(row["pnl"], 4)
    ci = None
    if count:
        p, z = len(wins)/count, 1.96
        den = 1+z*z/count
        mid = (p+z*z/(2*count))/den
        half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
        ci = [round(max(0, mid-half)*100, 2), round(min(1, mid+half)*100, 2)]
    return {
        "total_trades": count, "wins": len(wins), "losses": len(losses), "breakeven_count": count-len(wins)-len(losses),
        "open_count": sum(t.get("outcome") == "OPEN" for t in trades), "skipped_count": sum(str(t.get("outcome", "")).startswith("SKIPPED") for t in trades),
        "win_rate_pct": round(len(wins)/count*100, 2) if count else 0.0, "win_rate_ci95_pct": ci,
        "target_hit_count": target_hits, "target_hit_rate_pct": round(target_hits/count*100, 2) if count else 0.0,
        "profit_factor": round(profit/loss, 4) if loss else (None if profit else 0.0), "profit_factor_unbounded": bool(profit and not loss),
        "avg_trade_return_pct": round(sum(t["pnl_pct"] for t in closed)/count, 4) if count else 0.0,
        "median_trade_return_pct": round(float(np.median([t["pnl_pct"] for t in closed])), 4) if count else 0.0,
        "avg_win_pct": round(profit/len(wins), 4) if wins else 0.0, "avg_loss_pct": round(-loss/len(losses), 4) if losses else 0.0,
        "avg_hold_days": round(sum(t.get("hold_candles", 0) for t in closed)/count, 2) if count else 0.0,
        "max_drawdown_pct": round(max_dd, 4), "max_consecutive_losses": max_consecutive, "total_return_pct": round(equity-100, 4),
        "equity_curve_basis": "Fixed initial capital split equally across tickers; realized P&L only",
        "drawdown_basis": "Realized closed-trade P&L; excludes open-position drawdowns",
        "equity_curve": curve, "monthly_stats": [monthly[m] for m in sorted(monthly)],
    }


async def run_equity_backtest(
    tickers: List[str], years: int = 2, target_pct: float = 3.0,
    max_hold_candles: int = 20, compare_fair_value: bool = True,
    slippage_bps: float = 5.0, commission_bps: float = 10.0, require_uptrend: bool = True, progress=None,
) -> Dict[str, Any]:
    clean_tickers = list(dict.fromkeys(normalize_ticker(t) for t in tickers if t.strip()))
    def download_progress(info):
        if progress:
            progress({**info, "percent": 40 * info["completed"] / max(1, info["total"]),
                      "message": "Fetching historical daily bars"})
    kwargs = {"progress": download_progress} if progress else {}
    ticker_dfs = await asyncio.to_thread(fetch_batch_ohlcv, clean_tickers, "1d", years*365+365, **kwargs)
    trades, skipped = [], []
    cutoff = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize()-pd.Timedelta(days=years*365)
    for completed, ticker in enumerate(clean_tickers, 1):
        if progress:
            progress({"phase": "analysis", "completed": completed - 1, "total": len(clean_tickers),
                      "ticker": ticker, "percent": 40 + 59 * (completed - 1) / len(clean_tickers),
                      "failed_count": len(skipped), "message": "Replaying trades with entry, exit and cost rules"})
        df = ticker_dfs.get(ticker)
        if df is None or len(df) < 50:
            skipped.append({"ticker": ticker, "reason": "MISSING_OR_SHORT_HISTORY"})
            continue
        ticker_trades = await asyncio.to_thread(backtest_single_ticker, ticker, df, target_pct=target_pct, max_hold_candles=max_hold_candles, slippage_bps=slippage_bps, commission_bps=commission_bps, require_uptrend=require_uptrend, start_date=cutoff)
        trades.extend(ticker_trades)
    if progress:
        progress({"phase": "analysis", "completed": len(clean_tickers), "total": len(clean_tickers),
                  "ticker": None, "percent": 99, "failed_count": len(skipped), "message": "Calculating performance statistics"})
    stats = compute_performance_stats(trades, capital_tickers=clean_tickers)
    breakdowns = []
    for ticker in clean_tickers:
        s = compute_performance_stats([t for t in trades if t["ticker"] == ticker])
        if s["total_trades"] or s["open_count"]:
            breakdowns.append({"ticker": ticker, "trades": s["total_trades"], "wins": s["wins"], "losses": s["losses"], "open_count": s["open_count"], "win_rate_pct": s["win_rate_pct"], "net_pnl_pct": s["total_return_pct"], "expectancy_pct": s["avg_trade_return_pct"]})
    reason = "Historical fair-value comparison requires valuations indexed by publication dates. Today's fair value cannot be used for past entries."
    return {
        "parameters": {"tickers_count": len(clean_tickers), "years": years, "target_pct": target_pct, "max_hold_candles": max_hold_candles, "slippage_bps": slippage_bps, "commission_bps": commission_bps, "require_uptrend": require_uptrend},
        "technical_strategy": {"stats": stats, "trades_count": len(trades)},
        "aligned_fair_value_strategy": {"available": False, "stats": None, "trades_count": 0, "reason": reason} if compare_fair_value else None,
        "execution_assumptions": ["Completed daily candles only", "Confirmed signal enters next session open with buy slippage", "EMA close breach exits following open with sell slippage", "Targets are fixed from actual entry and filled by sell limits", "Unresolved trades remain open", "Net returns include commission on both sides", "One open position per ticker"],
        "warnings": ([reason] if compare_fair_value else [])+["Equity and drawdown use realized fixed-allocation P&L, not marked-to-market portfolio returns.", "The selected current ticker universe may contain survivorship bias.", "Technical backtest replays EMA confirmation and the selected trend filter; live volume, resistance and valuation selection gates are not included."],
        "data_skipped": skipped, "ticker_breakdowns": sorted(breakdowns, key=lambda t: t["expectancy_pct"], reverse=True),
        "recent_trades": sorted(trades, key=lambda t: t["entry_date"], reverse=True)[:100],
    }
