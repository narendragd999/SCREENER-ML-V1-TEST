"""Daily EMA9 setups with causal outcomes and explainable selection gates."""
import asyncio
import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from core.data_fetcher import clean_ohlcv, fetch_batch_ohlcv, normalize_ticker, load_ticker_registry, market_now
from core.fair_value import compute_fair_value
from core.strategy import indicator_frame, confirmed_breakout_mask, compute_trend


def _number(value, digits=2):
    try:
        return round(float(value), digits) if value is not None and math.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def analyze_ticker_ema9(ticker: str, df: pd.DataFrame, max_candles_ago: int = 10,
                        target_pct: float = 3.0) -> Optional[Dict[str, Any]]:
    if not 1 <= max_candles_ago <= 60 or not math.isfinite(target_pct) or target_pct <= 0:
        raise ValueError("Invalid breakout window or target percentage")
    bars = clean_ohlcv(df)
    if len(bars) < 31:
        return None
    bars = indicator_frame(bars)
    n = len(bars)
    confirmations = np.flatnonzero(confirmed_breakout_mask(bars).to_numpy())
    confirmations = [int(i) for i in confirmations if n - 1 - i <= max_candles_ago]
    if not confirmations:
        return None
    confirm_idx = confirmations[-1]
    breakout_idx = confirm_idx - 1
    row, conf = bars.iloc[-1], bars.iloc[confirm_idx]
    current, ema = float(row.Close), float(row.ema9)
    entry_idx = confirm_idx + 1
    entered = entry_idx < n
    entry = float(bars.Open.iloc[entry_idx]) if entered else float(conf.Close)
    target = entry * (1 + target_pct / 100)
    date = lambda i: str(bars.index[i].date())
    lifecycle = {"status": "FRESH", "exit_date": None, "exit_price": None, "pending_reason": None, "pnl_pct": None}
    if entered:
        from core.backtest import simulate_long_trade
        lifecycle = simulate_long_trade(bars, entry_idx, entry, target_pct, slippage_bps=0, commission_bps=0)
    status = lifecycle["status"]
    remaining = (target / current - 1) * 100
    age = n - 1 - confirm_idx
    if status == "ACTIVE":
        status = "FRESH" if age <= 1 else ("NEARING_TARGET" if 0 < remaining <= .8 else "ACTIVE")
    risk = (current - ema) / current * 100
    rr = remaining / risk if remaining > 0 and risk > 0 else None
    atr = float(row.atr14) if pd.notna(row.atr14) else 0
    net_reward = remaining - .30
    risk_estimate = max(risk, .5 * atr/current*100) if risk > 0 else None
    net_rr = net_reward / (risk_estimate + .30) if net_reward > 0 and risk_estimate else None
    extension = (current - ema) / atr if atr > 0 else None
    prior_volume = bars.Volume.iloc[max(0, confirm_idx-20):confirm_idx]
    volume_available = bool(bars.attrs.get("volume_available", True)
                            and (prior_volume > 0).sum() >= 15 and conf.Volume > 0
                            and (bars.Volume.iloc[-20:] > 0).sum() >= 15)
    volume_ratio = _number(conf.volume_ratio) if volume_available else None
    turnover = _number(row.turnover20) if volume_available else None
    trend = compute_trend(bars)
    stale_days = max(0, (market_now().date() - bars.index[-1].date()).days)
    resistance = []
    for i in range(max(2, n-90), n-2):
        high = float(bars.High.iloc[i])
        if high > current and high >= float(bars.High.iloc[i-2:i+3].max()):
            resistance.append(high)
    nearest = min(resistance) if resistance else None
    blockers, evidence = [], ["EMA cross followed by a higher close and higher high"]
    if status not in ("FRESH", "ACTIVE", "NEARING_TARGET"):
        blockers.append(f"Setup is {status}")
    if current <= ema:
        blockers.append("Latest close is below EMA9")
    if age > 5:
        blockers.append("Confirmation is older than five sessions")
    if trend["trend_regime"] != "UPTREND":
        blockers.append("Full 200-session uptrend is unverified")
    else:
        evidence.append("Above SMA50 and SMA200 with rising SMA50 and positive 60-session slope")
    if stale_days > 5:
        blockers.append("Daily price history is stale")
    if not volume_available:
        blockers.append("Volume evidence is missing or mostly zero")
    elif turnover is None or turnover < 10_000_000:
        blockers.append("20-session average turnover is below Rs 1 crore")
    if volume_ratio is None or volume_ratio < 1:
        blockers.append("Confirmation volume does not exceed its prior 20-session average")
    else:
        evidence.append(f"Confirmation volume is {volume_ratio:.2f}x its prior average")
    if extension is None or extension < 0 or extension > 1.5:
        blockers.append("Price is invalid or extended more than 1.5 ATR above EMA9")
    if net_rr is None or net_rr < 1.5:
        blockers.append("Remaining reward/risk after estimated costs is below 1.5")
    if nearest is not None and nearest < target:
        blockers.append("Confirmed resistance stands before the setup target")
    if bars.attrs.get("invalid_bars_removed", 0):
        blockers.append("Invalid or duplicate price bars were removed")
    score = (25 * (trend["trend_regime"] == "UPTREND") + 15
             + 15 * (volume_ratio is not None and volume_ratio >= 1)
             + 20 * (net_rr is not None and net_rr >= 1.5)
             + 10 * (turnover is not None and turnover >= 10_000_000)
             + 10 * (extension is not None and 0 <= extension <= 1.5) + 5 * (age <= 5))
    if nearest is not None and nearest < target:
        score = max(0, score - 15)
    if status not in ("FRESH", "ACTIVE", "NEARING_TARGET") or stale_days > 5:
        score = min(score, 35)
    candles = []
    for j in range(max(0, n-60), n):
        bar = bars.iloc[j]
        candles.append({"date": date(j), "open": _number(bar.Open), "high": _number(bar.High),
                        "low": _number(bar.Low), "close": _number(bar.Close), "volume": int(bar.Volume),
                        "ema9": _number(bar.ema9), "sma50": _number(bar.sma50),
                        "is_breakout": j == breakout_idx, "is_confirm": j == confirm_idx})
    return {"ticker": normalize_ticker(ticker), "current_price": _number(current), "ema9": _number(ema),
            "ema9_dist_pct": _number((current/ema-1)*100), "target_3pct": _number(target), "target_pct": target_pct,
            "distance_to_target_pct": _number(remaining), "entry_ref_price": _number(entry),
            "entry_date": date(entry_idx) if entered else None,
            "entry_basis": "next_session_open" if entered else "confirmation_close_estimate",
            "current_gain_pct": _number((current/entry-1)*100), "new_entry_target": _number(current*(1+target_pct/100)),
            "stop_loss_price": _number(ema), "stop_mode": "EMA9_CLOSE",
            "risk_pct": _number(risk) if risk > 0 else None, "risk_reward": _number(rr),
            "net_risk_reward": _number(net_rr), "cost_estimate_pct": .30,
            "risk_budget_estimate_pct": _number(risk_estimate), "risk_is_estimate": True,
            "risk_assumption": "EMA close invalidation fills next open; risk estimate uses at least 0.5 ATR plus costs. Gaps can exceed this estimate.",
            "breakout_date": date(breakout_idx), "breakout_close": _number(bars.Close.iloc[breakout_idx]),
            "breakout_high": _number(bars.High.iloc[breakout_idx]), "confirm_date": date(confirm_idx),
            "confirm_close": _number(conf.Close), "candles_ago": age, "signal_status": status,
            "exit_date": lifecycle.get("exit_date"), "exit_price": lifecycle.get("exit_price"),
            "realized_gain_pct": lifecycle.get("pnl_pct"), "pending_reason": lifecycle.get("pending_reason"),
            "atr14": _number(atr, 4), "atr_pct": _number(atr/current*100), "rsi14": _number(row.rsi14),
            "volume_ratio": volume_ratio, "average_turnover_20d": turnover, "ema_extension_atr": _number(extension),
            "nearest_resistance": _number(nearest), "setup_score": int(score), "score_is_probability": False,
            "is_actionable": not blockers, "evidence": evidence, "warnings": blockers, "as_of": date(n-1),
            "data_age_days": stale_days, "history_bars": n, "completed_candles_only": True, "candles": candles, **trend}


async def run_ema9_screener(tickers: Optional[List[str]] = None, source: str = "fno", lookback_days: int = 365,
                           max_candles_ago: int = 10, target_pct: float = 3.0, require_uptrend: bool = True,
                           allow_sideways: bool = False, require_fair_value_undervalued: bool = False,
                           actionable_only: bool = False, min_setup_score: int = 0,
                           require_market_uptrend: bool = False, require_relative_strength: bool = False,
                           enrich_watchlist: bool = False, progress=None) -> Dict[str, Any]:
    registry = load_ticker_registry(source)
    names = dict(zip(registry.symbol, registry.company_name)) if not registry.empty else {}
    if tickers is None:
        tickers = registry.symbol.tolist() if not registry.empty else []
    tickers = list(dict.fromkeys(normalize_ticker(t) for t in tickers if normalize_ticker(t)))
    if not tickers:
        return {"error": "No tickers provided or found.", "signals": [], "prime_targets": []}
    def download_progress(info):
        if progress:
            progress({**info, "percent": 40 * info["completed"] / max(1, info["total"]),
                      "message": "Fetching completed price bars"})
    kwargs = {"progress": download_progress} if progress else {}
    frames = await asyncio.to_thread(fetch_batch_ohlcv, tickers + ["^NSEI"], "1d", max(365, lookback_days), **kwargs)
    benchmark = clean_ohlcv(frames.get("^NSEI"))
    market = compute_trend(benchmark) if len(benchmark) >= 200 else {"trend_regime": "UNKNOWN"}
    if not benchmark.empty and (market_now().date() - benchmark.index[-1].date()).days > 5:
        market = {**market, "trend_regime": "UNKNOWN", "warning": "Benchmark history is stale"}
    if benchmark.attrs.get("invalid_bars_removed", 0):
        market = {**market, "trend_regime": "UNKNOWN", "warning": "Invalid benchmark price bars were removed"}
    signals, trend_filtered, failed = [], [], []
    for completed, ticker in enumerate(tickers, 1):
        await asyncio.sleep(0)
        if progress:
            progress({"phase": "analysis", "completed": completed - 1, "total": len(tickers),
                      "ticker": ticker, "percent": 40 + 45 * (completed - 1) / len(tickers),
                      "failed_count": len(failed), "message": "Checking EMA, trend, liquidity and risk"})
        frame = frames.get(ticker)
        if frame is None or len(frame) < 31:
            failed.append({"ticker": ticker, "reason": "Missing or insufficient daily history"})
            continue
        frame = clean_ohlcv(frame)
        if len(frame) < 31:
            failed.append({"ticker": ticker, "reason": "Insufficient validated completed daily history"})
            continue
        try:
            result = analyze_ticker_ema9(ticker, frame, max_candles_ago, target_pct)
        except Exception as exc:
            failed.append({"ticker": ticker, "reason": str(exc)})
            continue
        if result is None:
            continue
        result["company_name"] = names.get(ticker, ticker)
        if not benchmark.empty and result["as_of"] < str(benchmark.index[-1].date()):
            result["warnings"].append("Stock history is behind the latest completed Nifty session")
            result["is_actionable"] = False
        regime = result["trend_regime"]
        if require_uptrend and regime != "UPTREND" and not (allow_sideways and regime == "SIDEWAYS"):
            trend_filtered.append({"ticker": ticker, "regime": regime, "reason": "Trend filter"})
            continue
        pair = pd.concat([clean_ohlcv(frame).Close.rename("stock"), benchmark.Close.rename("market")], axis=1).dropna() if not benchmark.empty else pd.DataFrame()
        relative = None
        if len(pair) >= 61 and str(pair.index[-1].date()) == result["as_of"]:
            relative = (pair.stock.iloc[-1]/pair.stock.iloc[-61] - pair.market.iloc[-1]/pair.market.iloc[-61])*100
        result["relative_strength_60d_pct"] = _number(relative)
        result["market_regime"] = market["trend_regime"]
        if require_relative_strength and (relative is None or relative <= 0):
            result["warnings"].append("Stock has not outperformed Nifty over matched 60-session history")
            result["is_actionable"] = False
        if require_market_uptrend and market["trend_regime"] != "UPTREND":
            result["warnings"].append("Nifty uptrend is unverified")
            result["is_actionable"] = False
        signals.append(result)
    if progress:
        progress({"phase": "analysis", "completed": len(tickers), "total": len(tickers),
                  "ticker": None, "percent": 85, "failed_count": len(failed)})
    for completed, sig in enumerate(signals, 1):
        if progress:
            progress({"phase": "valuation", "completed": completed - 1, "total": len(signals),
                      "ticker": sig["ticker"], "percent": 85 + 14 * (completed - 1) / max(1, len(signals)),
                      "message": "Checking valuation evidence for candidate setups"})
        if not sig["is_actionable"] and not enrich_watchlist:
            fv = {"valuation_warnings": ["Valuation deferred for watch-only setups; use stock research for the full valuation."]}
        else:
            try:
                fv = await asyncio.to_thread(compute_fair_value, sig["ticker"], current_price=sig["current_price"])
            except Exception as exc:
                fv = {"valuation_warnings": [str(exc)]}
        price = _number(fv.get("composite_fair_price"))
        gap = _number((price/sig["current_price"]-1)*100) if price and price > 0 else None
        confidence = fv.get("valuation_confidence", "unavailable")
        sig.update({"composite_fair_price": price, "composite_gain_pct": gap, "fair_gap_pct": gap,
                    "valuation_bucket": "UNDERVALUED" if gap is not None and gap >= 5 else
                        ("OVERVALUED" if gap is not None and gap <= -5 else "FAIR" if gap is not None else "N/A"),
                    "fv_model_count": fv.get("model_count", 0), "valuation_confidence": confidence,
                    "valuation_as_of": fv.get("valuation_as_of"), "valuation_warnings": fv.get("valuation_warnings", []),
                    "is_recommended": bool(sig["is_actionable"] and gap is not None and gap >= 5 and confidence in ("high", "medium"))})
    if progress:
        progress({"phase": "valuation", "completed": len(signals), "total": len(signals),
                  "ticker": None, "percent": 99, "failed_count": len(failed), "message": "Preparing ranked results"})
    signals.sort(key=lambda s: (s["is_actionable"], s["setup_score"], -s["candles_ago"]), reverse=True)
    eligible = [s for s in signals if s["setup_score"] >= min_setup_score and (not actionable_only or s["is_actionable"])]
    prime = [s for s in eligible if s["is_recommended"]]
    technical = [s for s in eligible if s["is_actionable"] and not s["is_recommended"]]
    watchlist = [s for s in eligible if not s["is_actionable"]]
    selected = prime if require_fair_value_undervalued else eligible
    return {"signals": selected, "prime_targets": prime, "technical_only": technical, "watchlist": watchlist,
            "total_scanned": len(tickers), "total_signals": len(signals), "returned_count": len(selected),
            "prime_count": len(prime), "technical_count": len(technical), "watchlist_count": len(watchlist),
            "trend_filtered_count": len(trend_filtered), "trend_filtered": trend_filtered,
            "data_failed_count": len(failed), "data_failures": failed,
            "market_context": {**market, "benchmark": "^NSEI", "as_of": str(benchmark.index[-1].date()) if not benchmark.empty else None},
            "scan_policy": {"completed_daily_bars_only": True, "target_basis": "next_session_open",
                            "score_is_probability": False, "minimum_history_days_requested": max(365, lookback_days),
                            "require_market_uptrend": require_market_uptrend, "require_relative_strength": require_relative_strength}}
