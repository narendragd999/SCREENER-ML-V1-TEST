"""No-stop EMA duration research, with explicit horizons and censored outcomes.

The close-reference study preserves the user's target-touch question. The
next-open comparison models executable entries and one position per ticker.
Neither mode compounds overlapping positions into a fictional portfolio.
"""
import asyncio
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.data_fetcher import clean_ohlcv, fetch_batch_ohlcv, market_now, normalize_ticker
from core.fair_value import get_cached_fair_value
from core.strategy import indicator_frame, trend_regime_mask


MODES = ("legacy_target_study", "duration_next_open")


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _number(value, digits=4):
    number = _finite(value)
    return round(number, digits) if number is not None else None


def _date(value):
    return str(value.date()) if hasattr(value, "date") else str(value)[:10]


def _validate(target_pct, max_hold_sessions, mode, slippage_bps, commission_bps):
    if mode not in MODES:
        raise ValueError("Unknown duration mode")
    if _finite(target_pct) is None or not 0 < target_pct <= 100:
        raise ValueError("target_pct must be finite and between zero and 100")
    if not isinstance(max_hold_sessions, (int, np.integer)) or not 1 <= max_hold_sessions <= 250:
        raise ValueError("max_hold_sessions must be an integer from 1 to 250")
    if any(_finite(value) is None or not 0 <= value < 10000 for value in (slippage_bps, commission_bps)):
        raise ValueError("Execution costs must be finite basis points from zero to below 10000")


def simulate_duration_trade(
    df: pd.DataFrame, entry_idx: int, entry_price: float,
    target_pct: float = 3.0, max_hold_sessions: int = 90,
    mode: str = "legacy_target_study", slippage_bps: float = 5.0,
    commission_bps: float = 10.0,
) -> Dict[str, Any]:
    """Observe a no-stop trade without inventing an incomplete horizon exit.

    Legacy H means H subsequent sessions after a close-reference entry. In the
    executable comparison the entry session is session one; an unhit Hth close
    decides a market exit at the next available open. Target barriers and fills
    use unrounded prices. MAE conservatively includes the target day's full low.
    """
    _validate(target_pct, max_hold_sessions, mode, slippage_bps, commission_bps)
    if not isinstance(entry_idx, (int, np.integer)) or not 0 <= entry_idx < len(df):
        raise ValueError("entry_idx must identify an available price bar")
    if _finite(entry_price) is None or entry_price <= 0:
        raise ValueError("entry_price must be finite and positive")
    entry_idx = int(entry_idx)
    legacy = mode == "legacy_target_study"
    slip, fee = slippage_bps / 10000.0, commission_bps / 10000.0
    target = float(entry_price) * (1.0 + target_pct / 100.0)
    first = entry_idx + 1 if legacy else entry_idx
    horizon_idx = first + max_hold_sessions - 1
    full_horizon = horizon_idx < len(df)
    modeled_entry = entry_price * (1.0 + slip) if legacy else entry_price
    result = {
        "mode": mode, "entry_idx": entry_idx, "entry_date": _date(df.index[entry_idx]),
        "entry_price": float(entry_price), "target_price": target,
        "modeled_entry_fill": float(modeled_entry), "modeled_exit_fill": None,
        "max_hold_sessions": int(max_hold_sessions), "full_horizon_available": bool(full_horizon),
        "exit_idx": None, "exit_date": None, "exit_price": None,
        "status": "CENSORED", "status_reason": "INCOMPLETE_TARGET_OBSERVATION_HORIZON",
        "outcome": "OPEN", "target_hit": False, "target_hit_sessions": None,
        "hold_sessions": 0, "observation_sessions": 0,
        "gross_pnl_pct": None, "net_pnl_pct": None, "pnl_pct": None,
        "mae_pct": 0.0, "as_of": _date(df.index[-1]),
        "mae_policy": "Daily lows through observation/exit; target-day low is conservative because intraday order is unknown",
    }
    lowest = float(entry_price)

    def finish(j, exit_price, modeled_exit, status, reason, hold):
        gross = (exit_price / entry_price - 1.0) * 100.0
        net = (modeled_exit * (1.0 - fee) / (modeled_entry * (1.0 + fee)) - 1.0) * 100.0
        result.update({
            "exit_idx": int(j), "exit_date": _date(df.index[j]), "exit_price": float(exit_price),
            "modeled_exit_fill": float(modeled_exit), "status": status, "status_reason": reason,
            "target_hit": status == "TARGET_HIT", "target_hit_sessions": int(hold) if status == "TARGET_HIT" else None,
            "hold_sessions": int(hold), "gross_pnl_pct": _number(gross),
            "net_pnl_pct": _number(net), "pnl_pct": _number(net),
            "outcome": "WIN" if net > 0 else "LOSS" if net < 0 else "BREAKEVEN",
            "mae_pct": _number((min(lowest, entry_price) / entry_price - 1.0) * 100.0),
        })
        return result

    for j in range(first, min(len(df), horizon_idx + 1)):
        row = df.iloc[j]
        hold = j - first + 1
        lowest = min(lowest, float(row.Low))
        result["observation_sessions"] = int(hold)
        if float(row.High) >= target:
            # The study retains the old target reference, while next-open fills
            # can receive conservative price improvement on a later target gap.
            gap = not legacy and float(row.Open) >= target
            exit_price = max(target, float(row.Open) * (1.0 - slip)) if gap else target
            return finish(j, exit_price, exit_price, "TARGET_HIT", "TARGET_GAP" if gap else "TARGET_HIT", hold)
        if j == horizon_idx:
            if legacy:
                closing = float(row.Close)
                return finish(j, closing, closing * (1.0 - slip), "TIME_EXIT", "HORIZON_CLOSE_REFERENCE", hold)
            if j + 1 < len(df):
                opening = float(df.Open.iloc[j + 1])
                # Only the exit open belongs to this exposure, not that day's
                # later high/low after the market exit has already occurred.
                lowest = min(lowest, opening)
                fill = opening * (1.0 - slip)
                return finish(j + 1, fill, fill, "TIME_EXIT", "HORIZON_CLOSE_DECISION_NEXT_OPEN", hold + 1)
            result.update(status="TIME_EXIT_PENDING", status_reason="AWAITING_NEXT_OPEN_AFTER_HORIZON_CLOSE")
    result.update({
        "hold_sessions": int(result["observation_sessions"]),
        "mae_pct": _number((min(lowest, entry_price) / entry_price - 1.0) * 100.0),
        "unrealized_pnl_pct": _number((float(df.Close.iloc[-1]) / entry_price - 1.0) * 100.0),
    })
    return result


def backtest_duration_ticker(
    ticker: str, df: pd.DataFrame, target_pct: float = 3.0,
    max_hold_sessions: int = 90, mode: str = "legacy_target_study",
    require_uptrend: bool = False, slippage_bps: float = 5.0,
    commission_bps: float = 10.0, start_date: Optional[Any] = None,
    end_date: Optional[Any] = None, research_filters: bool = False,
    avoid_near_resistance: bool = False, require_pullback: bool = False,
) -> List[Dict[str, Any]]:
    """Causal strict EMA crosses; optional SMA50 and untuned research gates."""
    _validate(target_pct, max_hold_sessions, mode, slippage_bps, commission_bps)
    bars = clean_ohlcv(df, completed_only=True)
    if len(bars) < 60:
        return []
    bars = indicator_frame(bars)
    cross = (bars.Close.shift() < bars.ema9.shift()) & (bars.Close > bars.ema9)
    # Preserve the legacy 50-session warmup rather than searching an unstable
    # EMA immediately after the first observed close.
    cross.iloc[:51] = False
    trend = trend_regime_mask(bars) if research_filters else None
    start = pd.Timestamp(start_date).date() if start_date is not None else None
    end = pd.Timestamp(end_date).date() if end_date is not None else None
    if start and end and start > end:
        raise ValueError("start_date must not be after end_date")
    trades, occupied_until = [], -1
    for i in np.flatnonzero(cross.fillna(False).to_numpy()):
        i = int(i)
        row = bars.iloc[i]
        recent_high = float(bars.High.iloc[max(0, i - 20):i].max())
        resistance = recent_high if recent_high > float(row.Close) else None
        resistance_distance = (resistance / float(row.Close) - 1.0) * 100.0 if resistance else None
        near_resistance = resistance is not None and resistance <= float(row.Close) * 1.03
        prior_high = float(bars.High.iloc[max(0, i - 60):i].max())
        pullback_distance = (prior_high - float(row.Close)) / prior_high * 100.0 if prior_high > 0 else None
        pullback = prior_high > 0 and float(row.Close) <= prior_high * 0.90
        # Prior-window annotations use only information at the completed cross
        # close. These predefined legacy hypotheses do not inspect outcomes.
        if avoid_near_resistance and near_resistance:
            continue
        if require_pullback and not pullback:
            continue
        if require_uptrend and (pd.isna(row.sma50) or row.Close <= row.sma50):
            continue
        if research_filters:
            extension = (row.Close - row.ema9) / row.atr14 if row.atr14 > 0 else float("nan")
            if (trend.iloc[i] != "UPTREND" or pd.isna(row.volume_ratio) or row.volume_ratio < 1.5
                    or not math.isfinite(extension) or not 0 <= extension <= 1.0):
                continue
        entry_idx = i if mode == "legacy_target_study" else i + 1
        if entry_idx >= len(bars):
            # There is no executable buy price yet; this signal is not an entry.
            continue
        entry_day = bars.index[entry_idx].date()
        if (start is not None and entry_day < start) or (end is not None and entry_day > end):
            continue
        if mode == "duration_next_open" and entry_idx <= occupied_until:
            continue
        entry = float(row.Close) if mode == "legacy_target_study" else float(bars.Open.iloc[entry_idx]) * (1.0 + slippage_bps / 10000.0)
        trade = simulate_duration_trade(bars, entry_idx, entry, target_pct,
                                       max_hold_sessions, mode, slippage_bps, commission_bps)
        trade.update({
            "ticker": normalize_ticker(ticker), "breakout_date": _date(bars.index[i]),
            "signal_date": _date(bars.index[i]), "entry_basis": "breakout_close_reference" if mode == "legacy_target_study" else "next_session_open_with_slippage",
            "volume_ratio_at_signal": _number(row.volume_ratio),
            "ema_extension_atr_at_signal": _number((row.Close - row.ema9) / row.atr14) if row.atr14 > 0 else None,
            "research_filters": bool(research_filters),
            "resistance_price": _number(resistance), "resistance_dist_pct": _number(resistance_distance),
            "is_near_resistance": bool(near_resistance), "prior_high": _number(prior_high),
            "pullback_dist_pct": _number(pullback_distance), "is_pullback": bool(pullback),
            "entry_gap_pct": _number((float(bars.Open.iloc[entry_idx]) / float(row.Close) - 1.0) * 100.0) if mode == "duration_next_open" else None,
        })
        trades.append(trade)
        if mode == "duration_next_open":
            occupied_until = trade["exit_idx"] if trade["exit_idx"] is not None else len(bars)
    return trades


def _distribution(values):
    values = [float(v) for v in values if _finite(v) is not None]
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None, "max": None}
    return {"count": len(values), "min": _number(min(values)), "median": _number(np.median(values)),
            "p90": _number(np.percentile(values, 90)), "max": _number(max(values))}


def compute_duration_stats(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Target cohorts and equal-weight net trade diagnostics, without equity."""
    entries = [t for t in trades if _finite(t.get("entry_price")) is not None and t["entry_price"] > 0]
    closed = [t for t in entries if _finite(t.get("net_pnl_pct")) is not None and t.get("exit_date") is not None]
    target_hits = [t for t in entries if t.get("target_hit") is True]
    mature = [t for t in entries if t.get("full_horizon_available") is True]
    mature_hits = [t for t in mature if t.get("target_hit") is True]
    returns = [float(t["net_pnl_pct"]) for t in closed]
    profit, loss = sum(r for r in returns if r > 0), -sum(r for r in returns if r < 0)
    ratio = lambda numerator, denominator: _number(numerator / denominator * 100.0) if denominator else None
    target_holds = [t.get("target_hit_sessions", t.get("hold_sessions")) for t in target_hits]
    maes = [t.get("mae_pct") for t in entries if _finite(t.get("mae_pct")) is not None]
    horizon = max([int(t.get("max_hold_sessions", 90)) for t in entries], default=90)
    points = sorted({n for n in (1, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 250, horizon) if n <= horizon})
    curve = [{"sessions": n, "target_hits": sum(t.get("target_hit") is True and (t.get("target_hit_sessions") or t.get("hold_sessions", 0)) <= n for t in mature),
              "mature_entries": len(mature), "target_hit_rate_pct": ratio(sum(t.get("target_hit") is True and (t.get("target_hit_sessions") or t.get("hold_sessions", 0)) <= n for t in mature), len(mature))}
             for n in points]
    # Inclusive daily exposures are a conservative overlap bound; intraday
    # target times cannot be recovered from daily OHLC. No returns compound.
    events = []
    for trade in entries:
        events.append((trade["entry_date"], 0, 1))
        if trade.get("exit_date"):
            events.append((trade["exit_date"], 1, -1))
    exposure = maximum = 0
    for _, _, delta in sorted(events):
        exposure += delta
        maximum = max(maximum, exposure)
    target_distribution = _distribution(target_holds)
    stats = {
        "total_entries": len(entries), "closed_count": len(closed), "target_hits": len(target_hits),
        "all_entry_target_hit_rate_pct": ratio(len(target_hits), len(entries)),
        "mature_entries": len(mature), "mature_target_hits": len(mature_hits),
        "mature_target_hit_rate_pct": ratio(len(mature_hits), len(mature)),
        "closed_target_hit_rate_pct": ratio(sum(t.get("target_hit") is True for t in closed), len(closed)),
        "censored_count": sum(t.get("status") == "CENSORED" for t in entries),
        "pending_time_exit_count": sum(t.get("status") == "TIME_EXIT_PENDING" for t in entries),
        "open_count": len(entries) - len(closed),
        "net_wins": sum(r > 0 for r in returns), "net_losses": sum(r < 0 for r in returns),
        "net_win_rate_pct": ratio(sum(r > 0 for r in returns), len(closed)),
        "profit_factor": _number(profit / loss) if loss > 0 else None,
        "profit_factor_status": "AVAILABLE" if loss > 0 else "NO_CLOSED_LOSSES" if closed else "NO_CLOSED_TRADES",
        "avg_net_return_pct": _number(np.mean(returns)) if returns else None,
        "worst_net_loss_pct": _number(min([r for r in returns if r < 0], default=None)),
        "worst_mae_pct": _number(min(maes)) if maes else None,
        "median_mae_pct": _number(np.median(maes)) if maes else None,
        "median_target_hold_sessions": target_distribution["median"], "p90_target_hold_sessions": target_distribution["p90"],
        "hold_sessions_distribution": {"all_observed": _distribution([t.get("hold_sessions") for t in entries]),
                                      "closed": _distribution([t.get("hold_sessions") for t in closed]), "target_hits": target_distribution},
        "mae_pct_distribution": _distribution(maes), "max_concurrent_positions": int(maximum),
        "max_concurrent_positions_policy": "Conservative daily overlap bound with inclusive entry/exit dates; no capital allocation implied",
        "horizon_target_curve": curve,
        "net_return_policy": "Equal-weight per-entry returns; unresolved entries excluded from closed net statistics",
        "mature_cohort_policy": "Full configured target-observation horizon available even when an entry hit its target early",
        "portfolio_return_pct": None,
    }
    # Readable aliases support callers without collapsing distinct denominators.
    stats.update(target_hit_rate_all_entries_pct=stats["all_entry_target_hit_rate_pct"],
                 mature_cohort_target_hit_rate_pct=stats["mature_target_hit_rate_pct"],
                 closed_entries=stats["closed_count"], worst_closed_net_loss_pct=stats["worst_net_loss_pct"])
    return stats


async def run_duration_backtest(
    tickers: List[str], years: int = 5, target_pct: float = 3.0,
    max_hold_sessions: int = 90, mode: str = "legacy_target_study",
    require_uptrend: bool = False, slippage_bps: float = 5.0,
    commission_bps: float = 10.0, research_filters: bool = False,
    include_current_fair_value: bool = False, avoid_near_resistance: bool = False,
    require_pullback: bool = False, progress=None,
) -> Dict[str, Any]:
    """Full requested universe, completed history and actual ticker progress."""
    _validate(target_pct, max_hold_sessions, mode, slippage_bps, commission_bps)
    if not isinstance(years, int) or not 1 <= years <= 10:
        raise ValueError("years must be an integer from 1 to 10")
    symbols = list(dict.fromkeys(normalize_ticker(t) for t in tickers if normalize_ticker(t)))
    if not symbols:
        raise ValueError("At least one ticker is required")
    cutoff = pd.Timestamp(market_now().date()) - pd.Timedelta(days=years * 365)

    def download_progress(update):
        if progress:
            progress({**update, "percent": 40 * update["completed"] / max(1, update["total"]),
                      "message": "Fetching completed bars for the no-stop duration study"})

    kwargs = {"progress": download_progress} if progress else {}
    frames = await asyncio.to_thread(fetch_batch_ohlcv, symbols, "1d", years * 365 + 365, **kwargs)
    trades, skipped, coverage, snapshots = [], [], [], []
    retrospective = []
    for completed, ticker in enumerate(symbols, 1):
        if progress:
            progress({"phase": "analysis", "completed": completed - 1, "total": len(symbols), "ticker": ticker,
                      "percent": 40 + 59 * (completed - 1) / len(symbols), "failed_count": len(skipped),
                      "message": "Observing target horizons, censored entries and adverse excursions"})
        bars = clean_ohlcv(frames.get(ticker), completed_only=True)
        if len(bars) < 60:
            skipped.append({"ticker": ticker, "reason": "MISSING_OR_SHORT_COMPLETED_HISTORY", "available_bars": len(bars)})
            continue
        ticker_trades = await asyncio.to_thread(backtest_duration_ticker, ticker, bars, target_pct,
                                               max_hold_sessions, mode, require_uptrend, slippage_bps,
                                               commission_bps, cutoff, None, research_filters,
                                               avoid_near_resistance, require_pullback)
        row = {"ticker": ticker, "available_bars": len(bars), "start_date": _date(bars.index[0]), "as_of": _date(bars.index[-1]),
               "requested_entry_start_date": str(cutoff.date()), "entries": len(ticker_trades),
               "data_source": bars.attrs.get("data_source", "unknown"),
               "invalid_bars_removed": int(bars.attrs.get("invalid_bars_removed", 0)),
               "incomplete_bars_removed": int(bars.attrs.get("incomplete_bars_removed", 0))}
        coverage.append(row)
        if include_current_fair_value:
            current = float(bars.Close.iloc[-1])
            fv = get_cached_fair_value(ticker, current_price=current) or {}
            price = _finite(fv.get("composite_fair_price"))
            if price is not None and price > 0:
                snapshots.append({"ticker": ticker, "composite_fair_price": price,
                                  "valuation_as_of": fv.get("valuation_as_of"), "valuation_confidence": fv.get("valuation_confidence"),
                                  "current_price_as_of": row["as_of"], "retrieved_at": market_now().isoformat()})
                for trade in ticker_trades:
                    trade.update(current_fair_value_snapshot=price,
                                 current_fv_snapshot_valuation_as_of=fv.get("valuation_as_of"),
                                 retrospective_current_fv_undervalued=trade["entry_price"] < price)
                    if trade["entry_price"] < price:
                        retrospective.append(trade)
        trades.extend(ticker_trades)
    stats = compute_duration_stats(trades)
    # A no-entry run still needs the selected horizon for a useful empty curve.
    if not trades:
        stats["horizon_target_curve"] = [{"sessions": n, "target_hits": 0, "mature_entries": 0, "target_hit_rate_pct": None}
                                         for n in sorted({n for n in (1, 3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 250, max_hold_sessions) if n <= max_hold_sessions})]
    breakdowns = [{"ticker": t, "stats": compute_duration_stats([trade for trade in trades if trade["ticker"] == t]),
                   "as_of": next((c["as_of"] for c in coverage if c["ticker"] == t), None)} for t in symbols]
    legacy = mode == "legacy_target_study"
    warnings = [
        "No stop-loss is applied in either duration mode. Large adverse excursions, overnight gaps and capital lockup are intentional risks of this research question.",
        "Target-hit frequency is distinct from a profitable closed-trade rate. All-entry, mature-cohort and closed-trade denominators are reported separately.",
        "The horizon curve uses only entries with the full configured observation horizon, including early winners, available at the data cutoff.",
        "MAE conservatively uses the target day's whole daily low; intraday high/low sequence is unknown.",
        "Per-entry statistics are not a portfolio return. Current registries can create survivorship bias; ticker/time-correlated entries are not independent observations.",
        "Current valuation is never used as a historical entry gate. Its optional cohort is retrospective research only and uses available cached snapshots.",
    ]
    if legacy:
        warnings.append("Same-close entries are idealized reference prices. Modeled net estimates apply requested execution costs without changing the reference target-touch membership.")
        warnings.append("Every cross may overlap earlier entries. Final-candle close-reference signals remain censored rather than being omitted as in the old route.")
    else:
        warnings.append("Signals without a subsequent opening price are not counted as executed entries. Time exits decided at the final horizon close remain pending until the next opening is observed.")
        warnings.append("Optional resistance/pullback gates are evaluated at the signal close. A next-open gap can change the actual entry's resistance distance and reward; entry_gap_pct is reported separately.")
    if research_filters:
        warnings.append("The SMA200/trend, signal-volume and ATR-extension research variant is predefined but unvalidated; fewer signals do not establish better performance.")
    if avoid_near_resistance or require_pullback:
        warnings.append("The prior-20-session resistance and prior-60-session pullback gates are predefined legacy hypotheses, not thresholds fitted to this result.")
    if progress:
        progress({"phase": "analysis", "completed": len(symbols), "total": len(symbols), "ticker": None,
                  "percent": 99, "failed_count": len(skipped), "message": "Preparing duration cohorts and coverage"})
    return {
        "parameters": {"mode": mode, "tickers_count": len(symbols), "years": years, "target_pct": target_pct,
                       "max_hold_sessions": max_hold_sessions, "require_uptrend": bool(require_uptrend),
                       "slippage_bps": slippage_bps, "commission_bps": commission_bps,
                       "research_filters": bool(research_filters), "include_current_fair_value": bool(include_current_fair_value),
                       "avoid_near_resistance": bool(avoid_near_resistance), "require_pullback": bool(require_pullback)},
        "stats": stats, "trades": sorted(trades, key=lambda t: (t["entry_date"], t["ticker"])),
        "ticker_breakdowns": breakdowns, "data_skipped": skipped, "coverage": coverage,
        "as_of": max([c["as_of"] for c in coverage], default=None),
        "execution_assumptions": [
            "Completed validated daily bars; strict EMA9 crossover with 50-session warmup",
            "Optional original trend gate is close above SMA50 at the signal",
            "Breakout close reference; overlapping entries; H subsequent target sessions" if legacy else
                "Signal known at close; slipped next-open entry; one position per ticker; entry session counts as one",
            "Fixed unrounded target; no stop-loss; ideal sell-limit fills at observed target barriers",
            "Reference close at horizon, modeled sell costs; incomplete horizons remain censored" if legacy else
                "Hth close decides time exit at next available slipped open; absent execution stays pending",
            "Requested per-side commission; legacy modeled fills reserve requested slippage without changing reference target",
            "No compounding of overlapping positions and no implied portfolio equity curve",
        ],
        "research_filter_policy": {"enabled": bool(research_filters), "verified_sma50_sma200_uptrend": True,
                                   "minimum_signal_volume_ratio": 1.5, "maximum_signal_extension_atr": 1.0,
                                   "avoid_near_resistance": bool(avoid_near_resistance), "resistance_lookback_sessions": 20,
                                   "near_resistance_threshold_pct": 3.0, "require_pullback": bool(require_pullback),
                                   "pullback_lookback_sessions": 60, "minimum_pullback_pct": 10.0,
                                   "validated": False},
        "retrospective_current_fair_value_cohort": {
            "requested": bool(include_current_fair_value), "available": bool(snapshots),
            "label": "Retrospective entries below a cached current fair-value estimate",
            "stats": compute_duration_stats(retrospective) if snapshots else None, "snapshots": snapshots,
            "tickers_with_snapshot": len(snapshots), "tickers_without_snapshot": len(coverage) - len(snapshots),
            "historical_validation": False,
            "caveat": "The same current snapshot is compared with historical entries. It was not known at those entry dates and cannot establish historical valuation effectiveness.",
        },
        "warnings": warnings,
    }
