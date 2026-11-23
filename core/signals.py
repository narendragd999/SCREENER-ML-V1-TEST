"""Saved equity entries, causal outcome replay, and closed-trade statistics."""
import asyncio
import datetime as dt
import json
import math
import os
import uuid
from typing import Any, Dict, List

import pandas as pd

from core.backtest import simulate_long_trade
from core.data_fetcher import clean_ohlcv, fetch_ticker_ohlcv, normalize_ticker

SIGNALS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "signals.json")
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
CLOSED_STATUSES = ("TARGET_HIT", "STOPPED_OUT", "TIME_EXIT")


def _now() -> dt.datetime:
    return dt.datetime.now(IST)


def load_signals() -> List[Dict[str, Any]]:
    if not os.path.exists(SIGNALS_FILE):
        return []
    try:
        with open(SIGNALS_FILE, "r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except Exception as exc:
        print(f"[Signals] Error loading {SIGNALS_FILE}: {exc}")
        return []


def save_signals(signals: List[Dict[str, Any]]) -> bool:
    try:
        with open(SIGNALS_FILE, "w", encoding="utf-8") as handle:
            json.dump(signals, handle, indent=2, allow_nan=False)
        return True
    except Exception as exc:
        print(f"[Signals] Error saving {SIGNALS_FILE}: {exc}")
        return False


def add_or_update_signal(sig: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Track a real entry; preserve prior closed trades when re-entering a ticker.

    Without an explicit entry date/time, a saved pick is a new entry today.
    Today's entire high cannot be used for an entry made partway through today.
    """
    incoming = {k: v for k, v in sig.items() if v is not None}
    ticker = normalize_ticker(incoming.get("ticker", ""))
    if not ticker:
        raise ValueError("ticker is required")
    entry = float(incoming.get("entry_price", 0))
    target_pct = float(incoming.get("target_pct", 3.0))
    if not math.isfinite(entry) or entry <= 0 or not math.isfinite(target_pct) or target_pct <= 0:
        raise ValueError("Entry price and target percentage must be finite and positive")
    if incoming.get("stop_mode", "EMA9_CLOSE") not in ("EMA9_CLOSE", "PRICE"):
        raise ValueError("stop_mode must be EMA9_CLOSE or PRICE")
    if incoming.get("stop_mode") == "PRICE":
        stop = float(incoming.get("stop_loss_price", 0))
        if not math.isfinite(stop) or not 0 < stop < entry:
            raise ValueError("PRICE stop must be positive and below entry")
    now = _now()
    stamp = now.isoformat(timespec="seconds")
    signals = load_signals()
    existing_idx = next((i for i, row in enumerate(signals) if normalize_ticker(row.get("ticker", "")) == ticker and row.get("status") not in CLOSED_STATUSES), None)
    if existing_idx is not None:
        existing = signals[existing_idx]
        changed_entry = not math.isclose(float(existing.get("entry_price", entry)), entry)
        if changed_entry:
            # A corrected buy fill needs a new replay; previous exits/marks are stale.
            for key in ("exit_date", "exit_price", "final_pnl_pct", "gross_final_pnl_pct", "pending_reason", "evaluated_through"):
                existing.pop(key, None)
        incoming = {**existing, **incoming}
    incoming.update({
        "ticker": ticker, "entry_price": entry, "target_pct": target_pct,
        "target_3pct": round(entry*(1+target_pct/100), 4),
        "updated_at": stamp,
    })
    incoming.setdefault("trade_id", uuid.uuid4().hex)
    incoming.setdefault("created_at", stamp)
    incoming.setdefault("entry_date", now.date().isoformat())
    incoming.setdefault("entry_at_open", False)
    incoming.setdefault("entry_time", stamp)
    incoming.setdefault("stop_mode", "EMA9_CLOSE")
    incoming.setdefault("max_hold_candles", 20)
    incoming.setdefault("slippage_bps", 5.0)
    incoming.setdefault("commission_bps", 10.0)
    incoming["status"] = "ACTIVE" if existing_idx is None or changed_entry else incoming.get("status", "ACTIVE")
    if existing_idx is None:
        signals.insert(0, incoming)
    else:
        signals[existing_idx] = incoming
    if not save_signals(signals):
        raise OSError("Failed to save tracked signal")
    return signals


def delete_signal(ticker: str) -> List[Dict[str, Any]]:
    rows = load_signals()
    if any(row.get("trade_id") == ticker for row in rows):
        signals = [row for row in rows if row.get("trade_id") != ticker]
        save_signals(signals)
        return signals
    target = normalize_ticker(ticker)
    signals = [row for row in rows if normalize_ticker(row.get("ticker", "")) != target]
    save_signals(signals)
    return signals


def clear_signals() -> bool:
    return save_signals([])


def evaluate_signal(sig: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Replay all eligible bars after a saved entry, without touching storage."""
    result = dict(sig)
    if result.get("status") in CLOSED_STATUSES:
        return result
    df = clean_ohlcv(df, completed_only=True)
    if df.empty:
        result["tracking_warning"] = "No completed price history available"
        return result
    entry = float(result.get("entry_price", 0))
    if not math.isfinite(entry) or entry <= 0:
        result["tracking_warning"] = "Missing actual entry price; outcome not inferred"
        return result
    # Legacy rows must have a trustworthy time boundary; confirmation date alone
    # does not establish that the user entered on that date.
    boundary = result.get("entry_date") or result.get("entry_time") or result.get("created_at")
    if not boundary:
        result["tracking_warning"] = "Missing entry date; set it before evaluating outcomes"
        return result
    try:
        entry_date = pd.Timestamp(boundary)
        if entry_date.tzinfo is not None:
            entry_date = entry_date.tz_convert("Asia/Kolkata").tz_localize(None)
        entry_date = entry_date.normalize()
    except (ValueError, TypeError):
        result["tracking_warning"] = "Invalid entry date; outcome not inferred"
        return result
    dates = pd.DatetimeIndex(df.index).normalize()
    if dates[0] > entry_date and (dates[0]-entry_date).days > 7:
        result["tracking_warning"] = "Entry predates downloaded history; earlier outcomes cannot be verified"
        return result
    eligible = dates >= entry_date if result.get("entry_at_open", False) else dates > entry_date
    positions = [i for i, allowed in enumerate(eligible) if allowed]
    current = float(df["Close"].iloc[-1])
    target_pct = float(result.get("target_pct", 3.0))
    result.update({
        "current_price": round(current, 4),
        "ema9_current": round(float(df["Close"].ewm(span=9, adjust=False).mean().iloc[-1]), 4),
        "current_gain_pct": round((current/entry-1)*100, 4),
        "target_3pct": round(entry*(1+target_pct/100), 4),
        "target_progress_pct": round((current/entry-1)*100/target_pct*100, 2),
        "updated_at": _now().isoformat(timespec="seconds"),
        "data_as_of": str(dates[-1].date()),
    })
    if not positions:
        result["tracking_warning"] = "Awaiting completed bars after entry; entry-day high excluded unless entry_at_open is set"
        return result
    if positions[0] < 29:
        result["tracking_warning"] = "Insufficient pre-entry history to initialize EMA9 reliably"
        return result
    replay = simulate_long_trade(
        df, positions[0], entry, target_pct,
        int(result.get("max_hold_candles", 20)),
        float(result.get("slippage_bps", 5.0)),
        float(result.get("commission_bps", 10.0)),
        stop_mode=result.get("stop_mode", "EMA9_CLOSE"),
        stop_loss_price=result.get("stop_loss_price"),
        entry_date=str(entry_date.date()),
    )
    result.pop("tracking_warning", None)
    result.update({"status": replay["status"], "pending_reason": replay["pending_reason"], "status_reason": replay["status_reason"], "evaluated_through": str(dates[-1].date()), "hold_candles": replay["hold_candles"]})
    if replay["exit_date"]:
        result.update({"exit_date": replay["exit_date"], "exit_price": replay["exit_price"], "final_pnl_pct": replay["net_pnl_pct"], "gross_final_pnl_pct": replay["gross_pnl_pct"], "stop_reason": replay["stop_reason"]})
    return result


async def refresh_active_signals() -> List[Dict[str, Any]]:
    signals = load_signals()
    changed = False
    for i, sig in enumerate(signals):
        if sig.get("status") in CLOSED_STATUSES or not sig.get("ticker"):
            continue
        try:
            boundary = sig.get("entry_date") or sig.get("entry_time") or sig.get("created_at")
            days = 365
            if boundary:
                days = max(days, (_now().date()-pd.Timestamp(boundary).date()).days+365)
            df = await asyncio.to_thread(fetch_ticker_ohlcv, sig["ticker"], "1d", days, True)
            if df is None or df.empty:
                continue
            signals[i] = evaluate_signal(sig, df)
            changed = True
        except Exception as exc:
            sig["tracking_warning"] = str(exc)
            changed = True
    if changed:
        save_signals(signals)
    return signals


def get_signals_summary() -> Dict[str, Any]:
    signals = load_signals()
    closed = [row for row in signals if row.get("status") in CLOSED_STATUSES]
    known_closed = [row for row in closed if isinstance(row.get("final_pnl_pct"), (int, float)) and math.isfinite(row["final_pnl_pct"])]
    winners = [row for row in known_closed if row["final_pnl_pct"] > 0]
    losses = [row for row in known_closed if row["final_pnl_pct"] < 0]
    marks = [row.get("current_gain_pct") for row in signals if row.get("status") not in CLOSED_STATUSES]
    marks = [value for value in marks if isinstance(value, (int, float)) and math.isfinite(value)]
    hits = sum(row.get("status") == "TARGET_HIT" for row in closed)
    return {
        "total_signals": len(signals),
        "active_count": sum(row.get("status") not in CLOSED_STATUSES for row in signals),
        "target_hit_count": hits, "stopped_out_count": sum(row.get("status") == "STOPPED_OUT" for row in closed),
        "time_exit_count": sum(row.get("status") == "TIME_EXIT" for row in closed),
        "pending_exit_count": sum(row.get("status") in ("STOP_PENDING", "TIME_EXIT_PENDING") for row in signals),
        "closed_count": len(closed), "closed_with_pnl_count": len(known_closed),
        "wins": len(winners), "losses": len(losses),
        "win_rate_pct": round(len(winners)/len(known_closed)*100, 2) if known_closed else 0.0,
        "target_hit_rate_pct": round(hits/len(closed)*100, 2) if closed else 0.0,
        "avg_gain_pct": round(sum(row["final_pnl_pct"] for row in known_closed)/len(known_closed), 4) if known_closed else 0.0,
        "avg_closed_net_gain_pct": round(sum(row["final_pnl_pct"] for row in known_closed)/len(known_closed), 4) if known_closed else 0.0,
        "avg_active_gain_pct": round(sum(marks)/len(marks), 4) if marks else 0.0,
        "metrics_basis": "Win rate uses positive net realized P&L; target-hit rate is reported separately",
    }
