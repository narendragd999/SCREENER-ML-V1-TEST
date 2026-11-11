"""
tracker.py — Active No-Stop Position Tracker & Trade Journal
"""
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from core.data_fetcher import fetch_ticker_ohlcv, normalize_ticker, market_now

BASE_DIR = Path(__file__).resolve().parent.parent
TRADES_FILE = BASE_DIR / "active_trades.json"


def _load_raw_trades() -> List[Dict[str, Any]]:
    if not TRADES_FILE.exists():
        return []
    try:
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {TRADES_FILE}: {e}")
        return []


def _save_raw_trades(trades: List[Dict[str, Any]]) -> bool:
    try:
        with open(TRADES_FILE, "w", encoding="utf-8") as f:
            json.dump(trades, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving {TRADES_FILE}: {e}")
        return False


def get_tracked_positions() -> Dict[str, Any]:
    """Retrieve all tracked positions with summary statistics."""
    trades = _load_raw_trades()
    open_trades = [t for t in trades if t.get("status") == "OPEN"]
    hit_trades = [t for t in trades if t.get("status") == "TARGET_HIT"]

    total_hits = len(hit_trades)
    hit_days = [t["hold_sessions"] for t in hit_trades if t.get("hold_sessions")]
    avg_hit_days = round(sum(hit_days) / len(hit_days), 1) if hit_days else 0.0

    return {
        "total_trades": len(trades),
        "open_trades_count": len(open_trades),
        "target_hit_count": total_hits,
        "hit_rate_pct": round((total_hits / len(trades) * 100.0), 1) if trades else 0.0,
        "avg_sessions_to_hit": avg_hit_days,
        "positions": trades,
    }


def add_tracked_position(
    symbol: str,
    entry_price: float,
    entry_date: Optional[str] = None,
    target_pct: float = 3.0,
    capital: float = 100000.0,
    notes: str = "",
) -> Dict[str, Any]:
    """Add a new position to track under the No-Stop policy."""
    trades = _load_raw_trades()
    trade_id = str(uuid.uuid4())[:8]
    date_str = entry_date or str(market_now().date())
    target_price = round(entry_price * (1.0 + target_pct / 100.0), 2)
    shares = int(capital / entry_price) if entry_price > 0 else 0

    new_trade = {
        "id": trade_id,
        "symbol": symbol.strip().upper(),
        "entry_date": date_str,
        "entry_price": round(entry_price, 2),
        "target_pct": float(target_pct),
        "target_price": target_price,
        "capital": capital,
        "shares": shares,
        "notes": notes,
        "status": "OPEN",  # OPEN or TARGET_HIT
        "current_price": round(entry_price, 2),
        "unrealized_pnl_pct": 0.0,
        "mae_pct": 0.0,
        "lowest_price": round(entry_price, 2),
        "hold_sessions": 1,
        "exit_date": None,
        "exit_price": None,
        "realized_pnl_pct": None,
        "created_at": datetime.now().isoformat(),
    }

    trades.insert(0, new_trade)
    _save_raw_trades(trades)
    return new_trade


def remove_tracked_position(trade_id: str) -> bool:
    """Remove a tracked trade by ID."""
    trades = _load_raw_trades()
    filtered = [t for t in trades if t.get("id") != trade_id]
    if len(filtered) != len(trades):
        _save_raw_trades(filtered)
        return True
    return False


def refresh_tracked_positions() -> Dict[str, Any]:
    """
    Downloads latest market data for all OPEN positions:
    - Checks if High >= target_price (Marks TARGET_HIT)
    - Updates mark-to-market unrealized P&L
    - Updates MAE (maximum adverse excursion)
    """
    trades = _load_raw_trades()
    updated = False

    for trade in trades:
        if trade.get("status") != "OPEN":
            continue

        sym = trade["symbol"]
        df = fetch_ticker_ohlcv(sym, period="6mo")
        if df.empty:
            continue

        entry_date_str = trade["entry_date"]
        entry_price = trade["entry_price"]
        target_price = trade["target_price"]

        # Slice data from entry date onwards
        bars = df[df.index >= pd.to_datetime(entry_date_str)]
        if bars.empty:
            continue

        lowest = float(bars.Low.min())
        mae_pct = round(((lowest / entry_price) - 1.0) * 100.0, 2)
        trade["lowest_price"] = round(lowest, 2)
        trade["mae_pct"] = mae_pct

        # Check session by session for target hit
        target_hit = False
        for idx, bar in bars.iterrows():
            if float(bar.High) >= target_price:
                trade["status"] = "TARGET_HIT"
                trade["exit_date"] = str(idx.date())
                trade["exit_price"] = target_price
                trade["realized_pnl_pct"] = trade["target_pct"]
                trade["hold_sessions"] = len(bars.loc[:idx])
                trade["current_price"] = target_price
                trade["unrealized_pnl_pct"] = trade["target_pct"]
                target_hit = True
                updated = True
                break

        if not target_hit:
            latest_close = float(bars.Close.iloc[-1])
            trade["current_price"] = round(latest_close, 2)
            trade["unrealized_pnl_pct"] = round(((latest_close / entry_price) - 1.0) * 100.0, 2)
            trade["hold_sessions"] = len(bars)
            updated = True

    if updated:
        _save_raw_trades(trades)

    return get_tracked_positions()
