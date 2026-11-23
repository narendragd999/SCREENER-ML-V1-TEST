"""Deterministic execution cases that expose inflated screener outcomes."""
import json
import math

import pandas as pd
import pytest

from core.backtest import backtest_single_ticker, compute_performance_stats, simulate_long_trade
from core import signals


def bars(closes, start="2024-01-01"):
    frame = pd.DataFrame({"Open": closes, "High": [c+0.5 for c in closes], "Low": [c-0.5 for c in closes], "Close": closes, "Volume": 100000}, index=pd.bdate_range(start, periods=len(closes)))
    return frame.astype(float)


def set_bar(frame, pos, opening, high, low, close):
    frame.iloc[pos, frame.columns.get_indexer(["Open", "High", "Low", "Close"])] = [opening, high, low, close]


def test_confirmation_day_high_cannot_hit_a_post_confirmation_entry():
    frame = bars([100]*30+[99, 101, 102, 103, 103])
    set_bar(frame, 32, 101.5, 120, 101, 102)
    trades = backtest_single_ticker("TEST", frame, require_uptrend=False, slippage_bps=0, commission_bps=0)
    assert len(trades) == 1
    trade = trades[0]
    assert trade["entry_date"] == str(frame.index[33].date())
    assert trade["entry_price"] == 103
    assert trade["target_price"] == pytest.approx(106.09)
    assert trade["outcome"] == "OPEN"
    assert trade["exit_price"] is None


def test_ema_close_stop_executes_next_open_before_new_day_target():
    frame = bars([100]*31+[98, 101])
    set_bar(frame, 32, 97, 120, 96, 101)
    result = simulate_long_trade(frame, 30, 100, slippage_bps=0, commission_bps=0)
    assert result["status"] == "STOPPED_OUT"
    assert result["exit_date"] == str(frame.index[32].date())
    assert result["exit_price"] == 97
    assert result["net_pnl_pct"] == -3


def test_last_bar_ema_breach_stays_pending():
    result = simulate_long_trade(bars([100]*31+[98]), 30, 100)
    assert result["status"] == "STOP_PENDING"
    assert result["pending_reason"] == "EMA9_BREACH"
    assert result["exit_price"] is None
    assert result["net_pnl_pct"] is None


def test_target_gap_gets_open_improvement_and_fixed_target():
    frame = bars([100]*30+[108])
    result = simulate_long_trade(frame, 29, 100, include_entry_bar=False, slippage_bps=0, commission_bps=0)
    assert result["status"] == "TARGET_HIT"
    assert result["target_price"] == 103
    assert result["exit_price"] == 108
    assert result["gross_pnl_pct"] == 8


def test_ambiguous_price_stop_bar_uses_stop_first():
    frame = bars([100])
    set_bar(frame, 0, 100, 105, 95, 100)
    result = simulate_long_trade(frame, 0, 100, stop_mode="PRICE", stop_loss_price=97, slippage_bps=0, commission_bps=0)
    assert result["status"] == "STOPPED_OUT"
    assert result["status_reason"] == "AMBIGUOUS_BAR_STOP_FIRST"
    assert result["exit_price"] == 97


def test_price_stop_gap_fills_open_not_unavailable_stop():
    frame = bars([90])
    result = simulate_long_trade(frame, 0, 100, stop_mode="PRICE", stop_loss_price=97, slippage_bps=0, commission_bps=0)
    assert result["exit_price"] == 90
    assert result["status_reason"] == "PRICE_STOP_GAP"


def test_entry_day_high_is_excluded_for_intraday_saved_entry():
    frame = bars([100]*32)
    set_bar(frame, 30, 100, 120, 99, 100)
    result = simulate_long_trade(frame, 30, 100, include_entry_bar=False, slippage_bps=0, commission_bps=0)
    assert result["status"] == "ACTIVE"


def test_commission_reduces_target_return_and_limit_has_no_negative_slippage():
    frame = bars([100])
    set_bar(frame, 0, 100, 104, 99, 102)
    result = simulate_long_trade(frame, 0, 100, commission_bps=10, slippage_bps=50)
    assert result["exit_price"] == 103
    assert result["gross_pnl_pct"] == 3
    assert result["net_pnl_pct"] == pytest.approx((103*0.999/(100*1.001)-1)*100, abs=0.0001)


def test_holding_expiry_needs_next_available_open():
    frame = bars([100]*32)
    pending = simulate_long_trade(frame, 30, 100, max_hold_candles=2, slippage_bps=0, commission_bps=0)
    assert pending["status"] == "TIME_EXIT_PENDING"
    assert pending["exit_price"] is None
    frame = bars([100]*32+[102])
    exited = simulate_long_trade(frame, 30, 100, max_hold_candles=2, slippage_bps=0, commission_bps=0)
    assert exited["status"] == "TIME_EXIT"
    assert exited["exit_price"] == 102


def test_static_fair_price_cannot_be_used_for_historical_gate():
    frame = bars([100]*30+[99, 101, 102, 103, 103])
    trades = backtest_single_ticker("TEST", frame, fair_price=1000, require_fair_value=True, require_uptrend=False)
    assert trades[0]["outcome"] == "SKIPPED_FAIR_VALUE_UNAVAILABLE"
    assert compute_performance_stats(trades)["total_trades"] == 0


def test_historical_fair_value_is_available_only_after_publication():
    frame = bars([100]*30+[99, 101, 102, 103, 103])
    history = pd.Series([200.0], index=[frame.index[34]])
    trade = backtest_single_ticker("TEST", frame, fair_value_history=history, require_fair_value=True, require_uptrend=False)[0]
    assert trade["outcome"] == "SKIPPED_FAIR_VALUE_UNAVAILABLE"


def test_same_day_after_close_valuation_is_not_known_at_confirmation():
    frame = bars([100]*30+[99, 101, 102, 103, 103])
    history = pd.Series([200.0], index=[frame.index[32]+pd.Timedelta(hours=16)])
    trade = backtest_single_ticker("TEST", frame, fair_value_history=history, require_fair_value=True, require_uptrend=False)[0]
    assert trade["outcome"] == "SKIPPED_FAIR_VALUE_UNAVAILABLE"


def test_warmup_signals_do_not_open_positions_before_requested_start():
    frame = bars([100]*30+[99, 101, 102, 103, 103])
    assert backtest_single_ticker("TEST", frame, require_uptrend=False, start_date=frame.index[34]) == []


def test_overlapping_ticker_returns_are_not_sequentially_compounded():
    trades = [{"ticker": ticker, "entry_date": "2024-01-01", "exit_date": "2024-01-03", "outcome": "WIN", "pnl_pct": 10, "hold_candles": 3, "status": "TIME_EXIT"} for ticker in ("A", "B")]
    result = compute_performance_stats(trades)
    assert result["total_return_pct"] == 10
    assert result["equity_curve"][-1]["equity"] == 110
    assert result["profit_factor"] is None
    assert result["profit_factor_unbounded"] is True
    assert result["target_hit_rate_pct"] == 0
    json.dumps(result, allow_nan=False)


def test_open_trades_excluded_from_win_rate_and_profit_factor():
    trades = [{"ticker": "A", "entry_date": "2024-01-01", "exit_date": "2024-01-03", "outcome": "WIN", "pnl_pct": 1, "hold_candles": 3}, {"ticker": "B", "outcome": "OPEN", "pnl_pct": None}]
    result = compute_performance_stats(trades)
    assert result["total_trades"] == 1
    assert result["open_count"] == 1
    assert result["win_rate_pct"] == 100
    assert result["win_rate_ci95_pct"][0] < 30


def test_tracker_replays_missed_historical_target_before_latest_price():
    frame = bars([100]*36)
    set_bar(frame, 32, 100, 105, 99, 100)
    sig = {"ticker": "TEST", "entry_price": 100, "entry_date": str(frame.index[30].date()), "entry_at_open": False, "status": "ACTIVE", "commission_bps": 0, "slippage_bps": 0}
    result = signals.evaluate_signal(sig, frame)
    assert result["status"] == "TARGET_HIT"
    assert result["exit_date"] == str(frame.index[32].date())
    assert result["final_pnl_pct"] == 3


def test_tracker_does_not_retroactively_use_pre_entry_high():
    frame = bars([100]*36)
    set_bar(frame, 30, 100, 120, 99, 100)
    sig = {"ticker": "TEST", "entry_price": 100, "entry_date": str(frame.index[30].date()), "status": "ACTIVE"}
    result = signals.evaluate_signal(sig, frame)
    assert result["status"] == "ACTIVE"
    assert result["target_3pct"] == 103


def test_tracker_missing_entry_date_does_not_infer_confirmation_trade():
    sig = {"ticker": "TEST", "entry_price": 100, "confirm_date": "2024-01-01", "status": "ACTIVE"}
    result = signals.evaluate_signal(sig, bars([100]*40))
    assert result["status"] == "ACTIVE"
    assert "Missing entry date" in result["tracking_warning"]


def test_signal_save_anchors_target_and_preserves_closed_trade(tmp_path, monkeypatch):
    path = tmp_path/"signals.json"
    monkeypatch.setattr(signals, "SIGNALS_FILE", str(path))
    old = {"ticker": "TEST", "entry_price": 50, "entry_date": "2024-01-01", "status": "TARGET_HIT", "final_pnl_pct": 2.8}
    path.write_text(json.dumps([old]), encoding="utf-8")
    rows = signals.add_or_update_signal({"ticker": "TEST.NS", "entry_price": 100, "target_3pct": 200})
    assert len(rows) == 2
    assert rows[0]["status"] == "ACTIVE"
    assert rows[0]["target_3pct"] == 103
    assert rows[0]["entry_at_open"] is False
    assert rows[1]["status"] == "TARGET_HIT"
    assert signals.get_signals_summary()["win_rate_pct"] == 100


def test_summary_counts_profitable_ema_exit_as_win_not_target(tmp_path, monkeypatch):
    path = tmp_path/"signals.json"
    monkeypatch.setattr(signals, "SIGNALS_FILE", str(path))
    path.write_text(json.dumps([{"ticker": "A", "status": "STOPPED_OUT", "final_pnl_pct": 1.0}, {"ticker": "B", "status": "ACTIVE", "current_gain_pct": 100.0}]), encoding="utf-8")
    summary = signals.get_signals_summary()
    assert summary["win_rate_pct"] == 100
    assert summary["target_hit_rate_pct"] == 0
    assert summary["avg_gain_pct"] == 1
    assert summary["avg_active_gain_pct"] == 100


@pytest.mark.parametrize("bad", [0, -1, math.nan, math.inf])
def test_invalid_buy_fill_is_rejected(bad):
    with pytest.raises(ValueError):
        simulate_long_trade(bars([100]), 0, bad)
