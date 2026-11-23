"""Duration-study oracles: target frequency, execution, censoring and risk."""
import asyncio
import datetime as dt
import json

import pandas as pd
import pytest

from core import duration_backtest
from core.duration_backtest import (
    backtest_duration_ticker,
    compute_duration_stats,
    simulate_duration_trade,
)


def bars(closes, start="2024-01-01"):
    return pd.DataFrame({
        "Open": closes,
        "High": [price + 0.25 for price in closes],
        "Low": [price - 0.25 for price in closes],
        "Close": closes,
        "Volume": 100000.,
    }, index=pd.bdate_range(start, periods=len(closes))).astype(float)


def set_bar(frame, index, opening, high, low, close):
    frame.iloc[index, frame.columns.get_indexer(["Open", "High", "Low", "Close"])] = (
        opening, high, low, close
    )


def study(frame, entry=0, horizon=3, mode="legacy_target_study", **kwargs):
    return simulate_duration_trade(
        frame, entry, float(frame.Close.iloc[entry]), target_pct=3,
        max_hold_sessions=horizon, mode=mode,
        slippage_bps=0, commission_bps=0, **kwargs,
    )


def test_legacy_close_entry_cannot_use_entry_day_target_or_entry_day_mae():
    frame = bars([100, 100, 100])
    set_bar(frame, 0, 100, 140, 1, 100)
    trade = study(frame, horizon=2)
    assert trade["status"] == "TIME_EXIT"
    assert trade["target_hit"] is False
    assert trade["exit_idx"] == 2
    assert trade["mae_pct"] == pytest.approx(-0.25)


def test_legacy_horizon_includes_last_allowed_bar():
    frame = bars([100, 100, 100, 102])
    set_bar(frame, 3, 100, 103, 99, 102)
    trade = study(frame, entry=1, horizon=2)
    assert trade["status"] == "TARGET_HIT"
    assert trade["exit_idx"] == 3
    assert trade["target_price"] == 103
    assert trade["hold_sessions"] == 2
    assert trade["full_horizon_available"] is True


def test_legacy_target_after_expiry_does_not_rewrite_the_loss():
    frame = bars([100, 100, 100, 98, 110])
    trade = study(frame, entry=1, horizon=2)
    assert trade["status"] == "TIME_EXIT"
    assert trade["exit_idx"] == 3
    assert trade["exit_price"] == 98
    assert trade["net_pnl_pct"] == -2
    assert trade["outcome"] == "LOSS"
    assert trade["target_hit"] is False


def test_legacy_short_tail_is_censored_without_a_fabricated_exit():
    frame = bars([100, 98])
    trade = study(frame, horizon=3)
    assert trade["status"] == "CENSORED"
    assert trade["outcome"] == "OPEN"
    assert trade["target_hit"] is False
    assert trade["full_horizon_available"] is False
    assert trade["exit_price"] is None
    assert trade["net_pnl_pct"] is None
    assert trade["hold_sessions"] == 1
    assert trade["mae_pct"] == pytest.approx(-2.25)


def test_early_target_win_with_short_followup_is_not_a_mature_entry():
    frame = bars([100, 102])
    set_bar(frame, 1, 100, 104, 99, 102)
    trade = study(frame, horizon=90)
    assert trade["status"] == "TARGET_HIT"
    assert trade["target_hit"] is True
    assert trade["full_horizon_available"] is False
    stats = compute_duration_stats([trade])
    assert stats["target_hits"] == 1
    assert stats["all_entry_target_hit_rate_pct"] == 100
    assert stats["mature_entries"] == 0
    assert stats["mature_target_hit_rate_pct"] is None


def test_no_stop_can_report_a_target_win_after_a_sixty_percent_drawdown():
    frame = bars([100, 60, 70, 89, 103])
    set_bar(frame, 1, 90, 99, 50, 60)
    set_bar(frame, 2, 60, 80, 40, 70)
    set_bar(frame, 3, 70, 90, 55, 89)
    set_bar(frame, 4, 95, 104, 95, 103)
    trade = study(frame, horizon=4)
    assert trade["status"] == "TARGET_HIT"
    assert trade["net_pnl_pct"] == 3
    assert trade["hold_sessions"] == 4
    assert trade["mae_pct"] == -60
    stats = compute_duration_stats([trade])
    assert stats["all_entry_target_hit_rate_pct"] == 100
    assert stats["worst_mae_pct"] == -60


def test_legacy_cost_diagnostic_preserves_close_anchored_target():
    frame = bars([100, 102])
    set_bar(frame, 1, 100, 104, 99, 102)
    trade = simulate_duration_trade(
        frame, 0, 100, target_pct=3, max_hold_sessions=2,
        mode="legacy_target_study", slippage_bps=5, commission_bps=10,
    )
    assert trade["entry_price"] == 100
    assert trade["target_price"] == 103
    assert trade["gross_pnl_pct"] == 3
    assert trade["net_pnl_pct"] == pytest.approx(
        (103 * .999 / (100 * 1.0005 * 1.001) - 1) * 100, abs=.0002
    )


def test_next_open_entry_can_hit_a_target_on_its_entry_bar_without_double_slippage():
    frame = bars([100])
    set_bar(frame, 0, 100, 104, 99, 102)
    fill = 100 * 1.0005
    trade = simulate_duration_trade(
        frame, 0, fill, target_pct=3, max_hold_sessions=2,
        mode="duration_next_open", slippage_bps=5, commission_bps=10,
    )
    assert trade["status"] == "TARGET_HIT"
    assert trade["exit_idx"] == 0
    assert trade["hold_sessions"] == 1
    assert trade["target_price"] == pytest.approx(fill * 1.03, abs=.0002)
    assert trade["net_pnl_pct"] == pytest.approx(
        (fill * 1.03 * .999 / (fill * 1.001) - 1) * 100, abs=.0002
    )


def test_next_open_expiry_waits_for_a_fill_and_overrides_next_day_target():
    pending_frame = bars([100, 100, 100])
    pending = study(pending_frame, entry=1, horizon=2, mode="duration_next_open")
    assert pending["status"] == "TIME_EXIT_PENDING"
    assert pending["outcome"] == "OPEN"
    assert pending["exit_price"] is None
    assert pending["net_pnl_pct"] is None

    frame = bars([100, 100, 100, 110])
    trade = study(frame, entry=1, horizon=2, mode="duration_next_open")
    assert trade["status"] == "TIME_EXIT"
    assert trade["target_hit"] is False
    assert trade["exit_idx"] == 3
    assert trade["exit_price"] == 110
    assert trade["outcome"] == "WIN"


def test_strict_cross_requires_previous_close_below_ema_and_no_confirmation():
    equal_previous = bars([100] * 60 + [101, 101])
    assert backtest_duration_ticker(
        "TEST", equal_previous, require_uptrend=False, slippage_bps=0, commission_bps=0
    ) == []

    unconfirmed = bars([100] * 60 + [99, 101, 100])
    trades = backtest_duration_ticker(
        "TEST", unconfirmed, require_uptrend=False, slippage_bps=0, commission_bps=0
    )
    assert len(trades) == 1
    assert trades[0]["entry_idx"] == 61
    assert trades[0]["entry_price"] == 101


def test_legacy_final_cross_is_censored_but_next_open_without_a_bar_is_not_entered():
    frame = bars([100] * 60 + [99, 101])
    legacy = backtest_duration_ticker("TEST", frame, mode="legacy_target_study")
    assert len(legacy) == 1
    assert legacy[0]["status"] == "CENSORED"
    assert legacy[0]["hold_sessions"] == 0
    assert legacy[0]["mae_pct"] == 0
    assert backtest_duration_ticker("TEST", frame, mode="duration_next_open") == []


def test_legacy_crosses_overlap_but_next_open_mode_has_one_position_per_ticker():
    frame = bars([100] * 60 + [99, 101, 99, 101] + [101] * 10)
    set_bar(frame, 62, 103, 104, 98, 99)
    legacy = backtest_duration_ticker(
        "TEST", frame, mode="legacy_target_study", require_uptrend=False,
        slippage_bps=0, commission_bps=0,
    )
    practical = backtest_duration_ticker(
        "TEST", frame, mode="duration_next_open", require_uptrend=False,
        slippage_bps=5, commission_bps=0,
    )
    assert [trade["entry_idx"] for trade in legacy] == [61, 63]
    assert len(practical) == 1
    assert practical[0]["entry_idx"] == 62
    assert practical[0]["entry_price"] == pytest.approx(103 * 1.0005, abs=.0002)


def test_entry_window_is_inclusive_and_does_not_truncate_trade_followup():
    frame = bars([100] * 60 + [99, 101, 99, 101, 99, 101] + [101] * 95)
    date = str(frame.index[63].date())
    trades = backtest_duration_ticker(
        "TEST", frame, start_date=date, end_date=date, require_uptrend=False,
        slippage_bps=0, commission_bps=0,
    )
    assert len(trades) == 1
    assert trades[0]["entry_date"] == date
    assert trades[0]["entry_idx"] == 63
    assert trades[0]["full_horizon_available"] is True
    assert trades[0]["exit_idx"] == 153


def test_stats_separate_all_entry_mature_closed_and_net_profit_denominators():
    specifications = [
        ("TARGET_HIT", True, True, 2.7, 5, -8),
        ("TIME_EXIT", False, True, 1.0, 90, -6),
        ("TIME_EXIT", False, True, -10.0, 90, -15),
        ("TARGET_HIT", True, False, 2.7, 2, -4),
        ("CENSORED", False, False, None, 10, -20),
        ("TIME_EXIT_PENDING", False, False, None, 90, -12),
    ]
    trades = []
    for index, (status, hit, mature, net, hold, mae) in enumerate(specifications):
        trades.append({
            "ticker": f"T{index}", "entry_date": "2024-01-01", "entry_price": 100.,
            "target_price": 103., "entry_idx": 0,
            "exit_date": "2024-05-01" if net is not None else None,
            "exit_price": 100. + net if net is not None else None,
            "exit_idx": hold if net is not None else None, "status": status,
            "target_hit": hit, "full_horizon_available": mature, "hold_sessions": hold,
            "mae_pct": mae, "gross_pnl_pct": net, "net_pnl_pct": net,
            "outcome": "OPEN" if net is None else "WIN" if net > 0 else "LOSS",
            "max_hold_sessions": 90,
        })
    stats = compute_duration_stats(trades)
    assert stats["total_entries"] == 6
    assert stats["closed_count"] == 4
    assert stats["target_hits"] == 2
    assert stats["all_entry_target_hit_rate_pct"] == pytest.approx(100 / 3, abs=.05)
    assert stats["mature_entries"] == 3
    assert stats["mature_target_hit_rate_pct"] == pytest.approx(100 / 3, abs=.05)
    assert stats["closed_target_hit_rate_pct"] == 50
    assert stats["open_count"] == 2
    assert stats["net_win_rate_pct"] == 75
    assert stats["profit_factor"] == pytest.approx(.64)
    assert stats["avg_net_return_pct"] == pytest.approx(-.9)
    assert stats["worst_net_loss_pct"] == -10
    assert stats["worst_mae_pct"] == -20
    assert stats["median_target_hold_sessions"] == 3.5
    assert stats.get("total_return_pct") is None
    json.dumps(stats, allow_nan=False)


def test_overlapping_same_ticker_returns_are_not_an_investable_compounded_curve():
    frame = bars([100] * 60 + [99, 101, 99, 101] + [101] * 10)
    legacy = backtest_duration_ticker("TEST", frame, mode="legacy_target_study")
    stats = compute_duration_stats(legacy)
    assert stats["max_concurrent_positions"] == 2
    assert stats.get("total_return_pct") is None
    assert not stats.get("equity_curve")


def test_prior_resistance_gate_is_causal_and_rejects_before_practical_occupancy():
    frame = bars([100] * 60 + [99, 104, 99, 101] + [101] * 10)
    set_bar(frame, 55, 100, 105, 99, 100)
    legacy = backtest_duration_ticker(
        "TEST", frame, mode="legacy_target_study", slippage_bps=0, commission_bps=0,
    )
    assert [trade["entry_idx"] for trade in legacy] == [61, 63]
    assert legacy[0]["resistance_price"] == 105
    assert legacy[0]["resistance_dist_pct"] == pytest.approx((105 / 104 - 1) * 100, abs=.0002)
    assert legacy[0]["is_near_resistance"] is True
    assert legacy[1]["is_near_resistance"] is False

    filtered = backtest_duration_ticker(
        "TEST", frame, mode="duration_next_open", avoid_near_resistance=True,
        slippage_bps=0, commission_bps=0,
    )
    assert [trade["entry_idx"] for trade in filtered] == [64]
    assert filtered[0]["entry_gap_pct"] == 0

    future = frame.copy()
    set_bar(future, 68, 101, 200, 50, 101)
    changed = backtest_duration_ticker(
        "TEST", future, mode="duration_next_open", avoid_near_resistance=True,
        slippage_bps=0, commission_bps=0,
    )
    assert [trade["entry_idx"] for trade in changed] == [64]
    for key in ("resistance_price", "resistance_dist_pct", "is_near_resistance"):
        assert changed[0][key] == filtered[0][key]


def test_prior_pullback_gate_is_causal_and_rejects_before_practical_occupancy():
    frame = bars([100] * 60 + [99, 110, 99, 105] + [105] * 10)
    set_bar(frame, 55, 100, 120, 99, 100)
    legacy = backtest_duration_ticker(
        "TEST", frame, mode="legacy_target_study", slippage_bps=0, commission_bps=0,
    )
    assert [trade["entry_idx"] for trade in legacy] == [61, 63]
    assert legacy[0]["prior_high"] == 120
    assert legacy[0]["pullback_dist_pct"] == pytest.approx(100 / 12, abs=.0002)
    assert legacy[0]["is_pullback"] is False
    assert legacy[1]["pullback_dist_pct"] == 12.5
    assert legacy[1]["is_pullback"] is True

    filtered = backtest_duration_ticker(
        "TEST", frame, mode="duration_next_open", require_pullback=True,
        slippage_bps=0, commission_bps=0,
    )
    assert [trade["entry_idx"] for trade in filtered] == [64]

    future = frame.copy()
    set_bar(future, 68, 105, 200, 50, 105)
    changed = backtest_duration_ticker(
        "TEST", future, mode="duration_next_open", require_pullback=True,
        slippage_bps=0, commission_bps=0,
    )
    assert [trade["entry_idx"] for trade in changed] == [64]
    for key in ("prior_high", "pullback_dist_pct", "is_pullback"):
        assert changed[0][key] == filtered[0][key]


def test_async_runner_processes_all_305_requested_tickers_and_reports_coverage(monkeypatch):
    requested = [f"TEST{index:03d}" for index in range(305)]
    captured, updates = {}, []

    def missing_download(tickers, interval, lookback_days, progress=None):
        captured.update(tickers=tickers, interval=interval, lookback_days=lookback_days)
        if progress:
            for completed, ticker in enumerate(tickers, 1):
                progress({"phase": "download", "completed": completed,
                          "total": len(tickers), "ticker": ticker})
        return {}

    monkeypatch.setattr(duration_backtest, "fetch_batch_ohlcv", missing_download)
    result = asyncio.run(duration_backtest.run_duration_backtest(requested, years=5, progress=updates.append))
    assert captured["tickers"] == requested
    assert captured["interval"] == "1d"
    assert captured["lookback_days"] == 5 * 365 + 365
    assert result["parameters"]["tickers_count"] == 305
    assert len(result["data_skipped"]) == 305
    assert [row["ticker"] for row in result["data_skipped"]] == requested
    assert result["stats"]["total_entries"] == 0
    assert result["stats"]["closed_count"] == 0
    assert result["stats"]["all_entry_target_hit_rate_pct"] is None
    assert result["trades"] == []
    assert result["coverage"] == []
    assert updates[-1]["completed"] == updates[-1]["total"] == 305
    assert updates[-1]["failed_count"] == 305
    assert all(update["total"] == 305 for update in updates)


def test_current_fair_value_only_changes_the_labeled_retrospective_cohort(monkeypatch):
    frame = bars([100] * 60 + [99, 101, 99, 101] + [101] * 95)
    current_snapshot = {"price": 200.}
    cache_reads = []

    def download(tickers, interval, lookback_days, progress=None):
        return {"TEST": frame.copy()}

    def cached_value(ticker, current_price=None):
        cache_reads.append((ticker, current_price))
        return {"composite_fair_price": current_snapshot["price"],
                "valuation_as_of": "2024-03-31", "valuation_confidence": "medium"}

    monkeypatch.setattr(duration_backtest, "fetch_batch_ohlcv", download)
    monkeypatch.setattr(duration_backtest, "get_cached_fair_value", cached_value)
    monkeypatch.setattr(duration_backtest, "market_now", lambda: dt.datetime(2025, 7, 1, tzinfo=dt.timezone.utc))
    baseline = asyncio.run(duration_backtest.run_duration_backtest(["TEST"]))
    assert cache_reads == []
    above = asyncio.run(duration_backtest.run_duration_backtest(["TEST"], include_current_fair_value=True))
    current_snapshot["price"] = 50.
    below = asyncio.run(duration_backtest.run_duration_backtest(["TEST"], include_current_fair_value=True))

    assert baseline["stats"]["total_entries"] == 2
    assert baseline["stats"] == above["stats"] == below["stats"]
    entry_fields = ("entry_date", "entry_price", "target_price", "status", "target_hit", "net_pnl_pct")
    entries = lambda result: [[trade[key] for key in entry_fields] for trade in result["trades"]]
    assert entries(baseline) == entries(above) == entries(below)
    above_cohort = above["retrospective_current_fair_value_cohort"]
    below_cohort = below["retrospective_current_fair_value_cohort"]
    assert above_cohort["stats"]["total_entries"] == 2
    assert below_cohort["stats"]["total_entries"] == 0
    assert above_cohort["historical_validation"] is False
    assert below_cohort["historical_validation"] is False
    assert "Retrospective" in above_cohort["label"]
    assert "not known" in above_cohort["caveat"]
    assert above_cohort["snapshots"][0]["composite_fair_price"] == 200
    assert below_cohort["snapshots"][0]["composite_fair_price"] == 50


