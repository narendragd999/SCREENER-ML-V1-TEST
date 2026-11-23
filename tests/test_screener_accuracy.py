"""Offline numerical regressions for screening and user-facing analysis."""
import asyncio
import datetime as dt
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from core import data_fetcher, screener
from core.strategy import confirmed_breakout_mask, compute_trend
from main import app


def frame(closes=None):
    closes = closes or [100]*30 + [99, 101, 102, 103, 103]
    result = pd.DataFrame({"Open": closes, "High": np.array(closes)+.5,
                           "Low": np.array(closes)-.5, "Close": closes,
                           "Volume": 1_000_000}, index=pd.bdate_range("2025-01-01", periods=len(closes)))
    return result.astype(float)


def test_incomplete_session_cannot_create_daily_signal():
    bars = frame()
    bars.index = pd.bdate_range(end="2026-11-10", periods=len(bars), tz="Asia/Kolkata")
    now = dt.datetime(2026, 11, 10, 14, 0, tzinfo=data_fetcher.IST)
    cleaned = data_fetcher.clean_ohlcv(bars, now=now)
    assert str(cleaned.index[-1].date()) == "2026-11-09"
    assert cleaned.attrs["incomplete_bars_removed"] == 1
    after = data_fetcher.clean_ohlcv(bars, now=now.replace(hour=16))
    assert str(after.index[-1].date()) == "2026-11-10"


def test_exchange_dates_do_not_shift_to_previous_day():
    bars = frame()
    bars.index = bars.index.tz_localize("Asia/Kolkata")
    cleaned = data_fetcher.clean_ohlcv(bars)
    assert list(cleaned.index.date) == list(bars.index.date)


def test_historical_weekend_special_session_is_preserved():
    bars = frame()
    special = bars.iloc[:1].copy()
    special.index = pd.to_datetime(["2025-11-09"])
    cleaned = data_fetcher.clean_ohlcv(pd.concat([bars, special]))
    assert pd.Timestamp("2025-11-09") in cleaned.index


def test_chart_uses_the_selected_journal_entry():
    import main
    entry = {"ticker": "TEST", "trade_id": "old", "status": "TARGET_HIT",
             "entry_price": 90, "target_3pct": 99, "stop_loss_price": 85, "stop_mode": "PRICE"}
    with patch.object(main, "fetch_ticker_ohlcv", return_value=frame()), \
         patch.object(main, "load_signals", return_value=[entry]), \
         patch.object(main, "compute_fair_value", return_value={}):
        result = TestClient(app).get("/api/chart/TEST?trade_id=old")
    assert result.status_code == 200
    assert result.json()["target_3pct"] == 99
    assert result.json()["stop_loss"] == 85
    assert result.json()["tracked_entry"]["status"] == "TARGET_HIT"


def test_invalid_bars_duplicates_and_sorting_are_handled():
    bars = frame()
    bars.loc[bars.index[1], "High"] = 90
    bars.loc[bars.index[2], "Close"] = np.inf
    bars = pd.concat([bars, bars.iloc[3:4]]).iloc[::-1]
    cleaned = data_fetcher.clean_ohlcv(bars)
    assert cleaned.index.is_monotonic_increasing and cleaned.index.is_unique
    assert len(cleaned) == 33
    assert cleaned.attrs["invalid_bars_removed"] == 3


def test_batch_date_padding_is_not_a_corrupt_candle():
    bars = frame()
    padded = pd.DataFrame({name: np.nan for name in bars.columns}, index=[pd.Timestamp("2024-12-31")])
    cleaned = data_fetcher.clean_ohlcv(pd.concat([padded, bars]))
    assert len(cleaned) == len(bars)
    assert cleaned.attrs["invalid_bars_removed"] == 0
    assert cleaned.attrs["empty_alignment_rows_removed"] == 1


def test_single_ticker_column_layouts_supported_multiple_tickers_rejected():
    bars = frame()
    bars.columns = pd.MultiIndex.from_product([bars.columns, ["TCS.NS"]])
    assert len(data_fetcher.clean_ohlcv(bars)) == 35
    doubled = pd.concat([bars, bars.rename(columns={"TCS.NS": "INFY.NS"})], axis=1)
    assert data_fetcher.clean_ohlcv(doubled).empty
    lower = frame().rename(columns=str.lower)
    assert "Close" in data_fetcher.clean_ohlcv(lower)


def test_cache_invalidates_at_close_and_correct_exchange_symbols():
    before = dt.datetime(2026, 11, 10, 15, 25, tzinfo=data_fetcher.IST).timestamp()
    after = dt.datetime(2026, 11, 10, 15, 31, tzinfo=data_fetcher.IST).timestamp()
    assert not data_fetcher._usable_cache(before, after, "1d")
    assert data_fetcher.yahoo_symbol("TCS.NS") == "TCS.NS"
    assert data_fetcher.yahoo_symbol("500325.BO") == "500325.BO"
    assert data_fetcher.yahoo_symbol("^NSEI") == "^NSEI"
    assert data_fetcher.normalize_ticker(np.nan) == ""


def test_confirmation_requires_higher_high_and_is_causal():
    bars = frame()
    bars.loc[bars.index[31], "High"] = 110
    assert not confirmed_breakout_mask(bars).iloc[32]
    original = confirmed_breakout_mask(frame().iloc[:33])
    extended = confirmed_breakout_mask(frame())
    pd.testing.assert_series_equal(original, extended.iloc[:33])


def test_short_history_never_claims_verified_uptrend():
    assert compute_trend(frame(list(np.linspace(100, 150, 80))))["trend_regime"] == "UNKNOWN"
    bullish = compute_trend(frame(list(np.linspace(100, 180, 240))))
    assert bullish["trend_regime"] == "UPTREND" and bullish["above_200sma"]


def test_target_anchored_to_next_open_and_remaining_distance_changes():
    bars = frame()
    result = screener.analyze_ticker_ema9("TEST", bars)
    assert result["entry_ref_price"] == 103
    assert result["target_3pct"] == 106.09
    bars.loc[bars.index[-1], ["Open", "High", "Low", "Close"]] = [104, 104.5, 103.5, 104]
    moved = screener.analyze_ticker_ema9("TEST", bars)
    assert moved["target_3pct"] == result["target_3pct"]
    assert moved["distance_to_target_pct"] < result["distance_to_target_pct"]


def test_intraday_target_hit_is_terminal_even_after_price_falls():
    bars = frame()
    bars.loc[bars.index[33], "High"] = 110
    result = screener.analyze_ticker_ema9("TEST", bars)
    assert result["signal_status"] == "TARGET_HIT"
    assert not result["is_actionable"]


def test_historical_ema_failure_does_not_reactivate_on_recovery():
    bars = frame([100]*30+[99,101,102,101,98,103])
    result = screener.analyze_ticker_ema9("TEST", bars)
    assert result["signal_status"] == "STOPPED_OUT"
    assert result["exit_price"] == 103
    assert not result["is_actionable"]


def test_missing_volume_is_watchlist_only():
    result = screener.analyze_ticker_ema9("TEST", frame().drop(columns="Volume"))
    assert result["volume_ratio"] is None
    assert not result["is_actionable"]
    assert any("Volume" in text for text in result["warnings"])


def test_breakout_window_boundary_and_fresh_without_entry():
    bars = frame().iloc[:33]
    result = screener.analyze_ticker_ema9("TEST", bars, 1)
    assert result["candles_ago"] == 0 and result["entry_date"] is None
    assert result["entry_basis"] == "confirmation_close_estimate"
    assert screener.analyze_ticker_ema9("TEST", frame(), 1) is None


def test_scan_gap_uses_screen_price_and_reports_missing_tickers():
    bars = frame()
    with patch.object(screener, "fetch_batch_ohlcv", return_value={"TEST": bars}), \
         patch.object(screener, "compute_fair_value", return_value={"composite_fair_price": 120,
             "composite_gain_pct": -50, "valuation_confidence": "medium", "model_count": 2}) as valuation:
        results = asyncio.run(screener.run_ema9_screener(["TEST", "MISSING"], require_uptrend=False, enrich_watchlist=True))
    signal = results["signals"][0]
    assert signal["fair_gap_pct"] == pytest.approx(16.5)
    assert valuation.call_args.kwargs["current_price"] == 103
    assert results["data_failed_count"] == 1
    assert not signal["is_recommended"]
    assert signal["score_is_probability"] is False


def test_position_size_respects_cash_risk_and_costs():
    client = TestClient(app)
    response = client.post("/api/position-size", json={"capital": 100_000, "risk_pct": 1,
                           "entry_price": 100, "stop_price": 95, "target_price": 110})
    assert response.status_code == 200
    data = response.json()
    assert data["quantity"] == 188
    assert data["max_loss"] <= 1000 and data["capital_required"] <= 100_000
    assert data["net_reward_risk"] == 1.83


def test_api_rejects_impossible_trade_and_invalid_parameters():
    client = TestClient(app)
    bad = client.post("/api/position-size", json={"capital": 1000, "entry_price": 100,
                       "stop_price": 101, "target_price": 110})
    assert bad.status_code == 400
    assert client.post("/api/screener/scan", json={"target_pct": 0}).status_code == 422
    assert client.post("/api/signals", json={"ticker": "TCS", "entry_price": 0}).status_code == 422


def test_config_fallback_is_valid_python():
    import main
    with patch.object(main, "CONFIG_FILE") as config:
        config.exists.return_value = False
        assert main._read_config()["require_uptrend"] is True


def test_score_filter_applies_to_every_returned_partition():
    with patch.object(screener, "fetch_batch_ohlcv", return_value={"TEST": frame()}), \
         patch.object(screener, "compute_fair_value", return_value={}):
        result = asyncio.run(screener.run_ema9_screener(["TEST"], require_uptrend=False, min_setup_score=99))
    assert not result["signals"] and not result["watchlist"] and not result["technical_only"]
    assert not result["prime_targets"] and result["total_signals"] == 1


def test_corrupt_history_is_reported_as_failure():
    bars = frame()
    bars.loc[bars.index[:10], "High"] = 1
    with patch.object(screener, "fetch_batch_ohlcv", return_value={"TEST": bars}):
        result = asyncio.run(screener.run_ema9_screener(["TEST"], require_uptrend=False))
    assert result["data_failed_count"] == 1
    assert "validated" in result["data_failures"][0]["reason"]


def test_journal_delete_by_trade_id_preserves_reentry(monkeypatch):
    from core import signals
    rows = [{"trade_id": "closed", "ticker": "TEST", "status": "TARGET_HIT"},
            {"trade_id": "new", "ticker": "TEST", "status": "ACTIVE"}]
    monkeypatch.setattr(signals, "load_signals", lambda: rows)
    monkeypatch.setattr(signals, "save_signals", lambda rows: True)
    assert signals.delete_signal("closed") == [rows[1]]


def test_fixed_allocation_keeps_nontrading_tickers_in_cash():
    from core.backtest import compute_performance_stats
    trade = {"ticker": "A", "entry_date": "2025-01-01", "exit_date": "2025-01-02",
             "outcome": "WIN", "pnl_pct": 10, "hold_candles": 1}
    result = compute_performance_stats([trade], capital_tickers=["A", "B"])
    assert result["total_return_pct"] == 5
