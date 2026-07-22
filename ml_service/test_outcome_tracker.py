"""
Tests for outcome_tracker.py — the most critical module.

Run: python -m pytest tests/test_outcome_tracker.py -v
"""

import sys
import os
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Use a temp DB for tests
@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Use a fresh temp DB for each test."""
    from db import _db_lock
    import db as db_module

    # Reset singleton, pool, AND module-level `db` reference
    with _db_lock:
        new_db = db_module.Database(tmp_path / "test.db")
        db_module._db_singleton = new_db
        db_module._pool = new_db.pool
        db_module.db = new_db  # critical: update module-level `db` reference

    # Update references in modules that imported `db` at module level
    import outcome_tracker as ot_module
    monkeypatch.setattr(ot_module, "db", new_db)
    monkeypatch.setattr(ot_module, "_tracker", None)

    yield

    with _db_lock:
        db_module._db_singleton = None
        db_module._pool = None
        db_module.db = None


def make_synthetic_df(ticker: str, days: int = 100, uptrend: bool = True) -> pd.DataFrame:
    """Make synthetic OHLCV data."""
    np.random.seed(hash(ticker) % 2**32)
    dates = pd.date_range(end=datetime.now().strftime("%Y-%m-%d"), periods=days, freq="B")
    trend = 0.001 if uptrend else -0.001
    returns = np.random.normal(trend, 0.02, days)
    prices = 100 * np.cumprod(1 + returns)
    df = pd.DataFrame({
        "Open": prices * (1 + np.random.uniform(-0.005, 0.005, days)),
        "High": prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
        "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
        "Close": prices,
        "Volume": np.random.randint(100000, 1000000, days).astype(float),
    }, index=dates)
    df.index.name = "date"
    return df


def stub_feature_extractor(df, normalize=False):
    """Stub that returns a 70-element feature vector."""
    n_features = 70
    if df is None or len(df) < 30:
        return np.zeros(n_features, dtype=np.float32)
    feats = np.random.normal(0, 1, n_features).astype(np.float32)
    # Make features depend on actual data (so they're meaningful)
    if len(df) >= 5:
        feats[0] = (df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100
    return feats


# ── Tests ────────────────────────────────────────────────────────────────────

def test_outcome_tracker_initialization():
    """Tracker can be created."""
    from outcome_tracker import OutcomeTracker
    tracker = OutcomeTracker(lookahead_days=5, target_pct=3.0)
    assert tracker.lookahead_days == 5
    assert tracker.target_pct == 3.0


def test_log_prediction_and_resolve():
    """End-to-end: log a prediction, resolve it, verify outcome."""
    from db import db
    from outcome_tracker import OutcomeTracker

    # Create synthetic data for a ticker
    ticker = "TEST1"
    df = make_synthetic_df(ticker, days=250)  # need >= min_bars_required
    rows = []
    for idx, r in df.iterrows():
        rows.append({
            "date": str(idx.date()),
            "open": float(r["Open"]),
            "high": float(r["High"]),
            "low": float(r["Low"]),
            "close": float(r["Close"]),
            "volume": int(r["Volume"]),
        })
    db.upsert_ohlcv(ticker, rows)

    # Log a prediction 10 days ago (so 5-day outcome is available)
    pred_date = df.index[-10]
    pred_ts = str(pred_date.date()) + " 10:00:00"
    pred_id = db.log_prediction(
        ticker=ticker,
        prediction_ts=pred_ts,
        model_version="test_v1",
        source="test",
        signal_quality=0.6,
        trend_regime="UPTREND",
        expected_return=2.5,
        max_drawdown=-3.0,
        ml_score=65,
        actionable="BUY",
    )

    # Verify prediction was logged
    assert pred_id > 0
    pending = db.get_pending_outcomes(lookahead_days=5)
    assert len(pending) >= 1

    # Resolve outcomes (use min_bars=50 so 100-bar test data works)
    tracker = OutcomeTracker(lookahead_days=5, target_pct=3.0, min_bars_required=50)
    result = tracker.resolve_all_pending(feature_extractor=stub_feature_extractor)

    assert result["status"] == "ok"
    assert result["resolved"] >= 1

    # Verify outcome was stored
    stats = db.get_outcome_stats(days=30)
    assert stats["total"] >= 1
    assert "hit_rate" in stats
    assert "direction_accuracy" in stats


def test_historical_label_generation():
    """Historical labels can be generated from raw OHLCV."""
    from db import db
    from outcome_tracker import OutcomeTracker

    ticker_dfs = {
        "TEST2": make_synthetic_df("TEST2", days=300),
        "TEST3": make_synthetic_df("TEST3", days=300),
    }

    tracker = OutcomeTracker(lookahead_days=5, target_pct=3.0)
    n_samples = tracker.generate_historical_labels(
        ticker_dfs=ticker_dfs,
        feature_extractor=stub_feature_extractor,
        step=3,
    )

    assert n_samples > 0
    # Verify samples are in the queue
    X, y = db.get_training_samples(n=1000)
    assert len(X) == n_samples
    assert "hit_target" in y
    assert "return_pct" in y
    assert "regime" in y


def test_outcome_computes_correct_values():
    """Outcome values are computed correctly from OHLCV."""
    from db import db
    from outcome_tracker import OutcomeTracker

    # Create data with known outcome (30 bars, so use min_bars=20)
    ticker = "TEST4"
    dates = pd.date_range(end=datetime.now().strftime("%Y-%m-%d"), periods=30, freq="B")
    # First 20 bars: stable around 100
    # Then jump to 110 (+10%) over next 5 bars
    prices = list(np.linspace(99, 101, 20)) + list(np.linspace(101, 110, 10))
    df = pd.DataFrame({
        "Open": prices,
        "High": [p * 1.005 for p in prices],
        "Low": [p * 0.995 for p in prices],
        "Close": prices,
        "Volume": [1000000] * 30,
    }, index=dates)

    rows = []
    for idx, r in df.iterrows():
        rows.append({
            "date": str(idx.date()),
            "open": float(r["Open"]),
            "high": float(r["High"]),
            "low": float(r["Low"]),
            "close": float(r["Close"]),
            "volume": int(r["Volume"]),
        })
    db.upsert_ohlcv(ticker, rows)

    # Make a prediction on day 20 (1 day before the jump)
    pred_date = dates[20]
    pred_ts = str(pred_date.date()) + " 10:00:00"
    entry_price = float(df["Close"].iloc[20])  # ~101
    db.log_prediction(
        ticker=ticker,
        prediction_ts=pred_ts,
        model_version="test_v1",
        source="test",
        signal_quality=0.5,
        expected_return=5.0,
        max_drawdown=-2.0,
    )

    # Resolve
    tracker = OutcomeTracker(lookahead_days=5, target_pct=3.0, min_bars_required=20)
    result = tracker.resolve_all_pending()

    assert result["resolved"] >= 1

    # Check the outcome values
    outcomes = db.query("SELECT * FROM outcomes")
    assert len(outcomes) >= 1
    o = outcomes[0]
    # Actual return over 5 days = (price[25] / price[20] - 1) * 100
    expected_return = (df["Close"].iloc[25] / df["Close"].iloc[20] - 1) * 100
    assert abs(o["actual_return"] - round(expected_return, 3)) < 0.1
    # Should have hit the 3% target
    assert o["hit_target"] == 1
    assert o["days_to_target"] is not None


def test_resolve_skips_insufficient_future_data():
    """Predictions without enough future data are skipped (not resolved).

    A prediction made 6 days ago (so it's in pending list) but with no
    OHLCV data for that ticker should be skipped.
    """
    from db import db
    from outcome_tracker import OutcomeTracker

    # Make a prediction dated 6 days ago (so it's in the pending list)
    pred_date = datetime.now() - timedelta(days=6)
    pred_ts = pred_date.strftime("%Y-%m-%d") + " 10:00:00"
    db.log_prediction(
        ticker="TEST5_NODATA",  # No OHLCV data for this ticker
        prediction_ts=pred_ts,
        model_version="test_v1",
        source="test",
    )

    tracker = OutcomeTracker(lookahead_days=5, target_pct=3.0, min_bars_required=20)
    result = tracker.resolve_all_pending()

    # Should skip because no OHLCV data available
    assert result["resolved"] == 0
    assert result["skipped"] >= 1


def test_outcome_idempotent():
    """Resolving the same prediction twice doesn't create duplicate outcomes."""
    from db import db
    from outcome_tracker import OutcomeTracker

    ticker = "TEST6"
    df = make_synthetic_df(ticker, days=250)
    rows = []
    for idx, r in df.iterrows():
        rows.append({
            "date": str(idx.date()),
            "open": float(r["Open"]),
            "high": float(r["High"]),
            "low": float(r["Low"]),
            "close": float(r["Close"]),
            "volume": int(r["Volume"]),
        })
    db.upsert_ohlcv(ticker, rows)

    pred_date = df.index[-10]
    pred_ts = str(pred_date.date()) + " 10:00:00"
    db.log_prediction(
        ticker=ticker,
        prediction_ts=pred_ts,
        model_version="test_v1",
        source="test",
    )

    tracker = OutcomeTracker(lookahead_days=5, target_pct=3.0, min_bars_required=50)
    # First resolution
    result1 = tracker.resolve_all_pending()
    # Second resolution
    result2 = tracker.resolve_all_pending()

    # Second run should find nothing to resolve
    assert result1["resolved"] >= 1
    assert result2["resolved"] == 0

    # Only one outcome in DB
    outcomes = db.query("SELECT * FROM outcomes")
    assert len(outcomes) == 1
