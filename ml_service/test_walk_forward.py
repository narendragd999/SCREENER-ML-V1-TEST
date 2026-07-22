"""
Tests for walk_forward.py — time-series validation.

Run: python -m pytest tests/test_walk_forward.py -v
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_df():
    """A DataFrame with 300 days of samples."""
    dates = pd.date_range(end=datetime.now().strftime("%Y-%m-%d"), periods=300, freq="B")
    return pd.DataFrame({
        "date": dates,
        "ticker": "TEST",
        "value": np.random.randn(300),
    })


def test_simple_time_split_no_overlap(sample_df):
    """Train and val should NOT overlap in time."""
    from walk_forward import simple_time_split
    train, val = simple_time_split(sample_df, train_days=180, val_days=30)

    assert len(train) > 0
    assert len(val) > 0
    # Max train date < min val date
    assert train["date"].max() < val["date"].min()


def test_simple_time_split_returns_correct_sizes(sample_df):
    """Split sizes should roughly match the requested day windows."""
    from walk_forward import simple_time_split
    train, val = simple_time_split(sample_df, train_days=180, val_days=30)

    # Allow some slack for weekends
    train_days = (train["date"].max() - train["date"].min()).days
    val_days = (val["date"].max() - val["date"].min()).days
    assert 150 <= train_days <= 180
    assert 25 <= val_days <= 35


def test_simple_time_split_empty_df():
    """Empty DataFrame should return empty splits, not crash."""
    from walk_forward import simple_time_split
    empty = pd.DataFrame({"date": [], "ticker": [], "value": []})
    train, val = simple_time_split(empty, train_days=180, val_days=30)
    assert len(train) == 0
    assert len(val) == 0


def test_purge_and_embargo_creates_gap():
    """Purge should create a gap between train and val."""
    from walk_forward import simple_time_split, purge_and_embargo

    dates = pd.date_range(end=datetime.now().strftime("%Y-%m-%d"), periods=300, freq="B")
    df = pd.DataFrame({"date": dates, "value": np.random.randn(300)})

    train, val = simple_time_split(df, train_days=180, val_days=30)
    pre_train_end = train["date"].max()

    train_purged, val_purged = purge_and_embargo(train, val, embargo_days=3)

    # Purged train should end at least 3 days before original
    assert train_purged["date"].max() < pre_train_end
    gap_days = (val["date"].min() - train_purged["date"].max()).days
    assert gap_days >= 3


def test_expanding_window_yields_multiple_folds():
    """Expanding window should yield the requested number of folds."""
    from walk_forward import expanding_window_cv

    dates = pd.date_range(end=datetime.now().strftime("%Y-%m-%d"), periods=365, freq="B")
    df = pd.DataFrame({"date": dates, "value": np.random.randn(365)})

    folds = list(expanding_window_cv(df, n_folds=5, val_days=30, min_train_days=90))

    # May get fewer than 5 if not enough data, but should be >= 3
    assert len(folds) >= 3
    for train, val in folds:
        assert len(train) > 0
        assert len(val) > 0
        assert train["date"].max() < val["date"].min()


def test_sliding_window_train_size_constant():
    """Sliding window should have roughly constant train size."""
    from walk_forward import sliding_window_cv

    dates = pd.date_range(end=datetime.now().strftime("%Y-%m-%d"), periods=500, freq="B")
    df = pd.DataFrame({"date": dates, "value": np.random.randn(500)})

    folds = list(sliding_window_cv(df, n_folds=3, train_days=180, val_days=30))
    assert len(folds) >= 2

    train_sizes = [len(t) for t, v in folds]
    # All train sizes should be similar (within 10%)
    assert max(train_sizes) / min(train_sizes) < 1.1


def test_arrays_to_dated_df_roundtrip():
    """Conversion to/from DataFrame should preserve data."""
    from walk_forward import arrays_to_dated_df, dated_df_to_arrays

    X = np.random.randn(50, 70).astype(np.float32)
    y = {
        "hit_target": np.random.randint(0, 2, 50).astype(np.float32),
        "return_pct": np.random.randn(50).astype(np.float32),
    }
    dates = [f"2025-01-{i+1:02d}" for i in range(50)]

    df = arrays_to_dated_df(X, y, dates)
    assert len(df) == 50
    assert "label_hit_target" in df.columns
    assert "label_return_pct" in df.columns

    X_back, y_back = dated_df_to_arrays(df)
    np.testing.assert_array_almost_equal(X, X_back)
    np.testing.assert_array_almost_equal(y["hit_target"], y_back["hit_target"])
