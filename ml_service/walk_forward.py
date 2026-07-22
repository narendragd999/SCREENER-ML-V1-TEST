"""
Walk-Forward Validation — Fixes Data Leakage in Time-Series ML
================================================================
The original code uses `validation_split=0.1` in Keras `fit()`, which does a
RANDOM split. For time-series data this is catastrophic — the model sees
future data during training and reports inflated validation accuracy.

This module provides proper time-series splits where validation data is
ALWAYS strictly AFTER training data.

Three split modes are supported:

  1. `simple_time_split` — single train/val split with cutoff date
  2. `expanding_window` — multiple folds, train window grows each fold
  3. `sliding_window` — multiple folds, train window is fixed size

For continuous learning, use `simple_time_split` (fast, one fine-tune cycle).
For thorough backtesting, use `expanding_window` (slow, robust).
"""

from __future__ import annotations
import logging
from typing import List, Tuple, Optional, Iterator
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger("walk_forward")


def simple_time_split(samples: pd.DataFrame,
                      date_col: str = "date",
                      train_days: int = 180,
                      val_days: int = 30) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Single train/val split with strict time ordering.

    Args:
        samples: DataFrame with one row per sample, must have `date_col`
        date_col: name of the date column
        train_days: number of days of history to use for training
        val_days: number of days after train_end to use for validation

    Returns:
        (train_df, val_df) — both are views into the original DataFrame.

    Example:
        If today is 2026-01-15 and train_days=180, val_days=30:
          train: 2025-06-19 to 2025-12-16 (180 days)
          val:   2025-12-16 to 2026-01-15 (30 days)
    """
    if date_col not in samples.columns:
        raise ValueError(f"Column '{date_col}' not found in samples")

    samples = samples.copy()
    samples[date_col] = pd.to_datetime(samples[date_col])
    samples = samples.sort_values(date_col).reset_index(drop=True)

    if len(samples) == 0:
        return samples, samples

    latest_date = samples[date_col].max()
    val_start = latest_date - timedelta(days=val_days)
    train_start = val_start - timedelta(days=train_days)

    train = samples[(samples[date_col] >= train_start) &
                    (samples[date_col] < val_start)]
    val = samples[samples[date_col] >= val_start]

    logger.info(f"[WalkForward] simple_time_split: "
                f"train={len(train)} ({train_start.date()} to {val_start.date()}), "
                f"val={len(val)} ({val_start.date()} to {latest_date.date()})")
    return train, val


def expanding_window_cv(samples: pd.DataFrame,
                        date_col: str = "date",
                        n_folds: int = 5,
                        val_days: int = 30,
                        min_train_days: int = 90) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding window cross-validation.

    Each fold uses MORE training data than the previous one. The validation
    window slides forward in time.

    Args:
        samples: DataFrame with one row per sample
        date_col: name of the date column
        n_folds: number of (train, val) pairs to yield
        val_days: width of each validation window in days
        min_train_days: minimum training window for the first fold

    Yields:
        (train_df, val_df) tuples, one per fold
    """
    if date_col not in samples.columns:
        raise ValueError(f"Column '{date_col}' not found in samples")

    samples = samples.copy()
    samples[date_col] = pd.to_datetime(samples[date_col])
    samples = samples.sort_values(date_col).reset_index(drop=True)

    if len(samples) == 0:
        return

    latest_date = samples[date_col].max()
    earliest_date = samples[date_col].min()

    # Generate fold cutoffs working backward from latest_date
    fold_starts: List[datetime] = []
    for k in range(n_folds):
        val_end = latest_date - timedelta(days=k * val_days)
        val_start = val_end - timedelta(days=val_days)
        if val_start < earliest_date + timedelta(days=min_train_days):
            break  # not enough data for this fold
        fold_starts.append((val_start, val_end))

    # Reverse so folds come in chronological order
    fold_starts.reverse()

    for val_start, val_end in fold_starts:
        train = samples[samples[date_col] < val_start]
        val = samples[(samples[date_col] >= val_start) &
                      (samples[date_col] < val_end)]
        if len(train) > 0 and len(val) > 0:
            logger.info(f"[WalkForward] expanding fold: "
                       f"train={len(train)} ({train[date_col].min().date()} to "
                       f"{train[date_col].max().date()}), "
                       f"val={len(val)} ({val_start.date()} to {val_end.date()})")
            yield train, val


def sliding_window_cv(samples: pd.DataFrame,
                      date_col: str = "date",
                      n_folds: int = 5,
                      train_days: int = 180,
                      val_days: int = 30) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Sliding window cross-validation.

    Both train and val windows slide forward. Train window is fixed size.

    Useful when recent data is most relevant (e.g., regime changes).
    """
    if date_col not in samples.columns:
        raise ValueError(f"Column '{date_col}' not found in samples")

    samples = samples.copy()
    samples[date_col] = pd.to_datetime(samples[date_col])
    samples = samples.sort_values(date_col).reset_index(drop=True)

    if len(samples) == 0:
        return

    latest_date = samples[date_col].max()

    for k in range(n_folds):
        val_end = latest_date - timedelta(days=k * val_days)
        val_start = val_end - timedelta(days=val_days)
        train_start = val_start - timedelta(days=train_days)

        train = samples[(samples[date_col] >= train_start) &
                        (samples[date_col] < val_start)]
        val = samples[(samples[date_col] >= val_start) &
                      (samples[date_col] < val_end)]
        if len(train) > 0 and len(val) > 0:
            logger.info(f"[WalkForward] sliding fold: "
                       f"train={len(train)}, val={len(val)}")
            yield train, val


def purge_and_embargo(train_df: pd.DataFrame,
                      val_df: pd.DataFrame,
                      date_col: str = "date",
                      embargo_days: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Purge overlapping samples and apply embargo period.

    In financial ML, samples near the train/val boundary are correlated
    (3-day lookahead labels overlap). This function:
      1. Removes training samples whose label window overlaps with val period
      2. Adds an `embargo_days` gap between train and val

    Reference: Marcos López de Prado, "Advances in Financial Machine Learning"
    """
    if date_col not in train_df.columns or date_col not in val_df.columns:
        return train_df, val_df

    val_start = pd.to_datetime(val_df[date_col]).min()
    purge_threshold = val_start - timedelta(days=embargo_days)

    purged_train = train_df[pd.to_datetime(train_df[date_col]) < purge_threshold]
    return purged_train, val_df


# ── Helper: convert numpy arrays + dates to a DataFrame for splitting ─────────
def arrays_to_dated_df(X: np.ndarray,
                       y: Dict[str, np.ndarray],
                       dates: List[str],
                       tickers: Optional[List[str]] = None) -> pd.DataFrame:
    """Convert raw arrays into a DataFrame suitable for the split functions.

    Args:
        X: feature matrix of shape (N, F)
        y: dict of label arrays, each shape (N,)
        dates: list of N date strings ('YYYY-MM-DD')
        tickers: optional list of N ticker strings

    Returns:
        DataFrame with columns: date, ticker, feature_idx, + one column per label
    """
    n = len(X)
    if tickers is None:
        tickers = ["UNKNOWN"] * n

    df = pd.DataFrame({
        "date": dates,
        "ticker": tickers,
    })
    for k, v in y.items():
        df[f"label_{k}"] = v
    # Store features as a list column (one row per sample)
    df["features"] = list(X)
    return df


def dated_df_to_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Inverse of `arrays_to_dated_df`."""
    X = np.stack(df["features"].values)
    y = {}
    for col in df.columns:
        if col.startswith("label_"):
            y[col.replace("label_", "")] = df[col].values
    return X, y
