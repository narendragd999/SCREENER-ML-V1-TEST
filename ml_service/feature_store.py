"""
Feature Store — Layer 2 of the Continuous Learning Pipeline
=============================================================
Caches feature vectors per (ticker, date) so they don't need to be
recomputed every time.

Why this matters:
  - Feature extraction for 1 ticker with 10y of history = ~50ms
  - For 5000 tickers × 2500 trading days = 12.5M feature extractions
  - Total CPU time: ~170 hours (without caching)
  - With caching: ~1 hour (one-time) + ~1ms per cache hit

Features are deterministic — given the same OHLCV data, you get the same
features. So we cache them in the `features` SQLite table.

Usage:
    from feature_store import FeatureStore
    from ml_features import extract_features

    fs = FeatureStore(feature_extractor=extract_features)

    # Get features for a specific date (cached or compute)
    feats = fs.get_features_at_date("RELIANCE", "2026-01-15")

    # Get features for ALL dates of a ticker (bulk populate)
    fs.populate_ticker_history("RELIANCE", df)

    # Get features for a date range (for backtesting)
    feats_by_date = fs.get_features_range("RELIANCE", "2025-01-01", "2025-12-31")
"""

from __future__ import annotations
import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from .db import db, now_ist
except ImportError:
    from db import db, now_ist

logger = logging.getLogger("feature_store")


class FeatureStore:
    """
    Persistent feature cache with on-demand computation.

    Features are keyed by (ticker, date). The date corresponds to the LAST
    bar in the OHLCV window used to compute the features.

    Example: features for ("RELIANCE", "2026-01-15") are computed from
    RELIANCE's OHLCV data up to and including 2026-01-15.
    """

    def __init__(self,
                 feature_extractor: Callable[[pd.DataFrame, bool], np.ndarray],
                 min_bars: int = 200):
        """
        Args:
            feature_extractor: callable(df, normalize=False) -> np.ndarray
                Should match the signature of ml_features.extract_features
            min_bars: minimum OHLCV bars required to compute features
        """
        self.extract = feature_extractor
        self.min_bars = min_bars
        # In-memory LRU cache (small, for hot tickers)
        self._mem_cache: Dict[Tuple[str, str], np.ndarray] = {}
        self._mem_cache_lock = threading.Lock()
        self._mem_cache_max = 1000  # max entries in memory

    def get_features_at_date(self,
                             ticker: str,
                             date_str: str,
                             df: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """Get features for a ticker at a specific date.

        Args:
            ticker: stock symbol
            date_str: 'YYYY-MM-DD' — the date of the LAST bar used
            df: optional DataFrame. If provided, used for cache miss.
                If None, loaded from DB.

        Returns:
            Feature vector as np.ndarray, or None if cannot compute.
        """
        ticker = ticker.strip().upper()
        cache_key = (ticker, date_str)

        # Check memory cache
        with self._mem_cache_lock:
            if cache_key in self._mem_cache:
                return self._mem_cache[cache_key]

        # Check DB cache
        cached = db.get_cached_features(ticker, date_str)
        if cached is not None:
            self._put_in_mem_cache(cache_key, cached)
            return cached

        # Cache miss — compute from OHLCV
        if df is None:
            df = db.get_ohlcv(ticker, end_date=date_str)
        else:
            # Filter df to only bars up to and including date_str
            try:
                if isinstance(df.index, pd.DatetimeIndex):
                    df = df[df.index <= pd.Timestamp(date_str)]
                else:
                    df = df[df.index.astype(str) <= date_str]
            except Exception:
                pass

        if df is None or len(df) < self.min_bars:
            return None

        # Compute features
        try:
            features = self.extract(df, normalize=False)
        except Exception as e:
            logger.debug(f"Feature extraction failed for {ticker}@{date_str}: {e}")
            return None

        if features is None or (isinstance(features, np.ndarray) and features.size == 0):
            return None

        # Cache for future use
        db.cache_features(ticker, date_str, features)
        self._put_in_mem_cache(cache_key, features)

        return features

    def get_features_range(self,
                           ticker: str,
                           start_date: str,
                           end_date: str,
                           step: int = 1) -> Dict[str, np.ndarray]:
        """Get features for a date range. Returns dict of {date_str: features}.

        Args:
            ticker: stock symbol
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
            step: stride (e.g., step=3 returns every 3rd day)
        """
        ticker = ticker.strip().upper()
        df = db.get_ohlcv(ticker, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            return {}

        results: Dict[str, np.ndarray] = {}
        dates = df.index.tolist()
        for i in range(self.min_bars - 1, len(df), step):
            date_str = str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i])[:10]
            window_df = df.iloc[:i + 1]
            feats = self.get_features_at_date(ticker, date_str, df=window_df)
            if feats is not None:
                results[date_str] = feats
        return results

    def populate_ticker_history(self,
                                ticker: str,
                                df: Optional[pd.DataFrame] = None,
                                step: int = 3,
                                force: bool = False) -> int:
        """Bulk-populate features for a ticker's entire history.

        Useful for initial bootstrap before continuous learning starts.

        Args:
            ticker: stock symbol
            df: optional DataFrame. If None, loaded from DB.
            step: stride (3 = every 3rd bar, reduces correlation)
            force: if True, recompute even if cached

        Returns:
            Number of feature vectors cached.
        """
        ticker = ticker.strip().upper()
        if df is None:
            df = db.get_ohlcv(ticker)
        if df is None or len(df) < self.min_bars:
            return 0

        count = 0
        dates = df.index.tolist()
        for i in range(self.min_bars - 1, len(df), step):
            date_str = str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i])[:10]

            if not force:
                # Skip if already cached
                cached = db.get_cached_features(ticker, date_str)
                if cached is not None:
                    continue

            window_df = df.iloc[:i + 1]
            try:
                features = self.extract(window_df, normalize=False)
                if features is not None and features.size > 0:
                    db.cache_features(ticker, date_str, features)
                    count += 1
            except Exception as e:
                logger.debug(f"Feature computation failed for {ticker}@{date_str}: {e}")
                continue

        logger.info(f"[FeatureStore] {ticker}: cached {count} feature vectors")
        return count

    def bulk_populate(self,
                      ticker_dfs: Dict[str, pd.DataFrame],
                      step: int = 3,
                      force: bool = False) -> Dict[str, int]:
        """Populate features for multiple tickers.

        Returns dict of {ticker: count_cached}.
        """
        results = {}
        for ticker, df in ticker_dfs.items():
            if df is None or df.empty:
                continue
            count = self.populate_ticker_history(ticker, df=df, step=step, force=force)
            results[ticker] = count
        return results

    def get_feature_drift(self,
                          ticker: str,
                          recent_window: int = 30,
                          baseline_window: int = 90) -> Dict[str, float]:
        """Compute feature drift between recent and baseline windows.

        Uses Population Stability Index (PSI) — values >0.25 indicate
        significant drift.

        Args:
            ticker: stock symbol
            recent_window: number of recent days to use as "recent"
            baseline_window: number of prior days to use as "baseline"

        Returns:
            Dict with overall_psi and per_feature_psi (if feature names available)
        """
        df = db.get_ohlcv(ticker)
        if df is None or len(df) < self.min_bars + recent_window + baseline_window:
            return {"overall_psi": 0.0, "status": "insufficient_data"}

        dates = df.index.tolist()
        n = len(df)

        # Get baseline features
        baseline_end_idx = n - recent_window - 1
        baseline_start_idx = baseline_end_idx - baseline_window
        recent_start_idx = n - recent_window

        baseline_features = []
        for i in range(baseline_start_idx, baseline_end_idx, 3):
            date_str = str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i])[:10]
            feats = self.get_features_at_date(ticker, date_str, df=df.iloc[:i+1])
            if feats is not None:
                baseline_features.append(feats)

        recent_features = []
        for i in range(recent_start_idx, n, 1):
            date_str = str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i])[:10]
            feats = self.get_features_at_date(ticker, date_str, df=df.iloc[:i+1])
            if feats is not None:
                recent_features.append(feats)

        if not baseline_features or not recent_features:
            return {"overall_psi": 0.0, "status": "no_features"}

        baseline = np.stack(baseline_features)
        recent = np.stack(recent_features)

        # Compute PSI per feature
        psi_values = []
        for col in range(baseline.shape[1]):
            psi = self._compute_psi(baseline[:, col], recent[:, col])
            psi_values.append(psi)

        overall_psi = float(np.mean(psi_values))
        return {
            "overall_psi": round(overall_psi, 4),
            "max_psi": round(float(np.max(psi_values)), 4),
            "baseline_samples": len(baseline_features),
            "recent_samples": len(recent_features),
            "status": "drift_detected" if overall_psi > 0.25 else "ok",
        }

    def _compute_psi(self, expected: np.ndarray, actual: np.ndarray,
                     n_bins: int = 10) -> float:
        """Compute Population Stability Index for a single feature."""
        # Use quantiles of expected to define bins
        eps = 1e-4
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        if len(expected) < 10 or len(actual) < 10:
            return 0.0

        # Use quantile-based bins
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.quantile(expected, quantiles)
        bins[0] = -np.inf
        bins[-1] = np.inf
        # Remove duplicate bins
        bins = np.unique(bins)
        if len(bins) < 3:
            return 0.0

        expected_counts = np.histogram(expected, bins=bins)[0] / len(expected)
        actual_counts = np.histogram(actual, bins=bins)[0] / len(actual)

        # Add small epsilon to avoid log(0)
        expected_counts = np.clip(expected_counts, eps, None)
        actual_counts = np.clip(actual_counts, eps, None)

        psi = np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts))
        return float(psi)

    def _put_in_mem_cache(self, key: Tuple[str, str], value: np.ndarray) -> None:
        """Add to in-memory LRU cache, evicting oldest if full."""
        with self._mem_cache_lock:
            if len(self._mem_cache) >= self._mem_cache_max:
                # Evict oldest (FIFO — simple approximation of LRU)
                oldest_key = next(iter(self._mem_cache))
                del self._mem_cache[oldest_key]
            self._mem_cache[key] = value


# ── Singleton ────────────────────────────────────────────────────────────────
_store: Optional[FeatureStore] = None
_store_lock = threading.Lock()


def get_feature_store(feature_extractor: Optional[Callable] = None) -> FeatureStore:
    """Get the singleton FeatureStore instance.

    Requires `feature_extractor` on first call.
    """
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                if feature_extractor is None:
                    raise ValueError("feature_extractor required on first call")
                _store = FeatureStore(feature_extractor=feature_extractor)
    return _store
