"""
Delta Downloader — Layer 1 of the Continuous Learning Pipeline
================================================================
Replaces the original `download_ticker_data()` which downloads 10y of data
every time it's called. This module:

  1. Stores all OHLCV in the `ohlcv` SQLite table
  2. On subsequent calls, only fetches bars SINCE the last stored date
  3. Concurrent downloads (10 workers by default)
  4. Auto-retry on yfinance rate limits

Performance:
  - Original: 100 tickers × 10y download every cycle = ~15 min, 1000s of API calls
  - Delta:    100 tickers × ~5 new bars each = ~30 sec, ~100 API calls
  - Speedup:  ~30x

Usage:
    from delta_downloader import DeltaDownloader
    dd = DeltaDownloader()

    # Update one ticker (returns full DataFrame)
    df = dd.update_ticker("RELIANCE")

    # Update all F&O tickers concurrently
    results = dd.update_tickers_batch(fno_tickers, max_workers=10)
"""

from __future__ import annotations
import os
import time
import logging
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

try:
    from .db import db, now_ist, IST_OFFSET_HOURS
except ImportError:
    from db import db, now_ist, IST_OFFSET_HOURS

logger = logging.getLogger("delta_downloader")

# yfinance is heavy and optional — import lazily
_yf = None
def _get_yf():
    global _yf
    if _yf is None:
        import yfinance as yf
        _yf = yf
    return _yf


# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_PERIOD = "10y"        # initial download depth
DEFAULT_INTERVAL = "1d"
RATE_LIMIT_SLEEP = 0.5        # seconds between API calls per ticker (increased from 0.15)
MAX_RETRIES = 5               # increased from 3
CACHE_TTL_HOURS = 4           # in-memory cache TTL
MAX_CONCURRENT_DOWNLOADS = 3  # reduced from 10 — yfinance rate limits aggressively


class DeltaDownloader:
    """
    Incremental OHLCV downloader with SQLite persistence.

    Usage:
        dd = DeltaDownloader()
        df = dd.update_ticker("RELIANCE")
        results = dd.update_tickers_batch(["RELIANCE", "INFY", "TCS"])
    """

    def __init__(self):
        # In-memory cache to avoid DB hits during batch operations
        self._mem_cache: Dict[str, Dict] = {}  # ticker -> {"df": DataFrame, "ts": float}
        self._mem_cache_lock = threading.Lock()
        self._ticker_locks: Dict[str, threading.Lock] = {}
        self._ticker_locks_lock = threading.Lock()

    def _get_ticker_lock(self, ticker: str) -> threading.Lock:
        """Get a per-ticker lock to prevent concurrent downloads of same ticker."""
        with self._ticker_locks_lock:
            if ticker not in self._ticker_locks:
                self._ticker_locks[ticker] = threading.Lock()
            return self._ticker_locks[ticker]

    # ─────────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def update_ticker(self, ticker: str, force_full: bool = False) -> Optional[pd.DataFrame]:
        """Update OHLCV for a single ticker. Returns the full DataFrame.

        Args:
            ticker: stock symbol without exchange suffix (e.g. 'RELIANCE')
            force_full: if True, re-download 10y even if data exists
        """
        ticker = ticker.strip().upper()
        lock = self._get_ticker_lock(ticker)

        with lock:
            # Check in-memory cache first
            with self._mem_cache_lock:
                cached = self._mem_cache.get(ticker)
                if cached and (time.time() - cached["ts"]) < CACHE_TTL_HOURS * 3600:
                    return cached["df"].copy()

            # Check DB
            last_date = db.get_last_ohlcv_date(ticker) if not force_full else None

            if last_date is None:
                # First time: download full history
                df = self._download_full(ticker)
            else:
                # Delta: only fetch new bars
                df_existing = db.get_ohlcv(ticker)
                df_new = self._download_delta(ticker, last_date)

                if df_new is not None and not df_new.empty:
                    # Persist new bars to DB
                    rows = self._df_to_rows(df_new)
                    db.upsert_ohlcv(ticker, rows)
                    # Combine
                    df = pd.concat([df_existing, df_new])
                    df = df[~df.index.duplicated(keep="last")]
                    df = df.sort_index()
                else:
                    df = df_existing

            if df is not None and not df.empty:
                with self._mem_cache_lock:
                    self._mem_cache[ticker] = {"df": df, "ts": time.time()}

            return df

    def update_tickers_batch(self,
                             tickers: List[str],
                             max_workers: int = MAX_CONCURRENT_DOWNLOADS,
                             force_full: bool = False) -> Dict[str, pd.DataFrame]:
        """Update multiple tickers concurrently.

        Args:
            tickers: list of ticker symbols
            max_workers: number of concurrent download threads (default 3 — yfinance rate limits)
            force_full: if True, re-download full history for each ticker

        Returns:
            dict of {ticker: DataFrame} for tickers that succeeded
        """
        tickers = [t.strip().upper() for t in tickers if t.strip()]
        results: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(self.update_ticker, t, force_full): t
                for t in tickers
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        results[ticker] = df
                    else:
                        failed.append(ticker)
                except Exception as e:
                    logger.warning(f"[DeltaDownloader] {ticker} failed: {e}")
                    failed.append(ticker)

        logger.info(f"[DeltaDownloader] Batch update: {len(results)}/{len(tickers)} "
                    f"succeeded, {len(failed)} failed")
        if failed and len(failed) < 20:
            logger.info(f"[DeltaDownloader] Failed: {failed}")

        return results

    def get_cached(self, ticker: str) -> Optional[pd.DataFrame]:
        """Get cached data without triggering a download. Returns None if not cached."""
        ticker = ticker.strip().upper()
        # Check memory first
        with self._mem_cache_lock:
            cached = self._mem_cache.get(ticker)
            if cached:
                return cached["df"].copy()
        # Then DB
        df = db.get_ohlcv(ticker)
        if df is not None and not df.empty:
            with self._mem_cache_lock:
                self._mem_cache[ticker] = {"df": df, "ts": time.time()}
            return df
        return None

    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """Clear in-memory cache. If ticker is None, clears all."""
        with self._mem_cache_lock:
            if ticker:
                self._mem_cache.pop(ticker, None)
            else:
                self._mem_cache.clear()

    def get_staleness_report(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """Get a report of how stale each ticker's data is."""
        report = []
        now = datetime.utcnow() + timedelta(hours=IST_OFFSET_HOURS)
        for ticker in tickers:
            ticker = ticker.strip().upper()
            last_date = db.get_last_ohlcv_date(ticker)
            if last_date:
                last_dt = datetime.strptime(last_date, "%Y-%m-%d")
                days_stale = (now - last_dt).days
                report.append({
                    "ticker": ticker,
                    "last_date": last_date,
                    "days_stale": days_stale,
                    "needs_update": days_stale >= 1,  # any weekday
                })
            else:
                report.append({
                    "ticker": ticker,
                    "last_date": None,
                    "days_stale": None,
                    "needs_update": True,
                })
        return report

    # ─────────────────────────────────────────────────────────────────────────
    #  INTERNALS
    # ─────────────────────────────────────────────────────────────────────────

    def _download_full(self, ticker: str) -> Optional[pd.DataFrame]:
        """Download full history for a ticker. yfinance-compatible with retries."""
        yf = _get_yf()
        last_error = None
        for suffix in [".NS", ".BO"]:
            for attempt in range(MAX_RETRIES):
                try:
                    # Add small delay before each attempt to avoid rate limiting
                    time.sleep(RATE_LIMIT_SLEEP)
                    t = yf.Ticker(f"{ticker}{suffix}")
                    hist = t.history(period=DEFAULT_PERIOD, auto_adjust=True)
                    if hist is not None and not hist.empty and len(hist) >= 50:
                        df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
                        df = df.dropna(how="all")
                        # Persist to DB
                        rows = self._df_to_rows(df)
                        db.upsert_ohlcv(ticker, rows)
                        logger.info(f"[DeltaDownloader] {ticker}{suffix}: full download "
                                    f"{len(df)} bars")
                        return df
                    else:
                        last_error = "empty data"
                except Exception as e:
                    last_error = str(e)
                    if attempt < MAX_RETRIES - 1:
                        # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                        sleep_time = 2.0 ** attempt
                        time.sleep(sleep_time)
                    continue
        logger.warning(f"[DeltaDownloader] {ticker}: full download failed (last error: {last_error})")
        return None

    def _download_delta(self, ticker: str, last_date: str) -> Optional[pd.DataFrame]:
        """Download only bars newer than `last_date`."""
        yf = _get_yf()
        # Parse last_date and start 1 day before to catch any revisions
        try:
            start_dt = datetime.strptime(last_date, "%Y-%m-%d") - timedelta(days=1)
            start_str = start_dt.strftime("%Y-%m-%d")
        except Exception:
            return self._download_full(ticker)

        for suffix in [".NS", ".BO"]:
            for attempt in range(MAX_RETRIES):
                try:
                    t = yf.Ticker(f"{ticker}{suffix}")
                    hist = t.history(start=start_str, interval=DEFAULT_INTERVAL, auto_adjust=True)
                    if hist is None or hist.empty:
                        return pd.DataFrame()
                    df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df = df.dropna(how="all")
                    # Filter to only truly new bars
                    df = df[df.index > pd.Timestamp(last_date)]
                    if not df.empty:
                        logger.debug(f"[DeltaDownloader] {ticker}: {len(df)} new bars")
                    return df
                except Exception as e:
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(0.5 * (attempt + 1))
                    continue
        return None

    def _df_to_rows(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert a DataFrame to a list of row dicts for DB insertion."""
        rows = []
        for idx, r in df.iterrows():
            date_str = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
            rows.append({
                "date": date_str,
                "open": float(r["Open"]),
                "high": float(r["High"]),
                "low": float(r["Low"]),
                "close": float(r["Close"]),
                "volume": int(r["Volume"]) if not pd.isna(r["Volume"]) else 0,
            })
        return rows


# ── Singleton ────────────────────────────────────────────────────────────────
_downloader: Optional[DeltaDownloader] = None
_downloader_lock = threading.Lock()


def get_downloader() -> DeltaDownloader:
    """Get the singleton DeltaDownloader instance."""
    global _downloader
    if _downloader is None:
        with _downloader_lock:
            if _downloader is None:
                _downloader = DeltaDownloader()
    return _downloader
