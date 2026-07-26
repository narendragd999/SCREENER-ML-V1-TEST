"""
9EMA Breakout Screener v3.3 — FastAPI Router
═════════════════════════════════════════════
Fixes applied IN ema9_router.py (without modifying sma_router.py):
  • Global yf.download monkey-patch: Fixes the future end-date bug (2027-03-31)
    that causes "possibly delisted" errors in sma_router.py's FV fetch.
  • Global yf.download monkey-patch: Adds .BO fallback for single-ticker calls.
  • Replaced yf.download batch mode with yf.Ticker().history() loop for
    rock-solid EMA9 price data downloads.
"""

import os, io, asyncio, datetime as dt_module, time, math, logging, pickle, json
from typing import Optional, List, Dict
import httpx
import warnings as _warnings

# Suppress noisy yfinance "possibly delisted" warnings that print to stdout.
# These are harmless (the ticker just gets None data and moves to the failed
# list), but with 2000+ tickers they flood the server console.
_warnings.filterwarnings("ignore", message=".*possibly delisted.*")
_warnings.filterwarnings("ignore", message=".*No data found.*")

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────
#  YFINANCE GLOBAL PATCH (Fixes sma_router.py bugs from here)
# ─────────────────────────────────────────────────────────────
_original_yf_download = yf.download

def _safe_yf_download(tickers, start=None, end=None, **kwargs):
    """
    Intercepts yf.download calls globally.
    1. Clamps future end dates (e.g., 2027-03-31) to today+7 days.
    2. Adds .BO fallback for single-ticker string calls that return empty.
    """
    # Fix 1: Future end dates cause "possibly delisted" errors
    if end is not None:
        try:
            end_dt = pd.to_datetime(end)
            if end_dt > pd.Timestamp.now() + pd.Timedelta(days=30):
                end = str((pd.Timestamp.now() + pd.Timedelta(days=7)).date())
        except Exception:
            pass

    # Call original download
    result = _original_yf_download(tickers, start=start, end=end, **kwargs)
    
    # Fix 2: .BO fallback for single-ticker string calls
    if isinstance(tickers, str) and (result is None or result.empty):
        if tickers.endswith(".NS"):
            bo_ticker = tickers.replace(".NS", ".BO")
            try:
                result = _original_yf_download(bo_ticker, start=start, end=end, **kwargs)
            except Exception:
                pass
                
    return result

# Apply the patch globally. Now when sma_router.py calls yf.download, it uses our safe version!
yf.download = _safe_yf_download


try:
    from sma_router import _analyze_ticker as _sma_analyze_ticker
    _FV_AVAILABLE = True
except ImportError:
    _FV_AVAILABLE = False

router = APIRouter()

# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger("ema9_router")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DATA_DIR = "data"
FNO_CSV  = "tickers.csv"
ALL_CSV  = "tickers_all.csv"
HEALTHY_CSV = "tickers-healthy-strict-4.csv"
os.makedirs(DATA_DIR, exist_ok=True)

# Cache persistence directory — survives server restarts.
# Pickle files are stored here so cached yfinance + FV data persists across
# restarts. Only cleared when user clicks "Clear Cache" button.
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
YF_CACHE_FILE      = os.path.join(CACHE_DIR, "yf_cache.pkl")
FV_CACHE_FILE      = os.path.join(CACHE_DIR, "fv_cache.pkl")
FV_FAIL_CACHE_FILE = os.path.join(CACHE_DIR, "fv_fail_cache.pkl")

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
YF_CHUNK_SIZE    = 100
YF_CHUNK_DELAY   = 2.0
# ── screener.in politeness settings ─────────────────────────────────────────
# User explicitly requested MORE delay to avoid rate-limit violations on
# screener.in (the FV engine's data source via sma_router). These values are
# deliberately conservative — slower but safe. The 12-hour success cache
# (FV_CACHE_TTL_SEC below) is the primary mitigator; these delays gate the
# sequential + concurrent paths when the cache misses.
#
# Old values → New values (all ~3x more conservative):
#   FV_INTER_DELAY: 1.5s  → 5.0s   (sequential delay between screener.in calls)
#   FV_MAX_RETRIES: 2     → 1      (fewer retries = less load when screener.in fails)
#   FV_RETRY_DELAY: 3.0s  → 8.0s   (longer backoff between retries)
# At ~2.7s per screener.in request + 5.0s inter-delay = ~7.7s per ticker.
# For 50 cache-miss tickers ≈ 6.4 minutes (was ~3.5 min). Acceptable per user.
FV_INTER_DELAY   = 5.0
FV_MAX_RETRIES   = 1
FV_RETRY_DELAY   = 8.0
MIN_HISTORY_DAYS = 60
DOWNLOAD_PERIOD  = "10y"

# Cache TTLs (seconds)
YF_CACHE_TTL_SEC       = 4 * 3600   # 4 hours — yfinance data updates once daily (EOD)
FV_CACHE_TTL_SEC       = 12 * 3600  # 12 hours — FV is based on fundamentals, changes slowly
FV_FAILURE_CACHE_TTL_SEC = 1 * 3600  # 1 hour — cache FV failures (MANKIND, ABFRL etc.) to avoid retrying every run
MAX_CACHE_ENTRIES      = 500        # LRU cap — evict oldest when exceeded


# ─────────────────────────────────────────────────────────────
#  IN-MEMORY CACHE (avoids re-fetching yfinance / FV data on repeat calls)
#
#  Key insight: when the user runs the same duration backtest twice (e.g.,
#  to test different target_pct or max_hold_days), the OHLCV data is
#  identical. Caching at the _download_single_ticker level means a 200-
#  ticker re-run that took 30s now takes <0.5s.
#
#  Cache is keyed by ticker symbol. Each entry stores the DataFrame plus
#  a timestamp. TTL is 4 hours for price data (yfinance updates EOD) and
#  12 hours for FV (fundamentals change slowly).
# ─────────────────────────────────────────────────────────────
import threading as _threading

_yf_cache: Dict[str, dict] = {}    # {ticker: {"df": DataFrame, "ts": float, "hits": int}}
_fv_cache: Dict[str, dict] = {}    # {ticker: {"fv": dict, "ts": float, "hits": int}} — successful FV only
_fv_fail_cache: Dict[str, dict] = {}  # {ticker: {"fv": dict, "ts": float}} — failed FV (short TTL)
_cache_lock = _threading.Lock()

# Stats counters (for /cache/stats endpoint)
_cache_stats = {
    "yf_hits": 0, "yf_misses": 0, "yf_evictions": 0,
    "fv_hits": 0, "fv_misses": 0, "fv_evictions": 0,
    "fv_fail_hits": 0, "fv_fail_misses": 0,
    "created_at": time.time(),
}


def _cache_get(cache: dict, key: str, ttl_sec: int) -> Optional[object]:
    """Get a value from cache if not expired. Returns None on miss/expiry."""
    with _cache_lock:
        entry = cache.get(key)
        if entry is None:
            return None
        age = time.time() - entry["ts"]
        if age > ttl_sec:
            # Expired — remove and report miss
            del cache[key]
            return None
        entry["hits"] = entry.get("hits", 0) + 1
        return entry["value"]


def _cache_put(cache: dict, key: str, value: object, stats_key: str) -> None:
    """Put a value into cache. Enforces MAX_CACHE_ENTRIES via simple FIFO
    eviction (oldest by timestamp)."""
    with _cache_lock:
        # Enforce max size — evict oldest entries
        while len(cache) >= MAX_CACHE_ENTRIES:
            oldest_key = min(cache.keys(), key=lambda k: cache[k]["ts"])
            del cache[oldest_key]
            _cache_stats[f"{stats_key}_evictions"] += 1
        cache[key] = {"value": value, "ts": time.time(), "hits": 0}


def _cache_clear() -> Dict[str, int]:
    """Clear all cache entries (in-memory AND disk). Returns counts of what was cleared.
    This is called when user clicks 'Clear Cache' button — the ONLY way cache is cleared."""
    with _cache_lock:
        yf_count = len(_yf_cache)
        fv_count = len(_fv_cache)
        fv_fail_count = len(_fv_fail_cache)
        _yf_cache.clear()
        _fv_cache.clear()
        _fv_fail_cache.clear()
    # Also delete the disk cache files so cleared state persists across restarts
    _cache_clear_disk()
    return {
        "yf_cleared": yf_count,
        "fv_cleared": fv_count,
        "fv_fail_cleared": fv_fail_count,
    }


def _cache_stats_snapshot() -> Dict:
    """Get a snapshot of cache stats (thread-safe)."""
    with _cache_lock:
        # Compute total size of cached DataFrames (approximate)
        yf_size_mb = 0.0
        for entry in _yf_cache.values():
            df = entry.get("value")
            if df is not None and hasattr(df, "memory_usage"):
                try:
                    yf_size_mb += df.memory_usage(deep=True).sum() / (1024 * 1024)
                except Exception:
                    pass
        # Check disk cache file sizes
        disk_files = {}
        for name, path in [("yf", YF_CACHE_FILE), ("fv", FV_CACHE_FILE), ("fv_fail", FV_FAIL_CACHE_FILE)]:
            try:
                if os.path.exists(path):
                    disk_files[name] = {
                        "exists": True,
                        "size_mb": round(os.path.getsize(path) / (1024 * 1024), 2),
                        "modified": os.path.getmtime(path),
                    }
                else:
                    disk_files[name] = {"exists": False}
            except Exception:
                disk_files[name] = {"exists": False}
        return {
            "yf_cache_entries": len(_yf_cache),
            "fv_cache_entries": len(_fv_cache),
            "fv_fail_cache_entries": len(_fv_fail_cache),
            "yf_cache_size_mb": round(yf_size_mb, 2),
            "yf_hits": _cache_stats["yf_hits"],
            "yf_misses": _cache_stats["yf_misses"],
            "yf_evictions": _cache_stats["yf_evictions"],
            "fv_hits": _cache_stats["fv_hits"],
            "fv_misses": _cache_stats["fv_misses"],
            "fv_evictions": _cache_stats["fv_evictions"],
            "fv_fail_hits": _cache_stats["fv_fail_hits"],
            "fv_fail_misses": _cache_stats["fv_fail_misses"],
            "yf_ttl_hours": YF_CACHE_TTL_SEC / 3600,
            "fv_ttl_hours": FV_CACHE_TTL_SEC / 3600,
            "fv_fail_ttl_minutes": FV_FAILURE_CACHE_TTL_SEC / 60,
            "max_entries": MAX_CACHE_ENTRIES,
            "uptime_hours": round((time.time() - _cache_stats["created_at"]) / 3600, 2),
            "disk_persisted": True,  # cache survives restarts
            "disk_cache_dir": CACHE_DIR,
            "disk_files": disk_files,
            "last_disk_save": _last_disk_save_ts if _last_disk_save_ts > 0 else None,
        }


# ─────────────────────────────────────────────────────────────
#  DISK PERSISTENCE — cache survives server restarts
#
#  The in-memory cache is fast but lost on restart. These functions save
#  the cache to pickle files on disk so it persists across restarts.
#  The cache is ONLY cleared when the user clicks "Clear Cache" (which
#  calls _cache_clear() → _cache_clear_disk()).
#
#  Strategy:
#    - Load from disk on startup (module import)
#    - Save to disk after each backtest/screen endpoint completes
#    - Also save on a 5-minute timer as a safety net
# ─────────────────────────────────────────────────────────────

_last_disk_save_ts = 0.0
_DISK_SAVE_MIN_INTERVAL = 30.0  # don't save more than once per 30s (avoids disk thrash)


def _save_cache_to_disk(force: bool = False) -> Dict[str, int]:
    """Save all three caches (yf, fv, fv_fail) to pickle files on disk.

    Returns counts of entries saved per cache. Throttled to once per 30s
    unless force=True (used by Clear Cache and shutdown).
    """
    global _last_disk_save_ts
    now = time.time()
    if not force and (now - _last_disk_save_ts) < _DISK_SAVE_MIN_INTERVAL:
        return {"skipped": True, "reason": "throttled"}

    with _cache_lock:
        try:
            # Save yfinance cache (DataFrames — use pickle, not JSON)
            with open(YF_CACHE_FILE, "wb") as f:
                pickle.dump(dict(_yf_cache), f, protocol=pickle.HIGHEST_PROTOCOL)
            # Save FV success cache
            with open(FV_CACHE_FILE, "wb") as f:
                pickle.dump(dict(_fv_cache), f, protocol=pickle.HIGHEST_PROTOCOL)
            # Save FV failure cache
            with open(FV_FAIL_CACHE_FILE, "wb") as f:
                pickle.dump(dict(_fv_fail_cache), f, protocol=pickle.HIGHEST_PROTOCOL)
            _last_disk_save_ts = now
            counts = {
                "yf_saved": len(_yf_cache),
                "fv_saved": len(_fv_cache),
                "fv_fail_saved": len(_fv_fail_cache),
            }
            logger.info(f"[Cache-Disk] Saved to disk: {counts}")
            return counts
        except Exception as e:
            logger.error(f"[Cache-Disk] Save failed: {e}")
            return {"error": str(e)}


def _load_cache_from_disk() -> Dict[str, int]:
    """Load all three caches from pickle files on disk. Called on startup.

    Skips entries that have expired (based on their stored timestamp + TTL).
    """
    loaded = {"yf_loaded": 0, "fv_loaded": 0, "fv_fail_loaded": 0, "expired": 0}
    now = time.time()

    for cache_file, cache_dict, ttl, key_prefix in [
        (YF_CACHE_FILE,      _yf_cache,      YF_CACHE_TTL_SEC,         "yf_loaded"),
        (FV_CACHE_FILE,      _fv_cache,      FV_CACHE_TTL_SEC,         "fv_loaded"),
        (FV_FAIL_CACHE_FILE, _fv_fail_cache, FV_FAILURE_CACHE_TTL_SEC, "fv_fail_loaded"),
    ]:
        if not os.path.exists(cache_file):
            continue
        try:
            with open(cache_file, "rb") as f:
                loaded_data = pickle.load(f)
            if not isinstance(loaded_data, dict):
                continue
            for key, entry in loaded_data.items():
                if not isinstance(entry, dict) or "ts" not in entry or "value" not in entry:
                    continue
                age = now - entry["ts"]
                if age > ttl:
                    loaded["expired"] += 1
                    continue  # skip expired entries
                entry["hits"] = entry.get("hits", 0)
                cache_dict[key] = entry
                loaded[key_prefix] += 1
        except Exception as e:
            logger.warning(f"[Cache-Disk] Load failed for {cache_file}: {e}")
            # Corrupt cache file — remove it so it doesn't cause issues
            try:
                os.remove(cache_file)
            except Exception:
                pass

    logger.info(f"[Cache-Disk] Loaded from disk: {loaded}")
    return loaded


def _cache_clear_disk() -> None:
    """Delete the cache pickle files from disk. Called by Clear Cache endpoint."""
    for cache_file in [YF_CACHE_FILE, FV_CACHE_FILE, FV_FAIL_CACHE_FILE]:
        try:
            if os.path.exists(cache_file):
                os.remove(cache_file)
        except Exception as e:
            logger.warning(f"[Cache-Disk] Failed to remove {cache_file}: {e}")
    logger.info("[Cache-Disk] Disk cache files removed")


# Load cache from disk on module import (server startup)
_load_cache_from_disk()

# ─────────────────────────────────────────────────────────────
#  TICKER LOADING
# ─────────────────────────────────────────────────────────────
_fno_df: Optional[pd.DataFrame] = None
_all_df: Optional[pd.DataFrame] = None
_healthy_df: Optional[pd.DataFrame] = None


def _load_csv_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["symbol", "company_name"])
    try:
        df = pd.read_csv(csv_path, dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]
        if "name of company" in df.columns:
            df = df.rename(columns={"name of company": "company_name"})
        elif "security" in df.columns:
            df = df.rename(columns={"security": "company_name"})
        elif "company_name" not in df.columns:
            df["company_name"] = ""
        if "symbol" not in df.columns:
            return pd.DataFrame(columns=["symbol", "company_name"])
        df["symbol"]       = df["symbol"].str.strip().str.upper()
        df["company_name"] = df.get("company_name", pd.Series([""] * len(df))).fillna("").str.strip()
        out = df[["symbol", "company_name"]].dropna(subset=["symbol"])
        return out[out["symbol"] != ""].reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["symbol", "company_name"])


def _load_fno_df() -> pd.DataFrame:
    global _fno_df
    if _fno_df is None:
        _fno_df = _load_csv_df(FNO_CSV)
    return _fno_df


def _load_all_df() -> pd.DataFrame:
    global _all_df
    if _all_df is None:
        _all_df = _load_csv_df(ALL_CSV)
    return _all_df


def _load_healthy_df() -> pd.DataFrame:
    """Load the pre-filtered healthy/strict ticker list.
    This CSV contains only actively-traded NSE tickers — no delisted,
    no suspended, no illiquid names. Used by the full-universe screener
    to avoid wasting API calls on dead tickers.
    Falls back to FNO list if the healthy CSV doesn't exist."""
    global _healthy_df
    if _healthy_df is not None:
        return _healthy_df
    # Try the healthy CSV first
    healthy_path = os.path.join(DATA_DIR, HEALTHY_CSV)
    if not os.path.exists(healthy_path):
        healthy_path = HEALTHY_CSV  # try cwd
    if os.path.exists(healthy_path):
        _healthy_df = _load_csv_df(healthy_path)
        logger.info(f"[TickerLoad] Loaded healthy ticker list: {len(_healthy_df)} symbols from {HEALTHY_CSV}")
        if not _healthy_df.empty:
            return _healthy_df
    # Fallback: use FNO list if healthy CSV not found
    logger.warning(f"[TickerLoad] {HEALTHY_CSV} not found — falling back to FNO list ({FNO_CSV})")
    _healthy_df = _load_fno_df()
    return _healthy_df


# ─────────────────────────────────────────────────────────────
#  ROBUST SINGLE-TICKER DOWNLOAD (.NS → .BO fallback)
# ─────────────────────────────────────────────────────────────
def _download_single_ticker(ticker: str, period: str = DOWNLOAD_PERIOD) -> Optional[pd.DataFrame]:
    """
    Download a single ticker using yf.Ticker().history().
    Much more reliable than yf.download batch mode.
    Tries .NS first, then .BO if that fails.

    CACHING: Results are cached for YF_CACHE_TTL_SEC (4 hours) keyed by ticker.
    A copy of the cached DataFrame is returned so callers can mutate freely.
    """
    # Cache key — include period so different periods don't collide
    cache_key = f"{ticker}:{period}"

    # Check cache first
    cached = _cache_get(_yf_cache, cache_key, YF_CACHE_TTL_SEC)
    if cached is not None:
        with _cache_lock:
            _cache_stats["yf_hits"] += 1
        logger.debug(f"[Cache-YF] HIT  {ticker} (cached, returning copy)")
        return cached.copy()  # return a copy so callers can mutate

    with _cache_lock:
        _cache_stats["yf_misses"] += 1

    for suffix in [".NS", ".BO"]:
        try:
            yf_sym = f"{ticker}{suffix}"
            t = yf.Ticker(yf_sym)
            hist = t.history(period=period, auto_adjust=True)
            if hist is not None and not hist.empty and len(hist) >= 10:
                df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
                df = df.dropna(how="all")
                if len(df) >= 10:
                    # Cache the result (cache the original, return a copy)
                    _cache_put(_yf_cache, cache_key, df, "yf")
                    return df.copy()
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────────────────────
#  RELIABLE BATCH DOWNLOAD (Loop instead of yf.download)
# ─────────────────────────────────────────────────────────────

def _batch_download(
    tickers: List[str],
    interval: str,
    lookback_days: int,
) -> Dict[str, pd.DataFrame]:
    """
    Synchronous batch download (kept for backward compatibility with
    _screen_ticker and other sync callers). For new code, prefer
    _batch_download_async which runs downloads concurrently.
    """
    if not tickers:
        return {}

    result: Dict[str, pd.DataFrame] = {}
    cache_hits = 0
    cache_misses = 0

    for ticker in tickers:
        cache_key = f"{ticker}:{DOWNLOAD_PERIOD}"
        with _cache_lock:
            entry = _yf_cache.get(cache_key)
            was_cached = (entry is not None and
                         time.time() - entry["ts"] <= YF_CACHE_TTL_SEC)

        df = _download_single_ticker(ticker, period=DOWNLOAD_PERIOD)
        if df is not None:
            result[ticker] = df

        if was_cached:
            cache_hits += 1
        else:
            cache_misses += 1
            time.sleep(0.15)  # Small delay to avoid yfinance rate limiting

    logger.info(f"[Batch-Download] {len(result)}/{len(tickers)} tickers downloaded | "
                f"cache: {cache_hits} hits, {cache_misses} misses")
    return result


# Concurrency limit for parallel yfinance downloads
# Keep LOW to avoid yfinance rate limiting (soft-blocks at ~2000 req/hr).
# At concurrency=5 with ~1.5s per request, we do ~200 req/min in bursts,
# which is under yfinance's burst tolerance (~100/min sustained, burst OK).
# The cache makes this a one-time cost — subsequent runs are instant.
YF_CONCURRENCY = 5


async def _batch_download_async(
    tickers: List[str],
    interval: str = "1d",
    lookback_days: int = 500,
) -> Dict[str, pd.DataFrame]:
    """
    CONCURRENT batch download — runs up to YF_CONCURRENCY downloads in parallel.

    For 200 tickers with empty cache, this turns a ~30s sequential download
    (200 × 0.15s sleep) into a ~5s concurrent batch. Cache hits are instant
    and don't count against the concurrency limit.

    This is the preferred download function for async endpoints (backtest,
    duration-backtest, screen).
    """
    if not tickers:
        return {}

    _yf_semaphore = asyncio.Semaphore(YF_CONCURRENCY)
    cache_hits = 0
    cache_misses = 0

    async def _download_one(ticker):
        nonlocal cache_hits, cache_misses
        cache_key = f"{ticker}:{DOWNLOAD_PERIOD}"
        with _cache_lock:
            entry = _yf_cache.get(cache_key)
            was_cached = (entry is not None and
                         time.time() - entry["ts"] <= YF_CACHE_TTL_SEC)

        if was_cached:
            cache_hits += 1
            # Cached returns are instant — no semaphore needed
            df = await asyncio.to_thread(_download_single_ticker, ticker, DOWNLOAD_PERIOD)
        else:
            cache_misses += 1
            # Real yfinance call — limit concurrency to avoid rate limiting
            async with _yf_semaphore:
                # Tiny stagger (50ms) so concurrent requests don't all hit
                # yfinance at the exact same millisecond
                await asyncio.sleep(0.05)
                df = await asyncio.to_thread(_download_single_ticker, ticker, DOWNLOAD_PERIOD)

        return (ticker, df)

    t_start = time.time()
    results = await asyncio.gather(*[_download_one(t) for t in tickers])
    t_end = time.time()

    result: Dict[str, pd.DataFrame] = {}
    for ticker, df in results:
        if df is not None:
            result[ticker] = df

    logger.info(f"[Batch-Download-Async] {len(result)}/{len(tickers)} tickers downloaded | "
                f"cache: {cache_hits} hits, {cache_misses} misses | "
                f"concurrency: {YF_CONCURRENCY} | "
                f"time: {t_end - t_start:.1f}s "
                f"(sequential would take ~{cache_misses * 0.15:.0f}s)")
    return result


# ─────────────────────────────────────────────────────────────
#  JSON SANITIZER — converts NaN / Infinity floats to None so the
#  response can be JSON-serialized without raising
#  "ValueError: Out of range float values are not JSON compliant".
#
#  Why this exists:
#    yfinance occasionally returns rows where Open/High/Low/Close/Volume
#    are NaN (delisted ticker, missing bar, network glitch). When our
#    signal processor blindly does `float(...) / round(...)` on those
#    values, NaN propagates into the response dict. Python's strict
#    JSON encoder (used by FastAPI/Starlette) refuses NaN with a 500.
#
#  This sanitizer is the LAST line of defense — every router response
#  is passed through it before being returned. Local guards in
#  _compute_trend / _process_ticker_df remain as a first line.
# ─────────────────────────────────────────────────────────────
import math as _math


def _json_safe(obj):
    """
    Recursively walk a dict/list/tuple/scalar and replace any non-finite
    float (NaN, +Inf, -Inf) with None. Also converts numpy scalar types
    (np.float64, np.int64, np.bool_) to native Python types so the
    default JSON encoder doesn't choke on them.

    Safe to call on any object — non-numeric values pass through unchanged.
    """
    # numpy scalar types — convert to native python first
    try:
        import numpy as _np
        if isinstance(obj, _np.generic):
            obj = obj.item()
    except ImportError:
        pass

    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int,)):
        return obj
    if isinstance(obj, float):
        # The critical check — NaN and Inf are not JSON-compliant
        if _math.isnan(obj) or _math.isinf(obj):
            return None
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    # Pandas Timestamp, datetime, etc. — stringify as fallback
    try:
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
    except Exception:
        pass
    # Last resort: try to coerce to float and re-check
    try:
        f = float(obj)
        if _math.isnan(f) or _math.isinf(f):
            return None
        return f if f.is_integer() else f
    except (TypeError, ValueError):
        return str(obj)


# ─────────────────────────────────────────────────────────────
#  SWING LOW & STRUCTURAL LOW DETECTION
# ─────────────────────────────────────────────────────────────
#
#  Computes the deepest unbroken swing low for a ticker — the most
#  important level for stop-loss placement. Uses the Williams Fractal
#  approach: a swing low is a bar whose Low is lower than both the
#  preceding and following bars.
#
#  Returns two levels:
#    • Recent swing low: the most recent (closest) unbroken swing low
#    • Structural swing low: the DEEPEST unbroken swing low in the
#      full lookback (the real trend-invalidation floor)
#
#  "Unbroken" means today's Close is still ABOVE that low — if Close
#  has fallen below it, the low is considered broken and excluded.
#
#  Fields injected into the signal dict:
#    swing_low_price, swing_low_date, swing_low_days_ago
#    swing_pullback, swing_pullback_proximity_pct, swing_pullback_days_ago
#    swing_pullback_is_higher_low, swing_pullback_score
#    structural_swing_low_price, structural_swing_low_date, structural_swing_days_ago
#    structural_swing_proximity_pct, structural_is_same_as_recent
# ─────────────────────────────────────────────────────────────
def _compute_swing_and_structural_low(df: pd.DataFrame, current_price: float) -> Dict:
    """
    Detect recent swing low and deepest structural (unbroken) swing low.
    Returns a flat dict with all fields the frontend expects.
    """
    out: Dict = {
        "swing_low_price":           None,
        "swing_low_date":            None,
        "swing_low_days_ago":        None,
        "swing_pullback":            False,
        "swing_pullback_proximity_pct": None,
        "swing_pullback_days_ago":   None,
        "swing_pullback_is_higher_low": None,
        "swing_pullback_score":       0.0,
        "structural_swing_low_price": None,
        "structural_swing_low_date":  None,
        "structural_swing_days_ago":  None,
        "structural_swing_proximity_pct": None,
        "structural_is_same_as_recent": None,
    }

    try:
        if df is None or len(df) < 10 or current_price is None or current_price <= 0:
            return out

        n = len(df)
        lows = df["Low"].values
        highs = df["High"].values
        closes = df["Close"].values
        dates = df.index

        # ── Step 1: Identify all swing lows (Williams Fractal) ──────
        # A swing low at bar i: Low[i] < Low[i-1] AND Low[i] < Low[i+1]
        all_swing_lows = []  # ALL swing lows (for structural low calc)
        unbroken_swing_lows = []  # only unbroken ones (for pullback calc)
        for i in range(1, n - 1):
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                sl_price = float(lows[i])
                d = dates[i]
                date_str = str(d.date()) if hasattr(d, "date") else str(d)[:10]
                entry = (i, sl_price, date_str)
                all_swing_lows.append(entry)
                # Unbroken: today's close still above this low
                if current_price >= sl_price:
                    unbroken_swing_lows.append(entry)

        if not all_swing_lows:
            return out

        # ── Step 2: Recent swing low (most recent UNBROKEN) ─────
        # Use unbroken swing lows for pullback detection (entry timing)
        swing_lows = unbroken_swing_lows
        if not swing_lows:
            # No unbroken lows — use all swing lows and mark proximity as negative
            swing_lows = all_swing_lows
        recent_idx, recent_price, recent_date = swing_lows[-1]
        recent_days_ago = n - 1 - recent_idx

        # Proximity: how far current price is ABOVE this swing low
        recent_prox_pct = round((current_price - recent_price) / recent_price * 100, 2)

        # Check if this is a "higher low" than the one before it
        is_higher_low = None
        if len(swing_lows) >= 2:
            prev_price = swing_lows[-2][1]
            is_higher_low = recent_price > prev_price

        # Pullback score: higher when closer to swing low and recent
        pullback_score = 0.0
        if recent_prox_pct <= 5.0 and recent_days_ago <= 30:
            # Score combines closeness (inverse of proximity) + recency
            closeness = max(0, 5.0 - recent_prox_pct) / 5.0  # 0..1
            recency = max(0, 30 - recent_days_ago) / 30.0    # 0..1
            pullback_score = round(closeness * 0.7 + recency * 0.3, 3)

        # A stock is in "pullback" mode if within 5% of recent swing low
        is_pullback = recent_prox_pct <= 5.0

        out["swing_low_price"]           = round(recent_price, 2)
        out["swing_low_date"]            = recent_date
        out["swing_low_days_ago"]        = recent_days_ago
        out["swing_pullback"]            = is_pullback
        out["swing_pullback_proximity_pct"] = recent_prox_pct
        out["swing_pullback_days_ago"]   = recent_days_ago
        out["swing_pullback_is_higher_low"] = is_higher_low
        out["swing_pullback_score"]       = pullback_score

        # ── Step 3: Structural swing low (most-touched SUPPORT zone) ──
        # A structural low MUST be a SUPPORT level — meaning it must be
        # at or BELOW the current price.  A swing low above the current
        # price is resistance, not support.
        #
        # Among below-price swing lows, find the price zone that was
        # TESTED THE MOST NUMBER OF TIMES.  A "touch" = another swing
        # low within ±5% of the candidate's price.
        # The cluster with most touches = the real structural support.
        # Structural low = lowest price in that most-touched cluster.
        #
        # Tie-breaking: among clusters with equal touches, pick the one
        # closest to current price (most actionable for stop placement).
        #
        # Relevance guard: only consider swing lows within 1200 bars.

        STRUCTURAL_MAX_LOOKBACK = 1200
        _cutoff_idx = max(0, n - 1 - STRUCTURAL_MAX_LOOKBACK)

        # Only swing lows AT or BELOW current price (support, not resistance)
        # and within the lookback window
        _support_candidates = [
            (i, p, d) for i, p, d in all_swing_lows
            if p <= current_price and i >= _cutoff_idx
        ]

        if _support_candidates:
            def _cluster_touches(swing_lows_list, target_price, pct_band=5.0):
                """Count swing lows within ±pct_band% of target_price."""
                lo = target_price * (1 - pct_band / 100)
                hi = target_price * (1 + pct_band / 100)
                return sum(1 for _, p, _ in swing_lows_list if lo <= p <= hi)

            # Score each candidate: (touches, closeness_to_price, -price)
            # Higher touches = better. If tied, closer to current = better.
            # If still tied, lower price = better (deeper support).
            best = None
            best_score = (-1, 0.0, 0.0)
            for idx, price, date in _support_candidates:
                touches = _cluster_touches(_support_candidates, price)
                closeness = -abs(current_price - price) / current_price
                score = (touches, closeness, -price)
                if score > best_score:
                    best_score = score
                    best = (idx, price, date)

            # Find the LOWEST price within the same cluster (±5% of best)
            _band_lo = best[1] * 0.95
            _band_hi = best[1] * 1.05
            _cluster_members = [
                (i, p, d) for i, p, d in _support_candidates
                if _band_lo <= p <= _band_hi
            ]
            deepest = min(_cluster_members, key=lambda x: x[1])
            deepest_idx, deepest_price, deepest_date = deepest
            deepest_days_ago = n - 1 - deepest_idx
            struct_touches = len(_cluster_members)

        else:
            # No below-price swing low in lookback → absolute deepest
            _recent_candidates = [
                (i, p, d) for i, p, d in all_swing_lows if i >= _cutoff_idx
            ]
            if _recent_candidates:
                deepest = min(_recent_candidates, key=lambda x: x[1])
            else:
                deepest = min(all_swing_lows, key=lambda x: x[1])
            deepest_idx, deepest_price, deepest_date = deepest
            deepest_days_ago = n - 1 - deepest_idx
            struct_touches = 1

        # Proximity from structural low (positive = price is above = normal)
        struct_prox_pct = round((current_price - deepest_price) / deepest_price * 100, 2)

        # Check if structural == recent (same swing low)
        is_same = (deepest_idx == recent_idx)

        out["structural_swing_low_price"]       = round(deepest_price, 2)
        out["structural_swing_low_date"]         = deepest_date
        out["structural_swing_days_ago"]         = deepest_days_ago
        out["structural_swing_proximity_pct"]    = struct_prox_pct
        out["structural_is_same_as_recent"]      = is_same
        out["structural_touches"]               = struct_touches

    except Exception as exc:
        import traceback
        traceback.print_exc()
        # Return the default (all-None) dict on any error — graceful degradation

    return out


# ─────────────────────────────────────────────────────────────
#  TREND DETECTION
# ─────────────────────────────────────────────────────────────
def _compute_trend(df: pd.DataFrame) -> Dict:
    n = len(df)
    close = df["Close"]

    # NaN-safe scalar extraction helper
    def _safe_float(val):
        try:
            f = float(val)
            return f if not _math.isnan(f) and not _math.isinf(f) else None
        except (TypeError, ValueError):
            return None

    sma50  = _safe_float(close.rolling(50).mean().iloc[-1])  if n >= 50  else None
    sma200 = _safe_float(close.rolling(200).mean().iloc[-1]) if n >= 200 else None
    price  = _safe_float(close.iloc[-1])

    lr_slope_pct = 0.0
    if n >= 30:
        window = min(60, n)
        y  = close.iloc[-window:].values
        x  = np.arange(window, dtype=float)
        y_mean = _safe_float(y.mean())
        if y_mean not in (None, 0):
            slope = _safe_float(np.polyfit(x, y, 1)[0])
            if slope is not None:
                lr_slope_pct = round(slope / y_mean * 100, 4)

    above_50sma  = (price > sma50)  if (sma50  is not None and price is not None) else None
    above_200sma = (price > sma200) if (sma200 is not None and price is not None) else None

    if sma50 is not None and sma200 is not None:
        if above_50sma and sma50 > sma200 and lr_slope_pct > -0.05:
            regime = "UPTREND"
        elif not above_50sma and sma50 < sma200 and lr_slope_pct < 0.05:
            regime = "DOWNTREND"
        else:
            regime = "SIDEWAYS"
    elif sma50 is not None:
        if above_50sma and lr_slope_pct > 0:
            regime = "UPTREND"
        elif not above_50sma and lr_slope_pct < 0:
            regime = "DOWNTREND"
        else:
            regime = "SIDEWAYS"
    else:
        regime = "UPTREND" if lr_slope_pct > 0.05 else ("DOWNTREND" if lr_slope_pct < -0.05 else "SIDEWAYS")

    return {
        "trend_regime":   regime,
        "sma50":          round(sma50,  2) if sma50  is not None else None,
        "sma200":         round(sma200, 2) if sma200 is not None else None,
        "lr_slope_pct":   lr_slope_pct,
        "above_50sma":    above_50sma,
        "above_200sma":   above_200sma,
    }


# ─────────────────────────────────────────────────────────────
#  EMA9 BREAKOUT CORE LOGIC
# ─────────────────────────────────────────────────────────────
def _process_ticker_df(ticker: str, df: pd.DataFrame, max_candles_ago: int) -> Dict:
    df = df.copy().dropna()

    if len(df) < MIN_HISTORY_DAYS:
        return {
            "ticker":  ticker,
            "status":  "NO_DATA",
            "error":   f"Insufficient data ({len(df)} days, need ≥ {MIN_HISTORY_DAYS})",
        }

    df["ema9"]  = df["Close"].ewm(span=9, adjust=False).mean()
    df["sma50"] = df["Close"].rolling(50).mean()

    # ── ATR (Average True Range) for Volatility-Adjusted Target ──────────────
    # Wilder's smoothing (EMA with alpha=1/period). Used to compute per-signal
    # target_price when ema9_vol_target_mode="atr". Always computed so the
    # atr_pct_at_entry annotation is available even in "fixed" mode (lets the
    # frontend show volatility context for every signal).
    _high  = df["High"]
    _low   = df["Low"]
    _close = df["Close"]
    _prev_close = _close.shift(1)
    _tr = pd.concat([
        (_high - _low).abs(),
        (_high - _prev_close).abs(),
        (_low  - _prev_close).abs(),
    ], axis=1).max(axis=1)
    _atr_period_live = max(1, _load_vol_target_atr_period_from_config())
    df["atr"] = _tr.ewm(alpha=1.0 / _atr_period_live, adjust=False, min_periods=_atr_period_live).mean()

    # NaN-safe float helper (local to this function — short name for readability)
    def _sf(val):
        """Convert to float; return None if NaN/Inf."""
        try:
            f = float(val)
            return f if not _math.isnan(f) and not _math.isinf(f) else None
        except (TypeError, ValueError):
            return None

    n             = len(df)
    current_price = _sf(df["Close"].iloc[-1])
    current_ema9  = _sf(df["ema9"].iloc[-1])
    current_sma50 = _sf(df["sma50"].iloc[-1])

    # If current price/ema9 are NaN (corrupted bar from yfinance), bail out gracefully
    if current_price is None or current_ema9 is None:
        return {
            "ticker":  ticker,
            "status":  "NO_DATA",
            "error":   "Latest bar has NaN price/EMA9 (yfinance data corruption)",
        }

    # Round for display after the None-check
    current_price = round(current_price, 2)
    current_ema9  = round(current_ema9,  2)
    current_sma50 = round(current_sma50, 2) if current_sma50 is not None else None

    above_sma50 = current_sma50 is not None and current_price > current_sma50

    # ── Volatility-Adjusted Target config (loaded here, applied after breakout_idx is known) ─
    # When ema9_vol_target_mode="atr", target_price is computed from ATR at the
    # BREAKOUT day (not today) — that's the moment the signal was generated.
    # When mode="fixed", use a flat 3% target (legacy behavior). The mode is
    # read fresh from config.json on every call so the user's UI change takes
    # effect immediately.
    _vol_mode = _load_vol_target_mode_from_config()
    _vol_mult = _load_vol_target_multiplier_from_config()
    _vol_floor = _load_vol_target_floor_pct_from_config()
    _vol_cap  = _load_vol_target_cap_pct_from_config()
    # Legacy flat-3% target — used in NO_SIGNAL branch + as a fallback when ATR
    # is missing or mode="fixed". Computed upfront; replaced with the ATR-based
    # value later for the SIGNAL branch.
    target_3pct = round(current_price * 1.03, 2)
    trend_data  = _compute_trend(df)
    swing_data  = _compute_swing_and_structural_low(df, current_price)

    scan_start = max(1, n - max_candles_ago - 1)
    scan_end   = n - 2
    found        = False
    breakout_idx = None
    confirm_idx  = None

    for i in range(scan_end, scan_start - 1, -1):
        prev_close = float(df["Close"].iloc[i - 1])
        prev_ema   = float(df["ema9"].iloc[i - 1])
        curr_close = float(df["Close"].iloc[i])
        curr_ema   = float(df["ema9"].iloc[i])

        if not (prev_close < prev_ema and curr_close > curr_ema):
            continue

        conf_close = float(df["Close"].iloc[i + 1])
        if conf_close > curr_close:
            found        = True
            breakout_idx = i
            confirm_idx  = i + 1
            break

    if not found:
        return {
            "ticker":        ticker,
            "status":        "NO_SIGNAL",
            "current_price": current_price,
            "ema9":          current_ema9,
            "ema9_value":    current_ema9,
            "sma50_value":   current_sma50,
            "above_sma50":   above_sma50,
            "target_3pct":   target_3pct,
            **trend_data,
            **swing_data,
            # ── Momentum Sweep annotation (NO_SIGNAL branch) ──────────────────
            # Even with no fresh crossover, this ticker may already be
            # above 9EMA and forming higher-highs. The pipeline will
            # collect these into the momentum_sweep bucket.
            **_detect_momentum_sweep(df, current_price, current_ema9),
        }

    breakout_candle = df.iloc[breakout_idx]
    confirm_candle  = df.iloc[confirm_idx]

    def _date(idx):
        d = df.index[idx]
        return str(d.date()) if hasattr(d, "date") else str(d)[:10]

    bo_date  = _date(breakout_idx)
    con_date = _date(confirm_idx)

    entry_price       = float(confirm_candle["Close"])
    target_from_entry = entry_price * 1.03
    target_nearing    = current_price >= target_from_entry * 0.995

    # ── Apply Volatility-Adjusted Target (overrides flat 3% when mode="atr") ─
    # We compute ATR at the BREAKOUT day (when the signal was generated), then
    # derive the target % via:  effective_target_pct = clamp(mult × ATR/price × 100, floor, cap)
    # This becomes the per-signal target_price shown in the Prime Targets table
    # + used by the "NEARING" badge logic above (target_from_entry).
    # When mode="fixed" OR ATR is unavailable, fall back to the flat 3% target.
    atr_at_breakout = _sf(df["atr"].iloc[breakout_idx])
    atr_pct_at_breakout = None
    if atr_at_breakout is not None and entry_price > 0:
        atr_pct_at_breakout = round(atr_at_breakout / entry_price * 100, 2)
    vol_target_info = _compute_vol_target(
        entry_price          = entry_price,
        atr_at_entry         = atr_at_breakout,
        multiplier           = _vol_mult,
        floor_pct            = _vol_floor,
        cap_pct              = _vol_cap,
        fallback_target_pct  = 3.0,
    )
    if _vol_mode == "atr" and vol_target_info["target_mode"] == "atr":
        # Override the flat-3% values with the ATR-derived ones
        target_from_entry = vol_target_info["target_price"]
        target_nearing    = current_price >= target_from_entry * 0.995
        vol_target_mode_used = "atr"
    elif _vol_mode == "atr":
        # ATR requested but unavailable — fall back to flat 3%, mark it
        vol_target_mode_used = "atr_fallback_fixed"
    else:
        vol_target_mode_used = "fixed"
    # Round target_from_entry for display
    target_from_entry_rounded = round(float(target_from_entry), 2)
    target_3pct = target_from_entry_rounded  # also update the legacy field so the rest of the function uses the vol-adjusted value

    # ─── 9EMA distance tracking (for fresh-crossover buy-opportunity detection) ───────
    # initial_ema9_dist_pct: how far ABOVE the 9EMA the price closed on the BREAKOUT
    #   candle (the day of the crossover). Small + positive = clean breakout.
    # current_ema9_dist_pct_signed: same metric but for TODAY's price vs TODAY's 9EMA.
    #   Negative = price has fallen back below 9EMA (signal invalid).
    # ema9_dist_change_pct: how much the gap has widened/narrowed since the breakout day.
    # is_fresh_crossover: True when candles_ago <= 1 — i.e. breakout happened yesterday
    #   (confirmed today) or breakout happened today. This is the prime buy window.
    #
    # ALL math here uses _sf() so a corrupted bar (NaN) cannot leak into the JSON
    # response and trigger "Out of range float values are not JSON compliant".
    breakout_close_val = _sf(breakout_candle["Close"])
    breakout_ema9_val  = _sf(df["ema9"].iloc[breakout_idx])
    initial_ema9_dist_pct = (
        round((breakout_close_val - breakout_ema9_val) / breakout_ema9_val * 100, 2)
        if (breakout_close_val is not None and breakout_ema9_val is not None and breakout_ema9_val > 0)
        else None
    )
    current_ema9_dist_pct_signed = (
        round((current_price - current_ema9) / current_ema9 * 100, 2)
        if (current_price is not None and current_ema9 is not None and current_ema9 > 0)
        else None
    )
    ema9_dist_change_pct = (
        round(current_ema9_dist_pct_signed - initial_ema9_dist_pct, 2)
        if (initial_ema9_dist_pct is not None and current_ema9_dist_pct_signed is not None)
        else None
    )
    candles_ago_val = n - 1 - confirm_idx
    is_fresh_crossover = (candles_ago_val <= 1)

    # ─── Pullback-from-prior-high (entry-quality annotation) ─────────────────────
    # Live-screener parallel to the duration backtest's pullback logic.
    # Looks back `ema9_pullback_lookback_days` bars before the breakout candle,
    # finds the highest High in that window (= "prior high"), and computes how
    # far the entry price has pulled back from it.
    #
    # Config knobs live in config.json (set via the UI on the Screener tab):
    #   ema9_pullback_lookback_days (int, default 60)
    #   ema9_pullback_threshold_pct (float, default 10.0)
    #
    # is_pullback = True iff the pullback distance meets the threshold. The
    # actual filtering (drop non-pullback signals) is done by _run_screen_pipeline
    # based on the ema9_filter_pullback config flag — so the fields are always
    # present on every SIGNAL response, and the filter is applied at the batch
    # level (so partition into prime/other happens after the pullback filter).
    _pb_lookback = _load_pullback_lookback_days_from_config()
    _pb_threshold = _load_pullback_threshold_pct_from_config()
    _pb_fields = _compute_pullback_for_signal(
        df              = df,
        breakout_idx    = breakout_idx,
        entry_price     = entry_price,
        lookback_days   = _pb_lookback,
        threshold_pct   = _pb_threshold,
    )

    chart_start = max(0, n - 60)
    candles = []
    for j in range(chart_start, n):
        row = df.iloc[j]
        # Use _sf to NaN-proof every OHLC + EMA9 + SMA50 value in the candles array
        o = _sf(row["Open"]);  h = _sf(row["High"]);  l = _sf(row["Low"]);  c = _sf(row["Close"])
        e9 = _sf(df["ema9"].iloc[j]);  s50 = _sf(df["sma50"].iloc[j])
        try:
            vol = int(row["Volume"])
        except (TypeError, ValueError):
            vol = 0
        candles.append({
            "date":        str(df.index[j].date()) if hasattr(df.index[j], "date") else str(df.index[j])[:10],
            "open":        round(o, 2) if o is not None else None,
            "high":        round(h, 2) if h is not None else None,
            "low":         round(l, 2) if l is not None else None,
            "close":       round(c, 2) if c is not None else None,
            "volume":      vol,
            "ema9":        round(e9, 2) if e9 is not None else None,
            "sma50":       round(s50, 2) if s50 is not None else None,
            "is_breakout": (_date(j) == bo_date),
            "is_confirm":  (_date(j) == con_date),
        })

    # NaN-safe extraction of breakout/confirm candle values
    bo_close = _sf(breakout_candle["Close"])
    bo_high  = _sf(breakout_candle["High"])
    cf_close = _sf(confirm_candle["Close"])
    ema9_dist_pct_val = (
        round(abs(current_price - current_ema9) / current_ema9 * 100, 2)
        if (current_price is not None and current_ema9 is not None and current_ema9 > 0)
        else None
    )

    return {
        "ticker":                          ticker,
        "status":                          "SIGNAL",
        "current_price":                   current_price,
        "target_3pct":                     target_3pct,
        "target_nearing":                  target_nearing,
        "ema9":                            current_ema9,
        "ema9_value":                      current_ema9,
        "sma50_value":                     current_sma50,
        "above_sma50":                     above_sma50,
        "ema9_dist_pct":                   ema9_dist_pct_val,
        # ── New fields for fresh-crossover buy-opportunity detection ───────────────
        "breakout_ema9":                   round(breakout_ema9_val, 2) if breakout_ema9_val is not None else None,
        "initial_ema9_dist_pct":           initial_ema9_dist_pct,
        "current_ema9_dist_pct_signed":    current_ema9_dist_pct_signed,
        "ema9_dist_change_pct":            ema9_dist_change_pct,
        "is_fresh_crossover":              is_fresh_crossover,
        # ── Pullback-from-prior-high fields (entry-quality) ────────────────────────
        # Always present so the frontend can render them in the Prime Targets /
        # Other Signals tables. Filtering (when ema9_filter_pullback=True) is
        # applied by _run_screen_pipeline AFTER this function returns.
        "prior_high":                      _pb_fields["prior_high"],
        "pullback_dist_pct":               _pb_fields["pullback_dist_pct"],
        "is_pullback":                     _pb_fields["is_pullback"],
        "pullback_lookback_days":          _pb_lookback,
        "pullback_threshold_pct":          _pb_threshold,
        # ── Volatility-Adjusted Target fields (per-signal target_price) ───────────
        # Always present so the frontend can render them. When mode="fixed"
        # (default), effective_target_pct=3.0 and target_mode="fixed" — the
        # target_price is just entry × 1.03 (legacy behavior). When mode="atr",
        # effective_target_pct is computed from ATR/multiplier/floor/cap and
        # target_price = entry × (1 + effective_target_pct/100). atr_pct_at_entry
        # is the ATR as % of price (volatility context for sorting/filtering).
        "effective_target_pct":            vol_target_info["effective_target_pct"],
        "target_mode":                     vol_target_mode_used,
        "atr_at_entry":                    atr_at_breakout,
        "atr_pct_at_entry":                atr_pct_at_breakout,
        "vol_target_multiplier":           _vol_mult,
        "vol_target_floor_pct":            _vol_floor,
        "vol_target_cap_pct":              _vol_cap,
        # ────────────────────────────────────────────────────────────────────────────
        "breakout_date":                   bo_date,
        "breakout_close":                  round(bo_close, 2) if bo_close is not None else None,
        "breakout_high":                   round(bo_high,  2) if bo_high  is not None else None,
        "confirm_date":                    con_date,
        "confirm_close":                   round(cf_close, 2) if cf_close is not None else None,
        "candles_ago":                     candles_ago_val,
        "interval":                        "1d",
        "candles":                         candles,
        **trend_data,
        **swing_data,
        # ── Momentum Sweep annotation (always present) ───────────────────────
        # Stocks ALREADY above 9EMA making higher-highs with intact HH-LL
        # structure. Captured even without a fresh 9EMA crossover.
        **_detect_momentum_sweep(df, current_price, current_ema9),
    }


# ─────────────────────────────────────────────────────────────
#  SINGLE-TICKER SCREEN
# ─────────────────────────────────────────────────────────────
def _screen_ticker(ticker: str, max_candles_ago: int = 10) -> Dict:
    ticker = ticker.strip().upper()
    df = _download_single_ticker(ticker)
    if df is None or len(df) < MIN_HISTORY_DAYS:
        return {
            "ticker":  ticker,
            "status":  "NO_DATA",
            "error":   f"Insufficient data (need ≥ {MIN_HISTORY_DAYS} days)",
        }
    return _process_ticker_df(ticker, df, max_candles_ago)


# ─────────────────────────────────────────────────────────────
#  BACKTEST ENGINE  (9 EMA Breakout Walk-Forward)
# ─────────────────────────────────────────────────────────────
#
#  SIGNAL DEFINITION (mirrors _process_ticker_df):
#    • Breakout  : prev_close < prev_ema9 AND curr_close > curr_ema9
#    • Confirm   : next_close > breakout_close
#    • Entry     : confirm_close  (end of confirm candle)
#
#  EXIT RULES:
#    1. WIN → High ≥ entry × (1 + target_pct/100)
#    No force-exit. Trades that never hit target are dropped (open at end of data).
#
#  Trades are non-overlapping: no new signal while a trade is open.
#  Requires price > SMA50 at entry when require_uptrend=True.
# ─────────────────────────────────────────────────────────────

def _backtest_ticker(
    ticker:          str,
    df:              pd.DataFrame,
    target_pct:      float = 3.0,
    max_hold_days:   int   = 15,
    require_uptrend: bool  = True,
) -> Dict:
    """
    Walk-forward backtest for 9EMA breakout strategy on a single ticker.
    Returns a dict with per-trade list and aggregate statistics.
    """
    df = df.copy().dropna()
    n  = len(df)

    if n < MIN_HISTORY_DAYS:
        return {
            "ticker": ticker,
            "status": "NO_DATA",
            "error":  f"Insufficient data ({n} days, need ≥ {MIN_HISTORY_DAYS})",
            "trades": [],
        }

    df["ema9"]  = df["Close"].ewm(span=9, adjust=False).mean()
    df["sma50"] = df["Close"].rolling(50).mean()

    def _date(idx: int) -> str:
        d = df.index[idx]
        return str(d.date()) if hasattr(d, "date") else str(d)[:10]

    trades: List[Dict] = []
    skipped_signals: List[Dict] = []  # Signals generated while in_trade (for accuracy analysis)
    in_trade = False
    entry_idx: int = 0
    entry_price: float = 0.0
    breakout_date: str = ""
    confirm_date:  str = ""
    signal_idx:    int = 0   # confirm candle index (=entry)

    # Walk forward from day 51 (need 50 for SMA50) to leave room for exit
    for i in range(51, n - 1):
        # ── If in a trade, check exit conditions first ──────────────────
        if in_trade:
            days_held = i - entry_idx
            high_today  = float(df["High"].iloc[i])
            close_today = float(df["Close"].iloc[i])
            target_price = round(entry_price * (1 + target_pct / 100), 2)

            outcome  = None
            exit_px  = None
            exit_date = _date(i)

            # Only exit: Target hit (use intraday High). No force-exit ever.
            if high_today >= target_price:
                outcome = "WIN"
                exit_px = target_price

            if outcome:
                gain_pct = round((exit_px - entry_price) / entry_price * 100, 2)

                # ── Trend regime at trade entry ───────────────────────
                try:
                    sma50_entry = float(df["sma50"].iloc[entry_idx]) if not pd.isna(df["sma50"].iloc[entry_idx]) else None
                    sma200_series = df["Close"].rolling(200).mean()
                    sma200_entry = float(sma200_series.iloc[entry_idx]) if entry_idx >= 200 and not pd.isna(sma200_series.iloc[entry_idx]) else None
                    close_at_entry = float(df["Close"].iloc[entry_idx])
                    if sma50_entry and sma200_entry:
                        if close_at_entry > sma50_entry and sma50_entry > sma200_entry:
                            trend_at_entry = "UPTREND"
                        elif close_at_entry < sma50_entry and sma50_entry < sma200_entry:
                            trend_at_entry = "DOWNTREND"
                        else:
                            trend_at_entry = "SIDEWAYS"
                    elif sma50_entry:
                        trend_at_entry = "UPTREND" if close_at_entry > sma50_entry else "DOWNTREND"
                    else:
                        trend_at_entry = "SIDEWAYS"
                except Exception:
                    trend_at_entry = "SIDEWAYS"

                trades.append({
                    "ticker":        ticker,
                    "breakout_date": breakout_date,
                    "entry_date":    confirm_date,
                    "entry_price":   round(entry_price, 2),
                    "target_price":  target_price,
                    "exit_date":     exit_date,
                    "exit_price":    round(exit_px, 2),
                    "days_held":     days_held,
                    "gain_pct":      gain_pct,
                    "outcome":       outcome,
                    "is_win":        outcome == "WIN",
                    "trend_regime":  trend_at_entry,
                })
                in_trade = False
            # Trade still open — fall through to check for new breakout signals (record as skipped)

        # ── Look for new breakout signal ──────────────────────────────
        prev_close = float(df["Close"].iloc[i - 1])
        prev_ema   = float(df["ema9"].iloc[i - 1])
        curr_close = float(df["Close"].iloc[i])
        curr_ema   = float(df["ema9"].iloc[i])

        # Breakout condition
        if not (prev_close < prev_ema and curr_close > curr_ema):
            continue

        # Confirmation: need at least one more bar
        if i + 1 >= n:
            continue

        conf_close = float(df["Close"].iloc[i + 1])
        if conf_close <= curr_close:
            continue  # Confirmation candle didn't close higher

        # Uptrend filter: price > SMA50 at entry
        sma50_at_entry = df["sma50"].iloc[i + 1]
        if require_uptrend and (pd.isna(sma50_at_entry) or conf_close <= float(sma50_at_entry)):
            continue

        # ── If still in trade when a new signal fires → record as SKIPPED ──
        if in_trade:
            # Compute what the outcome of this skipped trade would have been
            skip_target = round(conf_close * (1 + target_pct / 100), 2)
            skip_outcome = "OPEN"  # will resolve below if we can scan forward
            skip_exit_price = None
            skip_exit_date  = None
            skip_days_held  = None

            # Scan forward to see if this skipped signal would have been a WIN/LOSS/TIMEOUT
            for fwd in range(i + 2, min(n, i + 2 + max_hold_days)):
                fwd_high  = float(df["High"].iloc[fwd])
                fwd_close = float(df["Close"].iloc[fwd])
                fwd_days  = fwd - (i + 1)
                if fwd_high >= skip_target:
                    skip_outcome    = "WIN"
                    skip_exit_price = skip_target
                    skip_exit_date  = _date(fwd)
                    skip_days_held  = fwd_days
                    break
                if fwd_days >= max_hold_days:
                    skip_outcome    = "TIMEOUT"
                    skip_exit_price = fwd_close
                    skip_exit_date  = _date(fwd)
                    skip_days_held  = fwd_days
                    break
            if skip_outcome == "OPEN":
                skip_exit_price = float(df["Close"].iloc[min(n - 1, i + 2)])
                skip_exit_date  = _date(min(n - 1, i + 2))
                skip_days_held  = 0

            skip_gain = round((skip_exit_price - conf_close) / conf_close * 100, 2) if skip_exit_price else 0.0

            # Trend at this skipped signal's entry
            try:
                sma50_skip  = float(df["sma50"].iloc[i + 1]) if not pd.isna(df["sma50"].iloc[i + 1]) else None
                sma200_skip_series = df["Close"].rolling(200).mean()
                sma200_skip = float(sma200_skip_series.iloc[i + 1]) if i + 1 >= 200 and not pd.isna(sma200_skip_series.iloc[i + 1]) else None
                if sma50_skip and sma200_skip:
                    if conf_close > sma50_skip and sma50_skip > sma200_skip:
                        trend_skip = "UPTREND"
                    elif conf_close < sma50_skip and sma50_skip < sma200_skip:
                        trend_skip = "DOWNTREND"
                    else:
                        trend_skip = "SIDEWAYS"
                elif sma50_skip:
                    trend_skip = "UPTREND" if conf_close > sma50_skip else "DOWNTREND"
                else:
                    trend_skip = "SIDEWAYS"
            except Exception:
                trend_skip = "SIDEWAYS"

            skipped_signals.append({
                "ticker":        ticker,
                "breakout_date": _date(i),
                "entry_date":    _date(i + 1),
                "entry_price":   round(conf_close, 2),
                "target_price":  skip_target,
                "exit_date":     skip_exit_date,
                "exit_price":    round(skip_exit_price, 2) if skip_exit_price else None,
                "days_held":     skip_days_held or 0,
                "gain_pct":      skip_gain,
                "outcome":       skip_outcome,
                "is_win":        skip_outcome == "WIN",
                "trend_regime":  trend_skip,
                "was_skipped":   True,
                "skipped_reason": "IN_TRADE",
            })
            continue  # Don't enter this signal — already in trade

        # Enter trade at confirmation close
        in_trade    = True
        entry_idx   = i + 1
        entry_price = conf_close
        breakout_date = _date(i)
        confirm_date  = _date(i + 1)
        # Note: Python for-loops ignore reassignment of i, so the confirm candle (i+1)
        # will be evaluated as the first exit-check bar on the next iteration.
        # days_held will be 0 on that bar — WIN check on confirm bar's High is intentional
        # (gap-up on confirm day should count as a win).

    # ── Summarise ────────────────────────────────────────────────────
    if not trades:
        return {
            "ticker": ticker,
            "status": "OK",
            "trades": [],
            "summary": {
                "total_trades":    0,
                "wins":            0,
                "losses":          0,
                "timeouts":        0,
                "win_rate_pct":    None,
                "avg_gain_pct":    None,
                "avg_win_pct":     None,
                "avg_loss_pct":    None,
                "max_win_pct":     None,
                "max_loss_pct":    None,
                "expectancy_pct":  None,
                "total_return_pct": None,
            },
        }

    wins      = [t for t in trades if t["outcome"] == "WIN"]
    losses    = [t for t in trades if t["outcome"] != "WIN"]
    timeouts  = [t for t in trades if t["outcome"] == "TIMEOUT"]

    gains     = [t["gain_pct"] for t in trades]
    win_gains = [t["gain_pct"] for t in wins]
    los_gains = [t["gain_pct"] for t in losses]

    win_rate   = round(len(wins) / len(trades) * 100, 1) if trades else None
    avg_gain   = round(float(np.mean(gains)), 2)          if gains     else None
    avg_win    = round(float(np.mean(win_gains)), 2)      if win_gains else None
    avg_loss   = round(float(np.mean(los_gains)), 2)      if los_gains else None
    max_win    = round(float(np.max(win_gains)), 2)       if win_gains else None
    max_loss   = round(float(np.min(los_gains)), 2)       if los_gains else None

    # Expectancy = WinRate × AvgWin + (1-WinRate) × AvgLoss
    expectancy = None
    if win_rate is not None and avg_win is not None and avg_loss is not None:
        wr = win_rate / 100
        expectancy = round(wr * avg_win + (1 - wr) * avg_loss, 2)

    # Simple compound return simulation (1 trade at a time, full re-invest)
    equity = 100.0
    for t in trades:
        equity *= (1 + t["gain_pct"] / 100)
    total_return = round(equity - 100, 2)

    return {
        "ticker": ticker,
        "status": "OK",
        "trades": trades,
        "skipped_signals": skipped_signals,
        "summary": {
            "total_trades":     len(trades),
            "wins":             len(wins),
            "losses":           len(losses) - len(timeouts),
            "timeouts":         len(timeouts),
            "win_rate_pct":     win_rate,
            "avg_gain_pct":     avg_gain,
            "avg_win_pct":      avg_win,
            "avg_loss_pct":     avg_loss,
            "max_win_pct":      max_win,
            "max_loss_pct":     max_loss,
            "expectancy_pct":   expectancy,
            "total_return_pct": total_return,
        },
    }


# ─────────────────────────────────────────────────────────────
#  DURATION BACKTEST ENGINE (v1.0 — date-range filtered)
#  Differs from _backtest_ticker in three ways:
#    1. Only counts trades whose ENTRY date is within [start_date, end_date]
#    2. Properly exits trades on WIN / LOSS / TIMEOUT (no silent drops)
#       - WIN     : intraday High >= target  → exit at target, gain = +target%
#       - LOSS    : max_hold_days reached, exit close < entry  → gain = actual %
#       - TIMEOUT : max_hold_days reached, exit close >= entry but < target
#    3. No stop-loss (per user spec) — trades run to target or max_hold_days
# ─────────────────────────────────────────────────────────────

def _duration_backtest_ticker(
    ticker:          str,
    df:              pd.DataFrame,
    target_pct:      float = 3.0,
    max_hold_days:   int   = 15,
    require_uptrend: bool  = True,
    start_date:      Optional[str] = None,
    end_date:        Optional[str] = None,
    resistance_lookback_days: int = 20,
    resistance_threshold_pct: float = 3.0,
    pullback_lookback_days:  int   = 60,
    pullback_threshold_pct:  float = 10.0,
    # ── Volatility-Adjusted Target ──────────────────────────────────────────
    # When vol_target_mode="atr", each trade's target_price is computed from
    # the stock's ATR at entry time instead of a flat target_pct:
    #   target_pct_for_trade = clamp(target_multiplier * ATR_at_entry / entry_price * 100,
    #                                target_floor_pct, target_cap_pct)
    # Default multiplier = 1.5 (catch ~1.5x the average daily move).
    # Floor 2%, cap 8% — keeps sane bounds on extreme volatilities.
    vol_target_mode:        str   = "fixed",   # "fixed" (use target_pct) | "atr" (compute from ATR)
    vol_target_atr_period:  int   = 14,        # ATR lookback (standard 14)
    vol_target_multiplier:  float = 1.5,       # target = multiplier × ATR / price × 100
    vol_target_floor_pct:   float = 2.0,       # min target % regardless of ATR
    vol_target_cap_pct:     float = 8.0,       # max target % regardless of ATR
) -> Dict:
    """
    Walk-forward backtest for the 9EMA breakout strategy, filtered to a
    specific date range. Indicators (EMA9, SMA50) are computed on the FULL
    history so they're accurate at the start of the user's date range — no
    warmup gap. Only trades whose BREAKOUT date falls within
    [start_date, end_date] are counted.

    DIFFERS FROM THE MAIN BACKTEST (_backtest_ticker) in two ways (per user spec):

      1. ENTRY ON BREAKOUT DAY (not confirmation day):
         The main backtest waits for a confirmation candle (next day closes
         higher) before entering. This duration backtest enters IMMEDIATELY
         on the breakout day's close — the day price crosses above the 9EMA.
         This captures the signal one day earlier and at a better entry price.
         entry_date = breakout_date = the day price closed above 9EMA.

      2. ALL SIGNALS RECORDED (no non-overlapping constraint):
         The main backtest skips new signals while a trade is open. This
         duration backtest records EVERY breakout signal as a separate trade,
         even if other trades are still open. Multiple trades can run
         simultaneously, each tracked independently with its own exit logic.

    Trade exits:
      - WIN     : intraday High >= target  → exit at target, gain = +target%
      - LOSS    : max_hold_days reached, exit close < entry  → actual negative %
      - TIMEOUT : max_hold_days reached, exit close >= entry but < target
      No stop-loss (per user spec).
    """
    df = df.copy().dropna()
    n  = len(df)

    if n < MIN_HISTORY_DAYS:
        return {
            "ticker": ticker,
            "status": "NO_DATA",
            "error":  f"Insufficient data ({n} days, need ≥ {MIN_HISTORY_DAYS})",
            "trades": [],
            "summary": {
                "total_trades": 0, "wins": 0, "losses": 0, "timeouts": 0,
                "win_rate_pct": None, "avg_gain_pct": None, "avg_win_pct": None,
                "avg_loss_pct": None, "max_win_pct": None, "max_loss_pct": None,
                "expectancy_pct": None, "total_return_pct": None,
            },
        }

    # Compute indicators on FULL history (so values at start_date are correct)
    df["ema9"]  = df["Close"].ewm(span=9, adjust=False).mean()
    df["sma50"] = df["Close"].rolling(50).mean()
    sma200_series = df["Close"].rolling(200).mean()

    # ── ATR (Average True Range) for Volatility-Adjusted Target ──────────────
    # Wilder's smoothing = EMA with alpha = 1/period (the standard ATR formula).
    # True Range = max(high-low, |high-prev_close|, |low-prev_close|).
    # We compute ATR on FULL history so values at start_date are correct.
    # vol_target_mode="fixed" → ATR is not needed, but it's cheap to compute
    # and harmless (used as annotation only on each trade dict).
    _high  = df["High"]
    _low   = df["Low"]
    _close = df["Close"]
    _prev_close = _close.shift(1)
    _tr = pd.concat([
        (_high - _low).abs(),
        (_high - _prev_close).abs(),
        (_low  - _prev_close).abs(),
    ], axis=1).max(axis=1)
    _atr_period = max(1, int(vol_target_atr_period))
    df["atr"] = _tr.ewm(alpha=1.0 / _atr_period, adjust=False, min_periods=_atr_period).mean()
    # ATR as % of price — used for annotation + vol-target computation
    df["atr_pct"] = (df["atr"] / _close) * 100.0

    # Parse date range (store as python date objects — tz-naive by design,
    # so we can compare against df.index.date which is also tz-naive)
    start_d = None
    end_d   = None
    if start_date:
        try:
            start_d = pd.to_datetime(start_date).date()
        except Exception:
            start_d = None
    if end_date:
        try:
            # +1 day to make end_date inclusive (breakout on end_date itself counts)
            end_d = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).date()
        except Exception:
            end_d = None

    def _date(idx: int) -> str:
        d = df.index[idx]
        return str(d.date()) if hasattr(d, "date") else str(d)[:10]

    def _in_range(idx: int) -> bool:
        d = df.index[idx]
        try:
            d_only = d.date() if hasattr(d, "date") else pd.Timestamp(d).date()
        except Exception:
            return True
        if start_d is not None and d_only < start_d:
            return False
        if end_d is not None and d_only >= end_d:
            return False
        return True

    def _trend_at(idx: int, close_price: float) -> str:
        """Determine trend regime at a given bar index."""
        try:
            sma50_val = float(df["sma50"].iloc[idx]) if not pd.isna(df["sma50"].iloc[idx]) else None
            sma200_val = float(sma200_series.iloc[idx]) if idx >= 200 and not pd.isna(sma200_series.iloc[idx]) else None
            if sma50_val and sma200_val:
                if close_price > sma50_val and sma50_val > sma200_val:
                    return "UPTREND"
                elif close_price < sma50_val and sma50_val < sma200_val:
                    return "DOWNTREND"
                else:
                    return "SIDEWAYS"
            elif sma50_val:
                return "UPTREND" if close_price > sma50_val else "DOWNTREND"
            else:
                return "SIDEWAYS"
        except Exception:
            return "SIDEWAYS"

    trades: List[Dict] = []
    open_trades: List[Dict] = []   # list of currently-open trade dicts

    # Walk forward from day 51 (need 50 for SMA50 warmup)
    for i in range(51, n):
        # ── 1. Check exits for ALL open trades (each tracked independently) ──
        # This runs BEFORE checking for new signals, so a trade entered on
        # day i-1 gets its first exit check on day i (days_held=1).
        still_open = []
        for trade in open_trades:
            days_held   = i - trade["_entry_idx"]
            high_today  = float(df["High"].iloc[i])
            close_today = float(df["Close"].iloc[i])

            outcome = None
            exit_px = None

            # WIN: target hit intraday
            if high_today >= trade["target_price"]:
                outcome = "WIN"
                exit_px = trade["target_price"]
            # Force exit at max_hold_days OR at last bar of data
            elif days_held >= max_hold_days or i >= n - 1:
                outcome = "LOSS" if close_today < trade["entry_price"] else "TIMEOUT"
                exit_px = close_today

            if outcome:
                gain_pct = round((exit_px - trade["entry_price"]) / trade["entry_price"] * 100, 2)

                # ── Peak analysis: highest High AFTER 3% target hit, UNTIL 9EMA breaks down ──
                # User's strategy clarification:
                #   "peak means after 3% target achieved, till 9ema not breakdown
                #    and that peak 9ema breakdown means momentum broken so we can
                #    cutoff trade so count peak as breakdown 9EMA line"
                #
                # This is a WHAT-IF analysis: "If I didn't exit at the 3% target
                # and instead held with a 9EMA trailing stop, what would my peak
                # have been?" The 9EMA breakdown (Close < EMA9) = momentum broken
                # = exit signal for this hypothetical hold.
                #
                # Logic:
                #   1. For WIN trades: target was hit on day i (exit day). Walk
                #      FORWARD from day i, tracking the highest High, until Close
                #      < EMA9 (9EMA breakdown = momentum broken) or end of data.
                #   2. For LOSS/TIMEOUT trades: target was NEVER hit, so there's
                #      no "post-target" period to analyze → peak = None.
                #
                # peak_high      = highest intraday High from target-hit day to
                #                  9EMA breakdown day (inclusive)
                # days_to_peak   = trading days from ENTRY to the peak day
                #                  (so user can compare with days_held)
                # peak_gain_pct  = % gain from entry to peak (compare with target_pct)
                #
                # df["ema9"] is already computed at line 1251 (before the main loop).
                entry_idx = trade["_entry_idx"]
                try:
                    if outcome == "WIN":
                        # Target was hit on day i — start tracking from here
                        target_hit_idx = i
                        peak_high = float(df["High"].iloc[target_hit_idx])
                        peak_idx  = target_hit_idx

                        # Walk FORWARD beyond the actual exit day, tracking
                        # highest High until 9EMA breaks down (Close < EMA9).
                        # This uses FUTURE data after the trade would have
                        # closed — that's intentional for the what-if analysis.
                        for j in range(target_hit_idx + 1, n):
                            high_j = float(df["High"].iloc[j])
                            if high_j > peak_high:
                                peak_high = high_j
                                peak_idx  = j
                            # 9EMA breakdown check: Close < EMA9 → momentum broken
                            close_j = float(df["Close"].iloc[j])
                            ema9_j  = float(df["ema9"].iloc[j])
                            if close_j < ema9_j:
                                break  # Momentum broken — stop tracking

                        days_to_peak  = peak_idx - entry_idx   # from ENTRY to peak
                        peak_gain_pct = round(
                            (peak_high - trade["entry_price"]) / trade["entry_price"] * 100, 2
                        )
                    else:
                        # LOSS / TIMEOUT: target never hit → no post-target peak
                        peak_high     = None
                        days_to_peak  = None
                        peak_gain_pct = None
                except Exception as _e_peak:
                    # Defensive: never let peak computation break the trade log
                    logger.warning(f"[Duration-BT] {ticker} peak-high computation failed: {_e_peak}")
                    peak_high     = None
                    days_to_peak  = None
                    peak_gain_pct = None

                trade["exit_date"]      = _date(i)
                trade["exit_price"]     = round(exit_px, 2)
                trade["days_held"]      = days_held
                trade["gain_pct"]       = gain_pct
                trade["outcome"]        = outcome
                trade["is_win"]         = outcome == "WIN"
                trade["peak_high"]      = round(peak_high, 2) if peak_high is not None else None
                trade["days_to_peak"]   = days_to_peak
                trade["peak_gain_pct"]  = peak_gain_pct
                # Remove the internal _entry_idx before appending
                del trade["_entry_idx"]
                trades.append(trade)
            else:
                still_open.append(trade)
        open_trades = still_open

        # ── 2. Check for new breakout signal on day i ──────────────
        # Don't enter new trades on the last bar (no future days to check exit)
        if i >= n - 1:
            continue

        prev_close = float(df["Close"].iloc[i - 1])
        prev_ema   = float(df["ema9"].iloc[i - 1])
        curr_close = float(df["Close"].iloc[i])
        curr_ema   = float(df["ema9"].iloc[i])

        # Breakout: price crosses ABOVE 9EMA (prev below, curr above)
        if not (prev_close < prev_ema and curr_close > curr_ema):
            continue

        # Uptrend filter at breakout day
        sma50_at_breakout = df["sma50"].iloc[i]
        if require_uptrend and (pd.isna(sma50_at_breakout) or curr_close <= float(sma50_at_breakout)):
            continue

        # Date-range filter: only count trades whose BREAKOUT date is in range
        if not _in_range(i):
            continue

        # ── Enter new trade at breakout day's close ──
        # NO confirmation required — entry is immediate on the breakout candle.
        # This captures the signal one day earlier than the main backtest.
        breakout_date = _date(i)

        # ── Compute overhead resistance (N-day high before breakout) ──
        # Look back `resistance_lookback_days` bars before the breakout day (i).
        # The highest High in that window = overhead resistance.
        # Only count it as resistance if it's ABOVE the entry price (overhead supply).
        # If the breakout itself made a new N-day high, there's no overhead resistance.
        resistance_price = None
        resistance_dist_pct = None
        is_near_resistance = False
        lookback_start = max(0, i - resistance_lookback_days)
        if i > lookback_start:
            recent_highs = df["High"].iloc[lookback_start:i]  # exclude breakout day itself
            if len(recent_highs) > 0:
                n_day_high = float(recent_highs.max())
                # Only count as resistance if ABOVE entry price (overhead supply)
                if n_day_high > curr_close:
                    resistance_price = round(n_day_high, 2)
                    resistance_dist_pct = round((n_day_high - curr_close) / curr_close * 100, 2)
                    # "Near resistance" = within threshold % of the N-day high
                    if resistance_dist_pct <= resistance_threshold_pct:
                        is_near_resistance = True

        # ── Compute pullback from previous high (entry-quality filter) ──────────
        # Look back `pullback_lookback_days` bars before the breakout day (i).
        # The highest High in that window = the "previous high" the stock is
        # recovering from. The pullback distance is how far the entry price
        # has fallen (corrected) from that prior high:
        #   pullback_dist_pct = (prior_high - entry_price) / prior_high * 100
        # A trade is `is_pullback=True` when the pullback distance meets the
        # user-configured threshold (default 10%). This filters out breakouts
        # that happen on already-extended stocks (no healthy correction first)
        # and highlights fresh breakouts AFTER a meaningful pullback — these
        # tend to have better risk:reward because the prior-high supply has
        # already been absorbed by the pullback.
        #
        # Edge case: if today's entry price happens to be ABOVE the prior
        # window's high (e.g., breakout made a new N-day high), pullback is
        # effectively 0 (no correction happened) — the trade does NOT qualify
        # as a pullback setup.
        prior_high = None
        pullback_dist_pct = 0.0    # default 0 (no pullback = no correction from prior high)
        is_pullback = False
        pb_lookback_start = max(0, i - pullback_lookback_days)
        if i > pb_lookback_start:
            pb_window_highs = df["High"].iloc[pb_lookback_start:i]  # exclude breakout day
            if len(pb_window_highs) > 0:
                pb_high_val = float(pb_window_highs.max())
                if pb_high_val > 0:
                    prior_high = round(pb_high_val, 2)
                    # Distance from prior high → current entry (always ≥ 0 if pb_high ≥ entry,
                    # negative only when entry is above the prior window high, which means
                    # today's breakout made a new high → not a pullback setup).
                    raw_pullback = (pb_high_val - curr_close) / pb_high_val * 100
                    pullback_dist_pct = round(raw_pullback, 2)
                    is_pullback = raw_pullback >= float(pullback_threshold_pct)

        # ── Compute target_price: fixed vs volatility-adjusted (ATR) ──────────
        # vol_target_mode="fixed" → use the user's target_pct as-is (legacy behavior).
        # vol_target_mode="atr"   → compute target_pct for THIS trade based on ATR:
        #     effective_target_pct = clamp(
        #       vol_target_multiplier × ATR_at_entry / entry_price × 100,
        #       vol_target_floor_pct, vol_target_cap_pct
        #     )
        # This adapts the target to each stock's volatility. A high-vol stock
        # (ATR/price = 5%) with multiplier=1.5 → target = 7.5% (capped at 8%).
        # A low-vol stock (ATR/price = 1.5%) with multiplier=1.5 → target = 2.25%.
        # Floor prevents trivial 0.5% targets; cap prevents absurd 25% targets
        # on penny stocks with huge ATR.
        atr_at_entry = None
        atr_pct_at_entry = None
        effective_target_pct = float(target_pct)
        target_mode_used = "fixed"
        try:
            atr_val = df["atr"].iloc[i]
            atr_pct_val = df["atr_pct"].iloc[i]
            if not pd.isna(atr_val) and float(atr_val) > 0:
                atr_at_entry = round(float(atr_val), 2)
            if not pd.isna(atr_pct_val) and float(atr_pct_val) > 0:
                atr_pct_at_entry = round(float(atr_pct_val), 2)
            # Apply ATR-based target if mode is "atr" AND we have a valid ATR
            if vol_target_mode == "atr" and atr_pct_at_entry is not None and curr_close > 0:
                raw_target_pct = float(vol_target_multiplier) * atr_pct_at_entry
                # Clamp to [floor, cap]
                effective_target_pct = max(
                    float(vol_target_floor_pct),
                    min(float(vol_target_cap_pct), raw_target_pct)
                )
                target_mode_used = "atr"
            elif vol_target_mode == "atr":
                # ATR not available (very early bars) — fall back to fixed target
                # but mark the mode so the frontend can show it was a fallback.
                target_mode_used = "atr_fallback_fixed"
        except Exception as _e_atr:
            # Defensive: never let ATR computation break the trade log
            logger.warning(f"[Duration-BT] {ticker} ATR computation failed at i={i}: {_e_atr}")

        target_price = round(curr_close * (1 + effective_target_pct / 100), 2)

        new_trade = {
            "ticker":               ticker,
            "breakout_date":        breakout_date,
            "entry_date":           breakout_date,   # same as breakout — entered at close
            "_entry_idx":           i,               # internal: for days_held calculation
            "entry_price":          round(curr_close, 2),
            "target_price":         target_price,
            "effective_target_pct": round(effective_target_pct, 2),   # the % actually applied (may differ from target_pct when mode=atr)
            "target_mode":          target_mode_used,                # "fixed" | "atr" | "atr_fallback_fixed"
            # Volatility (ATR) fields — always annotated so user can sort/filter:
            "atr_at_entry":         atr_at_entry,           # ATR value (₹) at breakout day
            "atr_pct_at_entry":     atr_pct_at_entry,       # ATR as % of price at breakout day
            "trend_regime":         _trend_at(i, curr_close),
            # Resistance fields (overhead supply):
            "resistance_price":     resistance_price,         # N-day high above entry, or None
            "resistance_dist_pct":  resistance_dist_pct,      # % above entry, or None
            "is_near_resistance":   is_near_resistance,       # True if within threshold %
            # Pullback-from-prior-high fields (entry-quality):
            "prior_high":           prior_high,               # N-day high before breakout, or None
            "pullback_dist_pct":    pullback_dist_pct,        # % pullback from prior high (≥0 if pulled back, ≤0 if extended)
            "is_pullback":          is_pullback,              # True if pullback_dist_pct >= threshold
            # Exit fields filled when trade closes:
            "exit_date":     None,
            "exit_price":    None,
            "days_held":     None,
            "gain_pct":      None,
            "outcome":       None,
            "is_win":        False,
            # Peak-during-holding analysis (filled at exit; None until then):
            # peak_high      = highest intraday High from entry to exit (₹)
            # days_to_peak   = trading days from entry until the peak day
            # peak_gain_pct  = % gain from entry to peak (compare with target_pct)
            "peak_high":     None,
            "days_to_peak":  None,
            "peak_gain_pct": None,
        }
        open_trades.append(new_trade)
        # NOTE: No "if in_trade: continue" here — ALL signals become trades.

    # ── Summarise ────────────────────────────────────────────
    if not trades:
        return {
            "ticker":   ticker,
            "status":   "OK",
            "trades":   [],
            "summary": {
                "total_trades": 0, "wins": 0, "losses": 0, "timeouts": 0,
                "win_rate_pct": None, "avg_gain_pct": None, "avg_win_pct": None,
                "avg_loss_pct": None, "max_win_pct": None, "max_loss_pct": None,
                "expectancy_pct": None, "total_return_pct": None,
            },
        }

    wins     = [t for t in trades if t["outcome"] == "WIN"]
    losses   = [t for t in trades if t["outcome"] == "LOSS"]
    timeouts = [t for t in trades if t["outcome"] == "TIMEOUT"]

    gains     = [t["gain_pct"] for t in trades]
    win_gains = [t["gain_pct"] for t in wins]
    los_gains = [t["gain_pct"] for t in losses]

    win_rate   = round(len(wins) / len(trades) * 100, 1) if trades else None
    avg_gain   = round(float(np.mean(gains)), 2)          if gains     else None
    avg_win    = round(float(np.mean(win_gains)), 2)      if win_gains else None
    avg_loss   = round(float(np.mean(los_gains)), 2)      if los_gains else None
    max_win    = round(float(np.max(win_gains)), 2)       if win_gains else None
    max_loss   = round(float(np.min(los_gains)), 2)       if los_gains else None

    # Expectancy = WinRate × AvgWin + (1-WinRate) × AvgNonWin
    non_win_gains = [t["gain_pct"] for t in trades if t["outcome"] != "WIN"]
    avg_non_win   = round(float(np.mean(non_win_gains)), 2) if non_win_gains else None
    expectancy = None
    if win_rate is not None and avg_win is not None and avg_non_win is not None:
        wr = win_rate / 100
        expectancy = round(wr * avg_win + (1 - wr) * avg_non_win, 2)

    # Compound return simulation (chronological order)
    equity = 100.0
    for t in sorted(trades, key=lambda x: x["entry_date"]):
        equity *= (1 + t["gain_pct"] / 100)
    total_return = round(equity - 100, 2)

    return {
        "ticker":   ticker,
        "status":   "OK",
        "trades":   trades,
        "summary": {
            "total_trades":     len(trades),
            "wins":             len(wins),
            "losses":           len(losses),
            "timeouts":         len(timeouts),
            "win_rate_pct":     win_rate,
            "avg_gain_pct":     avg_gain,
            "avg_win_pct":      avg_win,
            "avg_loss_pct":     avg_loss,
            "max_win_pct":      max_win,
            "max_loss_pct":     max_loss,
            "expectancy_pct":   expectancy,
            "total_return_pct": total_return,
        },
    }


# ─── Backtest Request Model ───────────────────────────────────
class Ema9BacktestRequest(BaseModel):
    tickers:         List[str]
    target_pct:      float = 3.0
    max_hold_days:   int   = 15
    require_uptrend: bool  = True
    fetch_fv:        bool  = False   # If True, enrich each ticker trade with Fair Value data


# ─── Backtest Route ───────────────────────────────────────────
@router.post("/api/ema9/backtest")
async def ema9_backtest(req: Ema9BacktestRequest):
    """
    Run walk-forward 9EMA breakout backtest across a list of tickers.
    Returns per-ticker summaries, all individual trades, and an
    aggregate report sorted by win-rate then total return.
    """
    tickers = [
        t.strip().upper().replace(".NS", "").replace(".BO", "")
        for t in req.tickers if t.strip()
    ][:300]  # Cap at 300 for reasonable runtime

    if not tickers:
        raise HTTPException(400, "No valid tickers provided.")

    logger.info(f"[Backtest] Starting: {len(tickers)} tickers | "
                f"target={req.target_pct}% | hold={req.max_hold_days}d | "
                f"uptrend={req.require_uptrend}")

    # Download all tickers (reuse existing batch downloader)
    ticker_dfs = await _batch_download_async(tickers, "1d", 500)

    # Run backtest on each ticker
    results      = []
    all_trades   = []
    all_skipped  = []
    failed       = []
    no_trades    = []

    for ticker in tickers:
        if ticker not in ticker_dfs:
            failed.append({"ticker": ticker, "error": "No price data"})
            continue
        try:
            bt = _backtest_ticker(
                ticker,
                ticker_dfs[ticker],
                target_pct      = req.target_pct,
                max_hold_days   = req.max_hold_days,
                require_uptrend = req.require_uptrend,
            )
            if bt["status"] == "NO_DATA":
                failed.append({"ticker": ticker, "error": bt.get("error", "NO_DATA")})
                continue
            if not bt["trades"]:
                no_trades.append(ticker)
                continue

            results.append(bt)
            all_trades.extend(bt["trades"])
            all_skipped.extend(bt.get("skipped_signals", []))
        except Exception as exc:
            logger.error(f"[Backtest] {ticker}: {exc}")
            failed.append({"ticker": ticker, "error": str(exc)[:120]})

    # ── Fair Value enrichment for backtest (optional, slow) ──────────────────
    # Fetch FV once per ticker, then annotate all trades for that ticker.
    # "Undervalued at entry" = entry_price < composite_fair_price (fundamental FV
    # is not time-varying at the daily scale so we use current FV as proxy).
    #
    # PERFORMANCE: FV fetches run CONCURRENTLY (up to 8 at a time) instead of
    # sequentially. Turns a 5-min sequential loop into ~30s for 200 tickers.
    fv_by_ticker: Dict[str, Dict] = {}
    if req.fetch_fv and _FV_AVAILABLE:
        logger.info(f"[Backtest-FV] Starting FV enrichment for {len(results)} tickers with trades")

        # Build list of (bt_result, ticker, cur_price)
        fv_tasks = []
        for bt_result in results:
            ticker = bt_result["ticker"]
            df_ref = ticker_dfs.get(ticker)
            cur_price = float(df_ref["Close"].iloc[-1]) if df_ref is not None else None
            fv_tasks.append((bt_result, ticker, cur_price))

        # Keep LOW: sma_router calls screener.in which has strict limits (~30 req/min).
        # User explicitly requested MORE conservative throttling to avoid rate-limit
        # violations. Reduced from 4 → 2 concurrent calls.
        # At concurrency=2 with ~2.7s per request, we do ~44 req/min — well under
        # the 30 req/min screener.in guideline (the cap is per-IP, not per-endpoint,
        # so staying at ~half the limit gives headroom for sma_router's internal
        # sub-requests). Slower but safe.
        FV_CONCURRENCY = 2
        _fv_semaphore = asyncio.Semaphore(FV_CONCURRENCY)

        async def _fetch_one_fv_bt(bt_result, ticker, cur_price):
            with _cache_lock:
                fv_entry = _fv_cache.get(ticker)
                was_cached = (fv_entry is not None and
                              time.time() - fv_entry["ts"] <= FV_CACHE_TTL_SEC)
                if not was_cached:
                    fv_fail_entry = _fv_fail_cache.get(ticker)
                    was_cached = (fv_fail_entry is not None and
                                  time.time() - fv_fail_entry["ts"] <= FV_FAILURE_CACHE_TTL_SEC)

            if was_cached:
                fv = await asyncio.to_thread(_enrich_fair_value, ticker, cur_price)
            else:
                async with _fv_semaphore:
                    fv = await asyncio.to_thread(_enrich_fair_value, ticker, cur_price)

            for k in _SAFE_FV_KEYS:
                bt_result[k] = fv.get(k)
            fv_price = fv.get("composite_fair_price")
            for trade in bt_result["trades"]:
                trade["composite_fair_price"] = fv_price
                if fv_price and fv_price > 0 and trade["entry_price"] > 0:
                    trade["is_undervalued"]       = trade["entry_price"] < fv_price
                    trade["entry_vs_fv_pct"]      = round((fv_price - trade["entry_price"]) / fv_price * 100, 2)
                else:
                    trade["is_undervalued"]  = None
                    trade["entry_vs_fv_pct"] = None
            return (ticker, fv, was_cached)

        t_fv_start = time.time()
        fv_results = await asyncio.gather(*[_fetch_one_fv_bt(bt, t, p) for bt, t, p in fv_tasks])
        t_fv_end = time.time()

        fv_hits = sum(1 for _, _, c in fv_results if c)
        fv_misses = sum(1 for _, _, c in fv_results if not c)
        for ticker, fv, _ in fv_results:
            fv_by_ticker[ticker] = fv

        # Re-sync all_trades list (it was built before FV enrichment)
        all_trades = []
        for bt_result in results:
            all_trades.extend(bt_result["trades"])
        logger.info(f"[Backtest-FV] FV enrichment complete for {len(fv_by_ticker)} tickers | "
                    f"cache: {fv_hits} hits, {fv_misses} misses | "
                    f"concurrency: {FV_CONCURRENCY} | "
                    f"time: {t_fv_end - t_fv_start:.1f}s "
                    f"(sequential would take ~{len(fv_tasks) * FV_INTER_DELAY:.0f}s)")
    else:
        # No FV requested — annotate trades with None so frontend knows
        for trade in all_trades:
            trade.setdefault("composite_fair_price", None)
            trade.setdefault("is_undervalued", None)
            trade.setdefault("entry_vs_fv_pct", None)

    # Sort by win_rate desc, then total_return desc
    results.sort(
        key=lambda r: (
            -(r["summary"]["win_rate_pct"]    or 0),
            -(r["summary"]["total_return_pct"] or 0),
        )
    )

    # Aggregate across all tickers
    all_wins     = [t for t in all_trades if t["is_win"]]
    all_losses   = [t for t in all_trades if not t["is_win"]]
    all_timeouts = [t for t in all_trades if t["outcome"] == "TIMEOUT"]

    agg_win_rate  = None
    agg_avg_gain  = None
    agg_avg_win   = None
    agg_avg_loss  = None
    agg_expectancy = None

    if all_trades:
        agg_win_rate  = round(len(all_wins) / len(all_trades) * 100, 1)
        gains         = [t["gain_pct"] for t in all_trades]
        agg_avg_gain  = round(float(np.mean(gains)), 2)
        if all_wins:
            agg_avg_win = round(float(np.mean([t["gain_pct"] for t in all_wins])), 2)
        if all_losses:
            agg_avg_loss = round(float(np.mean([t["gain_pct"] for t in all_losses])), 2)
        if agg_avg_win is not None and agg_avg_loss is not None:
            wr = agg_win_rate / 100
            agg_expectancy = round(wr * agg_avg_win + (1 - wr) * agg_avg_loss, 2)

    # ── Undervalued-only aggregate (for checkbox filter on frontend) ──────────
    uv_trades   = [t for t in all_trades if t.get("is_undervalued") is True]
    uv_wins     = [t for t in uv_trades if t["is_win"]]
    uv_losses   = [t for t in uv_trades if not t["is_win"]]
    uv_win_rate = round(len(uv_wins) / len(uv_trades) * 100, 1) if uv_trades else None
    uv_avg_win  = round(float(np.mean([t["gain_pct"] for t in uv_wins])),   2) if uv_wins   else None
    uv_avg_loss = round(float(np.mean([t["gain_pct"] for t in uv_losses])), 2) if uv_losses else None
    uv_expectancy = None
    if uv_win_rate is not None and uv_avg_win is not None and uv_avg_loss is not None:
        wr = uv_win_rate / 100
        uv_expectancy = round(wr * uv_avg_win + (1 - wr) * uv_avg_loss, 2)

    logger.info(
        f"[Backtest] Done: {len(results)} tickers with trades | "
        f"{len(all_trades)} total trades | WR={agg_win_rate}% | "
        f"UV trades={len(uv_trades)} | UV WR={uv_win_rate}%"
    )

    return _json_safe({
        "results":       results,
        "all_trades":    all_trades,
        "all_skipped":   all_skipped,
        "failed":        failed,
        "no_trades":     no_trades,
        "fv_enabled":    req.fetch_fv and _FV_AVAILABLE,
        "aggregate": {
            "tickers_screened":      len(tickers),
            "tickers_with_trades":   len(results),
            "tickers_no_trades":     len(no_trades),
            "tickers_failed":        len(failed),
            "total_trades":          len(all_trades),
            "total_wins":            len(all_wins),
            "total_losses":          len(all_losses),
            "total_timeouts":        len(all_timeouts),
            "win_rate_pct":          agg_win_rate,
            "avg_gain_pct":          agg_avg_gain,
            "avg_win_pct":           agg_avg_win,
            "avg_loss_pct":          agg_avg_loss,
            "expectancy_pct":        agg_expectancy,
        },
        "undervalued_aggregate": {
            "total_trades":    len(uv_trades),
            "total_wins":      len(uv_wins),
            "total_losses":    len(uv_losses),
            "win_rate_pct":    uv_win_rate,
            "avg_win_pct":     uv_avg_win,
            "avg_loss_pct":    uv_avg_loss,
            "expectancy_pct":  uv_expectancy,
        },
        "config": {
            "target_pct":      req.target_pct,
            "max_hold_days":   req.max_hold_days,
            "require_uptrend": req.require_uptrend,
            "fetch_fv":        req.fetch_fv,
        },
    })

    # Persist cache to disk (survives restarts) — throttled to once per 30s
    _save_cache_to_disk()


# ─────────────────────────────────────────────────────────────
#  DURATION BACKTEST — Date-Range Bounded Strategy Performance
# ─────────────────────────────────────────────────────────────

class Ema9DurationBacktestRequest(BaseModel):
    tickers:         List[str]
    target_pct:      float = 3.0
    max_hold_days:   int   = 15
    require_uptrend: bool  = True
    start_date:      Optional[str] = None   # YYYY-MM-DD (inclusive)
    end_date:        Optional[str] = None   # YYYY-MM-DD (inclusive)
    fetch_fv:        bool  = False   # If True, enrich each ticker trade with Fair Value data
    prime_only:      bool  = False   # If True, only count trades where entry_price < FV (undervalued = "prime")
    resistance_lookback_days: int = 20   # N-day high before breakout = overhead resistance
    filter_near_resistance:  bool = False  # If True, exclude trades within 3% of resistance
    resistance_threshold_pct: float = 3.0  # "Near resistance" = within this % of the N-day high
    # ── Pullback-from-previous-high (entry-quality filter) ────────────────────
    # Looks back `pullback_lookback_days` bars before the breakout candle,
    # finds the highest High in that window, and computes how far the entry
    # price has pulled back from that prior high:
    #   pullback_dist_pct = (prior_high - entry_price) / prior_high * 100
    # A trade is flagged `is_pullback=True` when pullback_dist_pct >=
    # `pullback_threshold_pct` (default 10%). When `filter_pullback=True`,
    # trades that don't meet the threshold are excluded from the primary
    # trade set + aggregate stats.
    pullback_lookback_days:  int   = 60    # window to find the prior high (default 60d ≈ 3 months)
    pullback_threshold_pct:  float = 10.0  # min % pullback required for the trade to qualify
    filter_pullback:         bool  = False # if True, drop trades that don't meet the pullback threshold
    # ── Volatility-Adjusted Target (per-trade target_pct from ATR) ────────────
    # When vol_target_mode="atr", each trade's target_price is computed from
    # the stock's ATR at entry time instead of a flat target_pct:
    #   effective_target_pct = clamp(
    #     vol_target_multiplier × ATR_at_entry / entry_price × 100,
    #     vol_target_floor_pct, vol_target_cap_pct
    #   )
    # This adapts the target to each stock's volatility — high-vol stocks get
    # larger targets (don't exit too early), low-vol stocks get smaller targets
    # (don't wait forever for an unreachable move).
    # vol_target_mode="fixed" (default) → use target_pct as-is (legacy behavior).
    vol_target_mode:        str   = "fixed"   # "fixed" | "atr"
    vol_target_atr_period:  int   = 14        # ATR lookback (standard 14)
    vol_target_multiplier:  float = 1.5       # target = multiplier × ATR/price × 100
    vol_target_floor_pct:   float = 2.0       # min target % (prevents 0.5% targets on tiny-ATR stocks)
    vol_target_cap_pct:     float = 8.0       # max target % (prevents 25% targets on huge-ATR stocks)


@router.post("/api/ema9/duration-backtest")
async def ema9_duration_backtest(req: Ema9DurationBacktestRequest):
    """
    Duration-bounded backtest for the 9EMA breakout strategy.

    Only trades whose ENTRY date falls within [start_date, end_date] are
    counted. Trades exit on WIN (target hit) OR at max_hold_days (classified
    as LOSS or TIMEOUT based on close vs entry). No stop-loss.

    Returns:
      - results:           per-ticker summaries + trades (sorted by win rate)
      - all_trades:        every trade across all tickers (chronological)
      - failed:            tickers that failed to download / had no data
      - no_trades:         tickers with valid data but no signals in range
      - aggregate:         consolidated multi-ticker stats
      - equity_curve:      chronological equity points (for Plotly chart)
      - monthly_breakdown: per-month W/L/T counts and avg gain
      - config:            echo of request parameters
    """
    tickers = [
        t.strip().upper().replace(".NS", "").replace(".BO", "")
        for t in req.tickers if t.strip()
    ][:300]

    if not tickers:
        raise HTTPException(400, "No valid tickers provided.")

    # Default end_date to today if not provided
    end_date = req.end_date or str(pd.Timestamp.now().date())

    logger.info(f"[Duration-BT] Starting: {len(tickers)} tickers | "
                f"target={req.target_pct}% | hold={req.max_hold_days}d | "
                f"uptrend={req.require_uptrend} | "
                f"start={req.start_date} | end={end_date} | "
                f"fetch_fv={req.fetch_fv} | prime_only={req.prime_only}")

    # If user requested prime_only but not fetch_fv, auto-enable FV (required
    # to determine whether each trade was undervalued at entry)
    fetch_fv_effective = req.fetch_fv or req.prime_only
    if req.prime_only and not req.fetch_fv:
        logger.info("[Duration-BT] prime_only=True → auto-enabling fetch_fv")

    # Download all tickers (full 5y history — needed for SMA50/EMA9 warmup
    # before the user's start_date)
    ticker_dfs = await _batch_download_async(tickers, "1d", 500)

    results    = []
    all_trades = []
    failed     = []
    no_trades  = []

    for ticker in tickers:
        if ticker not in ticker_dfs:
            failed.append({"ticker": ticker, "error": "No price data"})
            continue
        try:
            bt = _duration_backtest_ticker(
                ticker,
                ticker_dfs[ticker],
                target_pct      = req.target_pct,
                max_hold_days   = req.max_hold_days,
                require_uptrend = req.require_uptrend,
                start_date      = req.start_date,
                end_date        = end_date,
                resistance_lookback_days  = req.resistance_lookback_days,
                resistance_threshold_pct  = req.resistance_threshold_pct,
                pullback_lookback_days    = req.pullback_lookback_days,
                pullback_threshold_pct    = req.pullback_threshold_pct,
                vol_target_mode           = req.vol_target_mode,
                vol_target_atr_period     = req.vol_target_atr_period,
                vol_target_multiplier     = req.vol_target_multiplier,
                vol_target_floor_pct      = req.vol_target_floor_pct,
                vol_target_cap_pct        = req.vol_target_cap_pct,
            )
            if bt["status"] == "NO_DATA":
                failed.append({"ticker": ticker, "error": bt.get("error", "NO_DATA")})
                continue
            if not bt["trades"]:
                no_trades.append(ticker)
                results.append(bt)  # keep with empty trades for UI completeness
                continue

            results.append(bt)
            all_trades.extend(bt["trades"])
        except Exception as exc:
            logger.error(f"[Duration-BT] {ticker}: {exc}")
            failed.append({"ticker": ticker, "error": str(exc)[:120]})

    # ── Fair Value enrichment (optional, slow — one FV fetch per ticker) ─────
    # Annotates each trade with composite_fair_price, is_undervalued,
    # entry_vs_fv_pct. Same pattern as the existing /api/ema9/backtest.
    #
    # PERFORMANCE: FV fetches are run CONCURRENTLY (up to FV_CONCURRENCY at a
    # time) instead of sequentially. This turns a 5-minute sequential loop
    # (204 tickers × 1.5s sleep) into a ~30-second concurrent batch.
    # Cache hits are instant and don't count against the concurrency limit.
    fv_by_ticker: Dict[str, Dict] = {}
    if fetch_fv_effective and _FV_AVAILABLE:
        logger.info(f"[Duration-BT-FV] Starting FV enrichment for {len(results)} tickers with trades")

        # Build list of (index, ticker, cur_price) for tickers that have trades
        fv_tasks = []
        for bt_result in results:
            ticker = bt_result["ticker"]
            if not bt_result.get("trades"):
                continue
            df_ref = ticker_dfs.get(ticker)
            cur_price = float(df_ref["Close"].iloc[-1]) if df_ref is not None else None
            fv_tasks.append((bt_result, ticker, cur_price))

        # Semaphore to limit concurrent sma_router calls (avoids rate limiting)
        # Keep LOW: sma_router calls screener.in which has strict limits (~30 req/min).
        # User requested MORE conservative throttling — reduced from 4 → 2 concurrent
        # calls. Slower but safe; ~44 req/min vs the ~30 req/min guideline.
        FV_CONCURRENCY = 2

        async def _fetch_one_fv(bt_result, ticker, cur_price):
            """Fetch FV for one ticker, annotate its trades, return (ticker, fv, was_cached)."""
            # Peek at cache (success OR failure) — if hit, skip sma_router entirely
            with _cache_lock:
                fv_entry = _fv_cache.get(ticker)
                was_cached = (fv_entry is not None and
                              time.time() - fv_entry["ts"] <= FV_CACHE_TTL_SEC)
                if not was_cached:
                    fv_fail_entry = _fv_fail_cache.get(ticker)
                    was_cached = (fv_fail_entry is not None and
                                  time.time() - fv_fail_entry["ts"] <= FV_FAILURE_CACHE_TTL_SEC)

            # Cached fetches don't need the semaphore — run instantly
            if was_cached:
                fv = await asyncio.to_thread(_enrich_fair_value, ticker, cur_price)
            else:
                # Real sma_router call — acquire semaphore to limit concurrency
                async with _fv_semaphore:
                    fv = await asyncio.to_thread(_enrich_fair_value, ticker, cur_price)

            # Annotate the ticker-level result with FV summary
            for k in _SAFE_FV_KEYS:
                bt_result[k] = fv.get(k)
            # Annotate every trade for this ticker
            fv_price = fv.get("composite_fair_price")
            for trade in bt_result["trades"]:
                trade["composite_fair_price"] = fv_price
                if fv_price and fv_price > 0 and trade["entry_price"] > 0:
                    trade["is_undervalued"]  = trade["entry_price"] < fv_price
                    trade["entry_vs_fv_pct"] = round((fv_price - trade["entry_price"]) / fv_price * 100, 2)
                else:
                    trade["is_undervalued"]  = None
                    trade["entry_vs_fv_pct"] = None

            return (ticker, fv, was_cached)

        _fv_semaphore = asyncio.Semaphore(FV_CONCURRENCY)

        # Run all FV fetches concurrently (cache hits are instant, misses are
        # limited to FV_CONCURRENCY at a time)
        t_fv_start = time.time()
        fv_results = await asyncio.gather(*[_fetch_one_fv(bt, t, p) for bt, t, p in fv_tasks])
        t_fv_end = time.time()

        # Collect results and stats
        fv_cache_hits = sum(1 for _, _, was_cached in fv_results if was_cached)
        fv_cache_misses = sum(1 for _, _, was_cached in fv_results if not was_cached)
        for ticker, fv, _ in fv_results:
            fv_by_ticker[ticker] = fv

        # Re-sync all_trades list (annotations happened on bt_result["trades"])
        all_trades = []
        for bt_result in results:
            all_trades.extend(bt_result["trades"])
        logger.info(f"[Duration-BT-FV] FV enrichment complete for {len(fv_by_ticker)} tickers | "
                    f"cache: {fv_cache_hits} hits, {fv_cache_misses} misses | "
                    f"concurrency: {FV_CONCURRENCY} | "
                    f"time: {t_fv_end - t_fv_start:.1f}s "
                    f"(sequential would take ~{len(fv_tasks) * FV_INTER_DELAY:.0f}s)")
    else:
        # No FV requested — annotate trades with None so frontend knows
        for trade in all_trades:
            trade.setdefault("composite_fair_price", None)
            trade.setdefault("is_undervalued", None)
            trade.setdefault("entry_vs_fv_pct", None)

    # ── Prime-only filter: keep only trades where entry_price < FV ──────────
    # We track both the FULL set (all_trades) and the PRIME set (undervalued
    # only). The "prime_only" request flag controls which set drives the
    # primary aggregate / equity_curve / monthly_breakdown. The other set is
    # always returned for side-by-side comparison in the UI.
    prime_trades = [t for t in all_trades if t.get("is_undervalued") is True]
    overvalued_trades = [t for t in all_trades if t.get("is_undervalued") is False]

    if req.prime_only:
        # Replace per-ticker results' trades with prime-only subset, and
        # recompute their summaries so the per-ticker table reflects prime-only
        for bt_result in results:
            bt_result["trades_all_count"] = len(bt_result["trades"])
            bt_result["trades"] = [t for t in bt_result["trades"] if t.get("is_undervalued") is True]
            bt_result["summary"] = _recompute_summary(bt_result["trades"], req.target_pct)
        # Drive downstream with prime-only trades
        primary_trades = prime_trades
        logger.info(f"[Duration-BT] prime_only=True → using {len(prime_trades)}/{len(all_trades)} prime trades")
    else:
        primary_trades = all_trades

    # ── Resistance filter: exclude trades near overhead resistance ──────────
    # "Near resistance" = entry price within resistance_threshold_pct of the
    # N-day high (overhead supply). These trades are more likely to fail because
    # price hits the resistance ceiling and reverses.
    near_resistance_trades = [t for t in primary_trades if t.get("is_near_resistance") is True]
    away_resistance_trades = [t for t in primary_trades if t.get("is_near_resistance") is False]

    if req.filter_near_resistance:
        # Filter out near-resistance trades from per-ticker results
        for bt_result in results:
            bt_result["trades_before_res_filter"] = len(bt_result["trades"])
            bt_result["trades"] = [t for t in bt_result["trades"] if t.get("is_near_resistance") is False]
            bt_result["summary"] = _recompute_summary(bt_result["trades"], req.target_pct)
        primary_trades = away_resistance_trades
        # CRITICAL: re-sync all_trades so the trades table + equity curve + monthly
        # breakdown all reflect the filtered set. Without this, the frontend trades
        # table would still show near-resistance trades that were supposed to be filtered.
        all_trades = []
        for bt_result in results:
            all_trades.extend(bt_result["trades"])
        logger.info(f"[Duration-BT] filter_near_resistance=True → "
                    f"excluded {len(near_resistance_trades)} near-resistance trades, "
                    f"keeping {len(away_resistance_trades)}/{len(primary_trades) + len(near_resistance_trades)}")

    # ── Pullback filter: exclude trades that don't meet the pullback threshold ─
    # "Pullback" = entry price has fallen >= pullback_threshold_pct from the
    # N-day high in the pullback_lookback_days window before the breakout.
    # These are fresh breakouts AFTER a healthy correction — they tend to
    # outperform breakouts on already-extended stocks (no pullback).
    # Mirror the resistance filter pattern: partition into pullback vs
    # non_pullback sets, drive primary_trades from the pullback set when the
    # filter is enabled, and emit a side-by-side aggregate for A/B comparison.
    pullback_trades    = [t for t in primary_trades if t.get("is_pullback") is True]
    non_pullback_trades = [t for t in primary_trades if t.get("is_pullback") is False]

    if req.filter_pullback:
        for bt_result in results:
            bt_result["trades_before_pullback_filter"] = len(bt_result["trades"])
            bt_result["trades"] = [t for t in bt_result["trades"] if t.get("is_pullback") is True]
            bt_result["summary"] = _recompute_summary(bt_result["trades"], req.target_pct)
        primary_trades = pullback_trades
        # Re-sync all_trades so the trades table + equity curve + monthly
        # breakdown all reflect the pullback-filtered set.
        all_trades = []
        for bt_result in results:
            all_trades.extend(bt_result["trades"])
        logger.info(f"[Duration-BT] filter_pullback=True → "
                    f"kept {len(pullback_trades)} pullback trades, "
                    f"excluded {len(non_pullback_trades)} non-pullback trades "
                    f"(threshold={req.pullback_threshold_pct}%, lookback={req.pullback_lookback_days}d)")

    # Sort by win_rate desc, then total_return desc
    results.sort(
        key=lambda r: (
            -(r["summary"]["win_rate_pct"]    or 0),
            -(r["summary"]["total_return_pct"] or 0),
        )
    )

    # ── Aggregate across all tickers (primary set) ───────────
    all_wins     = [t for t in primary_trades if t["outcome"] == "WIN"]
    all_losses   = [t for t in primary_trades if t["outcome"] == "LOSS"]
    all_timeouts = [t for t in primary_trades if t["outcome"] == "TIMEOUT"]

    agg_win_rate    = None
    agg_avg_gain    = None
    agg_avg_win     = None
    agg_avg_loss    = None
    agg_avg_timeout = None
    agg_expectancy  = None
    agg_total_return = None

    if all_trades:
        agg_win_rate  = round(len(all_wins) / len(primary_trades) * 100, 1) if primary_trades else None
        gains         = [t["gain_pct"] for t in primary_trades]
        agg_avg_gain  = round(float(np.mean(gains)), 2) if gains else None
        if all_wins:
            agg_avg_win = round(float(np.mean([t["gain_pct"] for t in all_wins])), 2)
        if all_losses:
            agg_avg_loss = round(float(np.mean([t["gain_pct"] for t in all_losses])), 2)
        if all_timeouts:
            agg_avg_timeout = round(float(np.mean([t["gain_pct"] for t in all_timeouts])), 2)

        # Expectancy uses avg of non-win trades (losses + timeouts combined)
        non_win_gains = [t["gain_pct"] for t in primary_trades if t["outcome"] != "WIN"]
        avg_non_win   = round(float(np.mean(non_win_gains)), 2) if non_win_gains else None
        if agg_avg_win is not None and avg_non_win is not None and agg_win_rate is not None:
            wr = agg_win_rate / 100
            agg_expectancy = round(wr * agg_avg_win + (1 - wr) * avg_non_win, 2)

        # Compound total return across primary trades (chronological)
        if primary_trades:
            equity = 100.0
            for t in sorted(primary_trades, key=lambda x: (x["entry_date"], x["ticker"])):
                equity *= (1 + t["gain_pct"] / 100)
            agg_total_return = round(equity - 100, 2)

    # ── Equity curve points (chronological, multi-ticker) ────
    equity_curve = []
    if primary_trades:
        sorted_trades = sorted(primary_trades, key=lambda x: (x["entry_date"], x["ticker"]))
        eq = 100.0
        for t in sorted_trades:
            eq *= (1 + t["gain_pct"] / 100)
            equity_curve.append({
                "date":     t["entry_date"],
                "ticker":   t["ticker"],
                "gain_pct": t["gain_pct"],
                "outcome":  t["outcome"],
                "equity":   round(eq, 2),
                "is_prime": t.get("is_undervalued") is True,
            })

    # ── Monthly breakdown (primary set) ──────────────────────
    monthly: Dict[str, Dict] = {}
    for t in primary_trades:
        try:
            d = pd.to_datetime(t["entry_date"])
            mkey = f"{d.year}-{d.month:02d}"
        except Exception:
            continue
        if mkey not in monthly:
            monthly[mkey] = {
                "month": mkey, "trades": 0, "wins": 0,
                "losses": 0, "timeouts": 0, "gain_sum": 0.0,
                "prime_trades": 0, "prime_gain_sum": 0.0,
            }
        monthly[mkey]["trades"]   += 1
        monthly[mkey]["gain_sum"] += t["gain_pct"]
        if t.get("is_undervalued") is True:
            monthly[mkey]["prime_trades"]     += 1
            monthly[mkey]["prime_gain_sum"]   += t["gain_pct"]
        if t["outcome"] == "WIN":
            monthly[mkey]["wins"]     += 1
        elif t["outcome"] == "LOSS":
            monthly[mkey]["losses"]   += 1
        else:
            monthly[mkey]["timeouts"] += 1

    monthly_list = []
    for mkey in sorted(monthly.keys()):
        m = monthly[mkey]
        m["avg_gain_pct"] = round(m["gain_sum"] / m["trades"], 2) if m["trades"] else 0
        m["win_rate_pct"] = round(m["wins"] / m["trades"] * 100, 1) if m["trades"] else 0
        m["prime_trades"] = m.get("prime_trades", 0)
        m["prime_avg_gain_pct"] = round(m["prime_gain_sum"] / m["prime_trades"], 2) if m.get("prime_trades") else None
        del m["gain_sum"]
        del m["prime_gain_sum"]
        monthly_list.append(m)

    # ── Undervalued (Prime) aggregate — always computed when FV is fetched ─
    uv_agg = _compute_subset_aggregate(prime_trades)
    ov_agg = _compute_subset_aggregate(overvalued_trades)

    # ── Resistance aggregate — near-resistance vs away-from-resistance ──────
    # Shows whether avoiding near-resistance trades improves performance.
    near_res_agg = _compute_subset_aggregate(near_resistance_trades)
    away_res_agg = _compute_subset_aggregate(away_resistance_trades)

    # ── Pullback aggregate — pullback vs non-pullback breakouts ─────────────
    # Shows whether breakouts after a >= threshold % pullback from prior high
    # outperform breakouts on already-extended stocks. Always computed (the
    # pullback fields are computed on every trade) — the user toggles
    # filter_pullback to actually drop non-pullback trades from the primary set.
    pb_agg     = _compute_subset_aggregate(pullback_trades)
    non_pb_agg = _compute_subset_aggregate(non_pullback_trades)

    # ── Volatility-Adjusted Target aggregate — ATR-target vs fixed-target ────
    # When vol_target_mode="atr", the effective_target_pct varies per trade
    # (high-vol stocks get larger targets, low-vol stocks get smaller targets).
    # This A/B aggregate shows whether ATR-based targets beat the flat
    # target_pct baseline. Both subsets always exist when mode="atr" — we
    # split trades into "high-vol" (atr_pct_at_entry > median) vs "low-vol"
    # (atr_pct_at_entry <= median) so you can see if the strategy works
    # better on volatile or stable names.
    #
    # Note: when mode="fixed", all trades use the same target_pct and the
    # split still works — it just shows the strategy's natural performance
    # on high-vol vs low-vol stocks with a flat target.
    atr_pct_values = [t.get("atr_pct_at_entry") for t in primary_trades
                      if t.get("atr_pct_at_entry") is not None]
    if atr_pct_values:
        median_atr_pct = float(np.median(atr_pct_values))
    else:
        median_atr_pct = None
    high_vol_trades = [t for t in primary_trades
                       if t.get("atr_pct_at_entry") is not None
                       and median_atr_pct is not None
                       and t.get("atr_pct_at_entry") > median_atr_pct]
    low_vol_trades  = [t for t in primary_trades
                       if t.get("atr_pct_at_entry") is not None
                       and median_atr_pct is not None
                       and t.get("atr_pct_at_entry") <= median_atr_pct]
    high_vol_agg = _compute_subset_aggregate(high_vol_trades)
    low_vol_agg  = _compute_subset_aggregate(low_vol_trades)

    logger.info(
        f"[Duration-BT] Done: {len(primary_trades)} primary trades | "
        f"W={len(all_wins)} L={len(all_losses)} T={len(all_timeouts)} | "
        f"WR={agg_win_rate}% | "
        f"Prime={len(prime_trades)}/{len(all_trades)} "
        f"(UV WR={uv_agg.get('win_rate_pct')}%) | "
        f"NearRes={len(near_resistance_trades)} (WR={near_res_agg.get('win_rate_pct')}%) "
        f"AwayRes={len(away_resistance_trades)} (WR={away_res_agg.get('win_rate_pct')}%) | "
        f"Pullback={len(pullback_trades)} (WR={pb_agg.get('win_rate_pct')}%) "
        f"NonPullback={len(non_pullback_trades)} (WR={non_pb_agg.get('win_rate_pct')}%) | "
        f"target_mode={req.vol_target_mode} "
        f"HighVol={len(high_vol_trades)} (WR={high_vol_agg.get('win_rate_pct')}%) "
        f"LowVol={len(low_vol_trades)} (WR={low_vol_agg.get('win_rate_pct')}%)"
    )

    return _json_safe({
        "results":            results,
        "all_trades":         all_trades,   # full set (prime + overvalued), with FV annotations
        "failed":             failed,
        "no_trades":          no_trades,
        "aggregate": {
            "tickers_screened":      len(tickers),
            "tickers_with_trades":   len([r for r in results if r["trades"]]),
            "tickers_no_trades":     len(no_trades),
            "tickers_failed":        len(failed),
            "total_trades":          len(primary_trades),
            "total_wins":            len(all_wins),
            "total_losses":          len(all_losses),
            "total_timeouts":        len(all_timeouts),
            "win_rate_pct":          agg_win_rate,
            "avg_gain_pct":          agg_avg_gain,
            "avg_win_pct":           agg_avg_win,
            "avg_loss_pct":          agg_avg_loss,
            "avg_timeout_pct":       agg_avg_timeout,
            "expectancy_pct":        agg_expectancy,
            "total_return_pct":      agg_total_return,
        },
        "undervalued_aggregate": uv_agg,    # stats for PRIME (undervalued) trades only
        "overvalued_aggregate":  ov_agg,    # stats for OVERVALUED trades only (for comparison)
        "resistance_aggregate": {
            "near_resistance":  near_res_agg,   # stats for trades within threshold% of N-day high
            "away_resistance":  away_res_agg,   # stats for trades away from resistance
            "near_resistance_count":  len(near_resistance_trades),
            "away_resistance_count":  len(away_resistance_trades),
            "no_resistance_count":    len([t for t in primary_trades if t.get("resistance_price") is None]),
        },
        "pullback_aggregate": {
            "pullback":         pb_agg,         # stats for trades that pulled back >= threshold % from prior high
            "non_pullback":     non_pb_agg,     # stats for trades on already-extended stocks (no pullback)
            "pullback_count":        len(pullback_trades),
            "non_pullback_count":    len(non_pullback_trades),
            "pullback_threshold_pct": req.pullback_threshold_pct,
            "pullback_lookback_days": req.pullback_lookback_days,
        },
        "vol_target_aggregate": {
            # A/B split by volatility (median ATR% at entry). Shows whether
            # the strategy performs better on high-vol or low-vol stocks,
            # and (when mode="atr") whether the ATR-based target improves
            # outcomes on each subset vs the flat target_pct baseline.
            "high_vol":         high_vol_agg,   # stats for trades with atr_pct > median
            "low_vol":          low_vol_agg,    # stats for trades with atr_pct <= median
            "high_vol_count":   len(high_vol_trades),
            "low_vol_count":    len(low_vol_trades),
            "median_atr_pct":   round(median_atr_pct, 2) if median_atr_pct is not None else None,
            "mode":             req.vol_target_mode,           # "fixed" | "atr"
            "multiplier":       req.vol_target_multiplier,
            "atr_period":       req.vol_target_atr_period,
            "floor_pct":        req.vol_target_floor_pct,
            "cap_pct":          req.vol_target_cap_pct,
        },
        "fv_summary": {
            "fv_enabled":       fetch_fv_effective and _FV_AVAILABLE,
            "prime_only_mode":  req.prime_only,
            "total_trades_all": len(all_trades),
            "prime_trades":     len(prime_trades),
            "overvalued_trades": len(overvalued_trades),
            "no_fv_trades":     len([t for t in all_trades if t.get("is_undervalued") is None]),
        },
        "equity_curve":       equity_curve,
        "monthly_breakdown":  monthly_list,
        "config": {
            "target_pct":      req.target_pct,
            "max_hold_days":   req.max_hold_days,
            "require_uptrend": req.require_uptrend,
            "start_date":      req.start_date,
            "end_date":        end_date,
            "fetch_fv":        fetch_fv_effective,
            "prime_only":      req.prime_only,
            "resistance_lookback_days":  req.resistance_lookback_days,
            "filter_near_resistance":    req.filter_near_resistance,
            "resistance_threshold_pct":  req.resistance_threshold_pct,
            "pullback_lookback_days":    req.pullback_lookback_days,
            "pullback_threshold_pct":    req.pullback_threshold_pct,
            "filter_pullback":           req.filter_pullback,
            "vol_target_mode":           req.vol_target_mode,
            "vol_target_atr_period":     req.vol_target_atr_period,
            "vol_target_multiplier":     req.vol_target_multiplier,
            "vol_target_floor_pct":      req.vol_target_floor_pct,
            "vol_target_cap_pct":        req.vol_target_cap_pct,
        },
    })

    # Persist cache to disk (survives restarts) — throttled to once per 30s
    _save_cache_to_disk()


def _compute_subset_aggregate(trades: List[Dict]) -> Dict:
    """Compute W/L/T, win_rate, avg_gain, expectancy, total_return for a
    subset of trades (used for undervalued/overvalued comparison aggregates)."""
    if not trades:
        return {
            "total_trades":     0,
            "wins":             0,
            "losses":           0,
            "timeouts":         0,
            "win_rate_pct":     None,
            "avg_gain_pct":     None,
            "avg_win_pct":      None,
            "avg_loss_pct":     None,
            "expectancy_pct":   None,
            "total_return_pct": None,
        }
    wins     = [t for t in trades if t["outcome"] == "WIN"]
    losses   = [t for t in trades if t["outcome"] == "LOSS"]
    timeouts = [t for t in trades if t["outcome"] == "TIMEOUT"]

    win_rate   = round(len(wins) / len(trades) * 100, 1)
    avg_gain   = round(float(np.mean([t["gain_pct"] for t in trades])), 2)
    avg_win    = round(float(np.mean([t["gain_pct"] for t in wins])), 2)     if wins     else None
    avg_loss   = round(float(np.mean([t["gain_pct"] for t in losses])), 2)   if losses   else None

    non_win_gains = [t["gain_pct"] for t in trades if t["outcome"] != "WIN"]
    avg_non_win   = round(float(np.mean(non_win_gains)), 2) if non_win_gains else None
    expectancy = None
    if avg_win is not None and avg_non_win is not None:
        wr = win_rate / 100
        expectancy = round(wr * avg_win + (1 - wr) * avg_non_win, 2)

    equity = 100.0
    for t in sorted(trades, key=lambda x: (x["entry_date"], x["ticker"])):
        equity *= (1 + t["gain_pct"] / 100)
    total_return = round(equity - 100, 2)

    return {
        "total_trades":     len(trades),
        "wins":             len(wins),
        "losses":           len(losses),
        "timeouts":         len(timeouts),
        "win_rate_pct":     win_rate,
        "avg_gain_pct":     avg_gain,
        "avg_win_pct":      avg_win,
        "avg_loss_pct":     avg_loss,
        "expectancy_pct":   expectancy,
        "total_return_pct": total_return,
    }


def _recompute_summary(trades: List[Dict], target_pct: float) -> Dict:
    """Recompute the per-ticker summary dict from a (possibly filtered) trade
    list. Used when prime_only filters out non-prime trades from a ticker."""
    if not trades:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "timeouts": 0,
            "win_rate_pct": None, "avg_gain_pct": None, "avg_win_pct": None,
            "avg_loss_pct": None, "max_win_pct": None, "max_loss_pct": None,
            "expectancy_pct": None, "total_return_pct": None,
        }
    wins     = [t for t in trades if t["outcome"] == "WIN"]
    losses   = [t for t in trades if t["outcome"] == "LOSS"]
    timeouts = [t for t in trades if t["outcome"] == "TIMEOUT"]

    gains     = [t["gain_pct"] for t in trades]
    win_gains = [t["gain_pct"] for t in wins]
    los_gains = [t["gain_pct"] for t in losses]

    win_rate   = round(len(wins) / len(trades) * 100, 1) if trades else None
    avg_gain   = round(float(np.mean(gains)), 2)          if gains     else None
    avg_win    = round(float(np.mean(win_gains)), 2)      if win_gains else None
    avg_loss   = round(float(np.mean(los_gains)), 2)      if los_gains else None
    max_win    = round(float(np.max(win_gains)), 2)       if win_gains else None
    max_loss   = round(float(np.min(los_gains)), 2)       if los_gains else None

    non_win_gains = [t["gain_pct"] for t in trades if t["outcome"] != "WIN"]
    avg_non_win   = round(float(np.mean(non_win_gains)), 2) if non_win_gains else None
    expectancy = None
    if win_rate is not None and avg_win is not None and avg_non_win is not None:
        wr = win_rate / 100
        expectancy = round(wr * avg_win + (1 - wr) * avg_non_win, 2)

    equity = 100.0
    for t in sorted(trades, key=lambda x: x["entry_date"]):
        equity *= (1 + t["gain_pct"] / 100)
    total_return = round(equity - 100, 2)

    return {
        "total_trades":     len(trades),
        "wins":             len(wins),
        "losses":           len(losses),
        "timeouts":         len(timeouts),
        "win_rate_pct":     win_rate,
        "avg_gain_pct":     avg_gain,
        "avg_win_pct":      avg_win,
        "avg_loss_pct":     avg_loss,
        "max_win_pct":      max_win,
        "max_loss_pct":     max_loss,
        "expectancy_pct":   expectancy,
        "total_return_pct": total_return,
    }


# ─────────────────────────────────────────────────────────────
#  CACHE MANAGEMENT ENDPOINTS
# ─────────────────────────────────────────────────────────────

@router.get("/api/ema9/cache/stats")
async def ema9_cache_stats():
    """Return cache statistics: hit/miss counts, entries, size, TTLs, disk status.
    Useful for verifying the cache is working and diagnosing performance."""
    return _json_safe(_cache_stats_snapshot())


@router.post("/api/ema9/cache/clear")
async def ema9_cache_clear():
    """Clear ALL cached data — in-memory AND disk files. This is the ONLY way
    the cache is cleared (server restarts do NOT clear it).

    Use this if you suspect stale data (e.g., after a stock split or corporate
    action that yfinance hasn't picked up yet, or after quarterly earnings
    that change FV significantly)."""
    counts = _cache_clear()  # clears in-memory + deletes disk files
    logger.info(f"[Cache] Manually cleared (memory + disk): {counts}")
    return _json_safe({
        "status": "cleared",
        **counts,
        "disk_cleared": True,
        "message": f"Cleared {counts['yf_cleared']} yfinance + {counts['fv_cleared']} FV + "
                   f"{counts['fv_fail_cleared']} FV-failure entries (memory + disk). "
                   f"Cache will be empty until next backtest populates it.",
    })


@router.post("/api/ema9/cache/save")
async def ema9_cache_save():
    """Manually save cache to disk (forces immediate save, bypassing the
    30-second throttle). Useful before a planned server restart."""
    counts = _save_cache_to_disk(force=True)
    logger.info(f"[Cache] Manual save to disk: {counts}")
    return _json_safe({
        "status": "saved",
        **counts,
        "message": f"Cache saved to disk. Will survive server restart.",
    })


# ─────────────────────────────────────────────────────────────
#  FAIR VALUE ENRICHMENT (Now fixed by the global yf.download patch)
# ─────────────────────────────────────────────────────────────
_FV_NULL = {
    "composite_fair_price": None,
    "composite_gain_pct":   None,
    "fair_gap_pct":         None,
    "valuation_bucket":     "N/A",
    "fv_model_count":       0,
    "gap_to_fair_pct":      None,
    "fv_error":             "NOT_AVAILABLE",
}

_SAFE_FV_KEYS = {
    "composite_fair_price", "composite_gain_pct",
    "fair_gap_pct", "valuation_bucket", "fv_model_count",
    "gap_to_fair_pct", "fv_error",
}


def _enrich_fair_value(ticker: str, current_price: float = None) -> Dict:
    """Fetch Fair Value for a ticker via sma_router.

    CACHING:
      - Successful FV: cached for FV_CACHE_TTL_SEC (12 hours) since fundamentals
        change slowly. Only `gap_to_fair_pct` is recomputed per-call using the
        supplied current_price.
      - Failed FV (e.g., MANKIND, ABFRL, PNB): cached for FV_FAILURE_CACHE_TTL_SEC
        (1 hour) so we don't retry sma_router on every backtest run. This saves
        ~6-7 seconds per failing ticker per run (2 retries × 3s + inter-delay).
    """
    if not _FV_AVAILABLE:
        logger.warning(f"[FV] {ticker}: sma_router not available (import failed)")
        return {**_FV_NULL, "fv_error": "SMA_ROUTER_NOT_AVAILABLE"}

    # Check FAILURE cache first — if this ticker recently failed FV, return
    # the cached failure immediately (don't retry sma_router for 1 hour)
    cached_fail = _cache_get(_fv_fail_cache, ticker, FV_FAILURE_CACHE_TTL_SEC)
    if cached_fail is not None:
        with _cache_lock:
            _cache_stats["fv_fail_hits"] += 1
        logger.debug(f"[Cache-FV-FAIL] HIT  {ticker} (cached failure, skipping sma_router)")
        # Recompute gap_to_fair_pct (will be None since comp_fair is None)
        result = dict(cached_fail)
        result["gap_to_fair_pct"] = None
        return result

    # Check SUCCESS cache for the FV base values (everything except gap_to_fair_pct)
    cached_fv = _cache_get(_fv_cache, ticker, FV_CACHE_TTL_SEC)
    if cached_fv is not None:
        with _cache_lock:
            _cache_stats["fv_hits"] += 1
        logger.info(f"[Cache-FV] HIT  {ticker} (returning cached FV, no sma_router call)")
        # Recompute gap_to_fair_pct with the live current_price
        result = dict(cached_fv)
        comp_fair = result.get("composite_fair_price")
        if comp_fair and current_price and current_price > 0:
            result["gap_to_fair_pct"] = round((comp_fair - current_price) / current_price * 100, 2)
        else:
            result["gap_to_fair_pct"] = None
        return result

    with _cache_lock:
        _cache_stats["fv_misses"] += 1
        _cache_stats["fv_fail_misses"] += 1  # will become a fail hit if cached

    last_error = None

    for attempt in range(1, FV_MAX_RETRIES + 1):
        try:
            # This calls sma_router._analyze_ticker.
            # Because of our global monkey-patch at the top of this file,
            # the yf.download call inside sma_router will be safely intercepted!
            res = _sma_analyze_ticker(
                ticker,
                fy_start=2014,
                force=False,
                include_other_income=True,
            )

            if "error" in res:
                last_error = res["error"]
                logger.warning(f"[FV] {ticker} attempt {attempt}: sma_router returned error: {last_error}")
                if attempt < FV_MAX_RETRIES:
                    time.sleep(FV_RETRY_DELAY)
                    continue
                # Cache the failure (1h TTL) so we don't retry sma_router every run
                fail_result = {**_FV_NULL, "fv_error": f"SMA_ERROR: {str(last_error)[:120]}"}
                _cache_put(_fv_fail_cache, ticker, fail_result, "fv_fail")
                logger.info(f"[FV] {ticker}: FV failed (cached failure for {FV_FAILURE_CACHE_TTL_SEC//60}min)")
                return fail_result

            comp_fair = res.get("composite_fair_price")
            comp_gain = res.get("composite_gain_pct")
            bucket    = res.get("valuation_bucket", "N/A")
            model_cnt = res.get("model_count", 0)

            if comp_fair is None or comp_gain is None:
                logger.warning(f"[FV] {ticker} attempt {attempt}: No composite_fair_price in response.")
                if attempt < FV_MAX_RETRIES:
                    time.sleep(FV_RETRY_DELAY)
                    continue
                # Cache the failure (1h TTL)
                fail_result = {**_FV_NULL, "fv_error": "NO_COMPOSITE_FV_IN_RESPONSE"}
                _cache_put(_fv_fail_cache, ticker, fail_result, "fv_fail")
                logger.info(f"[FV] {ticker}: No composite FV (cached failure for {FV_FAILURE_CACHE_TTL_SEC//60}min)")
                return fail_result

            # Build the FV result (without gap_to_fair_pct, which is price-dependent)
            fv_base = {
                "composite_fair_price": comp_fair,
                "composite_gain_pct":   comp_gain,
                "fair_gap_pct":         comp_gain,
                "valuation_bucket":     bucket,
                "fv_model_count":       model_cnt,
                "gap_to_fair_pct":      None,  # filled below
                "fv_error":             None,
            }

            # Cache the base values (12h TTL) — gap_to_fair_pct recomputed per-call
            _cache_put(_fv_cache, ticker, fv_base, "fv")
            logger.info(f"[FV] {ticker}: FV=₹{comp_fair:.1f}, Bucket={bucket}, Models={model_cnt} (fetched + cached for 12h)")

            # Return with live gap_to_fair_pct
            result = dict(fv_base)
            if current_price and current_price > 0:
                result["gap_to_fair_pct"] = round((comp_fair - current_price) / current_price * 100, 2)
            return result

        except Exception as exc:
            last_error = str(exc)
            logger.error(f"[FV] {ticker} attempt {attempt}: Exception: {exc}")
            if attempt < FV_MAX_RETRIES:
                time.sleep(FV_RETRY_DELAY)
                continue

    # All retries exhausted — cache the failure (1h TTL)
    fail_result = {**_FV_NULL, "fv_error": f"RETRY_EXHAUSTED: {str(last_error)[:120]}"}
    _cache_put(_fv_fail_cache, ticker, fail_result, "fv_fail")
    logger.info(f"[FV] {ticker}: Retries exhausted (cached failure for {FV_FAILURE_CACHE_TTL_SEC//60}min)")
    return fail_result


# ─────────────────────────────────────────────────────────────
#  REQUEST MODEL
# ─────────────────────────────────────────────────────────────
class Ema9ScreenRequest(BaseModel):
    tickers:               List[str]
    interval:              str  = "1d"
    lookback_days:         int  = 180
    max_candles_ago:       int  = 10
    # Optional per-request override of the Require Uptrend (SMA50) filter.
    # If null/omitted, the backend falls back to config.json's
    # "ema9_require_uptrend" key (set via the Uptrend toggle on the Screener tab).
    # Pass True/False explicitly to override the global setting for one call.
    require_uptrend:       Optional[bool] = None
    allow_sideways:        bool = False
    # Optional per-request override of the Prime FV Gap tolerance.
    # If null/omitted, the backend falls back to config.json's
    # "ema9_prime_fv_gap_pct" (set via the UI input box → PUT /api/config).
    # Pass 0.0 to force strict undervalued-only for this one request.
    prime_fv_gap_pct:      Optional[float] = None
    # ── Pullback-from-prior-high (per-request override) ────────────────────
    # If null/omitted, the backend falls back to config.json's
    # "ema9_pullback_*" keys (set via the Pullback card on the Screener tab).
    # Pass explicit values here to override the global setting for one call.
    pullback_lookback_days: Optional[int]   = None
    pullback_threshold_pct: Optional[float] = None
    filter_pullback:        Optional[bool]  = None
    # ── Volatility-Adjusted Target (per-request override) ──────────────────
    # If null/omitted, the backend falls back to config.json's
    # "ema9_vol_target_*" keys (set via the Vol Target card on the Screener tab).
    # Pass explicit values here to override the global setting for one call.
    vol_target_mode:        Optional[str]   = None    # "fixed" | "atr"
    vol_target_atr_period:  Optional[int]   = None
    vol_target_multiplier:  Optional[float] = None
    vol_target_floor_pct:   Optional[float] = None
    vol_target_cap_pct:     Optional[float] = None


# ─────────────────────────────────────────────────────────────
#  SHARED SCREENING PIPELINE
# ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────
#  GOOGLE SHEETS LOGGER
# ─────────────────────────────────────────────────────────────
GSHEET_WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbz3Vj2-_xFkRhqXoySwxDyNZralsZ-XuZamHyuffo7INjtuNPNUSt4lxJg0aqOz7EAe/exec"

async def _log_signals_to_gsheet(signals: list, prime_targets: list):
    """Fire-and-forget: logs all scan signals to Google Sheets."""
    if not signals:
        return
    prime_set = {s.get("ticker") for s in prime_targets}
    scan_time = dt_module.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for s in signals:
        rows.append({
            "scan_time":       scan_time,
            "ticker":          s.get("ticker", ""),
            "price":           s.get("current_price", ""),
            "fair_value":      s.get("composite_fair_price", ""),
            "fv_gap_pct":      s.get("gap_to_fair_pct", ""),   # correct key from _enrich_fair_value
            "trend":           s.get("trend_regime", ""),
            "candles_ago":     s.get("candles_ago", ""),
            "valuation":       s.get("valuation_bucket", ""),  # was grade (always empty)
            "type":            "PRIME" if s.get("ticker") in prime_set else "OTHER",
            "interval":        s.get("interval", ""),
        })
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(GSHEET_WEBHOOK_URL, json=rows)
            if resp.status_code == 200:
                logger.info(f"[GSheet] Logged {len(rows)} signals successfully.")
            else:
                logger.warning(f"[GSheet] Unexpected response {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        logger.warning(f"[GSheet] Logging failed (non-blocking): {e}")


# ─────────────────────────────────────────────────────────────
#  Prime Target FV Gap Tolerance — config reader
# ─────────────────────────────────────────────────────────────
# Reads the user-configurable threshold from main.py's config.json.
# The threshold relaxes the "Prime Target" classification: by default a
# ticker must be strictly undervalued (current_price < fair_value, i.e.
# gap_to_fair_pct > 0). With a non-zero threshold N, a ticker is also
# classified as Prime if it is up to N% overvalued (gap_to_fair_pct >= -N).
#
# Config key: "ema9_prime_fv_gap_pct" (float, default 0.0)
# Set via: PUT /api/config  body: {"ema9_prime_fv_gap_pct": 10.0}
#
# This is read on EVERY call to _run_screen_pipeline + ema9_quick_scan
# so changes take effect immediately without a server restart. The

# ─────────────────────────────────────────────────────────────
#  NEAR-SUPPORT ENRICHMENT (Chartink Live Feed integration)
#  Adds multi-touch horizontal support detection fields to every
#  signal.  Detector lives in support_detector.py; this is the
#  inline enricher that _run_screen_pipeline calls after FV
#  enrichment so the Chartink watcher inherits the data for free.
# ─────────────────────────────────────────────────────────────
try:
    from near_support_enrich import enrich_signals_with_near_support
    _NS_ENRICH_AVAILABLE = True
except ImportError as _ns_err:
    _NS_ENRICH_AVAILABLE = False
    print(f"[ema9_router] near_support_enrich not importable — near-support enrichment disabled ({_ns_err})")


# Chartink watcher (which calls _run_screen_pipeline without explicitly
# passing prime_fv_gap_pct) inherits the config value automatically.
_CONFIG_FILE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.json"
)

def _load_prime_fv_gap_pct_from_config() -> float:
    """Read ema9_prime_fv_gap_pct from config.json. Returns 0.0 on any error."""
    try:
        if os.path.exists(_CONFIG_FILE_PATH):
            with open(_CONFIG_FILE_PATH) as f:
                cfg = json.load(f)
            val = float(cfg.get("ema9_prime_fv_gap_pct", 0.0))
            # Clamp to a sane range to prevent accidental mis-classification
            # (e.g., user typing 1000 would put everything in Prime).
            if val < 0:
                return 0.0
            if val > 50:
                return 50.0
            return val
    except Exception as e:
        logger.warning(f"[PrimeFvGap] Failed to read config.json: {e}")
    return 0.0


# ─────────────────────────────────────────────────────────────
#  Pullback-from-previous-high — config readers
# ─────────────────────────────────────────────────────────────
# Reads the user-configurable pullback parameters from main.py's config.json.
# These control the live screener pipeline's entry-quality filter:
#   • ema9_pullback_lookback_days (int, default 60)
#       Number of trading days to look back for the prior high. 60 ≈ 3 months.
#   • ema9_pullback_threshold_pct (float, default 10.0)
#       Min % pullback required from the prior high for the breakout to count
#       as a "pullback setup". 10.0 = stock must be ≥10% below its N-day high.
#   • ema9_filter_pullback (bool, default False)
#       When True, the live screener DROPS signals that don't meet the
#       pullback threshold (they don't appear in prime_targets/other_signals).
#       When False (default), every signal is annotated with the pullback
#       fields but none are filtered out — the user can sort/filter visually.
#
# Config keys: "ema9_pullback_lookback_days",
#              "ema9_pullback_threshold_pct",
#              "ema9_filter_pullback"
# Set via: PUT /api/config  body: {"ema9_pullback_threshold_pct": 10.0, ...}
#
# These are read on EVERY call to _run_screen_pipeline so changes take effect
# immediately without a server restart. The Chartink watcher (which calls
# _run_screen_pipeline without explicitly passing these) inherits the config
# values automatically — same pattern as prime_fv_gap_pct above.
def _load_pullback_lookback_days_from_config() -> int:
    """Read ema9_pullback_lookback_days from config.json. Returns 60 on any error."""
    try:
        if os.path.exists(_CONFIG_FILE_PATH):
            with open(_CONFIG_FILE_PATH) as f:
                cfg = json.load(f)
            val = int(cfg.get("ema9_pullback_lookback_days", 60))
            # Clamp to a sane range: too small (<5) makes the prior high
            # meaningless; too large (>250 ≈ 1 trading year) is excessive.
            if val < 5:
                return 5
            if val > 250:
                return 250
            return val
    except Exception as e:
        logger.warning(f"[Pullback-Lookback] Failed to read config.json: {e}")
    return 60


def _load_pullback_threshold_pct_from_config() -> float:
    """Read ema9_pullback_threshold_pct from config.json. Returns 10.0 on any error."""
    try:
        if os.path.exists(_CONFIG_FILE_PATH):
            with open(_CONFIG_FILE_PATH) as f:
                cfg = json.load(f)
            val = float(cfg.get("ema9_pullback_threshold_pct", 10.0))
            # Clamp 0-50: 0 would let every trade pass (no filter); >50 is
            # absurd (would require a >50% crash before buying).
            if val < 0:
                return 0.0
            if val > 50:
                return 50.0
            return val
    except Exception as e:
        logger.warning(f"[Pullback-Threshold] Failed to read config.json: {e}")
    return 10.0


def _load_filter_pullback_from_config() -> bool:
    """Read ema9_filter_pullback from config.json. Returns False on any error."""
    try:
        if os.path.exists(_CONFIG_FILE_PATH):
            with open(_CONFIG_FILE_PATH) as f:
                cfg = json.load(f)
            return bool(cfg.get("ema9_filter_pullback", False))
    except Exception as e:
        logger.warning(f"[FilterPullback] Failed to read config.json: {e}")
    return False


# ─────────────────────────────────────────────────────────────
#  Volatility-Adjusted Target — config readers
# ─────────────────────────────────────────────────────────────
# Reads the user-configurable vol-target parameters from main.py's config.json.
# These control the live screener's per-signal target_price computation:
#   • ema9_vol_target_mode (str, default "fixed")
#       "fixed" → use a flat target_pct for every signal (legacy behavior).
#       "atr"   → compute target_pct per signal from ATR:
#                    effective_target_pct = clamp(
#                      multiplier × ATR_at_entry / entry_price × 100,
#                      floor_pct, cap_pct
#                    )
#   • ema9_vol_target_atr_period (int, default 14) — ATR lookback (standard 14).
#   • ema9_vol_target_multiplier (float, default 1.5) — target = multiplier × ATR/price × 100.
#   • ema9_vol_target_floor_pct (float, default 2.0) — min target % (prevents
#     trivial 0.5% targets on tiny-ATR stocks like PSU banks).
#   • ema9_vol_target_cap_pct (float, default 8.0) — max target % (prevents
#     absurd 25% targets on huge-ATR stocks like volatile small-caps).
#
# Set via: PUT /api/config  body: {"ema9_vol_target_mode": "atr", "ema9_vol_target_multiplier": 1.5, ...}
#
# Read on EVERY call to _run_screen_pipeline + _process_ticker_df so changes
# take effect immediately (no server restart). The Chartink watcher inherits
# these via _run_screen_pipeline — same pattern as prime_fv_gap_pct + pullback.
def _load_vol_target_mode_from_config() -> str:
    """Read ema9_vol_target_mode from config.json. Returns 'fixed' on any error."""
    try:
        if os.path.exists(_CONFIG_FILE_PATH):
            with open(_CONFIG_FILE_PATH) as f:
                cfg = json.load(f)
            val = str(cfg.get("ema9_vol_target_mode", "fixed")).lower().strip()
            return val if val in ("fixed", "atr") else "fixed"
    except Exception as e:
        logger.warning(f"[VolTarget-Mode] Failed to read config.json: {e}")
    return "fixed"


def _load_vol_target_atr_period_from_config() -> int:
    """Read ema9_vol_target_atr_period from config.json. Returns 14 on any error."""
    try:
        if os.path.exists(_CONFIG_FILE_PATH):
            with open(_CONFIG_FILE_PATH) as f:
                cfg = json.load(f)
            val = int(cfg.get("ema9_vol_target_atr_period", 14))
            # Clamp: too short (<2) makes ATR noisy; too long (>50) is unresponsive.
            if val < 2:  return 2
            if val > 50: return 50
            return val
    except Exception as e:
        logger.warning(f"[VolTarget-ATRPeriod] Failed to read config.json: {e}")
    return 14


def _load_vol_target_multiplier_from_config() -> float:
    """Read ema9_vol_target_multiplier from config.json. Returns 1.5 on any error."""
    try:
        if os.path.exists(_CONFIG_FILE_PATH):
            with open(_CONFIG_FILE_PATH) as f:
                cfg = json.load(f)
            val = float(cfg.get("ema9_vol_target_multiplier", 1.5))
            # Clamp 0.5-5.0: <0.5 makes targets trivially small; >5 makes them
            # absurdly large (e.g., 5× ATR = 25%+ target on volatile stocks).
            if val < 0.5: return 0.5
            if val > 5.0: return 5.0
            return val
    except Exception as e:
        logger.warning(f"[VolTarget-Multiplier] Failed to read config.json: {e}")
    return 1.5


def _load_vol_target_floor_pct_from_config() -> float:
    """Read ema9_vol_target_floor_pct from config.json. Returns 2.0 on any error."""
    try:
        if os.path.exists(_CONFIG_FILE_PATH):
            with open(_CONFIG_FILE_PATH) as f:
                cfg = json.load(f)
            val = float(cfg.get("ema9_vol_target_floor_pct", 2.0))
            # Clamp 0.5-5.0: <0.5% target is too tight to be useful; >5% floor
            # defeats the purpose of vol-adjustment (forces large targets on
            # even the lowest-vol stocks).
            if val < 0.5: return 0.5
            if val > 5.0: return 5.0
            return val
    except Exception as e:
        logger.warning(f"[VolTarget-Floor] Failed to read config.json: {e}")
    return 2.0


def _load_vol_target_cap_pct_from_config() -> float:
    """Read ema9_vol_target_cap_pct from config.json. Returns 8.0 on any error."""
    try:
        if os.path.exists(_CONFIG_FILE_PATH):
            with open(_CONFIG_FILE_PATH) as f:
                cfg = json.load(f)
            val = float(cfg.get("ema9_vol_target_cap_pct", 8.0))
            # Clamp 3.0-20.0: <3% cap is too tight (defeats the purpose on
            # high-vol stocks); >20% target is too greedy (rarely hit).
            if val < 3.0:  return 3.0
            if val > 20.0: return 20.0
            return val
    except Exception as e:
        logger.warning(f"[VolTarget-Cap] Failed to read config.json: {e}")
    return 8.0


# ─────────────────────────────────────────────────────────────
#  Require Uptrend (SMA50) — config reader
# ─────────────────────────────────────────────────────────────
# Reads the user-configurable toggle from main.py's config.json.
# When True (default), the live screener + Chartink watcher REJECT any 9EMA
# crossover signal where current Close ≤ 50-day SMA. This is the legacy
# behavior — filters out breakouts in downtrends (likely to fail).
#
# When False, EVERY 9EMA crossover is accepted regardless of SMA50. Useful
# when you suspect valid signals are being missed because the stock is
# recovering from a deeper correction (Close still below SMA50 but momentum
# turning up). Typically you'd run with this OFF for a few days to see what
# you're missing, then turn it back ON if the additional signals are noise.
#
# ─────────────────────────────────────────────────────────────
#  Momentum Sweep — config readers + detector
# ─────────────────────────────────────────────────────────────
# Purpose: catch stocks ALREADY in a momentum phase (above 9EMA,
# making higher-highs, with HH-LL structure intact) even when no
# fresh 9EMA crossover has happened in the last 10 bars. These are
# momentum-continuation trade candidates for a quick 3% target.
#
# Detection (per ticker, every scan):
#   R1  Close > 9EMA                       (in momentum phase)
#   R2  Close > prev day's Low             (HH-LL structure intact)
#   R3  >=2 of last 3 highs are higher     (rising structure)
#   R4  EMA9 dist >= min_ema9_dist_pct     (not flat-lining on EMA)
#   R5  EMA9 dist <= max_ema9_dist_pct     (not extended / chase risk)
# R4 (SMA50) is inherited via the existing require_uptrend toggle.
#
# Config keys in config.json (set via PUT /api/config):
#   ema9_momentum_sweep_enabled         (bool,   default True)
#   ema9_momentum_sweep_min_ema9_dist_pct (float, default 0.5)
#   ema9_momentum_sweep_max_ema9_dist_pct (float, default 8.0)
#   ema9_momentum_sweep_target_pct      (float,  default 3.0)
def _load_momentum_sweep_enabled() -> bool:
    try:
        if os.path.exists(_CONFIG_FILE_PATH):
            with open(_CONFIG_FILE_PATH) as f:
                cfg = json.load(f)
            return bool(cfg.get("ema9_momentum_sweep_enabled", True))
    except Exception as e:
        logger.warning(f"[MomentumSweep] Failed to read enabled flag: {e}")
    return True


def _load_momentum_sweep_min_ema9_dist_pct() -> float:
    try:
        if os.path.exists(_CONFIG_FILE_PATH):
            with open(_CONFIG_FILE_PATH) as f:
                cfg = json.load(f)
            return float(cfg.get("ema9_momentum_sweep_min_ema9_dist_pct", 0.5))
    except Exception:
        pass
    return 0.5


def _load_momentum_sweep_max_ema9_dist_pct() -> float:
    try:
        if os.path.exists(_CONFIG_FILE_PATH):
            with open(_CONFIG_FILE_PATH) as f:
                cfg = json.load(f)
            return float(cfg.get("ema9_momentum_sweep_max_ema9_dist_pct", 8.0))
    except Exception:
        pass
    return 8.0


def _load_momentum_sweep_target_pct() -> float:
    try:
        if os.path.exists(_CONFIG_FILE_PATH):
            with open(_CONFIG_FILE_PATH) as f:
                cfg = json.load(f)
            return float(cfg.get("ema9_momentum_sweep_target_pct", 3.0))
    except Exception:
        pass
    return 3.0


def _detect_momentum_sweep(
    df: pd.DataFrame,
    current_price: Optional[float],
    current_ema9: Optional[float],
) -> Dict:
    """
    Detect whether a ticker qualifies as a momentum-sweep candidate.

    Designed to catch Adani Green-style momentum continuation:
      - Stock already in a strong uptrend (above rising 9EMA)
      - Consecutive green candles (strong buying pressure)
      - Body-dominant candles (small wicks = strong closes)
      - Higher highs forming (rising structure)
      - HH-LL intact (Close > prev day's Low)
      - EMA9 distance 0.5%-15% (catches extended momentum names)

    Returns a dict with:
      is_momentum_sweep              : bool
      momentum_higher_highs_count    : int   (0..2, # of last 3 highs that rose)
      momentum_prev_low              : float | None
      momentum_above_prev_low_pct    : float | None  (Close vs prev Low)
      momentum_target_price          : float | None  (price * (1 + target_pct/100))
      momentum_target_pct            : float | None  (echoes the active target %)
      momentum_ema9_dist_pct         : float | None  (signed: (close-ema9)/ema9*100)
      momentum_ema9_slope_pct        : float | None  (EMA9 slope: (today - 5d ago)/5d ago * 100)
      momentum_green_streak          : int   (0..3, # of last 3 candles that are green)
      momentum_body_wick_ratio       : float | None  (0..1, body/(body+wick) avg of last 3)
      momentum_score                 : float  (higher = stronger setup)
      momentum_grade                 : str   ("A" / "B" / "C" — setup quality)
    """
    out: Dict = {
        "is_momentum_sweep":              False,
        "momentum_higher_highs_count":    0,
        "momentum_prev_low":              None,
        "momentum_above_prev_low_pct":    None,
        "momentum_target_price":          None,
        "momentum_target_pct":            None,
        "momentum_ema9_dist_pct":         None,
        "momentum_ema9_slope_pct":        None,
        "momentum_green_streak":          0,
        "momentum_body_wick_ratio":       None,
        "momentum_score":                 0.0,
        "momentum_grade":                 "",
    }
    try:
        if current_price is None or current_ema9 is None:
            return out
        if len(df) < 8:   # need at least 8 bars: 3 for green streak + 5 for EMA9 slope
            return out

        target_pct = _load_momentum_sweep_target_pct()
        min_dist   = _load_momentum_sweep_min_ema9_dist_pct()
        max_dist   = _load_momentum_sweep_max_ema9_dist_pct()

        # ── R1: Close > 9EMA (in active momentum phase) ──────────────────
        if current_price <= current_ema9:
            return out

        # EMA9 distance % (signed) — how far above the 9EMA
        ema9_dist_pct = round((current_price - current_ema9) / current_ema9 * 100, 2)
        out["momentum_ema9_dist_pct"] = ema9_dist_pct

        # R7 + R8: distance window (0.5% to 15% by default)
        if ema9_dist_pct < min_dist or ema9_dist_pct > max_dist:
            return out

        # ── R2: 9EMA slope must be RISING (today's 9EMA > 9EMA 5 days ago) ──
        # This is the key filter that confirms the trend is accelerating,
        # not flat or rolling over. Matches the Adani Green chart where
        # the green 9EMA line is clearly sloping upward.
        ema9_today = current_ema9
        ema9_5d_ago = None
        try:
            ema9_5d_ago = float(df["ema9"].iloc[-6])   # 5 bars ago = -6 index
        except (IndexError, ValueError, TypeError):
            pass
        if ema9_5d_ago is None or ema9_5d_ago <= 0:
            return out
        ema9_slope_pct = round((ema9_today - ema9_5d_ago) / ema9_5d_ago * 100, 2)
        out["momentum_ema9_slope_pct"] = ema9_slope_pct
        if ema9_slope_pct <= 0:
            return out   # EMA9 must be sloping UP

        # ── R3: HH-LL intact — today's Close > previous day's Low ────────
        prev_low = float(df["Low"].iloc[-2])
        if prev_low is None or prev_low <= 0:
            return out
        above_prev_low_pct = round((current_price - prev_low) / prev_low * 100, 2)
        out["momentum_prev_low"]           = round(prev_low, 2)
        out["momentum_above_prev_low_pct"] = above_prev_low_pct
        if current_price <= prev_low:
            return out

        # ── R4: >=2 of last 3 candles are GREEN (Close > Open) ───────────
        # Matches the consecutive-green-candle pattern on the Adani Green chart.
        last3 = df.iloc[-3:]
        green_count = 0
        for _, row in last3.iterrows():
            try:
                if float(row["Close"]) > float(row["Open"]):
                    green_count += 1
            except (TypeError, ValueError):
                pass
        out["momentum_green_streak"] = green_count
        if green_count < 2:
            return out

        # ── R5: >=1 of last 3 highs is higher than its predecessor ───────
        # Loosened from 2 to 1 — even 1 higher high in the last 3 confirms
        # the structure is still rising (Adani Green has clear HH sequence
        # but a single bar pullback within the streak is normal).
        highs_last3 = df["High"].iloc[-3:].tolist()
        higher_count = 0
        for k in range(1, len(highs_last3)):
            try:
                if float(highs_last3[k]) > float(highs_last3[k - 1]):
                    higher_count += 1
            except (TypeError, ValueError):
                pass
        out["momentum_higher_highs_count"] = higher_count
        if higher_count < 1:
            return out

        # ── R6: Strong closes — body/wick ratio >= 0.6 over last 3 bars ──
        # Matches the "minimal wicks" characteristic on the Adani Green chart.
        # body = |Close - Open|, total range = High - Low, ratio = body/range.
        # Average over last 3 bars. Higher ratio = more body-dominant = stronger
        # close. Skip doji bars (body < 0.1% of price) to avoid div-by-zero noise.
        body_ratios = []
        for _, row in last3.iterrows():
            try:
                o = float(row["Open"]);  h = float(row["High"])
                l = float(row["Low"]);   c = float(row["Close"])
                rng = h - l
                if rng <= 0:
                    continue
                body = abs(c - o)
                body_ratios.append(body / rng)
            except (TypeError, ValueError):
                pass
        avg_body_wick = round(sum(body_ratios) / len(body_ratios), 3) if body_ratios else None
        out["momentum_body_wick_ratio"] = avg_body_wick
        if avg_body_wick is None or avg_body_wick < 0.6:
            return out

        # ─── All rules passed -> mark as momentum sweep ──────────────────
        out["is_momentum_sweep"]    = True
        out["momentum_target_pct"]  = target_pct
        out["momentum_target_price"] = round(current_price * (1.0 + target_pct / 100.0), 2)

        # ─── Composite score (0..10 scale) ───────────────────────────────
        # Weighted to surface the strongest Adani-Green-style setups:
        #   30% EMA9 slope strength     (trend acceleration)
        #   25% green candle streak     (buying pressure)
        #   20% body/wick ratio         (close strength)
        #   15% EMA9 distance           (extension = momentum)
        #   10% higher-high count       (structure)
        # Each component is normalized to its weight contribution.
        slope_score  = min(ema9_slope_pct / 5.0, 1.0)        # 5% slope = full marks
        green_score  = green_count / 3.0                      # 3/3 = full marks
        body_score   = (avg_body_wick - 0.6) / 0.4 if avg_body_wick >= 0.6 else 0.0   # 0.6->0, 1.0->1
        body_score   = min(max(body_score, 0.0), 1.0)
        dist_score   = min(ema9_dist_pct / 10.0, 1.0)        # 10% dist = full marks
        hh_score     = higher_count / 2.0                     # 2/2 = full marks

        score = (
            3.0 * slope_score
            + 2.5 * green_score
            + 2.0 * body_score
            + 1.5 * dist_score
            + 1.0 * hh_score
        )
        out["momentum_score"] = round(score, 3)

        # ─── Grade: A (>=7.5), B (>=5.5), C (>=4.0) ─────────────────────
        if score >= 7.5:
            out["momentum_grade"] = "A"
        elif score >= 5.5:
            out["momentum_grade"] = "B"
        else:
            out["momentum_grade"] = "C"
    except Exception as exc:
        logger.warning(f"[MomentumSweep] detection failed: {exc}")
    return out


# Config key: "ema9_require_uptrend" (bool, default True)
# Set via: PUT /api/config  body: {"ema9_require_uptrend": false}
#
# Read on EVERY call to _run_screen_pipeline so changes take effect
# immediately (no server restart). The Chartink watcher inherits this via
# _run_screen_pipeline — same pattern as prime_fv_gap_pct + pullback + vol-target.
def _load_require_uptrend_from_config() -> bool:
    """Read ema9_require_uptrend from config.json. Returns True on any error
    (preserves the legacy default — safer to filter than to admit noise)."""
    try:
        if os.path.exists(_CONFIG_FILE_PATH):
            with open(_CONFIG_FILE_PATH) as f:
                cfg = json.load(f)
            return bool(cfg.get("ema9_require_uptrend", True))
    except Exception as e:
        logger.warning(f"[RequireUptrend] Failed to read config.json: {e}")
    return True


def _compute_vol_target(
    entry_price: float,
    atr_at_entry: Optional[float],
    multiplier: float,
    floor_pct: float,
    cap_pct: float,
    fallback_target_pct: float = 3.0,
) -> Dict:
    """
    Compute the volatility-adjusted target_pct and target_price for a live
    screener signal.

    Mirrors the logic in _duration_backtest_ticker (effective_target_pct =
    clamp(multiplier × ATR/price × 100, floor, cap)).

    Returns a dict with:
      effective_target_pct : the % to apply (clamped)
      target_price         : entry_price × (1 + effective_target_pct/100)
      target_mode          : "atr" if ATR was used, "atr_fallback_fixed" if ATR
                             was missing and we fell back to fallback_target_pct,
                             or "fixed" if caller passed mode="fixed"
      atr_at_entry         : the ATR value used (echoed for the frontend)
      atr_pct_at_entry     : ATR as % of entry_price
    """
    out = {
        "effective_target_pct": float(fallback_target_pct),
        "target_price":         round(float(entry_price) * (1 + float(fallback_target_pct) / 100), 2),
        "target_mode":          "fixed",
        "atr_at_entry":         None,
        "atr_pct_at_entry":     None,
    }
    try:
        if atr_at_entry is not None and float(atr_at_entry) > 0 and float(entry_price) > 0:
            atr_pct = float(atr_at_entry) / float(entry_price) * 100.0
            out["atr_at_entry"]     = round(float(atr_at_entry), 2)
            out["atr_pct_at_entry"] = round(atr_pct, 2)
            raw_target_pct = float(multiplier) * atr_pct
            clamped = max(float(floor_pct), min(float(cap_pct), raw_target_pct))
            out["effective_target_pct"] = round(clamped, 2)
            out["target_price"]         = round(float(entry_price) * (1 + clamped / 100), 2)
            out["target_mode"]          = "atr"
    except Exception as e:
        logger.warning(f"[VolTarget-Compute] Failed: {e}")
    return out


def _compute_pullback_for_signal(
    df: pd.DataFrame,
    breakout_idx: int,
    entry_price: float,
    lookback_days: int,
    threshold_pct: float,
) -> Dict:
    """
    Compute pullback-from-prior-high fields for a live screener signal.

    Mirrors the logic in _duration_backtest_ticker (look back N bars before
    the breakout, find the highest High, compute pullback %).

    Returns a dict with:
      prior_high         : rounded ₹ value of the prior N-day high (or None)
      pullback_dist_pct  : % pullback from prior high (≥0 if pulled back)
      is_pullback        : True iff pullback_dist_pct >= threshold_pct
    """
    out = {"prior_high": None, "pullback_dist_pct": None, "is_pullback": False}
    try:
        n = len(df)
        pb_start = max(0, breakout_idx - int(lookback_days))
        if breakout_idx <= pb_start:
            return out
        window = df["High"].iloc[pb_start:breakout_idx]  # exclude breakout day
        if len(window) == 0:
            return out
        pb_high_val = float(window.max())
        if pb_high_val <= 0:
            return out
        out["prior_high"] = round(pb_high_val, 2)
        raw_pullback = (pb_high_val - float(entry_price)) / pb_high_val * 100
        out["pullback_dist_pct"] = round(raw_pullback, 2)
        out["is_pullback"] = raw_pullback >= float(threshold_pct)
    except Exception as e:
        logger.warning(f"[Pullback-Compute] Failed: {e}")
    return out


def _is_prime_target(fv: Optional[float], cp: Optional[float],
                     threshold_pct: float) -> bool:
    """
    Prime classification predicate, shared by the batch + single-ticker paths.

    Uses the SAME gap_to_fair_pct formula the rest of the codebase uses:
        gap = (fair_value - current_price) / current_price * 100
    So:
      • gap > 0  → undervalued (FV above price)
      • gap = 0  → fairly valued
      • gap < 0  → overvalued (price above FV)

    A ticker is Prime iff gap_to_fair_pct >= -threshold_pct.
    This makes the user-facing threshold match their mental model exactly:
    setting threshold = 10 means "include tickers down to −10% gap" (inclusive).

    Examples:
      threshold = 0   → gap >= 0  (undervalued OR fairly valued)
      threshold = 10  → gap >= -10 (allow up to 10% overvalued, inclusive)
      threshold = 50  → gap >= -50 (allow up to 50% overvalued)

    Note: at threshold = 0 this is *slightly* more permissive than the original
    `cp < fv` (which was strictly undervalued only). The difference is only the
    rare edge case where gap == 0 exactly (price == fair_value). In practice
    FV is rarely an exact integer match to live price, so this is negligible.
    """
    if not fv or not cp:
        return False
    try:
        fv_f = float(fv)
        cp_f = float(cp)
        if cp_f <= 0:
            return False
        gap_pct = (fv_f - cp_f) / cp_f * 100.0
        return gap_pct >= -float(threshold_pct)
    except (TypeError, ValueError, ZeroDivisionError):
        return False


async def _run_screen_pipeline(
    tickers: List[str],
    interval: str = "1d",
    lookback_days: int = 180,
    max_candles_ago: int = 10,
    require_uptrend: Optional[bool] = None,
    prime_fv_gap_pct: Optional[float] = None,
    pullback_lookback_days: Optional[int] = None,
    pullback_threshold_pct: Optional[float] = None,
    filter_pullback: Optional[bool] = None,
    # Volatility-Adjusted Target params (None = fall back to config.json)
    vol_target_mode: Optional[str] = None,
    vol_target_atr_period: Optional[int] = None,
    vol_target_multiplier: Optional[float] = None,
    vol_target_floor_pct: Optional[float] = None,
    vol_target_cap_pct: Optional[float] = None,
) -> Dict:
    tickers = [
        t.strip().upper().replace(".NS", "").replace(".BO", "")
        for t in tickers if t.strip()
    ][:2000]

    if not tickers:
        raise HTTPException(400, "No valid tickers provided.")

    # ── Resolve Require Uptrend (SMA50) filter ──────────────────────────────
    # None (default) → fall back to config.json's "ema9_require_uptrend" key,
    # which is what the UI toggle on the Screener tab updates. This lets the
    # Chartink watcher (which calls _run_screen_pipeline without explicitly
    # passing require_uptrend) inherit the user's global setting — same
    # pattern as prime_fv_gap_pct + pullback + vol-target.
    # Explicit True/False from a caller (e.g., Duration Backtest's checkbox)
    # overrides config and is respected as-is.
    if require_uptrend is None:
        require_uptrend = _load_require_uptrend_from_config()
    logger.info(f"[RequireUptrend] filter = {require_uptrend} "
                f"({'reject Close ≤ SMA50' if require_uptrend else 'accept ALL 9EMA crossovers'})")

    signals, filtered_by_trend, failed, momentum_sweep = [], [], [], []

    # Step 1: Batch download (using concurrent yf.Ticker loop)
    # Use _batch_download_async instead of sequential _batch_download.
    # With YF_CONCURRENCY=5 and 50ms stagger, 307 cache-miss tickers
    # take ~2 min instead of ~8 min sequential. Cache hits are instant.
    ticker_dfs = await _batch_download_async(tickers, interval, lookback_days)

    # Step 2: Process each ticker
    for ticker in tickers:
        if ticker not in ticker_dfs:
            failed.append({"ticker": ticker, "error": "No data available (tried .NS and .BO)"})
            continue
        try:
            res = _process_ticker_df(ticker, ticker_dfs[ticker], max_candles_ago)
            res["interval"] = interval

            if res["status"] in ("ERROR", "NO_DATA"):
                failed.append({"ticker": res["ticker"], "error": res.get("error", "")})
                continue

            # ── Momentum Sweep collection ────────────────────────────────
            # Captures tickers already above 9EMA making higher-highs with
            # intact HH-LL structure. Works for BOTH SIGNAL (fresh 9EMA
            # crossover) and NO_SIGNAL (already extended) tickers, so
            # the user sees momentum-continuation candidates even when
            # the breakout happened >10 days ago.
            if res.get("is_momentum_sweep") is True:
                momentum_sweep.append(res)

            if res["status"] != "SIGNAL":
                continue

            if require_uptrend and not res.get("above_sma50", False):
                filtered_by_trend.append({
                    "ticker":       res["ticker"],
                    "trend_regime": res.get("trend_regime", "UNKNOWN"),
                    "reason":       "Rejected: Close ≤ 50-day SMA (no uptrend)",
                })
                continue

            signals.append(res)

        except Exception as exc:
            failed.append({"ticker": ticker, "error": str(exc)})

    signals.sort(key=lambda r: r.get("candles_ago", 999))

    # ── Step 2.5: Pullback filter (entry-quality gate) ─────────────────────
    # Applied AFTER trend filter but BEFORE Fair Value enrichment, so we don't
    # waste FV API calls on signals that will be dropped anyway.
    #
    # When `filter_pullback` is True (either passed explicitly OR read from
    # config.json), signals whose `is_pullback` flag is False are moved to a
    # separate `filtered_by_pullback` list (mirroring `filtered_by_trend`) and
    # dropped from the `signals` list. The pullback fields themselves are
    # always present on every SIGNAL response regardless of this flag.
    #
    # Resolution order for the 3 pullback params:
    #   1. Explicit kwargs (passed by ema9_screen)
    #   2. config.json (read fresh on every call so UI changes take effect
    #      immediately — same pattern as prime_fv_gap_pct)
    if pullback_lookback_days is None:
        pullback_lookback_days = _load_pullback_lookback_days_from_config()
    if pullback_threshold_pct is None:
        pullback_threshold_pct = _load_pullback_threshold_pct_from_config()
    if filter_pullback is None:
        filter_pullback = _load_filter_pullback_from_config()

    # ── Resolve Volatility-Adjusted Target params (kwarg override → config.json) ──
    # _process_ticker_df reads these directly from config.json on every call,
    # so we don't strictly need to pass them through. But we resolve them here
    # too so we can echo the effective values in the response payload (for the
    # frontend's Chartink panel header to display).
    if vol_target_mode is None:
        vol_target_mode = _load_vol_target_mode_from_config()
    if vol_target_atr_period is None:
        vol_target_atr_period = _load_vol_target_atr_period_from_config()
    if vol_target_multiplier is None:
        vol_target_multiplier = _load_vol_target_multiplier_from_config()
    if vol_target_floor_pct is None:
        vol_target_floor_pct = _load_vol_target_floor_pct_from_config()
    if vol_target_cap_pct is None:
        vol_target_cap_pct = _load_vol_target_cap_pct_from_config()
    logger.info(f"[VolTarget] mode={vol_target_mode} | ATR period={vol_target_atr_period}d | "
                f"multiplier={vol_target_multiplier}× | floor={vol_target_floor_pct}% | "
                f"cap={vol_target_cap_pct}%")

    filtered_by_pullback: List[Dict] = []
    if filter_pullback:
        kept_signals = []
        for sig in signals:
            if sig.get("is_pullback") is True:
                kept_signals.append(sig)
            else:
                filtered_by_pullback.append({
                    "ticker":            sig.get("ticker", "?"),
                    "pullback_dist_pct": sig.get("pullback_dist_pct"),
                    "prior_high":        sig.get("prior_high"),
                    "current_price":     sig.get("current_price"),
                    "reason":            f"Rejected: pullback {sig.get('pullback_dist_pct')}% < "
                                         f"threshold {pullback_threshold_pct}% "
                                         f"(lookback={pullback_lookback_days}d)",
                })
        logger.info(f"[Pullback-Filter] filter_pullback=True "
                    f"(threshold={pullback_threshold_pct}%, lookback={pullback_lookback_days}d) → "
                    f"kept {len(kept_signals)}/{len(signals)} pullback signals, "
                    f"filtered {len(filtered_by_pullback)}")
        signals = kept_signals
    else:
        # Even when filter is off, log how many signals WOULD be kept — useful
        # for the user to gauge the filter's selectivity before enabling it.
        _pb_count = sum(1 for s in signals if s.get("is_pullback") is True)
        logger.info(f"[Pullback-Filter] filter_pullback=False (annotation only) → "
                    f"{_pb_count}/{len(signals)} signals meet the pullback threshold "
                    f"({pullback_threshold_pct}% over {pullback_lookback_days}d)")

    # Step 3: Fair Value enrichment
    fv_failures = []
    for i, sig in enumerate(signals):
        fv = await asyncio.to_thread(
            _enrich_fair_value, sig["ticker"], sig.get("current_price")
        )
        for k in _SAFE_FV_KEYS:
            sig[k] = fv.get(k)

        if fv.get("fv_error"):
            fv_failures.append({"ticker": sig["ticker"], "fv_error": fv["fv_error"]})

        if i < len(signals) - 1:
            await asyncio.sleep(FV_INTER_DELAY)

    if fv_failures:
        logger.info(f"[FV] {len(fv_failures)}/{len(signals)} tickers had FV errors: "
                     f"{[f['ticker'] for f in fv_failures]}")

    # Step 4.5: Near-Support enrichment (Chartink Live Feed integration)
    # ─────────────────────────────────────────────────────────────────────
    # Adds multi-touch horizontal support detection fields to every signal:
    #   near_support                  (bool)
    #   near_support_level            (float | None)
    #   near_support_touches          (int | None)
    #   near_support_proximity_pct    (float | None)
    #   near_support_lookback         (int | None)
    #   near_support_first_touch_date (str | None)
    #   near_support_last_touch_date  (str | None)
    #   near_support_score            (float | None)
    #
    # Runs at BOTH 120-day (short) and 500-day (long) lookbacks. Picks the
    # best (highest score) candidate per ticker.  Disabled when
    # config.json's "chartink_enrich_near_support" is False (default True).
    # The chartink_watcher inherits this for free — it calls _run_screen_pipeline
    # without changes, so /api/chartink-screener/results automatically serves
    # signals carrying these fields for the UI's "Near Support" column.
    near_support_enrich_meta = None
    if _NS_ENRICH_AVAILABLE and signals:
        try:
            # Read the master toggle from config.json (default True)
            _ns_enabled = True
            try:
                import os as _os, json as _json
                _cfg_p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "config.json")
                if _os.path.exists(_cfg_p):
                    with open(_cfg_p, "r") as _f:
                        _cfg = _json.load(_f)
                    _ns_enabled = bool(_cfg.get("chartink_enrich_near_support", True))
            except Exception:
                pass

            near_support_enrich_meta = await asyncio.to_thread(
                enrich_signals_with_near_support,
                signals,
                # Use defaults from the enricher module — they match
                # near_support_router.py (120, 500, 3, 0.7, 2.0, 1.5).
                # Callers can override via kwargs if needed in the future.
            )
            _ns_count = near_support_enrich_meta.get("enriched_count", 0)
            _ns_dur   = near_support_enrich_meta.get("duration_sec", 0)
            logger.info(f"[NearSupport] {_ns_count}/{len(signals)} signals "
                        f"are near a multi-touch support (took {_ns_dur}s)")
        except Exception as _ns_exc:
            logger.error(f"[NearSupport] enrichment failed: {_ns_exc}")
            near_support_enrich_meta = {"error": str(_ns_exc)}

    # Step 4: Partition — apply user-configured Prime FV Gap tolerance.
    # If the caller did NOT explicitly pass a threshold, fall back to the
    # value persisted in config.json (set via the UI input box → PUT /api/config).
    # This lets the Chartink watcher inherit the user's global setting without
    # needing any code changes to chartink_watcher.py.
    if prime_fv_gap_pct is None:
        prime_fv_gap_pct = _load_prime_fv_gap_pct_from_config()
    logger.info(f"[PrimeFvGap] Threshold = {prime_fv_gap_pct}% "
                f"(ticker is Prime if cp < fv * (1 + {prime_fv_gap_pct}/100))")

    prime_targets = []
    other_signals = []

    for sig in signals:
        fv = sig.get("composite_fair_price")
        cp = sig.get("current_price")
        if _is_prime_target(fv, cp, prime_fv_gap_pct):
            prime_targets.append(sig)
        else:
            other_signals.append(sig)

    # Step 5: GSheet logging is handled by chartink_watcher._log_primes_to_gsheet
    # which pushes the FULL daily-accumulated prime set after every scan.
    # The per-run logger below is intentionally disabled to avoid partial/duplicate rows.
    # asyncio.create_task(_log_signals_to_gsheet(signals, prime_targets))

    # Sanitize the WHOLE payload before returning — this is the data that
    # chartink_watcher caches and serves via /api/chartink-screener/results.
    # If even ONE NaN slips through here, that endpoint crashes with
    # "ValueError: Out of range float values are not JSON compliant".
    # ── Resolve Momentum Sweep config (for echo in response) ───────────
    _ms_enabled = _load_momentum_sweep_enabled()
    # When the master toggle is OFF, drop the captured momentum sweep
    # candidates so the UI hides the section cleanly.
    if not _ms_enabled:
        momentum_sweep = []
    # Sort: highest momentum_score first (strongest setup on top)
    momentum_sweep.sort(key=lambda r: r.get("momentum_score", 0.0), reverse=True)

    return _json_safe({
        "signals":             signals,
        "prime_targets":       prime_targets,
        "other_signals":       other_signals,
        "undervalued_signals": prime_targets,
        "filtered_by_trend":   filtered_by_trend,
        "filtered_by_pullback": filtered_by_pullback,   # NEW: pullback-filtered signals (when filter_pullback=True)
        "failed":              failed,
        "fv_failures":         fv_failures,
        # ── Momentum Sweep bucket (separate from prime/other) ────────────
        "momentum_sweep":      momentum_sweep,
        "momentum_sweep_count": len(momentum_sweep),
        "momentum_sweep_enabled": _ms_enabled,
        "momentum_sweep_target_pct": _load_momentum_sweep_target_pct(),
        "momentum_sweep_min_ema9_dist_pct": _load_momentum_sweep_min_ema9_dist_pct(),
        "momentum_sweep_max_ema9_dist_pct": _load_momentum_sweep_max_ema9_dist_pct(),
        "count":               len(signals),
        "prime_count":         len(prime_targets),
        "other_count":         len(other_signals),
        "undervalued_count":   len(prime_targets),
        "interval":            interval,
        # Echo the threshold that was actually applied so the frontend can
        # display it in the table header (e.g., "PRIME TARGETS (FV GAP ≥ −10%)")
        "prime_fv_gap_pct":    prime_fv_gap_pct,
        # Echo the pullback config that was actually applied (kwargs override
        # → config.json default) so the frontend can display the active filter
        # state in the Chartink panel header.
        "pullback_lookback_days":  pullback_lookback_days,
        "pullback_threshold_pct":  pullback_threshold_pct,
        "filter_pullback":         filter_pullback,
        "pullback_filtered_count": len(filtered_by_pullback),
        # Echo the Volatility-Adjusted Target config that was actually applied
        # (kwargs override → config.json default) so the frontend can display
        # "Target Mode: ATR ×1.5 (2-8%)" in the Chartink panel header.
        "vol_target_mode":           vol_target_mode,
        "vol_target_atr_period":     vol_target_atr_period,
        "vol_target_multiplier":     vol_target_multiplier,
        "vol_target_floor_pct":      vol_target_floor_pct,
        "vol_target_cap_pct":        vol_target_cap_pct,
        # Echo the Require Uptrend filter state so the frontend can show
        # "Uptrend: ON" or "Uptrend: OFF (accepting all crossovers)" in the
        # Chartink panel header.
        "require_uptrend":           require_uptrend,
        # Near-Support + Swing-Pullback enrichment summary — useful for
        # the Chartink UI header to show count badges.
        "near_support_count":         sum(1 for s in signals if s.get("near_support") is True),
        "swing_pullback_count":       sum(1 for s in signals if s.get("swing_pullback") is True),
        "support_and_swing_count":    sum(1 for s in signals if s.get("support_and_swing") is True),
        "near_support_enrich_meta":   near_support_enrich_meta,
    })


# ─────────────────────────────────────────────────────────────
#  CSV PARSING HELPER
# ─────────────────────────────────────────────────────────────
def _parse_csv_tickers(content: bytes) -> Dict:
    df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")

    symbol_col = None
    _TARGET_NAMES = {"symbol", "ticker", "stock", "ticker symbol", "stock symbol"}
    for col in df.columns:
        if col.strip().lower() in _TARGET_NAMES:
            symbol_col = col
            break

    if symbol_col is None:
        raise HTTPException(
            400,
            f"No Symbol/Ticker/Stock column found. Available: {list(df.columns)}",
        )

    tickers = (
        df[symbol_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".NS", "", regex=False)
        .str.replace(".BO", "", regex=False)
    )
    tickers = tickers[tickers != ""].unique().tolist()

    return {"tickers": tickers, "total_rows": len(df), "found_column": symbol_col}


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@router.get("/api/ema9/tickers")
async def ema9_tickers(q: str = "", source: str = "fno"):
    df = _load_all_df() if source == "all" else _load_fno_df()
    if df.empty:
        return _json_safe({"tickers": [], "total": 0})
    q = q.strip().upper()
    if q:
        mask = df["symbol"].str.contains(q, na=False)
        if "company_name" in df.columns:
            mask = mask | df["company_name"].str.upper().str.contains(q, na=False)
        filtered = df[mask].head(30)
    else:
        filtered = df.head(30)
    return _json_safe({
        "total": len(df),
        "tickers": [
            {"symbol": row["symbol"], "name": row["company_name"] or row["symbol"]}
            for _, row in filtered.iterrows()
        ],
    })


@router.get("/api/ema9/tickers/list")
async def ema9_tickers_list(source: str = "fno"):
    df = _load_all_df() if source == "all" else _load_fno_df()
    return _json_safe({
        "source":  source,
        "total":   len(df),
        "symbols": df["symbol"].tolist(),
        "tickers": [
            {"symbol": row["symbol"], "name": row["company_name"] or row["symbol"]}
            for _, row in df.iterrows()
        ],
    })


@router.post("/api/ema9/screen")
async def ema9_screen(req: Ema9ScreenRequest):
    result = await _run_screen_pipeline(
        tickers=req.tickers,
        interval=req.interval,
        lookback_days=req.lookback_days,
        max_candles_ago=req.max_candles_ago,
        require_uptrend=req.require_uptrend,
        # Pass through the optional per-request override. When null (the default),
        # _run_screen_pipeline falls back to config.json's value, which is what
        # the UI input box updates. So most calls leave this null.
        prime_fv_gap_pct=req.prime_fv_gap_pct,
        # Same pattern for pullback — pass through, fall back to config.json.
        pullback_lookback_days=req.pullback_lookback_days,
        pullback_threshold_pct=req.pullback_threshold_pct,
        filter_pullback=req.filter_pullback,
        # Same pattern for vol-target — pass through, fall back to config.json.
        vol_target_mode=req.vol_target_mode,
        vol_target_atr_period=req.vol_target_atr_period,
        vol_target_multiplier=req.vol_target_multiplier,
        vol_target_floor_pct=req.vol_target_floor_pct,
        vol_target_cap_pct=req.vol_target_cap_pct,
    )
    return _json_safe(result)


@router.post("/api/ema9/upload-csv")
async def ema9_upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are accepted.")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 10 MB).")

    csv_info = _parse_csv_tickers(content)
    tickers  = csv_info["tickers"]

    if not tickers:
        raise HTTPException(400, "No valid tickers found in CSV.")

    result = await _run_screen_pipeline(
        tickers=tickers,
        interval="1d",
        lookback_days=180,
        max_candles_ago=10,
        # require_uptrend defaults to None → falls back to config.json's
        # "ema9_require_uptrend" key (set via the Uptrend toggle on the
        # Screener tab). CSV upload now respects the user's global setting
        # instead of hardcoding True.
        require_uptrend=None,
    )

    result["csv_info"] = {
        "filename":       file.filename,
        "total_rows":     csv_info["total_rows"],
        "parsed_tickers": len(tickers),
        "found_column":   csv_info["found_column"],
    }

    return _json_safe(result)


# ─────────────────────────────────────────────────────────────
#  SAMPLE CSV DOWNLOAD
#  Serves a small reference CSV so users can see the expected
#  column format (Symbol / Company Name) before uploading their
#  own screener export.
# ─────────────────────────────────────────────────────────────
import pathlib as _pathlib

_SAMPLE_CSV_PATHS = [
    _pathlib.Path(__file__).parent / "sample_tickers.csv",          # same dir as router
    _pathlib.Path.cwd() / "sample_tickers.csv",                      # project root
    _pathlib.Path.cwd() / "download" / "sample_tickers.csv",         # download/ subdir
    _pathlib.Path("/home/z/my-project/download/sample_tickers.csv"), # absolute fallback
]


@router.get("/sample_tickers.csv")
async def download_sample_csv():
    """Download a sample CSV showing the expected ticker-list format."""
    for p in _SAMPLE_CSV_PATHS:
        if p.exists() and p.is_file():
            return FileResponse(
                path=str(p),
                media_type="text/csv",
                filename="sample_tickers.csv",
                headers={"Content-Disposition": 'attachment; filename="sample_tickers.csv"'},
            )
    raise HTTPException(404, "sample_tickers.csv not found on server")


@router.get("/api/ema9/quick-scan/{ticker}")
async def ema9_quick_scan(ticker: str):
    ticker = ticker.strip().upper().replace(".NS", "").replace(".BO", "")

    if not ticker or not ticker.isalnum():
        raise HTTPException(400, "Invalid ticker symbol.")

    result = await asyncio.to_thread(_screen_ticker, ticker)
    result["interval"] = "1d"

    fv = await asyncio.to_thread(
        _enrich_fair_value, ticker, result.get("current_price")
    )
    for k in _SAFE_FV_KEYS:
        result[k] = fv.get(k)

    if result.get("status") == "SIGNAL":
        fv_price = result.get("composite_fair_price")
        cp       = result.get("current_price")
        # Apply the same user-configured Prime FV Gap tolerance as the batch
        # pipeline (_run_screen_pipeline). Reads from config.json so the
        # threshold set in the UI input box applies to Quick Scan too.
        threshold_pct = _load_prime_fv_gap_pct_from_config()
        if _is_prime_target(fv_price, cp, threshold_pct):
            result["signal_type"] = "PRIME"
        else:
            result["signal_type"] = "TECHNICAL_ONLY"
        # Echo the threshold so the frontend Quick Scan card can show it.
        result["prime_fv_gap_pct"] = threshold_pct
    elif result.get("status") == "NO_SIGNAL":
        result["signal_type"] = "NO_SIGNAL"
    else:
        result["signal_type"] = result.get("status", "UNKNOWN")

    return _json_safe(result)


# ─────────────────────────────────────────────────────────────
#  SCAN ALL NSE F&O TICKERS — Full Universe Screener
# ─────────────────────────────────────────────────────────────
#  This endpoint loads ALL tickers from tickers-healthy-strict-4.csv
#  (the pre-filtered list of actively-traded NSE tickers — no delisted,
#  no suspended, no illiquid names) and runs the FULL EMA9 pipeline on
#  every single one — no pre-filter, no Chartink clause gate.
#
#  Why this exists:
#    The Chartink Live Feed only scans tickers that pass a Chartink
#    scan clause (e.g. "close > 9 EMA"). That means stocks BELOW 9EMA
#    that are about to cross over are NEVER caught. This endpoint
#    processes the ENTIRE healthy universe so you catch every fresh 9EMA
#    crossover, prime target, and momentum sweep candidate.
#
#  Ticker source priority:
#    1. data/tickers-healthy-strict-4.csv (pre-filtered, no dead names)
#    2. data/tickers.csv (F&O list, fallback)
#
#  Performance:
#    • yfinance data is cached (4hr TTL) — repeat runs are instant.
#    • FV enrichment is cached (12hr TTL).
#    • Near-support enrichment runs on SIGNAL results only.
#    • First run with ~307 tickers: ~2-3 min (async concurrent).
#    • Subsequent runs: <10 seconds (all cache hits).
# ─────────────────────────────────────────────────────────────

@router.post("/api/ema9/screen-all-nse")
async def ema9_screen_all_nse():
    """
    Scan the ENTIRE healthy NSE ticker universe through the full 9EMA
    pipeline. Returns the same structure as /api/ema9/screen.

    Ticker source: tickers-healthy-strict-4.csv (falls back to tickers.csv).

    No input body needed — tickers loaded from local CSV.
    """
    # Load healthy tickers (pre-filtered, no delisted/suspended names)
    source_df = _load_healthy_df()

    if source_df.empty:
        raise HTTPException(
            404,
            "No ticker CSV found. Place tickers-healthy-strict-4.csv or "
            "tickers.csv in the data/ directory next to the server."
        )

    tickers = source_df["symbol"].dropna().str.strip().str.upper().unique().tolist()

    if not tickers:
        raise HTTPException(404, "Ticker CSV is empty — no symbols to scan.")

    logger.info(f"[ScreenAllNSE] Starting full-universe scan: {len(tickers)} tickers from {HEALTHY_CSV}")

    # Run the FULL pipeline — same as /api/ema9/screen but with all tickers.
    # All config settings (uptrend, FV gap, pullback, vol target, momentum sweep)
    # are inherited from config.json automatically (None = fall back to config).
    result = await _run_screen_pipeline(
        tickers=tickers,
        interval="1d",
        lookback_days=180,
        max_candles_ago=10,
        require_uptrend=None,          # inherit from config
        prime_fv_gap_pct=None,         # inherit from config
        pullback_lookback_days=None,   # inherit from config
        pullback_threshold_pct=None,   # inherit from config
        filter_pullback=None,          # inherit from config
        vol_target_mode=None,          # inherit from config
        vol_target_atr_period=None,    # inherit from config
        vol_target_multiplier=None,    # inherit from config
        vol_target_floor_pct=None,     # inherit from config
        vol_target_cap_pct=None,       # inherit from config
    )

    # Add metadata so the frontend knows this was a full-universe scan
    result["scan_mode"] = "all_nse"
    result["scan_source"] = HEALTHY_CSV
    result["total_tickers_scanned"] = len(tickers)
    result["last_run_at"] = dt_module.datetime.now(IST if 'IST' in dir() else None).isoformat() if 'IST' in dir() else dt_module.datetime.utcnow().isoformat()

    logger.info(f"[ScreenAllNSE] Complete: {len(tickers)} scanned → "
                f"{result.get('prime_count', 0)} prime, "
                f"{result.get('other_count', 0)} other, "
                f"{result.get('momentum_sweep_count', 0)} momentum sweep, "
                f"{len(result.get('failed', []))} failed")

    return _json_safe(result)


@router.get("/api/ema9/fair-value/{ticker}")
async def ema9_fair_value(ticker: str):
    ticker = ticker.strip().upper().replace(".NS", "").replace(".BO", "")

    if not ticker or not ticker.isalnum():
        raise HTTPException(400, "Invalid ticker symbol.")

    if not _FV_AVAILABLE:
        raise HTTPException(503, "Fair Value engine (sma_router) is not available.")

    try:
        fv = await asyncio.to_thread(_enrich_fair_value, ticker, None)
    except Exception as exc:
        logger.error(f"[FV-OnDemand] {ticker}: {exc}")
        raise HTTPException(500, f"FV fetch failed: {str(exc)[:200]}")

    if fv.get("composite_fair_price") is None:
        error_msg = fv.get("fv_error", "UNKNOWN")
        logger.warning(f"[FV-OnDemand] {ticker}: No FV available. Error: {error_msg}")
        return _json_safe({
            "ticker":  ticker,
            "status":  "NO_FV",
            "fv_error": error_msg,
            **fv,
        })

    return _json_safe({
        "ticker":  ticker,
        "status":  "OK",
        **fv, 
    })