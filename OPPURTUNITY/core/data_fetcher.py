"""
data_fetcher.py — OHLCV Data Ingestion, Normalization & Fundamental Cache
"""
import os
import json
import asyncio
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent.parent
TICKERS_CSV = BASE_DIR / "tickers.csv"
FUNDAMENTALS_CACHE = BASE_DIR / "fundamentals_cache.json"

_DATA_CACHE: Dict[str, pd.DataFrame] = {}
_DATA_CACHE_TIMES: Dict[str, float] = {}
_DOWNLOAD_LOCK = threading.RLock()
CACHE_TTL = 300  # 5 minutes in memory


def market_now() -> datetime:
    """Current India Standard Time (IST)."""
    return datetime.now(timezone(timedelta(hours=5, minutes=30)))


# Symbol aliases for corporate ticker renames on NSE
SYMBOL_ALIASES = {
    "GUJGASLTD": "GUJENERGY",
    "GUJGAS": "GUJENERGY",
}


def normalize_ticker(symbol: str) -> str:
    """Normalize raw symbol into NSE Yahoo Finance ticker (e.g. TCS -> TCS.NS)."""
    s = str(symbol).strip().upper()
    if s.startswith("^"):
        return s

    # Strip exchange suffix for alias lookup
    core = s
    suffix = ".NS"
    if s.endswith(".NS"):
        core = s[:-3]
    elif s.endswith(".BO"):
        core = s[:-3]
        suffix = ".BO"

    if core in SYMBOL_ALIASES:
        core = SYMBOL_ALIASES[core]

    return f"{core}{suffix}"


def bare_symbol(ticker: str) -> str:
    """Strip exchange suffix for display (e.g. TCS.NS -> TCS)."""
    s = str(ticker).strip().upper()
    if s.endswith(".NS"):
        s = s[:-3]
    elif s.endswith(".BO"):
        s = s[:-3]
    return SYMBOL_ALIASES.get(s, s)


def load_ticker_registry() -> List[Dict[str, Any]]:
    """Load the 307 vetted healthy stocks from tickers.csv."""
    if not TICKERS_CSV.exists():
        return []
    try:
        df = pd.read_csv(TICKERS_CSV)
        # Standardize column names
        cols = {c.strip().lower(): c for c in df.columns}
        sym_col = cols.get("symbol", "SYMBOL")
        sec_col = cols.get("security", "SECURITY")
        sr_col = cols.get("sr. no.", cols.get("sr_no", "Sr. No."))

        records = []
        for _, row in df.iterrows():
            sym = str(row[sym_col]).strip() if pd.notna(row.get(sym_col)) else ""
            if not sym or sym.lower() == "nan":
                continue
            records.append({
                "sr_no": int(row[sr_col]) if sr_col in row and pd.notna(row[sr_col]) else len(records) + 1,
                "security": str(row[sec_col]).strip() if sec_col in row and pd.notna(row[sec_col]) else sym,
                "symbol": sym,
                "ticker": normalize_ticker(sym),
            })
        return records
    except Exception as e:
        print(f"Error loading tickers.csv: {e}")
        return []


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure standard OHLCV schema, sorted index, deduplicated bars."""
    if df is None or df.empty:
        return pd.DataFrame()
    bars = df.copy()

    # Handle MultiIndex columns from yfinance
    if isinstance(bars.columns, pd.MultiIndex):
        price_level = next((i for i in range(bars.columns.nlevels)
                            if {str(v).lower() for v in bars.columns.get_level_values(i)}
                            >= {"open", "high", "low", "close"}), None)
        if price_level is not None:
            bars.columns = bars.columns.get_level_values(price_level)
        else:
            bars.columns = [str(col[0]) for col in bars.columns]

    # De-duplicate columns if any repeat
    bars = bars.loc[:, ~bars.columns.duplicated(keep="first")]

    # Normalize column names
    canonical = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    bars = bars.rename(columns={c: canonical.get(str(c).strip().lower(), c) for c in bars.columns})

    required = ["Open", "High", "Low", "Close"]
    if any(c not in bars.columns for c in required):
        return pd.DataFrame()

    if "Volume" not in bars.columns:
        bars["Volume"] = 0.0

    # Ensure numeric conversion
    for col in required + ["Volume"]:
        bars[col] = pd.to_numeric(bars[col], errors="coerce")

    bars = bars.dropna(subset=required)
    bars = bars[(bars["Close"] > 0) & (bars["High"] >= bars["Low"])]
    bars.index = pd.to_datetime(bars.index, errors="coerce")
    bars = bars[~bars.index.isna()]
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()

    return bars[["Open", "High", "Low", "Close", "Volume"]]


def fetch_ticker_ohlcv(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch daily OHLCV bars using isolated yfinance Ticker instances with caching."""
    norm = normalize_ticker(ticker)
    now_ts = market_now().timestamp()

    with _DOWNLOAD_LOCK:
        if norm in _DATA_CACHE and (now_ts - _DATA_CACHE_TIMES.get(norm, 0)) < CACHE_TTL:
            return _DATA_CACHE[norm].copy()

    # 1. Primary: yf.Ticker(norm).history() provides isolated per-ticker data and prevents multi-thread cross-contamination
    try:
        t = yf.Ticker(norm)
        data = t.history(period=period, auto_adjust=True, timeout=15)
        cleaned = clean_ohlcv(data)
        if not cleaned.empty:
            with _DOWNLOAD_LOCK:
                _DATA_CACHE[norm] = cleaned
                _DATA_CACHE_TIMES[norm] = now_ts
            return cleaned.copy()
    except Exception:
        pass

    # 2. Fallback: single-threaded download with lock
    with _DOWNLOAD_LOCK:
        try:
            data = yf.download(norm, period=period, interval="1d", progress=False, auto_adjust=True, threads=False, timeout=15)
            cleaned = clean_ohlcv(data)
            if not cleaned.empty:
                _DATA_CACHE[norm] = cleaned
                _DATA_CACHE_TIMES[norm] = now_ts
                return cleaned.copy()
        except Exception as e:
            print(f"Failed to fetch OHLCV for {norm}: {e}")

    return pd.DataFrame()


async def fetch_ticker_ohlcv_async(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Async wrapper for non-blocking OHLCV fetch."""
    return await asyncio.to_thread(fetch_ticker_ohlcv, ticker, period)


async def fetch_batch_ohlcv_async(tickers: List[str], period: str = "1y", max_concurrency: int = 8) -> Dict[str, pd.DataFrame]:
    """Fetch batch OHLCV asynchronously with concurrency limit."""
    sem = asyncio.Semaphore(max_concurrency)
    results = {}

    async def _worker(sym: str):
        async with sem:
            df = await fetch_ticker_ohlcv_async(sym, period=period)
            if not df.empty:
                results[bare_symbol(sym)] = df

    tasks = [_worker(t) for t in tickers]
    await asyncio.gather(*tasks, return_exceptions=True)
    return results


def get_market_context() -> Dict[str, Any]:
    """Analyze Nifty 50 (^NSEI) trend and regime."""
    from core.strategy import compute_trend
    df = fetch_ticker_ohlcv("^NSEI", period="1y")
    if df.empty or len(df) < 50:
        return {"trend_regime": "UNKNOWN", "as_of": str(market_now().date())}
    trend = compute_trend(df)
    trend["as_of"] = str(df.index[-1].date())
    trend["close"] = round(float(df["Close"].iloc[-1]), 2)
    return trend


def load_cached_fundamentals() -> Dict[str, Any]:
    """Load pre-evaluated fundamental health checks."""
    if not FUNDAMENTALS_CACHE.exists():
        return {}
    try:
        with open(FUNDAMENTALS_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
        mapping = {}
        for item in results:
            sym = item.get("symbol")
            if sym:
                mapping[sym.upper()] = item
        return mapping
    except Exception as e:
        print(f"Error loading fundamentals cache: {e}")
        return {}
