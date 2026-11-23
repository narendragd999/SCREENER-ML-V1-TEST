"""
Core Data Fetcher & Ticker Registry for Indian Equity (NSE)
===========================================================
- High-resilience yfinance downloader with rate-limit protection.
- In-memory cache with configurable TTL to prevent redundant network calls.
- Fast symbol resolution for Nifty F&O and NSE All universes.
"""

import os
import time
import math
import asyncio
import datetime as dt
import threading
from typing import Callable, Dict, List, Optional, Tuple, Any
import pandas as pd
import yfinance as yf
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
FNO_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tickers.csv")
ALL_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tickers_all.csv")

# Memory Cache: (symbol, interval, lookback_days) -> (timestamp, DataFrame)
_PRICE_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}
CACHE_TTL_SECONDS = 600  # 10 minutes cache during session
# yfinance 0.2.x stores downloads in process-global dictionaries, even when
# threads=False. Overlapping calls can return a different stock's history.
_DOWNLOAD_LOCK = threading.Lock()
_CACHE_LOCK = threading.RLock()
_MAX_DOWNLOAD_CHUNK_SIZE = 4


class HistoryUnavailableError(RuntimeError):
    """The provider did not supply valid completed bars for this request."""


_FNO_DF: Optional[pd.DataFrame] = None
_ALL_DF: Optional[pd.DataFrame] = None

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def market_now() -> dt.datetime:
    return dt.datetime.now(IST)


def clean_ohlcv(df: pd.DataFrame, interval: str = "1d", completed_only: bool = True,
                now: Optional[dt.datetime] = None) -> pd.DataFrame:
    """Validate bars and exclude a daily candle while the NSE session is open.

    Keep exchange-local dates when removing time zones; UTC conversion can
    otherwise move Indian daily bars to the preceding calendar day.
    """
    if df is None:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    bars = df.copy()
    attrs = dict(df.attrs)
    original = len(bars)
    required = ["Open", "High", "Low", "Close"]
    if isinstance(bars.columns, pd.MultiIndex):
        price_level = next((i for i in range(bars.columns.nlevels)
                            if {str(v).lower() for v in bars.columns.get_level_values(i)}
                            >= {"open", "high", "low", "close"}), None)
        other_levels = [i for i in range(bars.columns.nlevels) if i != price_level]
        if price_level is None or any(len(bars.columns.get_level_values(i).unique()) != 1 for i in other_levels):
            return pd.DataFrame(columns=required + ["Volume"])
        bars.columns = bars.columns.get_level_values(price_level)
    canonical = {c.lower(): c for c in required + ["Volume"]}
    bars = bars.rename(columns={c: canonical.get(str(c).strip().lower(), c) for c in bars.columns})
    if any(c not in bars for c in required):
        return pd.DataFrame(columns=required + ["Volume"])
    # Batch downloads align symbols on a union of dates. Entirely empty OHLC
    # rows are alignment padding (often before an IPO), not corrupt candles.
    padding = bars[required].isna().all(axis=1)
    padding_count = int(padding.sum())
    bars = bars[~padding].copy()
    original -= padding_count
    idx = pd.to_datetime(bars.index, errors="coerce")
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(IST).tz_localize(None)
    bars.index = idx
    bars = bars[~bars.index.isna()].sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    missing_volume = "Volume" not in bars
    if missing_volume:
        bars["Volume"] = 0.0
    for col in required + ["Volume"]:
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    finite = np.isfinite(bars[required]).all(axis=1)
    valid = (finite & (bars[required] > 0).all(axis=1)
             & (bars.High >= bars[required].max(axis=1))
             & (bars.Low <= bars[required].min(axis=1)))
    bars = bars[valid].copy()
    bars["Volume"] = bars.Volume.replace([np.inf, -np.inf], np.nan).fillna(0).clip(lower=0)
    invalid_count = original - len(bars)
    incomplete_count = 0
    if completed_only and interval == "1d":
        current = now or market_now()
        if current.tzinfo is None:
            current = current.replace(tzinfo=IST)
        current = current.astimezone(IST)
        # Weekend rows can be legitimate special sessions (e.g. Muhurat).
        # Preserve historical provider bars; today's weekend session is only
        # considered complete on the next day because its schedule can differ.
        before_close = current.time().replace(tzinfo=None) < dt.time(15, 30) or current.weekday() >= 5
        dates = bars.index.date
        keep = dates < current.date() if before_close else dates <= current.date()
        incomplete_count = len(bars) - int(keep.sum())
        bars = bars[keep].copy()
    bars.attrs.update(attrs)
    bars.attrs.update({"invalid_bars_removed": attrs.get("invalid_bars_removed", 0) + invalid_count,
                       "empty_alignment_rows_removed": attrs.get("empty_alignment_rows_removed", 0) + padding_count,
                       "incomplete_bars_removed": attrs.get("incomplete_bars_removed", 0) + incomplete_count,
                       "volume_available": attrs.get("volume_available", not missing_volume),
        "completed_only": completed_only})
    return bars


def yahoo_symbol(ticker: str) -> str:
    sym = normalize_ticker(ticker)
    return sym if sym.startswith("^") or sym.endswith(".BO") else f"{sym}.NS"


def _usable_cache(timestamp: float, now: float, interval: str) -> bool:
    if now - timestamp >= CACHE_TTL_SECONDS:
        return False
    current = dt.datetime.fromtimestamp(now, IST)
    fetched = dt.datetime.fromtimestamp(timestamp, IST)
    # Fetch a final quote after the close instead of reusing a partial-session download.
    return not (interval == "1d" and current.date() == fetched.date()
                and current.time() >= dt.time(15, 30) > fetched.time())


def _extract_symbol_frame(raw: Optional[pd.DataFrame], expected_symbol: str,
                          allow_flat: bool = True) -> pd.DataFrame:
    """Select the requested Yahoo symbol before discarding column labels.

    Yahoo has returned both (Price, Ticker) and (Ticker, Price) layouts. Never
    flatten the ticker level first: that can relabel another stock's prices.
    """
    if raw is None or raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        price_level = next((i for i in range(frame.columns.nlevels)
                            if {str(v).strip().lower() for v in frame.columns.get_level_values(i)}
                            >= {"open", "high", "low", "close"}), None)
        if price_level is None:
            return pd.DataFrame()
        expected = expected_symbol.strip().upper()
        selected_level = selected_label = None
        for level in range(frame.columns.nlevels):
            if level == price_level:
                continue
            labels = frame.columns.get_level_values(level).unique()
            match = next((v for v in labels if str(v).strip().upper() == expected), None)
            if match is not None:
                selected_level, selected_label = level, match
                break
        if selected_level is None:
            return pd.DataFrame()
        frame = frame.xs(selected_label, axis=1, level=selected_level).copy()
        if isinstance(frame.columns, pd.MultiIndex):
            # clean_ohlcv accepts remaining single-valued auxiliary levels.
            frame = clean_ohlcv(frame, completed_only=False)
    elif not allow_flat:
        return pd.DataFrame()
    if not frame.columns.is_unique:
        return pd.DataFrame()
    return frame


def _history_context(frame: pd.DataFrame, symbol: str, source: str,
                     fetched_at: float, lookback_days: int,
                     min_bars: int = 15, warnings: Optional[List[str]] = None,
                     interval: str = "1d") -> pd.DataFrame:
    """Attach provenance and coverage without changing the available prices."""
    result = frame.copy()
    completed = clean_ohlcv(result, interval=interval, completed_only=True)
    history_warnings = list(result.attrs.get("history_warnings", []))
    history_warnings.extend(warnings or [])
    if len(completed) < min_bars:
        history_warnings.append(
            f"Only {len(completed)} completed bars are available; some indicators and setups need more history.")
    result.attrs.update({
        "ticker": symbol,
        "yahoo_symbol": yahoo_symbol(symbol),
        "data_source": source,
        "fetched_at": dt.datetime.fromtimestamp(fetched_at, IST).isoformat(),
        "available_history": {
            "bars": len(completed),
            "start_date": str(completed.index[0].date()) if len(completed) else None,
            "end_date": str(completed.index[-1].date()) if len(completed) else None,
            "requested_lookback_days": lookback_days,
        },
        "history_warnings": list(dict.fromkeys(history_warnings)),
    })
    return result


def _cached_history(symbol: str, interval: str, lookback_days: int, now: float,
                    min_bars: int = 15, allow_cached_shorter: bool = False) -> Optional[pd.DataFrame]:
    prefix = f"{symbol}_{interval}_"
    with _CACHE_LOCK:
        entries = [(key, ts, frame.copy()) for key, (ts, frame) in _PRICE_CACHE.items()
                   if key.startswith(prefix) and _usable_cache(ts, now, interval)]
    candidates = []
    for key, ts, frame in entries:
        try:
            cached_days = int(key[len(prefix):])
        except ValueError:
            continue
        if cached_days < lookback_days and not allow_cached_shorter:
            continue
        # Prefer the exact window, then the nearest wider one, then the widest
        # shorter window when the caller explicitly accepts partial coverage.
        priority = (0, 0) if cached_days == lookback_days else (
            (1, cached_days) if cached_days > lookback_days else (2, -cached_days))
        candidates.append((priority, cached_days, ts, frame))
    for _, cached_days, ts, frame in sorted(candidates, key=lambda item: item[0]):
        if frame.attrs.get("ticker", symbol) != symbol:
            continue
        frame = clean_ohlcv(frame, interval)
        if cached_days > lookback_days:
            start = pd.Timestamp(market_now().date() + dt.timedelta(days=1)
                                 - dt.timedelta(days=lookback_days + 30))
            frame = frame.loc[frame.index >= start].copy()
        if len(frame) < min_bars:
            continue
        warnings = []
        if cached_days < lookback_days:
            warnings.append(f"Using available cached history ({cached_days}-day request) for this analysis.")
        frame = _history_context(frame, symbol, frame.attrs.get("data_source", "memory_cache"),
                                 ts, lookback_days, min_bars, warnings, interval)
        frame.attrs["cache_reused"] = True
        return frame
    return None


def _save_history(symbol: str, interval: str, lookback_days: int,
                  frame: pd.DataFrame, source: str, min_bars: int = 15,
                  warnings: Optional[List[str]] = None) -> pd.DataFrame:
    fetched_at = time.time()
    frame = _history_context(frame, symbol, source, fetched_at, lookback_days, min_bars, warnings, interval)
    with _CACHE_LOCK:
        _PRICE_CACHE[f"{symbol}_{interval}_{lookback_days}"] = (fetched_at, frame.copy())
    return clean_ohlcv(frame, interval)


def normalize_ticker(symbol: str) -> str:
    """Use bare NSE tickers; preserve BSE suffixes and index symbols."""
    s = str(symbol or "").strip().upper()
    if s in ("NAN", "NONE", "NULL"):
        return ""
    if s.endswith(".NS"):
        s = s[:-3]
    return s


def load_ticker_registry(source: str = "fno") -> pd.DataFrame:
    """Load and parse ticker universe from CSV files."""
    global _FNO_DF, _ALL_DF
    if source == "all":
        if _ALL_DF is not None:
            return _ALL_DF
        csv_path = ALL_CSV
    else:
        if _FNO_DF is not None:
            return _FNO_DF
        csv_path = FNO_CSV

    if not os.path.exists(csv_path):
        # Fallback if csv missing: return default liquid Nifty tickers
        fallback_tickers = [
            ("RELIANCE", "Reliance Industries"), ("TCS", "Tata Consultancy Services"),
            ("HDFCBANK", "HDFC Bank Ltd"), ("INFY", "Infosys Ltd"),
            ("ICICIBANK", "ICICI Bank Ltd"), ("BHARTIARTL", "Bharti Airtel Ltd"),
            ("SBIN", "State Bank of India"), ("LICI", "Life Insurance Corp of India"),
            ("ITC", "ITC Ltd"), ("HINDUNILVR", "Hindustan Unilever Ltd"),
            ("LT", "Larsen & Toubro Ltd"), ("BAJFINANCE", "Bajaj Finance Ltd"),
            ("TATAMOTORS", "Tata Motors Ltd"), ("SUNPHARMA", "Sun Pharmaceutical"),
            ("TITAN", "Titan Company Ltd"), ("ADANIENT", "Adani Enterprises Ltd"),
            ("MARUTI", "Maruti Suzuki India"), ("AXISBANK", "Axis Bank Ltd"),
            ("ONGC", "Oil & Natural Gas Corp"), ("NTPC", "NTPC Ltd"),
            ("KALYANKJIL", "Kalyan Jewellers"), ("PERSISTENT", "Persistent Systems"),
            ("COFORGE", "Coforge Ltd"), ("DIXON", "Dixon Technologies"),
            ("POLYCAB", "Polycab India"), ("TRENT", "Trent Ltd")
        ]
        df = pd.DataFrame(fallback_tickers, columns=["symbol", "company_name"])
        return df

    try:
        raw_df = pd.read_csv(csv_path, dtype=str)
        raw_df.columns = [c.strip().lower() for c in raw_df.columns]
        
        name_col = "company_name"
        if "name of company" in raw_df.columns:
            raw_df = raw_df.rename(columns={"name of company": name_col})
        elif "security" in raw_df.columns:
            raw_df = raw_df.rename(columns={"security": name_col})
        elif name_col not in raw_df.columns:
            raw_df[name_col] = ""

        if "symbol" not in raw_df.columns:
            # Check first column
            raw_df = raw_df.rename(columns={raw_df.columns[0]: "symbol"})

        raw_df["symbol"] = raw_df["symbol"].fillna("").map(normalize_ticker)
        raw_df[name_col] = raw_df[name_col].fillna("").astype(str).str.strip()
        
        # Clean empty
        clean_df = raw_df[raw_df["symbol"].str.len() > 0][["symbol", name_col]].drop_duplicates(subset=["symbol"])
        clean_df = clean_df.reset_index(drop=True)

        if source == "all":
            _ALL_DF = clean_df
            return _ALL_DF
        else:
            _FNO_DF = clean_df
            return _FNO_DF
    except Exception as e:
        print(f"[DataFetcher] Error loading {csv_path}: {e}")
        return pd.DataFrame(columns=["symbol", "company_name"])


def get_tickers_list(source: str = "fno", query: str = "", limit: int = 50) -> List[Dict[str, str]]:
    """Return searchable list of tickers."""
    df = load_ticker_registry(source)
    if df.empty:
        return []
    
    q = query.strip().upper()
    if q:
        mask = df["symbol"].str.contains(q, na=False, regex=False)
        if "company_name" in df.columns:
            mask = mask | df["company_name"].str.upper().str.contains(q, na=False, regex=False)
        filtered = df[mask].head(limit)
    else:
        filtered = df.head(limit)

    results = []
    for _, row in filtered.iterrows():
        sym = row["symbol"]
        name = row["company_name"] if row["company_name"] else sym
        results.append({"symbol": sym, "name": name})
    return results


def fetch_ticker_ohlcv(
    ticker: str,
    interval: str = "1d",
    lookback_days: int = 365,
    force_refresh: bool = False,
    *,
    min_bars: int = 15,
    allow_cached_shorter: bool = False,
    raise_on_error: bool = False,
) -> Optional[pd.DataFrame]:
    """Fetch verified history, preserving short listings for partial research.

    min_bars governs cache reuse; a successful provider response with fewer
    bars remains available with a coverage warning. Only provider/data failures
    raise HistoryUnavailableError when raise_on_error is requested.
    """
    sym = normalize_ticker(ticker)
    if not sym:
        if raise_on_error:
            raise HistoryUnavailableError("A stock symbol is required")
        return None
    if not force_refresh:
        cached = _cached_history(sym, interval, lookback_days, time.time(), min_bars,
                                 allow_cached_shorter)
        if cached is not None:
            return cached
    yf_symbol = yahoo_symbol(sym)
    end_date = market_now().date() + dt.timedelta(days=1)
    start_date = end_date - dt.timedelta(days=lookback_days + 30)
    raw = None
    primary_error = None
    # A chart lookup must not queue behind an entire scan. Downloads stay
    # synchronous because Yahoo also touches shared error state in history().
    acquired = _DOWNLOAD_LOCK.acquire(timeout=2.0)
    if acquired:
        try:
            raw = yf.download(yf_symbol, start=str(start_date), end=str(end_date),
                              interval=interval, progress=False, auto_adjust=True,
                              threads=False, timeout=15)
        except Exception as exc:
            primary_error = str(exc)
        finally:
            _DOWNLOAD_LOCK.release()
    else:
        primary_error = "A batch history download is already running"
    try:
        frame = clean_ohlcv(_extract_symbol_frame(raw, yf_symbol), interval, completed_only=False)
        if not clean_ohlcv(frame, interval).empty:
            return _save_history(sym, interval, lookback_days, frame, "yfinance.download", min_bars)
    except Exception as exc:
        primary_error = str(exc)
    try:
        raw = yf.Ticker(yf_symbol).history(start=str(start_date), end=str(end_date),
                                          interval=interval, auto_adjust=True, timeout=15)
        frame = clean_ohlcv(_extract_symbol_frame(raw, yf_symbol), interval, completed_only=False)
        if not clean_ohlcv(frame, interval).empty:
            warning = "Primary history download was unavailable; recovered with individual ticker history."
            return _save_history(sym, interval, lookback_days, frame,
                                 "yfinance.Ticker.history", min_bars, [warning])
    except Exception as exc:
        primary_error = str(exc)
    if primary_error:
        print(f"[DataFetcher] History unavailable for {sym}: {primary_error}")
    if raise_on_error:
        raise HistoryUnavailableError(
            f"Daily price history for {sym} is temporarily unavailable. Retry and check your internet connection.")
    return None


def fetch_batch_ohlcv(
    tickers: List[str],
    interval: str = "1d",
    lookback_days: int = 365,
    chunk_size: int = 4,
    force_refresh: bool = False,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Download small groups, reporting each network attempt and completed ticker.

    With a progress callback, use one ticker per request so slow responses do
    not hide the completed work of an entire universe. Keep Yahoo's internal
    threads disabled: individual chart fallback errors also touch its shared
    dictionaries and can prematurely finish a threaded download.
    """
    clean_tickers = [normalize_ticker(t) for t in tickers if normalize_ticker(t)]
    clean_tickers = list(dict.fromkeys(clean_tickers))  # Preserve order, deduplicate
    if not clean_tickers:
        return {}
    chunk_size = 1 if progress is not None else max(1, min(int(chunk_size), _MAX_DOWNLOAD_CHUNK_SIZE))

    results: Dict[str, pd.DataFrame] = {}
    missing: List[str] = []
    now = time.time()
    completed = failed_count = 0

    def report(ticker=None):
        if progress is not None:
            try:
                progress({"phase": "download", "completed": completed,
                          "total": len(clean_tickers), "ticker": ticker,
                          "failed_count": failed_count})
            except Exception:
                # Display callbacks cannot invalidate correctly fetched prices.
                pass

    report()

    # Check cache first
    for t in clean_tickers:
        if not force_refresh:
            cached = _cached_history(t, interval, lookback_days, now)
            if cached is not None:
                results[t] = cached
                completed += 1
                report(t)
                continue
        missing.append(t)

    if not missing:
        return results

    end_date = market_now().date() + dt.timedelta(days=1)
    start_date = end_date - dt.timedelta(days=lookback_days + 30)

    n_chunks = math.ceil(len(missing) / chunk_size)
    for c_idx in range(n_chunks):
        chunk = missing[c_idx * chunk_size : (c_idx + 1) * chunk_size]
        yf_symbols = [yahoo_symbol(t) for t in chunk]
        # The count remains unchanged while this stock is in flight; the UI
        # can name the real pending request instead of showing unexplained 0%.
        report(chunk[0])

        try:
            with _DOWNLOAD_LOCK:
                raw = yf.download(
                    yf_symbols,
                    start=str(start_date),
                    end=str(end_date),
                    interval=interval,
                    group_by="ticker",
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                    timeout=15,
                )
        except Exception as e:
            print(f"[DataFetcher] Batch download exception on chunk {c_idx+1}: {e}")
            raw = None

        for ticker, yf_sym in zip(chunk, yf_symbols):
            try:
                sub_df = _extract_symbol_frame(raw, yf_sym, allow_flat=len(chunk) == 1)
                sub_df = clean_ohlcv(sub_df, interval, completed_only=False)
                if not clean_ohlcv(sub_df, interval).empty:
                    results[ticker] = _save_history(ticker, interval, lookback_days, sub_df,
                                                    "yfinance.download")
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1
            completed += 1
            report(ticker)

        # Yield briefly between requests so waiting chart lookups get the lock.
        if c_idx < n_chunks - 1:
            time.sleep(0.05 if progress is not None else 0.15)

    return results


def clear_cache():
    """Clear memory cache."""
    global _PRICE_CACHE
    with _CACHE_LOCK:
        _PRICE_CACHE.clear()
