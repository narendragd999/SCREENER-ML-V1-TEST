"""
Near-Support Router  (drop-in module for the EMA9 Screener backend)
═════════════════════════════════════════════════════════════════════
Adds a NEW set of endpoints that detect stocks CURRENTLY very near a
multi-touch horizontal support level — the same pattern the user
illustrated in the KALYANKJIL screenshot.

DESIGN
------
• Pure-function detector lives in `support_detector.py` (no FastAPI deps).
• This router wraps it in HTTP endpoints, mirrors the cache + scheduler
  patterns already used by ema9_router.py / chartink_watcher.py.
• Lives alongside the existing 9-EMA crossover pipeline — same stocks
  can appear in BOTH lists (9EMA breakout AND near-support).  This is
  the "separate category" integration the user asked for.
• Reuses yfinance for data — same source as `_batch_download()` in
  ema9_router.py.

ENDPOINTS
---------
GET  /api/near-support/scan
       Optional ?tickers=T1,T2,...  (default = NIFTY-50 universe)
       Optional ?lookback_short=120&lookback_long=500
       Optional ?min_touches=3&proximity_pct=2.0&cluster_pct=0.7
       Runs a fresh scan synchronously, returns the candidate list,
       AND caches the result on disk for /results polling.

GET  /api/near-support/results
       Returns the most recent cached scan (polled by the UI tab).

POST /api/near-support/scan-now
       Body: { "tickers": ["..."], "lookback_short": 120, ... }
       Same as GET /scan but accepts a ticker list in the body (for
       the Chartink Live Feed integration — feed the polled Chartink
       stocks in here).

GET  /api/near-support/export?format=csv
       Downloads the cached scan as a CSV (broker-paste friendly).

GET  /api/near-support/health
       Liveness probe — returns cache stats + scheduler status.

WIRING  (add 2 lines to main.py)
--------------------------------
    from near_support_router import router as near_support_router
    app.include_router(near_support_router)

SCHEDULER  (add 4 lines to main.py @app.on_event("startup"))
----------------------------------------------------------------
    from near_support_router import start_near_support_scheduler
    if _CHARTINK_FEED_ENABLED:           # piggyback on existing toggle
        start_near_support_scheduler()   # auto-poll every 5 min

The scheduler pulls the last-known Chartink Live Feed ticker list from
the chartink_watcher's cached results (GET /api/chartink-screener/results)
and runs near-support detection on it.  If the Chartink feed is empty or
disabled, it falls back to the default NIFTY-50 universe.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ─── local import ───
# support_detector.py lives next to this router file in the project root.
_HERE = os.path.dirname(os.path.abspath(__file__))
import sys as _sys
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
from support_detector import detect_near_support  # noqa: E402

router = APIRouter()
logger = logging.getLogger("near_support_router")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[NearSupport] %(asctime)s %(message)s"))
    logger.addHandler(_h)

# ─────────────────────────────────────────────────────────────
#  Constants / config
# ─────────────────────────────────────────────────────────────
DEFAULT_LOOKBACK_SHORT    = 120
DEFAULT_LOOKBACK_LONG     = 500
DEFAULT_MIN_TOUCHES       = 3
DEFAULT_CLUSTER_PCT       = 0.7
DEFAULT_PROXIMITY_PCT     = 2.0
DEFAULT_BREAKDOWN_BUFFER  = 1.5
SCAN_INTERVAL_SECONDS     = 5 * 60          # 5 min — matches Chartink cadence
DATA_DOWNLOAD_PERIOD      = "730d"          # buffer above 500d long lookback
HTTP_TIMEOUT_SECONDS      = 25

# Cache file location — same dir as config.json (project root)
_CACHE_PATH = Path(_HERE) / "near_support_cache.json"

# Default NIFTY-50-ish universe used when Chartink feed is unavailable.
DEFAULT_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "BHARTIARTL",
    "SBIN", "ITC", "LT", "HINDUNILVR", "AXISBANK", "KOTAKBANK",
    "BAJFINANCE", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO",
    "NESTLEIND", "WIPRO", "ONGC", "NTPC", "POWERGRID", "COALINDIA",
    "HCLTECH", "TECHM", "BAJAJFINSV", "TATAMOTORS", "TATASTEEL", "GRASIM",
    "JSWSTEEL", "HDFCLIFE", "SBILIFE", "TATACONSUM", "DIVISLAB", "CIPLA",
    "DRREDDY", "BRITANNIA", "EICHERMOT", "ADANIENT", "ADANIPORTS", "HEROMOTOCO",
    "BAJAJ-AUTO", "INDUSINDBK", "SHRIRAMFIN", "BPCL", "IOC", "DMART",
    "BEL", "HAL", "PFC", "RECLTD", "IRFC", "NHPC", "SAIL", "TATAPOWER",
    "ADANIPOWER", "IGL", "PETRONET", "MGL", "GUJGASLTD", "ATGL",
    "ANANTRAJ", "ASTRAL", "ASTRAZEN", "KANSAINER", "SUPREMEIND",
    "ZYDUSLIFE", "MUTHOOTFI", "KALYANKJIL",
]

# ─────────────────────────────────────────────────────────────
#  Cache helpers
# ─────────────────────────────────────────────────────────────
def _write_cache(payload: Dict[str, Any]) -> None:
    try:
        _CACHE_PATH.write_text(json.dumps(payload, indent=2, default=str))
    except Exception as e:
        logger.warning(f"cache write failed: {e}")


def _read_cache() -> Optional[Dict[str, Any]]:
    try:
        if _CACHE_PATH.exists():
            return json.loads(_CACHE_PATH.read_text())
    except Exception as e:
        logger.warning(f"cache read failed: {e}")
    return None


# ─────────────────────────────────────────────────────────────
#  yfinance download (mirrors ema9_router's _safe_yf_download pattern)
# ─────────────────────────────────────────────────────────────
def _to_yf_symbol(t: str) -> str:
    t = t.strip().upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return t
    return f"{t}.NS"


def _download_batch(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Robust sequential download using yf.Ticker().history() loop.
    
    IMPORTANT: We deliberately AVOID yf.download(batch) with threads=True
    because Yahoo rate-limits parallel batch requests and responds with
    fake 'possibly delisted' 404s even for liquid stocks like RELIANCE,
    HDFCBANK, TCS, ITC. This is the same pattern ema9_router.py uses
    (its top docstring says: 'Replaced yf.download batch mode with
    yf.Ticker().history() loop for rock-solid EMA9 price data downloads').
    
    Features:
      • Sequential yf.Ticker(t).history() calls (no parallel threads)
      • .BO fallback when .NS returns empty
      • Tiny sleep every 10 tickers to be polite to Yahoo
      • Per-ticker try/except so one failure doesn't kill the batch
    """
    cleaned: List[str] = []
    seen = set()
    for t in tickers:
        tc = (t or "").strip().upper()
        if tc and tc not in seen:
            seen.add(tc)
            cleaned.append(tc)
    if not cleaned:
        return {}

    import time
    out: Dict[str, pd.DataFrame] = {}
    fail_count = 0

    for i, raw_t in enumerate(cleaned):
        # Strip any existing exchange suffix so we can append fresh
        base = raw_t.replace(".NS", "").replace(".BO", "").strip()
        if not base:
            continue

        df = None
        # Try .NS first, then .BO as fallback (some stocks only on BSE)
        for suffix in (".NS", ".BO"):
            sym = f"{base}{suffix}"
            try:
                tk = yf.Ticker(sym)
                df = tk.history(period=DATA_DOWNLOAD_PERIOD, interval="1d", auto_adjust=False)
                if df is not None and not df.empty and "Close" in df.columns:
                    df = df.dropna(subset=["Close", "Low"])
                    if len(df) > 20:
                        out[base] = df
                        break  # success — don't try .BO
                df = None
            except Exception:
                df = None

        if df is None:
            fail_count += 1

        # Politeness delay every 10 tickers — avoids Yahoo rate-limit
        if i > 0 and i % 10 == 0:
            time.sleep(0.15)

    logger.info(f"downloaded {len(out)}/{len(cleaned)} tickers "
                f"({fail_count} failed, period={DATA_DOWNLOAD_PERIOD})")
    return out


# ─────────────────────────────────────────────────────────────
#  Core scan — shared by sync endpoint + scheduler
# ─────────────────────────────────────────────────────────────
def _run_near_support_scan(
    tickers: List[str],
    lookback_short: int = DEFAULT_LOOKBACK_SHORT,
    lookback_long: int  = DEFAULT_LOOKBACK_LONG,
    min_touches: int    = DEFAULT_MIN_TOUCHES,
    cluster_pct: float  = DEFAULT_CLUSTER_PCT,
    proximity_pct: float = DEFAULT_PROXIMITY_PCT,
    breakdown_buffer: float = DEFAULT_BREAKDOWN_BUFFER,
) -> Dict[str, Any]:
    """Synchronous scan — called from the endpoint via asyncio.to_thread."""
    t0 = time.time()
    logger.info(f"scan start | tickers={len(tickers)} "
                f"| lookbacks=({lookback_short},{lookback_long}) "
                f"| min_touches={min_touches} "
                f"| proximity=±{proximity_pct}% "
                f"| cluster=±{cluster_pct}%")

    ticker_dfs = _download_batch(tickers)
    logger.info(f"downloaded {len(ticker_dfs)}/{len(tickers)} tickers in "
                f"{time.time()-t0:.1f}s")

    candidates: List[Dict[str, Any]] = []
    for ticker, df in ticker_dfs.items():
        for lookback in (lookback_short, lookback_long):
            try:
                r = detect_near_support(
                    df,
                    lookback_days=lookback,
                    min_touches=min_touches,
                    cluster_pct=cluster_pct,
                    proximity_pct_threshold=proximity_pct,
                    breakdown_buffer_pct=breakdown_buffer,
                )
            except Exception as e:
                logger.debug(f"{ticker} LB={lookback}: {e}")
                r = None
            if r is None:
                continue
            candidates.append({
                "ticker":            ticker,
                "lookback_days":     lookback,
                "support_level":     r["support_level"],
                "touches":           r["touches"],
                "current_price":     r["current_price"],
                "proximity_pct":     r["proximity_pct"],
                "first_touch_date":  r["first_touch_date"],
                "last_touch_date":   r["last_touch_date"],
                "within_threshold":  r["within_threshold"],
                "score":             r["score"],
            })

    # Best first: highest score → most touches → smallest |proximity|
    candidates.sort(
        key=lambda c: (-c["score"], -c["touches"], abs(c["proximity_pct"]))
    )

    payload = {
        "scan_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scan_duration_sec":  round(time.time() - t0, 2),
        "universe_size":      len(tickers),
        "tickers_with_data":  len(ticker_dfs),
        "candidate_count":    len(candidates),
        "params": {
            "lookback_short":     lookback_short,
            "lookback_long":      lookback_long,
            "min_touches":        min_touches,
            "cluster_pct":        cluster_pct,
            "proximity_pct":      proximity_pct,
            "breakdown_buffer":   breakdown_buffer,
        },
        "candidates": candidates,
    }
    _write_cache(payload)
    logger.info(f"scan done in {payload['scan_duration_sec']}s | "
                f"{len(candidates)} candidates cached")
    return payload


# ─────────────────────────────────────────────────────────────
#  HTTP endpoints
# ─────────────────────────────────────────────────────────────
@router.get("/api/near-support/health")
async def near_support_health():
    cache = _read_cache()
    return {
        "status":             "ok",
        "cache_present":      cache is not None,
        "cache_timestamp":    cache.get("scan_timestamp_utc") if cache else None,
        "cache_candidate_count": cache.get("candidate_count") if cache else 0,
        "scheduler_running":  _SCHEDULER_RUNNING,
        "scheduler_interval": SCAN_INTERVAL_SECONDS,
        "cache_path":         str(_CACHE_PATH),
    }


@router.get("/api/near-support/results")
async def near_support_results():
    """Return the most recent cached scan (polled by the UI tab)."""
    cache = _read_cache()
    if cache is None:
        return {
            "scan_timestamp_utc": None,
            "candidates": [],
            "candidate_count": 0,
            "message": "No scan has been run yet. POST /api/near-support/scan-now "
                       "or wait for the scheduler to fire.",
        }
    return cache


@router.get("/api/near-support/scan")
async def near_support_scan_get(
    tickers: Optional[str] = Query(
        None, description="Comma-separated tickers (default = NIFTY-50 universe)"),
    lookback_short: int = Query(DEFAULT_LOOKBACK_SHORT, ge=30, le=1000),
    lookback_long:  int = Query(DEFAULT_LOOKBACK_LONG,  ge=60, le=1000),
    min_touches:    int = Query(DEFAULT_MIN_TOUCHES,    ge=2,  le=20),
    cluster_pct:    float = Query(DEFAULT_CLUSTER_PCT,  ge=0.1, le=5.0),
    proximity_pct:  float = Query(DEFAULT_PROXIMITY_PCT, ge=0.1, le=20.0),
):
    """Run a fresh scan synchronously and return the candidates.
    Caches the result on disk for /results polling."""
    if tickers:
        ticker_list = [t.strip() for t in tickers.split(",") if t.strip()]
    else:
        ticker_list = DEFAULT_UNIVERSE

    return await asyncio.to_thread(
        _run_near_support_scan,
        ticker_list,
        lookback_short,
        lookback_long,
        min_touches,
        cluster_pct,
        proximity_pct,
        DEFAULT_BREAKDOWN_BUFFER,
    )


class NearSupportScanRequest(BaseModel):
    tickers: List[str]
    lookback_short: int = DEFAULT_LOOKBACK_SHORT
    lookback_long: int  = DEFAULT_LOOKBACK_LONG
    min_touches: int    = DEFAULT_MIN_TOUCHES
    cluster_pct: float  = DEFAULT_CLUSTER_PCT
    proximity_pct: float = DEFAULT_PROXIMITY_PCT


@router.post("/api/near-support/scan-now")
async def near_support_scan_post(req: NearSupportScanRequest):
    """Run a scan on an explicit ticker list — used by the Chartink Live
    Feed integration to push the polled Chartink stocks through the
    near-support detector."""
    if not req.tickers:
        raise HTTPException(400, "tickers list is empty")
    if len(req.tickers) > 500:
        raise HTTPException(400, "tickers list too long (max 500)")
    return await asyncio.to_thread(
        _run_near_support_scan,
        req.tickers,
        req.lookback_short,
        req.lookback_long,
        req.min_touches,
        req.cluster_pct,
        req.proximity_pct,
        DEFAULT_BREAKDOWN_BUFFER,
    )


@router.get("/api/near-support/export")
async def near_support_export(format: str = Query("csv", pattern="^(csv)$")):
    """Download the cached scan as a CSV (broker-paste friendly)."""
    cache = _read_cache()
    if not cache or not cache.get("candidates"):
        raise HTTPException(404, "No cached scan — run /api/near-support/scan first.")
    df = pd.DataFrame(cache["candidates"])
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="near_support_{ts}.csv"'
        },
    )


# ─────────────────────────────────────────────────────────────
#  Scheduler  (auto-poll every 5 min, mirrors chartink_watcher)
# ─────────────────────────────────────────────────────────────
_SCHEDULER_RUNNING = False
_SCHEDULER_THREAD: Optional[threading.Thread] = None
_SCHEDULER_LOCK = threading.Lock()


def _scheduler_loop() -> None:
    global _SCHEDULER_RUNNING
    logger.info(f"scheduler started | interval={SCAN_INTERVAL_SECONDS}s")
    while _SCHEDULER_RUNNING:
        try:
            # Try to read the last Chartink Live Feed ticker list (so the
            # near-support scan covers the same stocks the user is already
            # watching on the Chartink tab).  Fall back to the default
            # universe if the Chartink feed is empty / unavailable.
            ticker_list: List[str] = list(DEFAULT_UNIVERSE)
            try:
                import httpx
                with httpx.Client(timeout=HTTP_TIMEOUT_SECONDS) as client:
                    resp = client.get(
                        "http://127.0.0.1:8000/api/chartink-screener/results"
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        # The chartink_watcher payload structure (best-effort
                        # extraction — works whether the result is a list of
                        # signals or a dict with 'signals'/'prime_targets').
                        ck_tickers: List[str] = []
                        if isinstance(data, dict):
                            for key in ("signals", "prime_targets",
                                        "undervalued_signals", "other_signals"):
                                for s in data.get(key, []) or []:
                                    t = s.get("ticker") or s.get("symbol")
                                    if t:
                                        ck_tickers.append(str(t).upper())
                        elif isinstance(data, list):
                            for s in data:
                                t = s.get("ticker") or s.get("symbol")
                                if t:
                                    ck_tickers.append(str(t).upper())
                        if ck_tickers:
                            ticker_list = list(dict.fromkeys(ck_tickers))
                            logger.info(f"scheduler using {len(ticker_list)} "
                                        f"tickers from Chartink feed")
            except Exception as e:
                logger.debug(f"Chartink feed fetch failed ({e}) — using default universe")

            _run_near_support_scan(
                ticker_list,
                DEFAULT_LOOKBACK_SHORT,
                DEFAULT_LOOKBACK_LONG,
                DEFAULT_MIN_TOUCHES,
                DEFAULT_CLUSTER_PCT,
                DEFAULT_PROXIMITY_PCT,
                DEFAULT_BREAKDOWN_BUFFER,
            )
        except Exception as e:
            logger.error(f"scheduler iteration failed: {e}")

        # Sleep in 5-second increments so we can stop quickly on shutdown.
        slept = 0
        while _SCHEDULER_RUNNING and slept < SCAN_INTERVAL_SECONDS:
            time.sleep(5)
            slept += 5

    logger.info("scheduler stopped")


def start_near_support_scheduler() -> None:
    """Start the background auto-poller (idempotent)."""
    global _SCHEDULER_RUNNING, _SCHEDULER_THREAD
    with _SCHEDULER_LOCK:
        if _SCHEDULER_RUNNING:
            logger.info("scheduler already running — no-op")
            return
        _SCHEDULER_RUNNING = True
        _SCHEDULER_THREAD = threading.Thread(
            target=_scheduler_loop, name="near-support-scheduler", daemon=True
        )
        _SCHEDULER_THREAD.start()


def stop_near_support_scheduler() -> None:
    """Stop the background auto-poller (idempotent)."""
    global _SCHEDULER_RUNNING, _SCHEDULER_THREAD
    with _SCHEDULER_LOCK:
        if not _SCHEDULER_RUNNING:
            return
        _SCHEDULER_RUNNING = False
        t = _SCHEDULER_THREAD
        _SCHEDULER_THREAD = None
    if t and t.is_alive():
        t.join(timeout=10)
    logger.info("scheduler stop requested")
