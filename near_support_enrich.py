"""
Near-Support Enrichment  —  inline mixin for the Chartink Live Feed pipeline
═════════════════════════════════════════════════════════════════════════════

PURPOSE
-------
Provides a single function — `enrich_signals_with_near_support()` — that
takes a list of signal dicts (the output of ema9_router._run_screen_pipeline)
and adds near-support fields to each one IN-PLACE.

This is what wires the multi-touch horizontal support detector INTO the
existing Chartink Live Feed screener.  When the chartink_watcher polls
Chartink → calls _run_screen_pipeline → which now calls this enricher
before returning → every signal served by /api/chartink-screener/results
automatically carries:

    near_support          : bool   — True if price is within ±2% of a support
    near_support_level    : float  — the horizontal support price (₹)
    near_support_touches  : int    — number of times support was tested
    near_support_proximity_pct : float  — (price - support) / support × 100
    near_support_lookback : int    — 120 or 500 (which lookback found it)

The frontend (Chartink tab) reads these fields and shows a "🎯" badge
in a new "Near Support" column, plus a filter toggle to show only
near-support candidates.

DESIGN
------
• ONE batched yfinance download for ALL signal tickers (no per-ticker HTTP).
  For ~30-50 chartink signals this completes in 3-5 seconds.
• Runs detect_near_support at BOTH 120d and 500d lookbacks for each ticker.
• Picks the best (highest score) candidate per ticker.
• Pure-function — no FastAPI, no global state.  Drop into any project.
• Robust: failures on individual tickers never crash the whole batch.
"""
from __future__ import annotations

import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ─── import the detector (same dir as this file) ───
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

try:
    from support_detector import detect_near_support, detect_swing_pullback
except ImportError as e:                    # pragma: no cover
    detect_near_support = None
    detect_swing_pullback = None
    logging.getLogger("near_support_enrich").warning(
        f"support_detector not importable — near-support enrichment disabled. ({e})"
    )

logger = logging.getLogger("near_support_enrich")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[NearSupportEnrich] %(asctime)s %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────
#  Config defaults (mirror near_support_router.py)
# ─────────────────────────────────────────────────────────────
DEFAULT_LOOKBACK_SHORT    = 120
DEFAULT_LOOKBACK_LONG     = 500
DEFAULT_MIN_TOUCHES       = 3
DEFAULT_CLUSTER_PCT       = 0.7
DEFAULT_PROXIMITY_PCT     = 2.0
DEFAULT_BREAKDOWN_BUFFER  = 1.5
DEFAULT_DOWNLOAD_PERIOD   = "730d"   # buffer above 500d long lookback

# ── Swing-pullback config (last unbroken swing low retest) ─────────
# These detect a FRESHER, TIGHTER entry than horizontal support.
# Combined with horizontal support, gives the highest-conviction setup.
DEFAULT_SWING_FRACTAL_SIZE     = 3       # 7-bar fractal (3 bars each side)
DEFAULT_SWING_MAX_AGE_BARS     = 20      # swing low must be within 1 month
DEFAULT_SWING_REQUIRE_HL       = True    # require higher-low (uptrend confirmation)
DEFAULT_SWING_BUY_ZONE_PCT     = 3.0     # price within 0% to +3% above swing low
DEFAULT_SWING_LOOKBACK         = 120     # bars of history to scan for swing lows


# ─────────────────────────────────────────────────────────────
#  yfinance batch download
# ─────────────────────────────────────────────────────────────
def _to_yf_symbol(t: str) -> str:
    t = t.strip().upper()
    if t.endswith(".NS") or t.endswith(".BO"):
        return t
    return f"{t}.NS"


def _batch_download(tickers: List[str], period: str = DEFAULT_DOWNLOAD_PERIOD) -> Dict[str, pd.DataFrame]:
    """Robust sequential download using yf.Ticker().history() loop.
    
    IMPORTANT: We deliberately AVOID yf.download(batch) with threads=True
    because Yahoo rate-limits parallel batch requests and responds with
    fake 'possibly delisted' 404s even for liquid stocks like RELIANCE,
    HDFCBANK, TCS, ITC. This is the same pattern ema9_router.py uses.
    
    Features:
      • Sequential yf.Ticker(t).history() calls (no parallel threads)
      • .BO fallback when .NS returns empty (mirrors ema9_router.py)
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

    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — skipping near-support download")
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
                df = tk.history(period=period, interval="1d", auto_adjust=False)
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
                f"({fail_count} failed, period={period})")
    return out


# ─────────────────────────────────────────────────────────────
#  Core enrichment
# ─────────────────────────────────────────────────────────────
def enrich_signals_with_near_support(
    signals: List[Dict[str, Any]],
    lookback_short: int = DEFAULT_LOOKBACK_SHORT,
    lookback_long: int  = DEFAULT_LOOKBACK_LONG,
    min_touches: int    = DEFAULT_MIN_TOUCHES,
    cluster_pct: float  = DEFAULT_CLUSTER_PCT,
    proximity_pct: float = DEFAULT_PROXIMITY_PCT,
    breakdown_buffer: float = DEFAULT_BREAKDOWN_BUFFER,
    enabled: bool = True,
) -> Dict[str, Any]:
    """
    Add near-support fields to every signal dict IN-PLACE.

    Parameters
    ----------
    signals : list of dict
        Each dict must have at least a 'ticker' key.  Fields added:
            near_support                  (bool)
            near_support_level            (float | None)
            near_support_touches          (int | None)
            near_support_proximity_pct    (float | None)
            near_support_lookback         (int | None)
            near_support_first_touch_date (str | None)
            near_support_last_touch_date  (str | None)
            near_support_score            (float | None)
    enabled : bool
        Master toggle.  When False, every signal gets near_support=False
        and the other fields set to None.  Used to honor the config flag
        `chartink_enrich_near_support` without changing call sites.

    Returns
    -------
    dict with enrichment metadata (counts, duration, errors) — useful for
    logging or for the API response payload.
    """
    t0 = time.time()

    # ── Defensive default: ensure every signal has the keys ───────────
    NS_FIELDS = {
        "near_support":                  False,
        "near_support_level":            None,
        "near_support_touches":          None,
        "near_support_proximity_pct":    None,
        "near_support_lookback":         None,
        "near_support_first_touch_date": None,
        "near_support_last_touch_date":  None,
        "near_support_score":            None,
        # ── Swing-pullback fields (last unbroken swing low retest) ──
        "swing_pullback":                False,
        "swing_low_price":               None,
        "swing_low_date":                None,
        "swing_pullback_proximity_pct":  None,
        "swing_pullback_is_higher_low":  None,
        "swing_pullback_prior_low":      None,
        "swing_pullback_days_ago":       None,
        "swing_pullback_score":          None,
        # ── Structural low (deepest unbroken swing — for STOP PLACEMENT) ──
        # This is the real trend-invalidation level.  The recent swing
        # (above) is for ENTRY TIMING; this one is for STOP PLACEMENT.
        "structural_swing_low_price":    None,
        "structural_swing_low_date":     None,
        "structural_swing_proximity_pct": None,
        "structural_swing_days_ago":     None,
        "structural_is_same_as_recent":  False,
        # ── Combined conviction flag (BOTH horizontal support AND swing) ──
        "support_and_swing":             False,
    }
    for sig in signals:
        for k, v in NS_FIELDS.items():
            sig.setdefault(k, v)

    if not enabled or detect_near_support is None or not signals:
        return {
            "enabled":              enabled,
            "detector_available":   detect_near_support is not None,
            "signals_count":        len(signals),
            "enriched_count":       0,
            "duration_sec":         round(time.time() - t0, 2),
            "skipped_reason":       ("disabled by config" if not enabled
                                     else "detector not available" if detect_near_support is None
                                     else "no signals"),
        }

    # ── Collect unique tickers ─────────────────────────────────────────
    tickers = list({(s.get("ticker") or "").strip().upper() for s in signals if s.get("ticker")})
    if not tickers:
        return {
            "enabled": True, "detector_available": True,
            "signals_count": len(signals), "enriched_count": 0,
            "duration_sec": round(time.time() - t0, 2),
            "skipped_reason": "no tickers in signals",
        }

    # ── Batch download ─────────────────────────────────────────────────
    logger.info(f"enriching {len(tickers)} tickers "
                f"(lookbacks {lookback_short}/{lookback_long}, "
                f"min_touches={min_touches}, proximity=±{proximity_pct}%)")
    ticker_dfs = _batch_download(tickers)
    if not ticker_dfs:
        logger.warning("no OHLCV data downloaded — skipping enrichment")
        return {
            "enabled": True, "detector_available": True,
            "signals_count": len(signals), "enriched_count": 0,
            "tickers_with_data": 0,
            "duration_sec": round(time.time() - t0, 2),
            "skipped_reason": "no OHLCV data downloaded",
        }

    # ── Run detectors at both lookbacks for each ticker ───────────────
    # Pick the BEST (highest score) candidate per ticker.
    # We run BOTH horizontal-support detection AND swing-pullback detection
    # on the SAME df — no extra network round-trip, just CPU.
    best_per_ticker: Dict[str, Dict[str, Any]] = {}
    swing_per_ticker: Dict[str, Dict[str, Any]] = {}
    for ticker, df in ticker_dfs.items():
        # ── Horizontal multi-touch support detection ──
        best: Optional[Dict[str, Any]] = None
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
            if best is None or r["score"] > best["score"]:
                best = r
                best["lookback_days"] = lookback
        if best:
            best_per_ticker[ticker] = best

        # ── Swing-pullback detection (last unbroken swing low retest) ──
        # Uses the SAME df — no second download.  Detects a fresher, tighter
        # entry signal than horizontal support.
        if detect_swing_pullback is not None:
            try:
                sw = detect_swing_pullback(
                    df,
                    fractal_size=DEFAULT_SWING_FRACTAL_SIZE,
                    max_age_bars=DEFAULT_SWING_MAX_AGE_BARS,
                    require_higher_low=DEFAULT_SWING_REQUIRE_HL,
                    buy_zone_pct=DEFAULT_SWING_BUY_ZONE_PCT,
                    lookback_days=DEFAULT_SWING_LOOKBACK,
                )
            except Exception as e:
                logger.debug(f"{ticker} swing: {e}")
                sw = None
            if sw:
                swing_per_ticker[ticker] = sw

    # ── Stamp each signal with its ticker's near-support + swing data ──
    enriched_count = 0
    swing_count = 0
    both_count = 0
    for sig in signals:
        t = (sig.get("ticker") or "").strip().upper()
        ns = best_per_ticker.get(t)
        sw = swing_per_ticker.get(t)
        if ns:
            sig["near_support"]                  = True
            sig["near_support_level"]            = ns["support_level"]
            sig["near_support_touches"]          = ns["touches"]
            sig["near_support_proximity_pct"]    = ns["proximity_pct"]
            sig["near_support_lookback"]         = ns["lookback_days"]
            sig["near_support_first_touch_date"] = ns.get("first_touch_date")
            sig["near_support_last_touch_date"]  = ns.get("last_touch_date")
            sig["near_support_score"]            = ns.get("score")
            enriched_count += 1
        if sw:
            sig["swing_pullback"]                = True
            sig["swing_low_price"]               = sw["swing_low_price"]
            sig["swing_low_date"]                = sw.get("swing_low_date")
            sig["swing_pullback_proximity_pct"]  = sw["proximity_pct"]
            sig["swing_pullback_is_higher_low"]  = sw.get("is_higher_low")
            sig["swing_pullback_prior_low"]      = sw.get("prior_swing_low_price")
            sig["swing_pullback_days_ago"]       = sw.get("days_since_swing")
            sig["swing_pullback_score"]          = sw.get("score")
            # ── Structural low (deepest unbroken swing — STOP PLACEMENT) ──
            # Always stamped when swing_pullback is True, even if the
            # structural low is the SAME as the recent swing (in which
            # case structural_is_same_as_recent = True).
            sig["structural_swing_low_price"]     = sw.get("structural_swing_low_price")
            sig["structural_swing_low_date"]      = sw.get("structural_swing_low_date")
            sig["structural_swing_proximity_pct"] = sw.get("structural_swing_proximity_pct")
            sig["structural_swing_days_ago"]      = sw.get("structural_swing_days_ago")
            sig["structural_is_same_as_recent"]   = sw.get("structural_is_same_as_recent", False)
            swing_count += 1
        # ── Highest-conviction flag: BOTH horizontal support AND swing ──
        if ns and sw:
            sig["support_and_swing"] = True
            both_count += 1

    duration = round(time.time() - t0, 2)
    logger.info(f"enrichment done in {duration}s | "
                f"near_support={enriched_count}/{len(signals)}, "
                f"swing_pullback={swing_count}/{len(signals)}, "
                f"BOTH={both_count}/{len(signals)}")

    return {
        "enabled":             True,
        "detector_available":  True,
        "signals_count":       len(signals),
        "tickers_with_data":   len(ticker_dfs),
        "enriched_count":      enriched_count,
        "swing_count":         swing_count,
        "both_count":          both_count,
        "duration_sec":        duration,
        "lookback_short":      lookback_short,
        "lookback_long":       lookback_long,
        "min_touches":         min_touches,
        "cluster_pct":         cluster_pct,
        "proximity_pct":       proximity_pct,
        "breakdown_buffer":    breakdown_buffer,
        # Swing-pullback params echoed for UI display
        "swing_fractal_size":  DEFAULT_SWING_FRACTAL_SIZE,
        "swing_max_age_bars":  DEFAULT_SWING_MAX_AGE_BARS,
        "swing_require_hl":    DEFAULT_SWING_REQUIRE_HL,
        "swing_buy_zone_pct":  DEFAULT_SWING_BUY_ZONE_PCT,
        "swing_lookback":      DEFAULT_SWING_LOOKBACK,
    }


# ─────────────────────────────────────────────────────────────
#  Config helper — reads from main.py's config.json
# ─────────────────────────────────────────────────────────────
def _load_config_toggle() -> bool:
    """Read chartink_enrich_near_support from config.json. Default True."""
    try:
        cfg_path = os.path.join(_HERE, "config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            return bool(cfg.get("chartink_enrich_near_support", True))
    except Exception:
        pass
    return True


# Lazy import of json (only needed by the helper above)
import json  # noqa: E402
