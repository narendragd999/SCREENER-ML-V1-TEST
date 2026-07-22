"""
Multi-Touch Horizontal Support Detector
═══════════════════════════════════════
Pure-function module — no network, no FastAPI.  Drop into any project.

ALGORITHM (mirrors the KALYANKJIL example screenshot)
-------------------------------------------------------
1. Take the last N days of OHLCV (lookback).
2. Find fractal pivot lows (5-bar: bar i is a pivot low if Low[i] is the
   lowest among Low[i-2 .. i+2]).  These are the candidate "touches".
3. Greedy-cluster pivots whose prices fall within ±`cluster_pct` of each
   other.  Each cluster = one horizontal support level.
4. Keep clusters with >= `min_touches` touches.
5. For each surviving cluster compute:
     support_level   = mean of pivot prices in the cluster
     touches         = count
     first_touch_date / last_touch_date  = oldest / newest touch
     proximity_pct   = (current_close - support_level) / support_level * 100
6. Filter to clusters where:
     -proximity_pct_threshold <= proximity_pct <= +proximity_pct_threshold
   (price is at-or-near support, allowing a slight breach)
   AND
     proximity_pct > -breakdown_buffer_pct
   (skip supports that have been clearly broken — those are now resistance).
7. Among valid clusters, rank by:
     score = touches * 1.0  -  abs(proximity_pct) * 0.3
   Higher touches + closer proximity  →  higher score.
8. Return the top cluster (or top N clusters — caller's choice).

USAGE
-----
    import yfinance as yf
    from support_detector import detect_near_support

    df = yf.download("KALYANKJIL.NS", period="500d", interval="1d", progress=False)
    result = detect_near_support(df, lookback_days=120, min_touches=3,
                                  proximity_pct_threshold=2.0)
    # result is a dict: { support_level, touches, proximity_pct, ... } or None
"""
from __future__ import annotations

import math
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
#  Core pivot detection
# ─────────────────────────────────────────────────────────────
def _find_pivot_lows(low_series: pd.Series, fractal_size: int = 2) -> List[tuple]:
    """
    5-bar fractal (default): bar i is a pivot low if Low[i] is the strict
    minimum of Low[i-fractal_size .. i+fractal_size].

    Returns a list of (index_label, price) tuples for each pivot low.
    """
    n = len(low_series)
    pivots: List[tuple] = []
    values = low_series.to_numpy()
    labels = low_series.index

    for i in range(fractal_size, n - fractal_size):
        window = values[i - fractal_size : i + fractal_size + 1]
        # Strict minimum at the centre — equal neighbours are OK because we
        # use <= for non-centre bars (so a flat shelf still counts).
        center = values[i]
        if math.isnan(center):
            continue
        is_min = True
        for j, v in enumerate(window):
            if j == fractal_size:  # skip centre
                continue
            if math.isnan(v):
                continue
            if v < center:
                is_min = False
                break
        if is_min:
            pivots.append((labels[i], float(center)))
    return pivots


# ─────────────────────────────────────────────────────────────
#  Greedy clustering of pivots into horizontal levels
# ─────────────────────────────────────────────────────────────
def _cluster_pivots(
    pivots: List[tuple], cluster_pct: float = 0.7
) -> List[List[tuple]]:
    """
    Sort pivots by price; greedily group pivots whose price is within
    `cluster_pct` of the running cluster's first (lowest) price.

    Each cluster is a list of (date, price) tuples.
    """
    if not pivots:
        return []
    sorted_pivots = sorted(pivots, key=lambda p: p[1])
    clusters: List[List[tuple]] = []
    current: List[tuple] = [sorted_pivots[0]]
    ref_price = sorted_pivots[0][1]

    for date, price in sorted_pivots[1:]:
        gap_pct = abs(price - ref_price) / ref_price * 100.0
        if gap_pct <= cluster_pct:
            current.append((date, price))
        else:
            clusters.append(current)
            current = [(date, price)]
            ref_price = price
    clusters.append(current)
    return clusters


# ─────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────
def detect_near_support(
    df: pd.DataFrame,
    lookback_days: int = 120,
    min_touches: int = 3,
    cluster_pct: float = 0.7,
    proximity_pct_threshold: float = 2.0,
    breakdown_buffer_pct: float = 1.5,
    fractal_size: int = 2,
) -> Optional[Dict[str, Any]]:
    """
    Detect whether the latest close is "very near" a multi-touch horizontal
    support level in the last `lookback_days` bars.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data — must contain at least a 'Close' and a 'Low' column,
        indexed by date.  yfinance-style: 'Open'/'High'/'Low'/'Close'/'Volume'.
    lookback_days : int
        Number of most-recent bars to scan for pivots.
    min_touches : int
        Minimum pivots required in a cluster to qualify as a support level.
    cluster_pct : float
        Max % spread between pivot prices to be in the same cluster.
    proximity_pct_threshold : float
        Max |close - support| / support * 100 to be considered "near".
    breakdown_buffer_pct : float
        If close is more than this % BELOW support, the level is treated as
        broken (skip it — broken support becomes resistance).
    fractal_size : int
        Bars on each side for the pivot detector (2 = 5-bar fractal).

    Returns
    -------
    dict  with keys:
        support_level       : float
        touches             : int
        first_touch_date    : str (ISO date)
        last_touch_date     : str (ISO date)
        current_price       : float
        proximity_pct       : float  (positive = above support, negative = below)
        within_threshold    : bool   (True if |proximity_pct| <= proximity_pct_threshold
                                       AND proximity_pct > -breakdown_buffer_pct)
        lookback_days       : int
        touch_dates         : list[str]   (ISO dates of each pivot in the cluster)
        touch_prices        : list[float] (low prices at each pivot)
        all_clusters        : list[dict]  (every valid cluster found, for debugging)
    OR None if no valid support found.
    """
    # ── Normalise column names (yfinance sometimes returns MultiIndex) ───
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance batch download returns ('Open', 'TICKER'), etc.  Flatten.
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    needed = {"Close", "Low"}
    if not needed.issubset(set(df.columns)):
        # try lowercase
        rename = {c: c.capitalize() for c in df.columns if c.lower() in {"close", "low"}}
        df = df.rename(columns=rename)
    if not needed.issubset(set(df.columns)):
        return None

    if len(df) < fractal_size * 2 + 5:
        return None  # not enough history to find pivots

    recent = df.tail(lookback_days).copy()
    close = float(recent["Close"].iloc[-1])
    if math.isnan(close) or close <= 0:
        return None

    # ── Sanity: filter out suspended / "stuck" stocks ──────────────────
    # If the coefficient of variation of close is < 1%, the stock hasn't
    # really traded (e.g. JETAIRWAYS parked at ₹34.16 for years). Skip.
    closes = recent["Close"].dropna().to_numpy()
    if len(closes) > 5:
        cv = float(np.std(closes) / np.mean(closes)) if np.mean(closes) > 0 else 0
        if cv < 0.01:
            return None
    # Also reject if every close equals the last close (zero volatility).
    if np.all(closes == close):
        return None

    pivots = _find_pivot_lows(recent["Low"], fractal_size=fractal_size)
    if len(pivots) < min_touches:
        return None

    clusters = _cluster_pivots(pivots, cluster_pct=cluster_pct)
    valid_clusters = [c for c in clusters if len(c) >= min_touches]
    if not valid_clusters:
        return None

    # ── Touch-density sanity: a healthy multi-touch support has touches ─
    # spread across distinct swings, not the same bar being a pivot every
    # day. If a cluster has touches > 30% of lookback bars, the stock is
    # trading in a degenerate micro-range (e.g. JETAIRWAYS parked at ₹34).
    max_touches = int(lookback_days * 0.3)
    valid_clusters = [c for c in valid_clusters if len(c) <= max_touches]
    if not valid_clusters:
        return None

    # ── Build a record for every valid cluster ──────────────────────────
    all_records: List[Dict[str, Any]] = []
    for c in valid_clusters:
        prices = [p for _, p in c]
        support_level = float(np.mean(prices))
        proximity_pct = (close - support_level) / support_level * 100.0
        dates = [d.isoformat() if hasattr(d, "isoformat") else str(d) for d, _ in c]
        all_records.append({
            "support_level":     round(support_level, 2),
            "touches":           len(c),
            "first_touch_date":  min(dates),
            "last_touch_date":   max(dates),
            "current_price":     round(close, 2),
            "proximity_pct":     round(proximity_pct, 2),
            "within_threshold":  (
                abs(proximity_pct) <= proximity_pct_threshold
                and proximity_pct > -breakdown_buffer_pct
            ),
            "touch_dates":       dates,
            "touch_prices":      [round(p, 2) for p in prices],
            # score = touches bonus  -  distance penalty
            "score": round(len(c) * 1.0 - abs(proximity_pct) * 0.3, 3),
        })

    # ── Pick best candidate ─────────────────────────────────────────────
    # Prefer within-threshold clusters; rank by score.
    within = [r for r in all_records if r["within_threshold"]]
    if not within:
        # No cluster currently within threshold — return None to caller,
        # but include all_records in a debug field if requested.
        return None

    best = max(within, key=lambda r: r["score"])
    best["lookback_days"] = lookback_days
    best["all_clusters"] = all_records
    return best


# ─────────────────────────────────────────────────────────────
#  Convenience wrapper — scan many tickers in parallel
# ─────────────────────────────────────────────────────────────
def scan_tickers(
    ticker_to_df: Dict[str, pd.DataFrame],
    lookback_days_list: Optional[List[int]] = None,
    min_touches: int = 3,
    cluster_pct: float = 0.7,
    proximity_pct_threshold: float = 2.0,
    breakdown_buffer_pct: float = 1.5,
) -> List[Dict[str, Any]]:
    """
    Run detect_near_support on every (ticker -> df) entry, for each lookback
    in lookback_days_list.  Returns a flat list of candidate dicts.

    Default lookback list = [120, 500]  (short + long, as user requested).
    Each result dict includes the source lookback so callers can group/sort.
    """
    if lookback_days_list is None:
        lookback_days_list = [120, 500]

    out: List[Dict[str, Any]] = []
    for ticker, df in ticker_to_df.items():
        if df is None or df.empty:
            continue
        for lb in lookback_days_list:
            try:
                res = detect_near_support(
                    df,
                    lookback_days=lb,
                    min_touches=min_touches,
                    cluster_pct=cluster_pct,
                    proximity_pct_threshold=proximity_pct_threshold,
                    breakdown_buffer_pct=breakdown_buffer_pct,
                )
            except Exception as e:
                res = None
            if res:
                res["ticker"] = ticker
                res["lookback_days"] = lb
                out.append(res)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  SWING-PULLBACK DETECTOR  (last unbroken swing low retest)
# ═══════════════════════════════════════════════════════════════════════════
#
# Purpose
# -------
# Catches stocks whose current price is sitting right at or just bouncing
# off the most recent UNBROKEN swing low.  This is a tighter, fresher
# entry signal than horizontal multi-touch support — useful for:
#   • Tighter stop placement (just below the swing low, which is fresh)
#   • Faster target achievement (entering at the bounce, not before it)
#   • Avoiding "falling knives" (price must be AT the swing, not above it)
#
# Combined with horizontal support detection, this gives the highest-
# conviction setup: a major structural level that's ALSO being retested
# right now via a fresh swing low.
#
# Algorithm
# ---------
# 1. Take last N bars of OHLCV.
# 2. Find all swing lows using a (2k+1)-bar fractal (bar i is the strict
#    minimum of Low[i-k..i+k]). Default k=3 (7-bar fractal).
# 3. The "current candidate" is the MOST RECENT swing low whose price
#    has NOT been breached by any subsequent bar's low.  (i.e. it's
#    still an unbroken low.)
# 4. Apply freshness filter: the swing low must be within `max_age_bars`
#    trading days of today (default 20 = ~1 month).
# 5. Apply higher-low filter (default ON): the candidate swing low must
#    be at a HIGHER price than the prior swing low (uptrend confirmation).
#    This filters out lower-lows in downtrends — counter-trend bounces
#    that often fail.
# 6. Apply buy-zone filter: current close must be within
#    [0%, +buy_zone_pct] above the swing low.  Negative values (price
#    below swing) mean the swing is broken — skip.  Values above
#    +buy_zone_pct mean price has bounced too far — entry missed.
# 7. Return the swing-low price, date, distance %, higher-low flag, and
#    days since the swing low was formed.
# ═══════════════════════════════════════════════════════════════════════════


def detect_swing_pullback(
    df: pd.DataFrame,
    fractal_size: int = 3,
    max_age_bars: int = 20,
    require_higher_low: bool = True,
    buy_zone_pct: float = 3.0,
    lookback_days: int = 120,
) -> Optional[Dict[str, Any]]:
    """
    Detect whether the latest close is sitting at / bouncing off the most
    recent unbroken swing low.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data — must contain 'Close' and 'Low' columns.
    fractal_size : int
        Bars on each side for the pivot detector (3 = 7-bar fractal).
    max_age_bars : int
        Max age of the swing low in trading days. Older swings = stale.
    require_higher_low : bool
        When True, the swing low must be a higher-low (uptrend confirmation).
    buy_zone_pct : float
        Max % above the swing low for price to qualify as "at the swing".
        Price below 0% = swing broken; price above buy_zone_pct = bounced too far.
    lookback_days : int
        Number of most-recent bars to scan for swing lows.  Should be at
        least 2 × max_age_bars so we have history for higher-low detection.

    Returns
    -------
    dict with keys:
        swing_low_price        : float
        swing_low_date         : str (ISO date)
        swing_low_index        : int  (position in df, 0-based from lookback)
        current_price          : float
        proximity_pct          : float  (how far above the swing low, in %)
        is_higher_low          : bool   (True if swing > prior swing)
        prior_swing_low_price  : float | None
        days_since_swing       : int    (bars between swing low and today)
        within_buy_zone        : bool   (True if 0 <= proximity_pct <= buy_zone_pct
                                         AND (not require_higher_low OR is_higher_low))
        score                  : float  (higher when closer to swing + fresher)
    OR None if no valid swing-pullback setup found.
    """
    # ── Normalise column names (same as detect_near_support) ───────────
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    needed = {"Close", "Low"}
    if not needed.issubset(set(df.columns)):
        rename = {c: c.capitalize() for c in df.columns if c.lower() in {"close", "low"}}
        df = df.rename(columns=rename)
    if not needed.issubset(set(df.columns)):
        return None
    if len(df) < fractal_size * 2 + 5:
        return None

    # Use enough history to find a prior swing low for HL comparison
    actual_lookback = max(lookback_days, max_age_bars * 3, fractal_size * 4)
    recent = df.tail(actual_lookback).copy().reset_index(drop=True)
    close = float(recent["Close"].iloc[-1])
    if math.isnan(close) or close <= 0:
        return None

    # ── Find all swing lows (same fractal logic, but indexed by position) ──
    n = len(recent)
    lows_arr = recent["Low"].to_numpy()
    pivots: List[tuple] = []   # list of (position_in_recent, price, date_label)
    for i in range(fractal_size, n - fractal_size):
        center = lows_arr[i]
        if math.isnan(center):
            continue
        is_min = True
        for j in range(i - fractal_size, i + fractal_size + 1):
            if j == i:
                continue
            v = lows_arr[j]
            if math.isnan(v):
                continue
            if v < center:
                is_min = False
                break
        if is_min:
            date_label = recent.index[i]
            # After reset_index(drop=True), recent.index is a RangeIndex of
            # integers — not dates.  Detect this and fall back to the original
            # df's DatetimeIndex to get the actual calendar date.
            if not hasattr(date_label, "isoformat"):
                # date_label is an integer (RangeIndex) — map back to df.index
                orig_pos = len(df) - len(recent) + i
                if 0 <= orig_pos < len(df):
                    date_label = df.index[orig_pos]
            pivots.append((i, float(center), date_label))

    if len(pivots) < 1:
        return None

    # ── Walk pivots from MOST RECENT backwards, find first UNBROKEN one ──
    # A swing low at position p is "unbroken" if no bar after p has a Low
    # strictly below pivots[p].price.  (Equal lows are OK — support shelf.)
    candidate = None
    prior_swing = None
    for idx in range(len(pivots) - 1, -1, -1):
        pos, price, date_label = pivots[idx]
        # Check no subsequent bar breaks this low
        broken = False
        for k in range(pos + 1, n):
            v = lows_arr[k]
            if not math.isnan(v) and v < price:
                broken = True
                break
        if broken:
            continue
        # Found an unbroken swing low.  Look for the prior swing low
        # (the one immediately before this one) for higher-low comparison.
        if idx > 0:
            prior_pos, prior_price, _ = pivots[idx - 1]
            prior_swing = (prior_pos, prior_price)
        candidate = (pos, price, date_label)
        break

    if candidate is None:
        return None  # every recent swing low has been broken — pure downtrend

    # ── ALSO find the DEEPEST (structural) unbroken swing low ──────────
    # This is the lowest-priced unbroken swing low within a WIDER window
    # (default 3× max_age_bars).  It represents the structural bottom of
    # the current move — the real stop-placement level.
    #
    # Why both?  The MOST RECENT swing tells you about ENTRY TIMING
    # ("are we bouncing right now?").  The DEEPEST swing tells you about
    # STOP PLACEMENT ("where is the true trend-invalidation level?").
    #
    # Example: PNB on 2026-06-30
    #   Most recent unbroken swing: 2026-06-12 @ ₹104.37 (+2.22% above) → entry timing
    #   Deepest unbroken swing:     2026-05-18 @ ₹98.50  (+8.31% above) → stop placement
    #   June 12 is just a higher-low climbing away from the May 18 structural bottom.
    structural_window = max(max_age_bars * 3, 60)
    structural_candidate = None   # (pos, price, date_label, days_since)
    for idx in range(len(pivots) - 1, -1, -1):
        pos, price, date_label = pivots[idx]
        days_since_this = n - 1 - pos
        if days_since_this > structural_window:
            continue
        # Check unbroken
        broken = False
        for k in range(pos + 1, n):
            v = lows_arr[k]
            if not math.isnan(v) and v < price:
                broken = True
                break
        if broken:
            continue
        # Pick the DEEPEST (lowest price)
        if structural_candidate is None or price < structural_candidate[1]:
            structural_candidate = (pos, price, date_label, days_since_this)

    pos, swing_price, swing_date = candidate
    days_since = n - 1 - pos   # bars between swing low and today

    # ── Apply freshness filter ─────────────────────────────────────────
    if days_since > max_age_bars:
        return None

    # ── Higher-low check ───────────────────────────────────────────────
    is_higher_low = False
    if prior_swing is not None:
        _, prior_price = prior_swing
        is_higher_low = swing_price > prior_price
    if require_higher_low and not is_higher_low:
        return None

    # ── Buy-zone check ─────────────────────────────────────────────────
    proximity_pct = (close - swing_price) / swing_price * 100.0
    within_buy_zone = (0.0 <= proximity_pct <= buy_zone_pct)
    if not within_buy_zone:
        return None

    # Date label to ISO string
    swing_date_str = swing_date.isoformat() if hasattr(swing_date, "isoformat") else str(swing_date)

    # Score: closer to swing (small proximity) + fresher (small days_since) = higher
    score = (buy_zone_pct - proximity_pct) * 1.0 + (max_age_bars - days_since) * 0.2

    # ── Build structural-low fields (deepest unbroken swing) ───────────
    # These are for STOP PLACEMENT — where to put the actual stop-loss.
    # The most-recent swing (above) is for ENTRY TIMING — when to buy.
    structural_fields: Dict[str, Any] = {
        "structural_swing_low_price":      None,
        "structural_swing_low_date":       None,
        "structural_swing_proximity_pct":  None,
        "structural_swing_days_ago":       None,
        "structural_is_same_as_recent":    False,
    }
    if structural_candidate is not None:
        s_pos, s_price, s_date, s_days = structural_candidate
        s_date_str = s_date.isoformat() if hasattr(s_date, "isoformat") else str(s_date)
        s_proximity = (close - s_price) / s_price * 100.0
        structural_fields = {
            "structural_swing_low_price":      round(s_price, 2),
            "structural_swing_low_date":       s_date_str,
            "structural_swing_proximity_pct":  round(s_proximity, 2),
            "structural_swing_days_ago":       int(s_days),
            "structural_is_same_as_recent":    (s_pos == pos),  # True if deepest == most-recent
        }

    return {
        "swing_low_price":        round(swing_price, 2),
        "swing_low_date":         swing_date_str,
        "swing_low_index":        int(pos),
        "current_price":          round(close, 2),
        "proximity_pct":          round(proximity_pct, 2),
        "is_higher_low":          is_higher_low,
        "prior_swing_low_price":  round(prior_swing[1], 2) if prior_swing else None,
        "days_since_swing":       int(days_since),
        "within_buy_zone":        True,   # already filtered above
        "score":                  round(score, 3),
        # ── Structural low (deepest unbroken swing — for stop placement) ──
        **structural_fields,
    }


# ─────────────────────────────────────────────────────────────
#  Convenience wrapper — scan many tickers for swing pullback
# ─────────────────────────────────────────────────────────────
def scan_tickers_swing(
    ticker_to_df: Dict[str, pd.DataFrame],
    fractal_size: int = 3,
    max_age_bars: int = 20,
    require_higher_low: bool = True,
    buy_zone_pct: float = 3.0,
    lookback_days: int = 120,
) -> List[Dict[str, Any]]:
    """Run detect_swing_pullback on every (ticker -> df) entry."""
    out: List[Dict[str, Any]] = []
    for ticker, df in ticker_to_df.items():
        if df is None or df.empty:
            continue
        try:
            res = detect_swing_pullback(
                df,
                fractal_size=fractal_size,
                max_age_bars=max_age_bars,
                require_higher_low=require_higher_low,
                buy_zone_pct=buy_zone_pct,
                lookback_days=lookback_days,
            )
        except Exception:
            res = None
        if res:
            res["ticker"] = ticker
            out.append(res)
    return out
