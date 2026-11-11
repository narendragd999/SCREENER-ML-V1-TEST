"""
support_detector.py — Multi-Touch Horizontal & Structural Low Support Detector
==============================================================================
Identifies institutional structural swing lows and horizontal support floors
(as illustrated in HCLTECH at ₹1,210.2 / ₹1,367.8, FINEORG at ₹4,020.4, and SBIN at ₹940 / ₹1,038).

A STRUCTURAL LOW is a major cycle swing low where institutional buyers defended
the price, initiating a major expansion (+10% to +50%+ subsequent rally).
"""
import math
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd


def find_pivot_lows(df: pd.DataFrame, fractal_size: int = 3) -> List[Tuple[int, str, float]]:
    """Identifies fractal pivot lows for general horizontal clustering."""
    n = len(df)
    pivots = []
    lows = df["Low"].to_numpy(dtype=float)
    dates = df.index

    for i in range(fractal_size, n - fractal_size):
        window = lows[i - fractal_size : i + fractal_size + 1]
        center = lows[i]
        if not np.isfinite(center):
            continue
        if center <= np.min(window):
            date_str = str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i])[:10]
            pivots.append((i, date_str, float(center)))
    return pivots


def find_macro_structural_lows(
    df: pd.DataFrame,
    window_half: int = 15,
    min_rally_pct: float = 8.0,
    lookback_bars: int = 750,
) -> List[Dict[str, Any]]:
    """
    Identifies MACRO STRUCTURAL LOWS:
    A price level that served as a significant multi-month turning point
    and triggered at least `min_rally_pct` subsequent expansion.
    """
    if df is None or len(df) < 30:
        return []

    window_df = df.iloc[-lookback_bars:] if len(df) > lookback_bars else df
    n = len(window_df)
    lows = window_df["Low"].to_numpy(dtype=float)
    highs = window_df["High"].to_numpy(dtype=float)
    dates = window_df.index

    structural_pivots = []
    for i in range(window_half, n - 4):
        w_start = max(0, i - window_half)
        w_end = min(n, i + window_half + 1)
        center_low = lows[i]
        if not np.isfinite(center_low):
            continue

        if center_low <= np.min(lows[w_start:w_end]):
            # Measure subsequent forward impulse / expansion
            forward_window = highs[i : min(n, i + 85)]
            if len(forward_window) > 0:
                fwd_high = float(np.max(forward_window))
                rally_pct = ((fwd_high - center_low) / center_low) * 100.0
                if rally_pct >= min_rally_pct:
                    date_str = str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i])[:10]
                    structural_pivots.append({
                        "index": i,
                        "date": date_str,
                        "low_price": float(center_low),
                        "rally_pct": float(rally_pct),
                    })
    return structural_pivots


def get_all_structural_low_floors(
    df: pd.DataFrame,
    lookback_bars: int = 750,
    cluster_pct: float = 2.5,
    min_rally_pct: float = 8.0,
) -> List[Dict[str, Any]]:
    """
    Groups macro structural swing lows into verified structural support floors,
    ranking them by touch frequency, expansion size, and proximity.
    """
    if df is None or len(df) < 30:
        return []

    current_close = float(df["Close"].iloc[-1])
    s_pivots = find_macro_structural_lows(df, window_half=12, min_rally_pct=min_rally_pct, lookback_bars=lookback_bars)
    if not s_pivots:
        return []

    # Sort by price to form price clusters
    sorted_pivots = sorted(s_pivots, key=lambda x: x["low_price"])
    clusters = []
    curr_cluster = [sorted_pivots[0]]

    for p in sorted_pivots[1:]:
        ref_price = curr_cluster[0]["low_price"]
        spread = abs(p["low_price"] - ref_price) / ref_price * 100.0
        if spread <= cluster_pct:
            curr_cluster.append(p)
        else:
            clusters.append(curr_cluster)
            curr_cluster = [p]
    if curr_cluster:
        clusters.append(curr_cluster)

    verified_structural_floors = []
    for cl in clusters:
        prices = [p["low_price"] for p in cl]
        rallies = [p["rally_pct"] for p in cl]
        dates = [p["date"] for p in cl]
        avg_floor = float(np.mean(prices))
        proximity_pct = ((current_close / avg_floor) - 1.0) * 100.0

        if -1.5 <= proximity_pct <= 2.5:
            status = "AT_STRUCTURAL_LOW"
        elif 2.5 < proximity_pct <= 6.0:
            status = "REBOUNDING_FROM_STRUCTURAL_LOW"
        elif proximity_pct < -1.5:
            status = "BROKEN_BELOW"
        else:
            status = "ABOVE_STRUCTURAL_LOW"

        verified_structural_floors.append({
            "structural_low_price": round(avg_floor, 2),
            "touches": len(cl),
            "touch_dates": dates,
            "proximity_pct": round(proximity_pct, 2),
            "max_historical_rally_pct": round(float(np.max(rallies)), 1),
            "status": status,
            "is_active_now": -2.0 <= proximity_pct <= 6.0,
            "min_touch_price": round(float(np.min(prices)), 2),
            "max_touch_price": round(float(np.max(prices)), 2),
        })

    # Sort active first, then closest to price, then touch count
    verified_structural_floors.sort(
        key=lambda s: (1 if s["is_active_now"] else 0, -abs(s["proximity_pct"]), s["touches"]),
        reverse=True,
    )
    return verified_structural_floors


def detect_structural_support(
    df: pd.DataFrame,
    lookback_bars: int = 750,
    max_proximity_pct: float = 6.0,
) -> Optional[Dict[str, Any]]:
    """
    Primary engine for detecting if current price is AT or REBOUNDING from a
    Major Structural Low Support Price (e.g. within -2.0% to +6.0%).
    """
    floors = get_all_structural_low_floors(df, lookback_bars=lookback_bars, min_rally_pct=8.0)
    if not floors:
        return None

    # Filter to active floors around current price
    active = [f for f in floors if -2.0 <= f["proximity_pct"] <= max_proximity_pct]
    if active:
        # Prioritize exact floor tests (-1.5% to +2.5%), then multi-touch, then proximity
        active.sort(
            key=lambda s: (1 if s["status"] == "AT_STRUCTURAL_LOW" else 0, s["touches"], -abs(s["proximity_pct"])),
            reverse=True,
        )
        best = active[0]
        return {
            "structural_low_price": best["structural_low_price"],
            "structural_touches": best["touches"],
            "structural_dates": best["touch_dates"],
            "structural_proximity_pct": best["proximity_pct"],
            "max_historical_rally_pct": best["max_historical_rally_pct"],
            "structural_status": best["status"],
            "is_at_structural_low": best["status"] == "AT_STRUCTURAL_LOW",
        }

    # If no cluster, check if nearest single macro structural low is active
    s_pivots = find_macro_structural_lows(df, window_half=12, min_rally_pct=8.0, lookback_bars=lookback_bars)
    if s_pivots:
        current_close = float(df["Close"].iloc[-1])
        closest_p = min(s_pivots, key=lambda p: abs((current_close / p["low_price"] - 1.0) * 100.0))
        prox = ((current_close / closest_p["low_price"]) - 1.0) * 100.0
        if -2.0 <= prox <= max_proximity_pct:
            status = "AT_STRUCTURAL_LOW" if -1.5 <= prox <= 2.5 else "REBOUNDING_FROM_STRUCTURAL_LOW"
            return {
                "structural_low_price": round(closest_p["low_price"], 2),
                "structural_touches": 1,
                "structural_dates": [closest_p["date"]],
                "structural_proximity_pct": round(prox, 2),
                "max_historical_rally_pct": round(closest_p["rally_pct"], 1),
                "structural_status": status,
                "is_at_structural_low": status == "AT_STRUCTURAL_LOW",
            }

    return None


def get_all_support_floors(
    df: pd.DataFrame,
    lookback_bars: int = 600,
    min_touches: int = 2,
    cluster_pct: float = 2.5,
    min_intervening_rebound_pct: float = 3.0,
) -> List[Dict[str, Any]]:
    """Scans historical data for all multi-touch horizontal support floors."""
    if df is None or len(df) < 40:
        return []

    window_df = df.iloc[-lookback_bars:] if len(df) > lookback_bars else df
    current_close = float(df["Close"].iloc[-1])

    pivots = find_pivot_lows(window_df, fractal_size=3)
    if len(pivots) < min_touches:
        pivots = find_pivot_lows(window_df, fractal_size=2)
        if len(pivots) < min_touches:
            return []

    sorted_pivots = sorted(pivots, key=lambda x: x[2])
    clusters = []
    curr_cluster = [sorted_pivots[0]]

    for p in sorted_pivots[1:]:
        ref_price = curr_cluster[0][2]
        spread_pct = abs(p[2] - ref_price) / ref_price * 100.0
        if spread_pct <= cluster_pct:
            curr_cluster.append(p)
        else:
            clusters.append(curr_cluster)
            curr_cluster = [p]
    if curr_cluster:
        clusters.append(curr_cluster)

    verified_floors = []
    for cl in clusters:
        if len(cl) < min_touches:
            continue

        cl_sorted = sorted(cl, key=lambda x: x[0])
        independent_touches = [cl_sorted[0]]

        for p in cl_sorted[1:]:
            prev = independent_touches[-1]
            idx_diff = p[0] - prev[0]
            if idx_diff < 5:
                continue

            sub_highs = window_df["High"].iloc[prev[0] : p[0] + 1]
            if not sub_highs.empty:
                max_high = float(sub_highs.max())
                rebound_pct = ((max_high / max(prev[2], p[2])) - 1.0) * 100.0
                if rebound_pct >= min_intervening_rebound_pct:
                    independent_touches.append(p)

        if len(independent_touches) >= min_touches:
            prices = [t[2] for t in independent_touches]
            support_level = float(np.mean(prices))
            proximity_pct = ((current_close / support_level) - 1.0) * 100.0
            touch_dates = [t[1] for t in independent_touches]

            if -1.5 <= proximity_pct <= 2.0:
                status = "AT_SUPPORT_FLOOR"
            elif 2.0 < proximity_pct <= 5.5:
                status = "REBOUNDING_FROM_SUPPORT"
            elif proximity_pct < -1.5:
                status = "BROKEN_BELOW"
            else:
                status = "ABOVE_FLOOR"

            verified_floors.append({
                "support_level": round(support_level, 2),
                "touches": len(independent_touches),
                "touch_dates": touch_dates,
                "proximity_pct": round(proximity_pct, 2),
                "current_price": round(current_close, 2),
                "status": status,
                "is_active_now": -2.0 <= proximity_pct <= 5.5,
                "first_touch_date": touch_dates[0],
                "last_touch_date": touch_dates[-1],
                "min_touch_price": round(float(np.min(prices)), 2),
                "max_touch_price": round(float(np.max(prices)), 2),
            })

    verified_floors.sort(key=lambda s: (1 if s["is_active_now"] else 0, -abs(s["proximity_pct"]), s["touches"]), reverse=True)
    return verified_floors


def detect_horizontal_support(
    df: pd.DataFrame,
    lookback_bars: int = 600,
    min_touches: int = 2,
    cluster_pct: float = 2.5,
    max_proximity_pct: float = 5.5,
    min_intervening_rebound_pct: float = 3.0,
) -> Optional[Dict[str, Any]]:
    """Finds active multi-touch horizontal support near current price."""
    floors = get_all_support_floors(
        df,
        lookback_bars=lookback_bars,
        min_touches=min_touches,
        cluster_pct=cluster_pct,
        min_intervening_rebound_pct=min_intervening_rebound_pct,
    )
    if not floors:
        return None

    active = [f for f in floors if -2.0 <= f["proximity_pct"] <= max_proximity_pct]
    if not active:
        return None

    active.sort(key=lambda s: (1 if s["status"] == "AT_SUPPORT_FLOOR" else 0, s["touches"], -abs(s["proximity_pct"])), reverse=True)
    return active[0]


def detect_powerful_macro_floor(df: pd.DataFrame, lookback_bars: int = 750) -> Optional[Dict[str, Any]]:
    """
    Identifies the Most Powerful Macro Structural Low (Rock-Bottom Cycle Floor)
    as illustrated by APOLLOTYRE at ₹377.15, HCLTECH at ₹1,210.2, and FINEORG at ₹4,020.4.
    
    This is the lowest major demand base of the 3-year market cycle where downside
    is mathematically capped and probability of gain back to cycle highs is highest.
    """
    if df is None or len(df) < 50:
        return None

    window_df = df.iloc[-lookback_bars:] if len(df) > lookback_bars else df
    n = len(window_df)
    lows = window_df["Low"].to_numpy(dtype=float)
    highs = window_df["High"].to_numpy(dtype=float)
    dates = window_df.index
    current_close = float(df["Close"].iloc[-1])

    cycle_low = float(np.min(lows))
    cycle_high = float(np.max(highs))
    range_span = cycle_high - cycle_low
    if range_span <= 0:
        return None

    range_percentile = ((current_close - cycle_low) / range_span) * 100.0

    # 1. Collect macro structural pivot lows in bottom 35% of the 3-year cycle range
    macro_pivots = []
    for i in range(12, n - 4):
        if lows[i] <= np.min(lows[max(0, i - 12) : min(n, i + 13)]):
            rel_height = (lows[i] - cycle_low) / range_span
            if rel_height <= 0.35:
                fwd_high = np.max(highs[i : min(n, i + 120)])
                rally = ((fwd_high - lows[i]) / lows[i]) * 100.0
                macro_pivots.append((i, str(dates[i].date()), float(lows[i]), float(rally)))

    if not macro_pivots:
        min_idx = int(np.argmin(lows))
        macro_pivots = [(min_idx, str(dates[min_idx].date()), cycle_low, 25.0)]

    # 2. Cluster into macro rock-bottom zones (4.5% spread)
    macro_pivots.sort(key=lambda x: x[2])
    clusters = []
    curr = [macro_pivots[0]]
    for p in macro_pivots[1:]:
        if abs(p[2] - curr[0][2]) / curr[0][2] * 100.0 <= 4.5:
            curr.append(p)
        else:
            clusters.append(curr)
            curr = [p]
    if curr:
        clusters.append(curr)

    # 3. Pick the lowest multi-touch cluster (the true rock-bottom institutional floor)
    multi_touch_clusters = [cl for cl in clusters if len(cl) >= 2]
    if multi_touch_clusters:
        multi_touch_clusters.sort(key=lambda cl: float(np.mean([p[2] for p in cl])))
        rock_bottom_cl = multi_touch_clusters[0]
    else:
        clusters.sort(key=lambda cl: float(np.mean([p[2] for p in cl])))
        rock_bottom_cl = clusters[0]

    macro_floor = float(np.mean([p[2] for p in rock_bottom_cl]))
    max_rally = float(max(p[3] for p in rock_bottom_cl))
    touches = len(rock_bottom_cl)
    dates_list = [p[1] for p in rock_bottom_cl]

    prox_to_macro_floor = ((current_close / macro_floor) - 1.0) * 100.0
    upside_to_high = ((cycle_high / current_close) - 1.0) * 100.0

    # Probability rating based on range percentile and macro floor proximity
    if -2.5 <= prox_to_macro_floor <= 4.0:
        gain_prob = "VERY HIGH (At Rock-Bottom Floor)"
    elif 4.0 < prox_to_macro_floor <= 16.0:
        gain_prob = "HIGH (Early Rebound from Floor)"
    elif range_percentile <= 25.0:
        gain_prob = "HIGH (Cycle Value Zone)"
    else:
        gain_prob = "MODERATE"

    return {
        "macro_floor_price": round(macro_floor, 2),
        "touches": touches,
        "touch_dates": dates_list,
        "max_historical_rally_pct": round(max_rally, 1),
        "prox_to_macro_floor_pct": round(prox_to_macro_floor, 2),
        "range_percentile": round(range_percentile, 1),
        "cycle_low": round(cycle_low, 2),
        "cycle_high": round(cycle_high, 2),
        "upside_to_cycle_high": round(upside_to_high, 1),
        "gain_probability_rating": gain_prob,
        "is_at_macro_floor": -2.5 <= prox_to_macro_floor <= 4.0,
        "is_rebounding_from_macro": 4.0 < prox_to_macro_floor <= 16.0,
    }
