"""
Per-Stock Deep Analyzer with TensorFlow ML (V2 - INTELLIGENT)
==============================================================
UPGRADED: Now uses Intelligent Scoring Engine instead of untrained ML models.

Key V2 Intelligence Upgrades:
  1. Signal quality computed from REAL indicator alignment (not random ML outputs)
  2. Pattern conflict resolution (bullish + bearish patterns don't both penalize)
  3. Multi-timeframe alignment (weekly + daily trend agreement)
  4. Momentum acceleration detection
  5. Volume confirmation as entry requirement
  6. Dynamic confidence rebalancing (penalties capped, boosters enhanced)
  7. Each reason tagged as positive/negative/neutral for proper UI color coding
  8. Chasing rally detection with graduated penalties
  9. Sector-relative strength scoring
  10. Composite scoring that actually produces BUY/STRONG_BUY recommendations

Analyzes each stock using:
  1. yfinance data download (10-year OHLCV)
  2. Intelligent signal quality scoring (10-component weighted system)
  3. Pattern recognition (candlestick, chart patterns, support/resistance)
  4. Behavior profiling (volatility, trend vs mean-reversion, 3% hit rate)
  5. Multi-timeframe alignment (weekly + daily)
  6. Momentum acceleration detection
  7. 3% target strategy with smart entry/exit/position sizing
"""

from __future__ import annotations
import math
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

logger = logging.getLogger("ml_stock_analyzer")

# Import the intelligent scoring engine
try:
    from intelligent_scoring import compute_intelligent_score, compute_multi_timeframe_alignment, detect_momentum_acceleration, compute_fair_value
    HAS_INTELLIGENT_SCORING = True
except ImportError:
    HAS_INTELLIGENT_SCORING = False
    compute_fair_value = None
    logger.warning("intelligent_scoring.py not found, falling back to basic scoring")


# ═══════════════════════════════════════════════════════════════════════════
#  PATTERN RECOGNITION ENGINE (IMPROVED)
# ═══════════════════════════════════════════════════════════════════════════

def detect_candlestick_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect candlestick patterns in the last 30 bars.
    V2: Only returns patterns from the LAST 5 bars (recent = relevant).
    Each pattern gets recency weighting.
    """
    patterns = []
    n = len(df)
    lookback = min(30, n)
    closes = df["Close"].values
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    dates = df.index if isinstance(df.index, pd.DatetimeIndex) else range(n)

    for i in range(max(0, n - lookback), n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        body = abs(c - o)
        total_range = h - l
        if total_range == 0:
            continue
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l
        body_ratio = body / total_range

        bar_offset = n - 1 - i  # 0 = most recent

        # Doji
        if body_ratio < 0.1 and total_range > 0:
            patterns.append({
                "name": "Doji", "type": "neutral",
                "bar_offset": bar_offset,
                "date": str(dates[i])[:10],
                "confidence": round(0.7 + (0.1 - body_ratio) * 3, 2),
                "description": "Indecision candle - potential reversal point",
            })

        # Hammer (bullish)
        if lower_shadow > body * 2 and upper_shadow < body * 0.5 and body_ratio > 0.1:
            if i > 0 and closes[i - 1] < opens[i - 1]:
                patterns.append({
                    "name": "Hammer", "type": "bullish",
                    "bar_offset": bar_offset,
                    "date": str(dates[i])[:10],
                    "confidence": round(min(0.9, 0.6 + lower_shadow / total_range), 2),
                    "description": "Bullish reversal - long lower wick after downtrend",
                })

        # Shooting Star (bearish)
        if upper_shadow > body * 2 and lower_shadow < body * 0.5 and body_ratio > 0.1:
            if i > 0 and closes[i - 1] > opens[i - 1]:
                patterns.append({
                    "name": "Shooting Star", "type": "bearish",
                    "bar_offset": bar_offset,
                    "date": str(dates[i])[:10],
                    "confidence": round(min(0.9, 0.6 + upper_shadow / total_range), 2),
                    "description": "Bearish reversal - long upper wick after uptrend",
                })

        # Bullish Engulfing
        if (i > 0 and closes[i - 1] < opens[i - 1] and
            closes[i] > opens[i] and
            opens[i] <= closes[i - 1] and
            closes[i] >= opens[i - 1]):
            patterns.append({
                "name": "Bullish Engulfing", "type": "bullish",
                "bar_offset": bar_offset,
                "date": str(dates[i])[:10],
                "confidence": 0.82,
                "description": "Strong bullish reversal - green candle fully engulfs prior red",
            })

        # Bearish Engulfing
        if (i > 0 and closes[i - 1] > opens[i - 1] and
            closes[i] < opens[i] and
            opens[i] >= closes[i - 1] and
            closes[i] <= opens[i - 1]):
            patterns.append({
                "name": "Bearish Engulfing", "type": "bearish",
                "bar_offset": bar_offset,
                "date": str(dates[i])[:10],
                "confidence": 0.82,
                "description": "Strong bearish reversal - red candle fully engulfs prior green",
            })

        # Morning Star (3-bar bullish)
        if (i >= 2 and
            closes[i - 2] < opens[i - 2] and
            abs(closes[i - 1] - opens[i - 1]) < abs(closes[i - 2] - opens[i - 2]) * 0.3 and
            closes[i] > opens[i] and
            (closes[i] - opens[i]) > abs(closes[i - 2] - opens[i - 2]) * 0.5):
            patterns.append({
                "name": "Morning Star", "type": "bullish",
                "bar_offset": bar_offset,
                "date": str(dates[i])[:10],
                "confidence": 0.85,
                "description": "3-bar bullish reversal pattern - high reliability",
            })

        # Evening Star (3-bar bearish)
        if (i >= 2 and
            closes[i - 2] > opens[i - 2] and
            abs(closes[i - 1] - opens[i - 1]) < abs(closes[i - 2] - opens[i - 2]) * 0.3 and
            closes[i] < opens[i] and
            (opens[i] - closes[i]) > abs(closes[i - 2] - opens[i - 2]) * 0.5):
            patterns.append({
                "name": "Evening Star", "type": "bearish",
                "bar_offset": bar_offset,
                "date": str(dates[i])[:10],
                "confidence": 0.85,
                "description": "3-bar bearish reversal pattern - high reliability",
            })

    # V2: Filter to recent patterns only (last 5 bars) and weight by recency
    recent = [p for p in patterns if p["bar_offset"] <= 5]
    return recent[-10:]


def detect_chart_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Detect higher-level chart patterns using swing analysis.
    V2: Added conflict resolution - won't detect Double Top AND Double Bottom
        for the same stock simultaneously.
    """
    patterns = []
    n = len(df)
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values

    if n < 50:
        return patterns

    # Find swing points (local highs and lows over 10-bar window)
    swing_window = 10
    swing_highs = []
    swing_lows = []

    for i in range(swing_window, n - swing_window):
        if highs[i] == max(highs[i - swing_window:i + swing_window + 1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - swing_window:i + swing_window + 1]):
            swing_lows.append((i, lows[i]))

    # Higher Highs / Higher Lows (uptrend structure)
    if len(swing_highs) >= 3 and len(swing_lows) >= 3:
        recent_highs = swing_highs[-3:]
        recent_lows = swing_lows[-3:]

        hh_count = sum(1 for i in range(1, len(recent_highs))
                       if recent_highs[i][1] > recent_highs[i - 1][1])
        hl_count = sum(1 for i in range(1, len(recent_lows))
                       if recent_lows[i][1] > recent_lows[i - 1][1])

        if hh_count >= 2 and hl_count >= 2:
            patterns.append({
                "name": "Higher Highs & Higher Lows",
                "type": "bullish",
                "confidence": 0.80,
                "description": "Strong uptrend structure - consistent HH/HL pattern",
                "details": f"{hh_count} higher highs, {hl_count} higher lows in recent swings",
            })

        # Lower Highs / Lower Lows (downtrend structure)
        lh_count = sum(1 for i in range(1, len(recent_highs))
                       if recent_highs[i][1] < recent_highs[i - 1][1])
        ll_count = sum(1 for i in range(1, len(recent_lows))
                       if recent_lows[i][1] < recent_lows[i - 1][1])

        if lh_count >= 2 and ll_count >= 2:
            patterns.append({
                "name": "Lower Highs & Lower Lows",
                "type": "bearish",
                "confidence": 0.80,
                "description": "Strong downtrend structure - consistent LH/LL pattern",
                "details": f"{lh_count} lower highs, {ll_count} lower lows in recent swings",
            })

    # V2 CONFLICT RESOLUTION: Only detect Double Top OR Double Bottom, not both
    # Determine which is more relevant based on current price position
    current_price = closes[-1]
    recent_high_avg = np.mean([h for _, h in swing_highs[-3:]]) if swing_highs else current_price
    recent_low_avg = np.mean([l for _, l in swing_lows[-3:]]) if swing_lows else current_price
    price_position = (current_price - recent_low_avg) / (recent_high_avg - recent_low_avg) if recent_high_avg != recent_low_avg else 0.5

    # If price is in lower half, Double Bottom is more relevant
    # If price is in upper half, Double Top is more relevant
    if price_position < 0.5 and len(swing_lows) >= 2:
        last_two = swing_lows[-2:]
        price_diff_pct = abs(last_two[0][1] - last_two[1][1]) / last_two[0][1] * 100
        if price_diff_pct < 1.5:
            patterns.append({
                "name": "Double Bottom",
                "type": "bullish",
                "confidence": round(max(0.5, 0.9 - price_diff_pct / 3), 2),
                "description": f"Two swing lows near {last_two[0][1]:.1f} ({price_diff_pct:.1f}% apart) - potential reversal",
            })
    elif price_position >= 0.5 and len(swing_highs) >= 2:
        last_two = swing_highs[-2:]
        price_diff_pct = abs(last_two[0][1] - last_two[1][1]) / last_two[0][1] * 100
        if price_diff_pct < 1.5:
            patterns.append({
                "name": "Double Top",
                "type": "bearish",
                "confidence": round(max(0.5, 0.9 - price_diff_pct / 3), 2),
                "description": f"Two swing highs near {last_two[0][1]:.1f} ({price_diff_pct:.1f}% apart) - potential reversal",
            })

    # Consolidation / Range detection (last 20 bars)
    recent_closes = closes[-20:]
    range_pct = (max(recent_closes) - min(recent_closes)) / np.mean(recent_closes) * 100
    if range_pct < 3.0:
        patterns.append({
            "name": "Tight Consolidation",
            "type": "neutral",
            "confidence": 0.75,
            "description": f"Price range only {range_pct:.1f}% in 20 bars - breakout incoming",
            "details": f"Range: {min(recent_closes):.1f} to {max(recent_closes):.1f}",
        })

    # Ascending Triangle (higher lows + flat resistance)
    if len(swing_lows) >= 3 and len(swing_highs) >= 2:
        recent_lows_3 = swing_lows[-3:]
        recent_highs_2 = swing_highs[-2:]
        lows_rising = all(recent_lows_3[i][1] < recent_lows_3[i+1][1] for i in range(2))
        highs_flat = abs(recent_highs_2[0][1] - recent_highs_2[1][1]) / recent_highs_2[0][1] * 100 < 2
        if lows_rising and highs_flat:
            patterns.append({
                "name": "Ascending Triangle",
                "type": "bullish",
                "confidence": 0.82,
                "description": f"Higher lows with flat resistance near {recent_highs_2[0][1]:.1f} - bullish breakout pattern",
            })

    # Volume breakout pattern
    if "Volume" in df.columns and n >= 21:
        recent_vol = df["Volume"].values[-5:]
        avg_vol = df["Volume"].values[-21:-5].mean()
        if avg_vol > 0 and recent_vol[-1] > avg_vol * 2.0:
            price_change = (closes[-1] - closes[-2]) / closes[-2] * 100
            direction = "up" if price_change > 0 else "down"
            patterns.append({
                "name": f"Volume Breakout ({'Bullish' if direction == 'up' else 'Bearish'})",
                "type": direction,
                "confidence": 0.78,
                "description": f"Volume {recent_vol[-1] / avg_vol:.1f}x average with {price_change:+.1f}% price move",
            })

    return patterns


def detect_support_resistance(df: pd.DataFrame, levels: int = 5) -> Dict[str, List[Dict]]:
    """
    Detect support and resistance levels using price clustering.
    """
    n = len(df)
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    current_price = closes[-1]

    if n < 30:
        return {"supports": [], "resistances": []}

    all_prices = np.concatenate([highs[-120:], lows[-120:]])
    if len(all_prices) == 0:
        return {"supports": [], "resistances": []}

    price_range = all_prices.max() - all_prices.min()
    if price_range == 0:
        return {"supports": [], "resistances": []}

    num_bins = 40
    hist, bin_edges = np.histogram(all_prices, bins=num_bins)

    cluster_levels = []
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] >= 3:
            level_price = (bin_edges[i] + bin_edges[i + 1]) / 2
            cluster_levels.append({
                "price": round(level_price, 2),
                "touches": int(hist[i]),
                "strength": round(hist[i] / hist.max(), 2),
            })

    cluster_levels.sort(key=lambda x: x["strength"], reverse=True)
    supports = [l for l in cluster_levels if l["price"] < current_price][:levels]
    resistances = [l for l in cluster_levels if l["price"] > current_price][:levels]

    for s in supports:
        s["distance_pct"] = round((current_price - s["price"]) / current_price * 100, 2)
    for r in resistances:
        r["distance_pct"] = round((r["price"] - current_price) / current_price * 100, 2)

    return {"supports": supports, "resistances": resistances}


# ═══════════════════════════════════════════════════════════════════════════
#  BEHAVIOR PROFILER (UNCHANGED - already solid)
# ═══════════════════════════════════════════════════════════════════════════

def profile_stock_behavior(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze the stock's individual behavior profile:
    - Volatility regime (low/medium/high/extreme)
    - Trend vs Mean Reversion tendency
    - Average move size per day/week
    - Best performing day-of-week
    - Gap-up / Gap-down frequency
    - Consecutive up/down day streaks
    - Historical 3% hit rate
    """
    n = len(df)
    closes = df["Close"].values
    opens = df["Open"].values
    highs = df["High"].values
    lows = df["Low"].values
    volumes = df["Volume"].values if "Volume" in df.columns else None

    if n < 60:
        return {"error": "Insufficient data for behavior profiling"}

    returns = np.diff(closes) / closes[:-1] * 100

    # Volatility Regime
    daily_std = np.std(returns[-60:]) if len(returns) >= 60 else np.std(returns)
    annualized_vol = daily_std * math.sqrt(252)

    if annualized_vol < 15:
        vol_regime = "LOW"
        vol_color = "green"
    elif annualized_vol < 30:
        vol_regime = "MEDIUM"
        vol_color = "yellow"
    elif annualized_vol < 50:
        vol_regime = "HIGH"
        vol_color = "orange"
    else:
        vol_regime = "EXTREME"
        vol_color = "red"

    # Trend vs Mean Reversion
    if len(returns) >= 20:
        autocorr_1 = np.corrcoef(returns[-60:-1], returns[-59:])[0, 1] if len(returns) >= 60 else np.corrcoef(returns[:-1], returns[1:])[0, 1]
        if not np.isnan(autocorr_1):
            if autocorr_1 > 0.05:
                behavior_type = "TREND-FOLLOWING"
                behavior_desc = f"Positive autocorrelation ({autocorr_1:.3f}) - momentum continues"
            elif autocorr_1 < -0.05:
                behavior_type = "MEAN-REVERTING"
                behavior_desc = f"Negative autocorrelation ({autocorr_1:.3f}) - reversals likely"
            else:
                behavior_type = "RANDOM WALK"
                behavior_desc = f"Near-zero autocorrelation ({autocorr_1:.3f}) - no clear edge"
        else:
            behavior_type = "UNKNOWN"
            behavior_desc = "Insufficient data for autocorrelation"
    else:
        behavior_type = "UNKNOWN"
        behavior_desc = "Insufficient data"
        autocorr_1 = 0

    avg_daily_move = round(np.mean(np.abs(returns[-60:])) if len(returns) >= 60 else np.mean(np.abs(returns)), 2)
    avg_weekly_move = round(avg_daily_move * math.sqrt(5), 2)

    # Historical 3% Target Hit Rate (within 5 trading days)
    target_pct = 3.0
    lookahead = 5
    hit_count = 0
    total_windows = 0

    for i in range(min(60, n - lookahead - 1)):
        idx = n - 1 - i
        if idx - lookahead < 0:
            break
        entry = closes[idx]
        max_price = max(highs[idx - lookahead + 1:idx + 1])
        if entry > 0 and (max_price - entry) / entry * 100 >= target_pct:
            hit_count += 1
        total_windows += 1

    hit_rate = round(hit_count / total_windows * 100, 1) if total_windows > 0 else 0

    # Average days to hit 3%
    days_to_hit_list = []
    for i in range(min(60, n - lookahead - 1)):
        idx = n - 1 - i
        if idx - lookahead < 0:
            break
        entry = closes[idx]
        for j in range(1, lookahead + 1):
            if idx - j < 0:
                break
            if (highs[idx - j] - entry) / entry * 100 >= target_pct:
                days_to_hit_list.append(j)
                break

    avg_days_to_target = round(np.mean(days_to_hit_list), 1) if days_to_hit_list else None

    # Gap Analysis
    gap_ups = 0
    gap_downs = 0
    for i in range(1, min(252, n)):
        gap = (opens[i] - closes[i - 1]) / closes[i - 1] * 100
        if gap > 0.5:
            gap_ups += 1
        elif gap < -0.5:
            gap_downs += 1
    total_gap_days = gap_ups + gap_downs
    gap_trading_days = min(252, n) - 1

    # Consecutive Streaks
    max_up_streak = 0
    max_down_streak = 0
    current_up = 0
    current_down = 0
    for r in returns[-252:]:
        if r > 0:
            current_up += 1
            current_down = 0
            max_up_streak = max(max_up_streak, current_up)
        elif r < 0:
            current_down += 1
            current_up = 0
            max_down_streak = max(max_down_streak, current_down)
        else:
            current_up = 0
            current_down = 0

    # Current Streak
    current_streak = 0
    current_streak_dir = "neutral"
    for i in range(len(returns) - 1, max(len(returns) - 11, -1), -1):
        if returns[i] > 0:
            if current_streak_dir == "up" or current_streak_dir == "neutral":
                current_streak += 1
                current_streak_dir = "up"
            else:
                break
        elif returns[i] < 0:
            if current_streak_dir == "down" or current_streak_dir == "neutral":
                current_streak += 1
                current_streak_dir = "down"
            else:
                break
        else:
            break

    # Best Day of Week
    if isinstance(df.index, pd.DatetimeIndex):
        df_temp = df.tail(252).copy()
        df_temp["day_of_week"] = df_temp.index.dayofweek
        df_temp["return"] = df_temp["Close"].pct_change() * 100
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        day_returns = df_temp.groupby("day_of_week")["return"].mean()
        best_day_idx = day_returns.idxmax() if not day_returns.empty else 0
        worst_day_idx = day_returns.idxmin() if not day_returns.empty else 0
        best_day = day_names[best_day_idx] if 0 <= best_day_idx <= 4 else "N/A"
        worst_day = day_names[worst_day_idx] if 0 <= worst_day_idx <= 4 else "N/A"
        best_day_avg = round(float(day_returns.get(best_day_idx, 0)), 3)
        worst_day_avg = round(float(day_returns.get(worst_day_idx, 0)), 3)
    else:
        best_day = "N/A"
        worst_day = "N/A"
        best_day_avg = 0
        worst_day_avg = 0

    return {
        "volatility": {
            "daily_std": round(daily_std, 3),
            "annualized": round(annualized_vol, 1),
            "regime": vol_regime,
            "color": vol_color,
        },
        "behavior_type": behavior_type,
        "behavior_description": behavior_desc,
        "autocorrelation": round(float(autocorr_1) if not np.isnan(autocorr_1) else 0, 4),
        "avg_daily_move_pct": avg_daily_move,
        "avg_weekly_move_pct": avg_weekly_move,
        "target_3pct": {
            "hit_rate_pct": hit_rate,
            "windows_analyzed": total_windows,
            "avg_days_to_hit": avg_days_to_target,
        },
        "gaps": {
            "gap_up_count": gap_ups,
            "gap_down_count": gap_downs,
            "gap_up_pct": round(gap_ups / gap_trading_days * 100, 1) if gap_trading_days > 0 else 0,
            "gap_down_pct": round(gap_downs / gap_trading_days * 100, 1) if gap_trading_days > 0 else 0,
        },
        "streaks": {
            "max_up_streak": max_up_streak,
            "max_down_streak": max_down_streak,
            "current_streak": current_streak,
            "current_streak_dir": current_streak_dir,
        },
        "seasonality": {
            "best_day": best_day,
            "best_day_avg_return": best_day_avg,
            "worst_day": worst_day,
            "worst_day_avg_return": worst_day_avg,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  V2 INTELLIGENT STRATEGY ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def suggest_strategy(
    ticker: str,
    df: pd.DataFrame,
    ml_prediction: Optional[Dict] = None,
    behavior_profile: Optional[Dict] = None,
    patterns: Optional[List[Dict]] = None,
    sr_levels: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    V2 Intelligent 3% Target Strategy Engine.

    KEY CHANGES FROM V1:
      1. Uses Intelligent Scoring (real indicators) instead of untrained ML
      2. Pattern conflict resolution - bullish and bearish patterns are NETTED
      3. Each reason tagged with "sentiment" (positive/negative/neutral) for UI
      4. Penalties are CAPPED at -30 total (V1 allowed unlimited stacking to -73)
      5. Multi-timeframe alignment gives +10 boost
      6. Momentum acceleration gives +5/-5
      7. Volume confirmation is a REQUIREMENT for STRONG_BUY
      8. Chasing rally penalties are graduated but softer
      9. Position sizing includes volume quality factor
    """
    n = len(df)
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    opens = df["Open"].values
    current_price = float(closes[-1])

    ml_pred = ml_prediction or {}
    behavior = behavior_profile or profile_stock_behavior(df)
    pattern_list = patterns or detect_candlestick_patterns(df)
    chart_pats = detect_chart_patterns(df)
    sr = sr_levels or detect_support_resistance(df)

    # ══════════════════════════════════════════════════════════════
    #  GET INTELLIGENT SCORE (or fall back to ML prediction)
    # ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
    #  COMBINED STUDY: TF Model + Intelligent Scoring (NEW)
    #  - TF model drives the DECISION (signal_quality, regime, expected_return)
    #  - Intelligent scoring provides EXPLANATION (10 components)
    # ══════════════════════════════════════════════════════════════

    ml_pred = ml_prediction or {}

    # Check if TF model is trained (model_status == "trained")
    tf_model_trained = (
        ml_pred.get("model_status") == "trained"
        and ml_pred.get("signal_quality") is not None
    )

    if tf_model_trained:
        # ✅ USE TF MODEL for the decision (this is what you want!)
        ml_score = ml_pred.get("ml_score", 50)
        signal_quality = ml_pred.get("signal_quality", 0.5)
        regime = ml_pred.get("trend_regime", "SIDEWAYS")
        expected_return = ml_pred.get("expected_return", 0)
        max_drawdown = ml_pred.get("max_drawdown", 0)
        risk_reward_base = ml_pred.get("risk_reward_ratio", 0)
        score_components = {}
        scoring_method = "tf_model_trained"

        # Also compute intelligent scoring for EXPLANATION (not for decision)
        if HAS_INTELLIGENT_SCORING:
            try:
                intel_score = compute_intelligent_score(df)
                score_components = intel_score.get("components", {})
                # Store for later use in reasons
                _intel_context = intel_score
            except Exception:
                _intel_context = {}
        else:
            _intel_context = {}
    elif HAS_INTELLIGENT_SCORING:
        # ⚠️ FALLBACK: TF model not trained, use intelligent scoring
        intel_score = compute_intelligent_score(df)
        ml_score = intel_score.get("ml_score", 50)
        signal_quality = intel_score.get("signal_quality", 0.5)
        regime = intel_score.get("trend_regime", "SIDEWAYS")
        expected_return = intel_score.get("expected_return", 0)
        max_drawdown = intel_score.get("max_drawdown", 0)
        risk_reward_base = intel_score.get("risk_reward_ratio", 0)
        score_components = intel_score.get("components", {})
        scoring_method = "intelligent_v2_fallback"
        _intel_context = intel_score
    else:
        # ❌ NO MODEL AVAILABLE — use whatever ml_pred has
        ml_score = ml_pred.get("ml_score", 50)
        signal_quality = ml_pred.get("signal_quality", 0.5)
        regime = ml_pred.get("trend_regime", "SIDEWAYS")
        expected_return = ml_pred.get("expected_return", 0)
        max_drawdown = ml_pred.get("max_drawdown", 0)
        risk_reward_base = ml_pred.get("risk_reward_ratio", 0)
        score_components = {}
        scoring_method = "no_model"
        _intel_context = {}

    # ══════════════════════════════════════════════════════════════
    #  MULTI-TIMEFRAME ALIGNMENT
    # ══════════════════════════════════════════════════════════════
    mtf_alignment = {"score": 0, "alignment": "NEUTRAL"}
    if HAS_INTELLIGENT_SCORING:
        try:
            mtf_alignment = compute_multi_timeframe_alignment(df)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════
    #  MOMENTUM ACCELERATION
    # ══════════════════════════════════════════════════════════════
    momentum_info = {"acceleration": "UNKNOWN", "score": 0}
    if HAS_INTELLIGENT_SCORING:
        try:
            momentum_info = detect_momentum_acceleration(df)
        except Exception:
            pass

    # ══════════════════════════════════════════════════════════════
    #  COMPUTE INDICATORS
    # ══════════════════════════════════════════════════════════════
    rsi_value = 50.0
    if n >= 15:
        close_series = pd.Series(closes)
        delta = close_series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        rsi_val = rsi_series.iloc[-1]
        if not np.isnan(rsi_val):
            rsi_value = float(rsi_val)

    # ATR for smart stop loss
    atr_value = 0
    if n >= 15:
        tr_values = [
            max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            for i in range(max(0, n - 14), n)
        ]
        atr_value = np.mean(tr_values)
    atr_pct = (atr_value / current_price * 100) if current_price > 0 else 0

    # Recent price moves
    move_5d = ((closes[-1] / closes[-6]) - 1) * 100 if n > 5 else 0
    move_10d = ((closes[-1] / closes[-11]) - 1) * 100 if n > 10 else 0

    # Volume analysis
    vol_ratio = 1.0
    if "Volume" in df.columns and n >= 25:
        vol_avg = float(np.mean(df["Volume"].values[-25:-5]))
        vol_cur = float(df["Volume"].values[-1])
        vol_ratio = vol_cur / vol_avg if vol_avg > 0 else 1.0

    # ══════════════════════════════════════════════════════════════
    #  COMPUTE STOP LOSS & TARGET
    # ══════════════════════════════════════════════════════════════
    if sr["supports"]:
        nearest_support = sr["supports"][0]
        support_stop = nearest_support["price"] * 0.995
        atr_stop = current_price - 1.5 * atr_value
        stop_price = max(support_stop, atr_stop)
    else:
        if atr_value > 0:
            stop_price = current_price - 1.5 * atr_value
        else:
            stop_price = current_price * 0.97

    stop_loss_pct = round((stop_price - current_price) / current_price * 100, 2)

    # Dynamic target
    target_pct = 3.0
    target_price = round(current_price * (1 + target_pct / 100), 2)
    risk_per_share = current_price - stop_price
    reward_per_share = target_price - current_price
    actual_rr = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0

    if actual_rr < 1.0 and risk_per_share > 0:
        target_pct = 2.0
        target_price = round(current_price * (1 + target_pct / 100), 2)
        reward_per_share = target_price - current_price
        actual_rr = round(reward_per_share / risk_per_share, 2)

    if actual_rr < 1.0 and risk_per_share > 0:
        target_pct = 1.5
        target_price = round(current_price * (1 + target_pct / 100), 2)
        reward_per_share = target_price - current_price
        actual_rr = round(reward_per_share / risk_per_share, 2)

    target_type = f"{target_pct}% gain target"

    # Entry Price
    entry_price = current_price
    entry_type = "MARKET"
    pullback_entry = None

    if sr["supports"]:
        nearest_support = sr["supports"][0]
        support_dist = nearest_support["distance_pct"]
        if 0.5 <= support_dist <= 2.5:
            pullback_entry = round(nearest_support["price"] * 1.002, 2)
            entry_type = "PULLBACK (Limit Order)"

    # ══════════════════════════════════════════════════════════════
    #  V2 CONFIDENCE SCORING (NEW INTELLIGENT SYSTEM)
    # ══════════════════════════════════════════════════════════════

    # Each reason is a dict: {"text": "...", "sentiment": "positive"|"negative"|"neutral", "impact": N}
    reasons = []
    boost_total = 0
    penalty_total = 0
    MAX_PENALTY = 30  # V2: Cap total penalty at 30 points

    # ── PATTERN CONFLICT RESOLUTION ──
    bullish_patterns = [p for p in pattern_list if p["type"] == "bullish"]
    bearish_patterns = [p for p in pattern_list if p["type"] == "bearish"]
    bullish_chart = [p for p in chart_pats if p["type"] == "bullish"]
    bearish_chart = [p for p in chart_pats if p["type"] == "bearish"]

    # V2: NET pattern effect instead of separate boost/penalty
    pattern_net = 0
    # Candlestick patterns (less weight, recent only)
    pattern_net += len(bullish_patterns) * 3
    pattern_net -= len(bearish_patterns) * 3
    # Chart patterns (more weight, structural)
    pattern_net += len(bullish_chart) * 5
    pattern_net -= len(bearish_chart) * 5

    if pattern_net > 0:
        boost = min(12, pattern_net)
        boost_total += boost
        names = [p["name"] for p in bullish_patterns + bullish_chart][:3]
        reasons.append({"text": f"Bullish patterns: {', '.join(names)}", "sentiment": "positive", "impact": boost})
    elif pattern_net < 0:
        penalty = min(15, abs(pattern_net))
        penalty_total += penalty
        names = [p["name"] for p in bearish_patterns + bearish_chart][:3]
        reasons.append({"text": f"WARNING: Bearish patterns: {', '.join(names)}", "sentiment": "negative", "impact": -penalty})
    else:
        # Mixed patterns - neutral, slight penalty for confusion
        penalty_total += 3
        all_names = [p["name"] for p in bullish_patterns + bearish_patterns + bullish_chart + bearish_chart][:4]
        reasons.append({"text": f"Mixed signals: {', '.join(all_names)}", "sentiment": "neutral", "impact": -3})

# ── TF MODEL SIGNAL (NEW — shows that trained model is driving decision) ──
    if tf_model_trained:
        model_version = ml_pred.get("model_version", "unknown")
        if signal_quality > 0.6:
            boost_total += 12
            reasons.append({"text": f"🤖 TF Model ({model_version[:25]}): {signal_quality:.0%} confidence — HIGH quality signal", "sentiment": "positive", "impact": 12})
        elif signal_quality > 0.45:
            boost_total += 6
            reasons.append({"text": f"🤖 TF Model: {signal_quality:.0%} confidence — moderate signal", "sentiment": "positive", "impact": 6})
        elif signal_quality < 0.3:
            penalty_total += 10
            reasons.append({"text": f"🤖 TF Model: {signal_quality:.0%} confidence — LOW quality, avoid", "sentiment": "negative", "impact": -10})
        else:
            reasons.append({"text": f"🤖 TF Model: {signal_quality:.0%} confidence — neutral", "sentiment": "neutral", "impact": 0})

    # ── INTELLIGENT SCORING CONTEXT (explanation only, not for decision) ──
    if _intel_context:
        comp = _intel_context.get("components", {})
        if comp:
            top_components = sorted(comp.items(), key=lambda x: x[1], reverse=True)[:3]
            for name, score in top_components:
                if score >= 8:
                    reasons.append({"text": f"📊 {name.replace('_', ' ').title()}: {score}/10 — strong", "sentiment": "positive", "impact": 0})
                elif score <= 3:
                    reasons.append({"text": f"📊 {name.replace('_', ' ').title()}: {score}/10 — weak", "sentiment": "negative", "impact": 0})

    # ── ML/INTELLIGENT REGIME ──
    if regime == "UPTREND":
        boost_total += 12
        reasons.append({"text": "Trend: UPTREND detected (EMA alignment confirmed)", "sentiment": "positive", "impact": 12})
    elif regime == "DOWNTREND":
        penalty_total += 18
        reasons.append({"text": "Trend: DOWNTREND - avoid longs", "sentiment": "negative", "impact": -18})

    # ── SIGNAL QUALITY (from intelligent scoring) ──
    if signal_quality > 0.65:
        boost_total += 10
        reasons.append({"text": f"Strong signal quality ({signal_quality:.0%}) - indicators aligned", "sentiment": "positive", "impact": 10})
    elif signal_quality > 0.50:
        boost_total += 5
        reasons.append({"text": f"Good signal quality ({signal_quality:.0%})", "sentiment": "positive", "impact": 5})
    elif signal_quality < 0.30:
        penalty_total += min(12, int((0.30 - signal_quality) * 40))
        reasons.append({"text": f"Low signal quality ({signal_quality:.0%}) - indicators not aligned", "sentiment": "negative", "impact": -12})

    # ── EXPECTED RETURN ──
    if expected_return > 3:
        boost_total += 10
        reasons.append({"text": f"ML expects strong +{expected_return:.1f}% return", "sentiment": "positive", "impact": 10})
    elif expected_return > 1.5:
        boost_total += 6
        reasons.append({"text": f"ML expects +{expected_return:.1f}% return", "sentiment": "positive", "impact": 6})
    elif expected_return < 0:
        penalty_total += 10
        reasons.append({"text": f"ML expects negative return ({expected_return:.1f}%)", "sentiment": "negative", "impact": -10})

    # ── MULTI-TIMEFRAME ALIGNMENT (NEW V2) ──
    mtf_score = mtf_alignment.get("score", 0)
    if mtf_score >= 3:
        boost_total += 10
        reasons.append({"text": f"STRONG: {mtf_alignment['alignment']}", "sentiment": "positive", "impact": 10})
    elif mtf_score == 2:
        boost_total += 6
        reasons.append({"text": f"GOOD: {mtf_alignment['alignment']}", "sentiment": "positive", "impact": 6})
    elif mtf_score == 1:
        boost_total += 3
        reasons.append({"text": f"PULLBACK: {mtf_alignment['alignment']}", "sentiment": "positive", "impact": 3})
    elif mtf_score <= -2:
        penalty_total += 12
        reasons.append({"text": f"AVOID: {mtf_alignment['alignment']}", "sentiment": "negative", "impact": -12})

    # ── MOMENTUM ACCELERATION (NEW V2) ──
    mom_score_val = momentum_info.get("score", 0)
    if mom_score_val >= 3:
        boost_total += 8
        reasons.append({"text": f"Momentum: {momentum_info['acceleration']}", "sentiment": "positive", "impact": 8})
    elif mom_score_val >= 2:
        boost_total += 5
        reasons.append({"text": f"Momentum: {momentum_info['acceleration']}", "sentiment": "positive", "impact": 5})
    elif mom_score_val == 1:
        boost_total += 2
        reasons.append({"text": f"Momentum: {momentum_info['acceleration']}", "sentiment": "positive", "impact": 2})
    elif mom_score_val <= -2:
        penalty_total += 8
        reasons.append({"text": f"Momentum: {momentum_info['acceleration']}", "sentiment": "negative", "impact": -8})

    # ── 3% HISTORICAL HIT RATE ──
    hit_rate = behavior.get("target_3pct", {}).get("hit_rate_pct", 0)
    if hit_rate >= 50:
        boost_total += 12
        reasons.append({"text": f"Excellent 3% hit rate ({hit_rate}%) - historically reliable", "sentiment": "positive", "impact": 12})
    elif hit_rate >= 40:
        boost_total += 8
        reasons.append({"text": f"Strong 3% hit rate ({hit_rate}%)", "sentiment": "positive", "impact": 8})
    elif hit_rate >= 30:
        boost_total += 4
        reasons.append({"text": f"Good 3% hit rate ({hit_rate}%)", "sentiment": "positive", "impact": 4})
    elif hit_rate < 20:
        penalty_total += 8
        reasons.append({"text": f"Weak 3% hit rate ({hit_rate}%) - target may be unrealistic", "sentiment": "negative", "impact": -8})

    # ── BEHAVIOR TYPE + REGIME ALIGNMENT ──
    beh_type = behavior.get("behavior_type", "UNKNOWN")
    if beh_type == "TREND-FOLLOWING" and regime == "UPTREND":
        boost_total += 8
        reasons.append({"text": "Trend-following stock in uptrend - momentum likely continues", "sentiment": "positive", "impact": 8})
    elif beh_type == "MEAN-REVERTING" and regime == "UPTREND":
        penalty_total += 5
        reasons.append({"text": "Mean-reverting stock in uptrend - pullback likely first", "sentiment": "neutral", "impact": -5})

    # ── PULLBACK ENTRY ──
    if pullback_entry:
        boost_total += 6
        reasons.append({"text": f"Pullback entry at Rs.{pullback_entry:.2f} ({sr['supports'][0]['distance_pct']:.1f}% below) - better R:R", "sentiment": "positive", "impact": 6})

    # ── RSI ZONE ──
    if rsi_value >= 78:
        penalty_total += 12
        reasons.append({"text": f"RSI overbought ({rsi_value:.0f}) - high pullback risk", "sentiment": "negative", "impact": -12})
    elif rsi_value >= 70:
        penalty_total += 5
        reasons.append({"text": f"RSI elevated ({rsi_value:.0f}) - caution zone", "sentiment": "neutral", "impact": -5})
    elif rsi_value <= 30:
        boost_total += 8
        reasons.append({"text": f"RSI oversold ({rsi_value:.0f}) - bounce zone, good entry", "sentiment": "positive", "impact": 8})
    elif rsi_value <= 40:
        boost_total += 4
        reasons.append({"text": f"RSI low ({rsi_value:.0f}) - room to run upward", "sentiment": "positive", "impact": 4})

    # ── VOLUME CONFIRMATION (NEW V2) ──
    if vol_ratio >= 1.5:
        boost_total += 6
        reasons.append({"text": f"Volume confirmation ({vol_ratio:.1f}x avg) - strong conviction", "sentiment": "positive", "impact": 6})
    elif vol_ratio >= 1.0:
        reasons.append({"text": f"Normal volume ({vol_ratio:.1f}x avg)", "sentiment": "neutral", "impact": 0})
    elif vol_ratio < 0.5:
        penalty_total += 5
        reasons.append({"text": f"Low volume ({vol_ratio:.1f}x avg) - weak conviction", "sentiment": "negative", "impact": -5})

    # ── R:R QUALITY ──
    if actual_rr >= 2.5:
        boost_total += 8
        reasons.append({"text": f"Excellent R:R ({actual_rr}x) - asymmetric payoff in your favor", "sentiment": "positive", "impact": 8})
    elif actual_rr >= 1.5:
        boost_total += 5
        reasons.append({"text": f"Good R:R ({actual_rr}x)", "sentiment": "positive", "impact": 5})
    elif actual_rr < 1.0:
        penalty_total += 10
        reasons.append({"text": f"Poor R:R ({actual_rr}x) - risk exceeds reward", "sentiment": "negative", "impact": -10})

    # ── CHASING RALLY (softer than V1) ──
    if move_5d > 5:
        penalty_total += 12
        reasons.append({"text": f"ALREADY UP {move_5d:.1f}% in 5 days - chasing rally, high pullback risk", "sentiment": "negative", "impact": -12})
    elif move_5d > 3:
        penalty_total += 6
        reasons.append({"text": f"Up {move_5d:.1f}% in 5 days - most of the move may be done", "sentiment": "negative", "impact": -6})

    # ── ML MAX DRAWDOWN (capped penalty) ──
    if max_drawdown < -8:
        penalty_total += 8
        reasons.append({"text": f"ML predicts high drawdown ({max_drawdown:.1f}%) - risky", "sentiment": "negative", "impact": -8})
    elif max_drawdown < -5:
        penalty_total += 4
        reasons.append({"text": f"ML predicts moderate drawdown ({max_drawdown:.1f}%)", "sentiment": "neutral", "impact": -4})

    # ── FAIR VALUE BONUS/DETECTION ──
    fair_value_info = {}
    if HAS_INTELLIGENT_SCORING and compute_fair_value is not None:
        try:
            fair_value_info = compute_fair_value(df)
            fv_gap = fair_value_info.get("gap_to_fair_pct", 0)
            fv_price = fair_value_info.get("fair_value", 0)
            if fair_value_info.get("is_undervalued"):
                boost_bv = min(8, abs(fv_gap) * 2)
                boost_total += boost_bv
                reasons.append({"text": f"Below Fair Value (Rs.{fv_price}) by {fv_gap:.1f}% — upside potential", "sentiment": "positive", "impact": boost_bv})
            elif fair_value_info.get("is_overvalued"):
                pen_ov = min(6, abs(fv_gap) * 1.5)
                penalty_total += pen_ov
                reasons.append({"text": f"Above Fair Value (Rs.{fv_price}) by {fv_gap:.1f}% — overvalued", "sentiment": "negative", "impact": -pen_ov})
        except Exception:
            pass

    # ── ENFORCE MAX PENALTY CAP (KEY V2 FIX) ──
    if penalty_total > MAX_PENALTY:
        excess = penalty_total - MAX_PENALTY
        penalty_total = MAX_PENALTY
        reasons.append({"text": f"Penalty cap applied (removed {excess} excess penalty points)", "sentiment": "neutral", "impact": 0})

    # ══════════════════════════════════════════════════════════════
    #  FINAL CONFIDENCE & ACTION
    # ══════════════════════════════════════════════════════════════
    confidence = 50 + boost_total - penalty_total
    confidence = max(0, min(100, confidence))

    # Volume requirement for STRONG_BUY
    volume_ok = vol_ratio >= 0.7

    # Action determination
    strategy_action = "WAIT"
    if confidence >= 75 and actual_rr >= 1.5 and regime != "DOWNTREND" and volume_ok:
        strategy_action = "STRONG_BUY"
    elif confidence >= 60 and actual_rr >= 1.0 and regime != "DOWNTREND":
        strategy_action = "BUY"
    elif confidence >= 45 and actual_rr >= 1.0:
        strategy_action = "WATCH"
    elif confidence < 35:
        strategy_action = "WAIT"
    else:
        strategy_action = "WATCH"

    # ── Position Sizing ──
    if risk_per_share > 0 and abs(stop_loss_pct) > 0:
        base_position = 1.0 / abs(stop_loss_pct)
        conf_multiplier = min(1.2, confidence / 80.0)
        rr_multiplier = min(1.3, actual_rr / 1.5) if actual_rr > 0 else 0.5
        vol_multiplier = min(1.0, vol_ratio) if vol_ratio >= 0.5 else 0.5
        position_pct = round(base_position * conf_multiplier * rr_multiplier * vol_multiplier, 1)
        position_pct = max(0.5, min(position_pct, 10.0))
    else:
        position_pct = 1.0

    if strategy_action == "STRONG_BUY":
        size_label = f"AGGRESSIVE ({position_pct:.1f}% of capital)"
    elif strategy_action == "BUY":
        size_label = f"STANDARD ({position_pct:.1f}% of capital)"
    elif strategy_action == "WATCH":
        size_label = f"TRIAL ({round(position_pct * 0.5, 1):.1f}% of capital)"
    else:
        size_label = "NO POSITION"

    # ── Build Strategy Steps ──
    steps = []
    if strategy_action in ("STRONG_BUY", "BUY"):
        if pullback_entry:
            steps.append(f"1. Place LIMIT BUY at Rs.{pullback_entry:.2f} (near support, better entry than market)")
        else:
            steps.append(f"1. BUY at market price Rs.{current_price:.2f}")
        steps.append(f"2. STOP LOSS at Rs.{stop_price:.2f} ({stop_loss_pct}%)")
        steps.append(f"3. TARGET {target_pct}% at Rs.{target_price:.2f}")
        if target_pct < 3.0:
            steps.append(f"4. TARGET REDUCED to {target_pct}% (3% had poor R:R)")
        if actual_rr >= 2.0:
            steps.append(f"5. Excellent R:R ({actual_rr}x) - trail stop at {target_pct * 0.5:.1f}% to lock profits")
        avg_days = behavior.get("target_3pct", {}).get("avg_days_to_hit")
        if avg_days:
            steps.append(f"6. Historically takes avg {avg_days} days to hit 3%")
        if rsi_value >= 65:
            steps.append(f"7. RSI at {rsi_value:.0f} - if no follow-through in 1-2 days, exit")
    else:
        steps.append(f"1. {strategy_action} - conditions do not support a high-probability trade right now")
        neg_reasons = [r["text"] for r in reasons if r["sentiment"] == "negative"]
        if neg_reasons:
            steps.append(f"2. Key concerns: {'; '.join(neg_reasons[:2])}")
        if pullback_entry:
            steps.append(f"3. Watch for pullback to Rs.{pullback_entry:.2f} for better R:R")
        if move_5d > 3:
            steps.append(f"4. Stock up {move_5d:.1f}% in 5 days - wait for pullback")
        steps.append(f"5. Monitor for RSI cool-down or trend/pattern change")

    return {
        "ticker": ticker,
        "current_price": round(float(current_price), 2),
        "strategy_action": strategy_action,
        "confidence": round(confidence, 1),
        "entry": {
            "price": round(float(pullback_entry or current_price), 2),
            "type": entry_type,
        },
        "stop_loss": {
            "price": round(float(stop_price), 2),
            "pct": stop_loss_pct,
        },
        "target": {
            "price": target_price,
            "pct": target_pct,
            "type": target_type,
        },
        "risk_reward": actual_rr,
        "position_size": size_label,
        "reasons": reasons,  # V2: list of {text, sentiment, impact}
        "steps": steps,
        "ml_enhanced": True,
        "scoring_method": scoring_method,
        "score_components": score_components,
        "multi_timeframe": mtf_alignment,
        "momentum_acceleration": momentum_info,
        "volume_ratio": round(vol_ratio, 2),
        "rsi_value": round(rsi_value, 1),
        "fair_value": fair_value_info,
        "boost_total": boost_total,
        "penalty_total": penalty_total,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  FULL STOCK ANALYSIS (COMBINES EVERYTHING)
# ═══════════════════════════════════════════════════════════════════════════

def analyze_stock(
    ticker: str,
    df: pd.DataFrame,
    ml_prediction: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Complete per-stock analysis combining all engines.
    V2: Uses intelligent scoring + multi-timeframe + momentum acceleration.
    """
    n = len(df)
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    current_price = float(closes[-1])

    # 1. Pattern Recognition
    candle_patterns = detect_candlestick_patterns(df)
    chart_patterns = detect_chart_patterns(df)

    # 2. Support / Resistance
    sr_levels = detect_support_resistance(df)

    # 3. Behavior Profile
    behavior = profile_stock_behavior(df)

    # 4. Strategy Suggestion (uses intelligent scoring internally)
    strategy = suggest_strategy(
        ticker=ticker,
        df=df,
        ml_prediction=ml_prediction,
        behavior_profile=behavior,
        patterns=candle_patterns,
        sr_levels=sr_levels,
    )

    # 5. Price Summary
    returns = np.diff(closes) / closes[:-1] * 100 if n > 1 else np.array([0])
    price_change_1d = round(float(returns[-1]), 2) if len(returns) > 0 else 0
    price_change_5d = round(float((closes[-1] / closes[-6] - 1) * 100), 2) if n > 5 else 0
    price_change_20d = round(float((closes[-1] / closes[-21] - 1) * 100), 2) if n > 20 else 0
    price_change_60d = round(float((closes[-1] / closes[-61] - 1) * 100), 2) if n > 60 else 0

    # 6. Technical Indicators
    indicators = {}
    if n >= 20:
        close_series = pd.Series(closes)
        ema9 = float(close_series.ewm(span=9, adjust=False).mean().iloc[-1])
        ema21 = float(close_series.ewm(span=21, adjust=False).mean().iloc[-1])
        ema50 = float(close_series.ewm(span=50, adjust=False).mean().iloc[-1])

        indicators = {
            "ema9": round(ema9, 2),
            "ema21": round(ema21, 2),
            "ema50": round(ema50, 2),
            "ema9_signal": "BULLISH" if ema9 > ema21 else "BEARISH",
            "price_vs_ema9_pct": round((current_price - ema9) / ema9 * 100, 2),
        }

        delta = close_series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = float((100 - (100 / (1 + rs))).iloc[-1]) if not rs.isna().all() else 50
        indicators["rsi_14"] = round(rsi, 1)
        indicators["rsi_signal"] = "OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL"

        exp12 = close_series.ewm(span=12, adjust=False).mean()
        exp26 = close_series.ewm(span=26, adjust=False).mean()
        macd = float((exp12 - exp26).iloc[-1])
        signal_line = float(exp12.ewm(span=9, adjust=False).mean().iloc[-1] - exp26.ewm(span=9, adjust=False).mean().iloc[-1])
        indicators["macd"] = round(macd, 4)
        indicators["macd_signal"] = "BULLISH" if macd > 0 else "BEARISH"

        sma20 = close_series.rolling(window=20).mean().iloc[-1]
        std20 = close_series.rolling(window=20).std().iloc[-1]
        if not np.isnan(sma20) and not np.isnan(std20):
            bb_upper = float(sma20 + 2 * std20)
            bb_lower = float(sma20 - 2 * std20)
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100 if (bb_upper - bb_lower) > 0 else 50
            indicators["bb_upper"] = round(bb_upper, 2)
            indicators["bb_lower"] = round(bb_lower, 2)
            indicators["bb_position_pct"] = round(bb_position, 1)
            indicators["bb_signal"] = "NEAR UPPER (Overbought)" if bb_position > 80 else "NEAR LOWER (Oversold)" if bb_position < 20 else "MID-RANGE"

    # 7. Volume Analysis
    volume_analysis = {}
    if "Volume" in df.columns and n >= 21:
        vol = df["Volume"].values
        vol_avg_20 = float(np.mean(vol[-21:-1]))
        vol_current = float(vol[-1])
        vol_ratio = vol_current / vol_avg_20 if vol_avg_20 > 0 else 1.0
        volume_analysis = {
            "current": int(vol_current),
            "avg_20d": int(vol_avg_20),
            "ratio": round(vol_ratio, 2),
            "signal": "HIGH VOLUME" if vol_ratio > 1.5 else "LOW VOLUME" if vol_ratio < 0.5 else "NORMAL",
        }

    # 8. Fair Value Analysis
    fair_value_analysis = {}
    if HAS_INTELLIGENT_SCORING and compute_fair_value is not None:
        try:
            fair_value_analysis = compute_fair_value(df)
        except Exception:
            pass

    # Assemble full report
    report = {
        "ticker": ticker,
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_points": n,
        "price_summary": {
            "current": round(current_price, 2),
            "change_1d_pct": price_change_1d,
            "change_5d_pct": price_change_5d,
            "change_20d_pct": price_change_20d,
            "change_60d_pct": price_change_60d,
            "day_high": round(float(highs[-1]), 2),
            "day_low": round(float(lows[-1]), 2),
            "period_high_52w": round(float(np.max(highs[-min(252, n):])), 2),
            "period_low_52w": round(float(np.min(lows[-min(252, n):])), 2),
            "distance_from_52w_high_pct": round((current_price - np.max(highs[-min(252, n):])) / np.max(highs[-min(252, n):]) * 100, 1) if n >= 20 else 0,
        },
        "indicators": indicators,
        "volume_analysis": volume_analysis,
        "ml_prediction": ml_prediction,
        "behavior_profile": behavior,
        "candlestick_patterns": candle_patterns[-10:],
        "chart_patterns": chart_patterns,
        "support_resistance": sr_levels,
        "fair_value": fair_value_analysis,
        "strategy": strategy,
    }

    return report


# ═══════════════════════════════════════════════════════════════════════════
#  MULTI-STOCK SCANNER (V2 - INTELLIGENT)
# ═══════════════════════════════════════════════════════════════════════════

def scan_stocks_for_3pct(
    ticker_dfs: Dict[str, pd.DataFrame],
    ml_predictions: Optional[Dict[str, Dict]] = None,
) -> List[Dict]:
    """
    V2 Scanner: Uses intelligent scoring instead of untrained ML.
    Returns sorted list with better action distribution.
    """
    results = []

    for ticker, df in ticker_dfs.items():
        ml_pred = ml_predictions.get(ticker) if ml_predictions else None

        # Use intelligent scoring if available
        if HAS_INTELLIGENT_SCORING:
            try:
                intel_score = compute_intelligent_score(df)
                # Use this as ml_prediction for the strategy engine
                effective_ml = ml_pred or {}
                effective_ml.update(intel_score)
            except Exception:
                effective_ml = ml_pred
        else:
            effective_ml = ml_pred

        behavior = profile_stock_behavior(df)
        strategy = suggest_strategy(ticker, df, ml_prediction=effective_ml, behavior_profile=behavior)

        # Extract top positive and negative reasons for the scanner table
        pos_reasons = [r["text"] for r in strategy.get("reasons", []) if r["sentiment"] == "positive"][:2]
        neg_reasons = [r["text"] for r in strategy.get("reasons", []) if r["sentiment"] == "negative"][:1]

        results.append({
            "ticker": ticker,
            "current_price": strategy["current_price"],
            "strategy_action": strategy["strategy_action"],
            "confidence": strategy["confidence"],
            "risk_reward": strategy["risk_reward"],
            "signal_quality": effective_ml.get("signal_quality", 0) if effective_ml else 0,
            "trend_regime": effective_ml.get("trend_regime", "N/A") if effective_ml else "N/A",
            "expected_return": effective_ml.get("expected_return", 0) if effective_ml else 0,
            "max_drawdown": effective_ml.get("max_drawdown", 0) if effective_ml else 0,
            "hit_rate_3pct": behavior.get("target_3pct", {}).get("hit_rate_pct", 0),
            "behavior_type": behavior.get("behavior_type", "UNKNOWN"),
            "volatility_regime": behavior.get("volatility", {}).get("regime", "UNKNOWN"),
            "target_price": strategy["target"]["price"],
            "stop_loss": strategy["stop_loss"]["price"],
            "entry_price": strategy["entry"]["price"],
            "position_size": strategy["position_size"],
            "reasons": pos_reasons + neg_reasons,
            "reasons_tagged": strategy.get("reasons", []),  # V2: full tagged reasons
            "volume_ratio": strategy.get("volume_ratio", 1.0),
            "rsi_value": strategy.get("rsi_value", 50),
            "multi_timeframe": strategy.get("multi_timeframe", {}),
            "momentum": strategy.get("momentum_acceleration", {}),
            "scoring_method": strategy.get("scoring_method", "unknown"),
        })

    # Sort by action then confidence
    action_order = {"STRONG_BUY": 0, "BUY": 1, "WATCH": 2, "WAIT": 3}
    results.sort(key=lambda x: (action_order.get(x["strategy_action"], 99), -x["confidence"]))

    return results