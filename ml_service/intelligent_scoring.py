"""
Intelligent Signal Quality Scoring Engine
==========================================
REPLACES untrained TensorFlow model outputs with REAL indicator-based scoring.

The old system used randomly-initialized Keras models that output garbage
signal quality (1-18%), killing every stock's confidence. This module computes
signal quality from ACTUAL technical indicator alignment.

Scoring Dimensions (10 components, each 0-10):
  1. EMA Alignment     - EMA9 > EMA21 > EMA50 (trend alignment)
  2. RSI Position      - 40-65 = sweet spot, <30 = oversold boost, >75 = penalty
  3. MACD Momentum     - MACD histogram direction and magnitude
  4. Volume Confirmation - Volume ratio vs 20D average
  5. Price Position    - Where price sits within Bollinger Bands
  6. Momentum          - Rate of change acceleration (5D vs 10D)
  7. Trend Strength    - ADX-like measure from EMA slope
  8. Pullback Quality  - Is price pulling back to EMA9/21 support?
  9. Recent Performance - 5D and 10D returns (not chasing)
  10. Volatility Fit    - Volatility appropriate for 3% target

Total: 0-100 signal quality score
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List


def compute_intelligent_score(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Master scoring function. Returns signal_quality (0-1), regime, and
    component breakdown for explainability.
    """
    n = len(df)
    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    opens = df["Open"].values
    volumes = df["Volume"].values if "Volume" in df.columns else None
    current_price = float(closes[-1])

    if n < 60:
        return {
            "signal_quality": 0.5,
            "trend_regime": "SIDEWAYS",
            "expected_return": 0,
            "max_drawdown": -5,
            "risk_reward_ratio": 1.0,
            "ml_score": 50,
            "components": {},
            "scoring_method": "intelligent_v2",
        }

    cs = pd.Series(closes)

    # ── Pre-compute indicators ──
    ema9 = float(cs.ewm(span=9, adjust=False).mean().iloc[-1])
    ema21 = float(cs.ewm(span=21, adjust=False).mean().iloc[-1])
    ema50 = float(cs.ewm(span=50, adjust=False).mean().iloc[-1])

    # RSI
    delta = cs.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    rsi_val = float(rsi_series.iloc[-1]) if not rsi_series.isna().all() else 50.0

    # MACD
    exp12 = cs.ewm(span=12, adjust=False).mean()
    exp26 = cs.ewm(span=26, adjust=False).mean()
    macd_line = exp12 - exp26
    macd_val = float(macd_line.iloc[-1])
    macd_signal = float(macd_line.ewm(span=9, adjust=False).mean().iloc[-1])
    macd_hist = macd_val - macd_signal
    # MACD histogram 3 days ago for direction
    macd_hist_3ago = float(macd_line.iloc[-4]) - float(macd_line.ewm(span=9, adjust=False).mean().iloc[-4]) if n >= 4 else 0

    # Bollinger Bands
    sma20 = float(cs.rolling(window=20).mean().iloc[-1])
    std20 = float(cs.rolling(window=20).std().iloc[-1])
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_range = bb_upper - bb_lower
    bb_position = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5

    # Volume
    vol_ratio = 1.0
    vol_trend = 0  # positive = increasing volume
    if volumes is not None and n >= 25:
        vol_avg_20 = float(np.mean(volumes[-25:-5]))
        vol_current = float(volumes[-1])
        vol_ratio = vol_current / vol_avg_20 if vol_avg_20 > 0 else 1.0
        # Volume trend: is volume increasing over last 5 days?
        vol_5d_avg = float(np.mean(volumes[-5:]))
        vol_20d_avg = float(np.mean(volumes[-25:-5]))
        vol_trend = (vol_5d_avg - vol_20d_avg) / vol_20d_avg if vol_20d_avg > 0 else 0

    # ── Score each component (0-10) ──
    components = {}

    # 1. EMA ALIGNMENT (0-10)
    ema_score = 0
    if ema9 > ema21 > ema50:
        ema_score = 10  # Perfect bullish alignment
    elif ema9 > ema21:
        ema_score = 7  # Short-term bullish
        if ema21 > ema50:
            ema_score = 9  # Near perfect
    elif ema9 > ema50:
        ema_score = 5  # Mixed
    elif ema9 > ema21 and ema21 < ema50:
        ema_score = 4  # Potential crossover
    elif ema9 < ema21 < ema50:
        ema_score = 1  # Bearish alignment
    else:
        ema_score = 3  # Messy
    # Bonus: price above all EMAs
    if current_price > ema9 > ema21 > ema50:
        ema_score = min(10, ema_score + 1)
    components["ema_alignment"] = ema_score

    # 2. RSI POSITION (0-10)
    # Sweet spot: 45-65. Oversold <35 = boost for reversal. Overbought >75 = penalty.
    rsi_score = 5  # neutral
    if 45 <= rsi_val <= 65:
        rsi_score = 9  # Sweet spot - room to run
    elif 40 <= rsi_val <= 70:
        rsi_score = 7  # Good
    elif 35 <= rsi_val <= 75:
        rsi_score = 5  # OK
    elif rsi_val < 30:
        rsi_score = 7  # Oversold = bounce potential, but risky
    elif rsi_val < 35:
        rsi_score = 6  # Near oversold
    elif rsi_val > 80:
        rsi_score = 1  # Very overbought
    elif rsi_val > 75:
        rsi_score = 3  # Overbought
    else:
        rsi_score = 4
    components["rsi_position"] = rsi_score

    # 3. MACD MOMENTUM (0-10)
    macd_score = 5
    if macd_hist > 0 and macd_hist_3ago > 0:
        macd_score = 8  # Bullish and strengthening
        if macd_hist > macd_hist_3ago:
            macd_score = 10  # Accelerating bullish
    elif macd_hist > 0 and macd_hist_3ago <= 0:
        macd_score = 7  # Just turned bullish (crossover!)
    elif macd_hist > 0:
        macd_score = 6  # Bullish but decelerating
    elif macd_hist < 0 and macd_hist_3ago < 0:
        macd_score = 2  # Bearish and worsening
    elif macd_hist < 0 and macd_hist_3ago >= 0:
        macd_score = 3  # Just turned bearish (death cross!)
    else:
        macd_score = 4
    components["macd_momentum"] = macd_score

    # 4. VOLUME CONFIRMATION (0-10)
    vol_score = 5
    if vol_ratio > 2.0:
        vol_score = 9  # Very high volume = strong conviction
    elif vol_ratio > 1.5:
        vol_score = 8  # Above average = good
    elif vol_ratio > 1.0:
        vol_score = 6  # Normal
    elif vol_ratio > 0.7:
        vol_score = 5  # Slightly below
    elif vol_ratio > 0.4:
        vol_score = 3  # Low volume
    else:
        vol_score = 1  # Very low = no conviction
    # Volume trend bonus
    if vol_trend > 0.2:
        vol_score = min(10, vol_score + 1)
    components["volume_confirmation"] = vol_score

    # 5. PRICE POSITION (Bollinger Band) (0-10)
    # Best: 40-60% (middle-upper half, room to run)
    bb_score = 5
    if 0.4 <= bb_position <= 0.65:
        bb_score = 9  # Sweet spot
    elif 0.3 <= bb_position <= 0.75:
        bb_score = 7  # Good
    elif 0.2 <= bb_position <= 0.85:
        bb_score = 5  # OK
    elif bb_position > 0.9:
        bb_score = 1  # At upper band = overbought
    elif bb_position > 0.8:
        bb_score = 3  # Near upper
    elif bb_position < 0.1:
        bb_score = 7  # At lower band = potential bounce
    elif bb_position < 0.2:
        bb_score = 6  # Near lower = oversold zone
    components["price_position"] = bb_score

    # 6. MOMENTUM (Rate of change) (0-10)
    mom_score = 5
    if n > 10:
        ret_5d = (closes[-1] / closes[-6] - 1) * 100 if n > 5 else 0
        ret_10d = (closes[-1] / closes[-11] - 1) * 100 if n > 10 else 0
        # Acceleration: 5D return vs 10D return
        # If 5D > 0 and > half of 10D = momentum accelerating
        if ret_5d > 0 and ret_10d > 0:
            if ret_5d > ret_10d * 0.6:
                mom_score = 9  # Strong accelerating momentum
            elif ret_5d > ret_10d * 0.3:
                mom_score = 7  # Good momentum
            else:
                mom_score = 6  # Decelerating but still up
        elif ret_5d > 0 and ret_10d <= 0:
            mom_score = 7  # Reversal momentum (turning positive)
        elif ret_5d > 0:
            mom_score = 5  # Slight positive
        elif ret_5d < -3:
            mom_score = 2  # Strong negative momentum
        elif ret_5d < 0:
            mom_score = 3  # Negative
        else:
            mom_score = 5

        # CHASING PENALTY: if up too much in 5 days
        if ret_5d > 5:
            mom_score = max(1, mom_score - 4)  # Severe chasing penalty
        elif ret_5d > 3:
            mom_score = max(2, mom_score - 2)
    components["momentum"] = mom_score

    # 7. TREND STRENGTH (EMA slope) (0-10)
    trend_score = 5
    if n > 20:
        # EMA9 slope over last 10 bars
        ema9_series = cs.ewm(span=9, adjust=False).mean()
        ema9_now = float(ema9_series.iloc[-1])
        ema9_10ago = float(ema9_series.iloc[-11]) if n >= 11 else ema9_now
        ema_slope_pct = (ema9_now - ema9_10ago) / ema9_10ago * 100 if ema9_10ago > 0 else 0

        # EMA21 slope
        ema21_series = cs.ewm(span=21, adjust=False).mean()
        ema21_now = float(ema21_series.iloc[-1])
        ema21_10ago = float(ema21_series.iloc[-11]) if n >= 11 else ema21_now
        ema21_slope_pct = (ema21_now - ema21_10ago) / ema21_10ago * 100 if ema21_10ago > 0 else 0

        if ema_slope_pct > 2 and ema21_slope_pct > 1:
            trend_score = 10  # Strong uptrend
        elif ema_slope_pct > 1 and ema21_slope_pct > 0:
            trend_score = 8  # Good uptrend
        elif ema_slope_pct > 0 and ema21_slope_pct > 0:
            trend_score = 6  # Mild uptrend
        elif ema_slope_pct < -2 and ema21_slope_pct < -1:
            trend_score = 1  # Strong downtrend
        elif ema_slope_pct < 0 and ema21_slope_pct < 0:
            trend_score = 3  # Downtrend
        elif ema_slope_pct > 0 and ema21_slope_pct < 0:
            trend_score = 5  # Conflicting
        else:
            trend_score = 4  # Sideways
    components["trend_strength"] = trend_score

    # 8. PULLBACK QUALITY (0-10)
    # Is price pulling back to a logical support (EMA9/21)?
    pull_score = 5
    dist_to_ema9 = (current_price - ema9) / ema9 * 100
    dist_to_ema21 = (current_price - ema21) / ema21 * 100

    if 0 < dist_to_ema9 < 0.5:
        pull_score = 9  # Right at EMA9 support (bouncing)
    elif -0.5 < dist_to_ema9 < 0:
        pull_score = 8  # Just touched EMA9 (perfect entry)
    elif 0.5 <= dist_to_ema9 < 1.5:
        pull_score = 7  # Slightly above EMA9, healthy
    elif 1.5 <= dist_to_ema9 < 3:
        pull_score = 5  # Extended above EMA9
    elif dist_to_ema9 >= 3:
        pull_score = 3  # Far above EMA9, chase zone
    elif dist_to_ema9 < -1:
        pull_score = 4  # Below EMA9, weak

    # Near EMA21 bonus (deeper pullback but at stronger support)
    if -0.5 < dist_to_ema21 < 1.0 and dist_to_ema9 > 1.0:
        pull_score = max(pull_score, 7)  # Pulling back to EMA21
    components["pullback_quality"] = pull_score

    # 9. RECENT PERFORMANCE (0-10)
    # Check 5D and 10D returns, score positively if moderate up, penalize chasing
    perf_score = 5
    if n > 10:
        ret_5d = (closes[-1] / closes[-6] - 1) * 100 if n > 5 else 0
        ret_10d = (closes[-1] / closes[-11] - 1) * 100 if n > 10 else 0

        if 0 < ret_5d <= 2 and ret_10d <= 3:
            perf_score = 9  # Moderate recent gains, not extended
        elif 0 < ret_5d <= 3 and ret_10d <= 5:
            perf_score = 7  # Good gains
        elif ret_5d <= 0 and ret_10d > 0:
            perf_score = 6  # Pulling back in longer uptrend (good for entry)
        elif ret_5d <= -1 and ret_10d > 2:
            perf_score = 7  # Pullback in uptrend = potential buy
        elif ret_5d > 5:
            perf_score = 2  # Chasing
        elif ret_5d > 3:
            perf_score = 4  # Extended
        elif ret_5d < -3:
            perf_score = 2  # Breaking down
        elif ret_5d < -1:
            perf_score = 4  # Weak
    components["recent_performance"] = perf_score

    # 10. VOLATILITY FIT (0-10)
    # For 3% target, we need stocks that CAN move 3% but don't have crazy volatility
    vol_score = 5
    if n > 20:
        c60 = closes[-60:]
        returns = np.diff(c60) / c60[:-1] * 100 if len(c60) > 1 else np.array([0])
        daily_std = float(np.std(returns))
        annualized_vol = daily_std * np.sqrt(252)

        if 15 <= annualized_vol <= 30:
            vol_score = 9  # Sweet spot: can move 3% in 1-2 weeks
        elif 10 <= annualized_vol <= 40:
            vol_score = 7  # Good
        elif annualized_vol < 10:
            vol_score = 4  # Too quiet, 3% target may take too long
        elif annualized_vol > 60:
            vol_score = 2  # Too wild, stop loss will be too wide
        elif annualized_vol > 40:
            vol_score = 5  # High but manageable
        else:
            vol_score = 5
    components["volatility_fit"] = vol_score

    # ── Aggregate Score ──
    # Weighted average (some components matter more)
    weights = {
        "ema_alignment": 1.3,      # Trend alignment is critical
        "rsi_position": 1.0,       # Entry timing
        "macd_momentum": 1.1,      # Momentum confirmation
        "volume_confirmation": 0.9,  # Volume validates
        "price_position": 0.8,     # BB position
        "momentum": 1.2,           # Acceleration matters
        "trend_strength": 1.2,     # Trend strength
        "pullback_quality": 1.0,   # Entry quality
        "recent_performance": 0.8, # Not chasing
        "volatility_fit": 0.7,     # Appropriate volatility
    }

    total_weight = sum(weights.values())
    weighted_sum = sum(components[k] * weights[k] for k in components)
    raw_score = weighted_sum / total_weight  # 0-10

    # Convert to 0-100 scale
    signal_quality = raw_score / 10.0  # 0.0 to 1.0

    # ── Determine Regime ──
    if ema_score >= 7 and trend_score >= 6:
        regime = "UPTREND"
    elif ema_score <= 2 and trend_score <= 3:
        regime = "DOWNTREND"
    else:
        regime = "SIDEWAYS"

    # ── Estimate Expected Return ──
    # Based on momentum + trend strength + RSI room
    if n > 10:
        ret_5d = (closes[-1] / closes[-6] - 1) * 100 if n > 5 else 0
        # Simple estimate: if all aligned, expect 3-5% in 1-2 weeks
        base_return = 1.5
        if regime == "UPTREND":
            base_return += 1.5
        if macd_hist > 0:
            base_return += 0.5
        if rsi_val < 50:
            base_return += 0.5  # Room to run
        if ret_5d > 3:
            base_return -= 1.0  # Already moved
        if regime == "DOWNTREND":
            base_return = -2.0
        expected_return = round(base_return, 1)
    else:
        expected_return = 0

    # ── Estimate Max Drawdown ──
    # Based on ATR and volatility
    if n > 14:
        tr_values = [
            max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            for i in range(max(0, n-14), n)
        ]
        atr = np.mean(tr_values)
        atr_pct = atr / current_price * 100
        max_drawdown = round(-atr_pct * 3, 1)  # Roughly 3x ATR as worst case
    else:
        max_drawdown = -5.0

    # ── Estimate Risk:Reward ──
    # Simplified: target 3% vs estimated stop (2x ATR)
    if n > 14:
        tr_values = [
            max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            for i in range(max(0, n-14), n)
        ]
        atr = np.mean(tr_values)
        stop_pct = atr / current_price * 100 * 1.5
        risk_reward = round(3.0 / stop_pct, 2) if stop_pct > 0 else 1.0
    else:
        risk_reward = 1.0

    # ML score (composite)
    ml_score = round(signal_quality * 60 + (10 if regime == "UPTREND" else -10 if regime == "DOWNTREND" else 0), 1)
    ml_score = max(0, min(100, ml_score))

    return {
        "signal_quality": round(signal_quality, 3),
        "trend_regime": regime,
        "expected_return": expected_return,
        "max_drawdown": max_drawdown,
        "risk_reward_ratio": risk_reward,
        "ml_score": ml_score,
        "components": {k: round(v, 1) for k, v in components.items()},
        "scoring_method": "intelligent_v2",
    }


def compute_multi_timeframe_alignment(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check if daily and weekly trends are aligned.
    Weekly uptrend + daily pullback = HIGH CONFIDENCE setup.
    """
    n = len(df)
    closes = df["Close"].values
    current_price = float(closes[-1])

    if n < 120:
        return {"weekly_regime": "UNKNOWN", "alignment": "NEUTRAL", "score": 0}

    cs = pd.Series(closes)

    # Daily EMAs
    daily_ema9 = float(cs.ewm(span=9, adjust=False).mean().iloc[-1])
    daily_ema21 = float(cs.ewm(span=21, adjust=False).mean().iloc[-1])
    daily_ema50 = float(cs.ewm(span=50, adjust=False).mean().iloc[-1])

    # Weekly trend: use 50-day and 100-day as proxy
    weekly_proxy_short = float(cs.ewm(span=50, adjust=False).mean().iloc[-1])
    weekly_proxy_long = float(cs.ewm(span=100, adjust=False).mean().iloc[-1])

    # Weekly regime
    if weekly_proxy_short > weekly_proxy_long:
        weekly_regime = "UPTREND"
    elif weekly_proxy_short < weekly_proxy_long:
        weekly_regime = "DOWNTREND"
    else:
        weekly_regime = "SIDEWAYS"

    # Daily regime
    if daily_ema9 > daily_ema21 > daily_ema50:
        daily_regime = "STRONG_UP"
    elif daily_ema9 > daily_ema21:
        daily_regime = "UP"
    elif daily_ema9 < daily_ema21 < daily_ema50:
        daily_regime = "STRONG_DOWN"
    elif daily_ema9 < daily_ema21:
        daily_regime = "DOWN"
    else:
        daily_regime = "SIDEWAYS"

    # Alignment scoring
    alignment_score = 0
    alignment_label = "NEUTRAL"

    if weekly_regime == "UPTREND" and daily_regime in ("UP", "STRONG_UP"):
        alignment_score = 3
        alignment_label = "STRONG BUY ALIGNMENT (Weekly + Daily UP)"
    elif weekly_regime == "UPTREND" and daily_regime == "SIDEWAYS":
        alignment_score = 2
        alignment_label = "GOOD (Weekly UP, Daily consolidating - watch for breakout)"
    elif weekly_regime == "UPTREND" and daily_regime in ("DOWN", "STRONG_DOWN"):
        alignment_score = 1
        alignment_label = "PULLBACK (Weekly UP, Daily DOWN - potential buy zone)"
    elif weekly_regime == "DOWNTREND" and daily_regime in ("DOWN", "STRONG_DOWN"):
        alignment_score = -2
        alignment_label = "AVOID (Weekly + Daily DOWN)"
    elif weekly_regime == "DOWNTREND" and daily_regime in ("UP", "STRONG_UP"):
        alignment_score = 0
        alignment_label = "COUNTER-TREND (Weekly DOWN, Daily UP - risky)"
    elif weekly_regime == "SIDEWAYS":
        alignment_score = 0
        alignment_label = "NEUTRAL (Weekly sideways, wait for direction)"

    return {
        "weekly_regime": weekly_regime,
        "daily_regime": daily_regime,
        "alignment": alignment_label,
        "score": alignment_score,
        "weekly_ema_short": round(weekly_proxy_short, 2),
        "weekly_ema_long": round(weekly_proxy_long, 2),
    }


def detect_momentum_acceleration(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect if momentum is accelerating, steady, or decelerating.
    Accelerating momentum = strong buy signal.
    """
    n = len(df)
    closes = df["Close"].values

    if n < 30:
        return {"acceleration": "UNKNOWN", "score": 0}

    # 3-day, 5-day, 10-day returns
    ret_3d = (closes[-1] / closes[-4] - 1) * 100 if n > 3 else 0
    ret_5d = (closes[-1] / closes[-6] - 1) * 100 if n > 5 else 0
    ret_10d = (closes[-1] / closes[-11] - 1) * 100 if n > 10 else 0

    # Rate of change of returns (acceleration)
    # If 3D return > 5D return > 10D return = accelerating up
    # If 3D return < 5D return < 10D return = decelerating
    accel_score = 0
    accel_label = "STEADY"

    if ret_3d > ret_5d > ret_10d and ret_3d > 0:
        accel_score = 3
        accel_label = "ACCELERATING UP (momentum building)"
    elif ret_3d > ret_5d and ret_5d <= ret_10d and ret_3d > 0:
        accel_score = 2
        accel_label = "RECOVERING (short-term bounce)"
    elif ret_3d > 0 and ret_5d > 0 and ret_10d > 0 and ret_3d < ret_5d:
        accel_score = 1
        accel_label = "STEADY UPTREND (decelerating but still up)"
    elif ret_3d < ret_5d < ret_10d and ret_3d < 0:
        accel_score = -3
        accel_label = "ACCELERATING DOWN (selling pressure)"
    elif ret_3d < 0 and ret_5d < 0:
        accel_score = -2
        accel_label = "DECLINING (momentum negative)"
    elif ret_3d < 0 and ret_5d > 0:
        accel_score = 0
        accel_label = "PULLING BACK (normal in uptrend)"
    else:
        accel_score = 0
        accel_label = "MIXED (no clear acceleration)"

    return {
        "acceleration": accel_label,
        "score": accel_score,
        "ret_3d": round(ret_3d, 2),
        "ret_5d": round(ret_5d, 2),
        "ret_10d": round(ret_10d, 2),
    }


def compute_fair_value(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Estimate Fair Value using a multi-method approach:
      1. PE Regression: Use EMA-normalized price as baseline (proxy for earnings-based FV)
      2. Mean Reversion: 200-day SMA as long-term fair value anchor
      3. Bollinger Band Mid: 20-day SMA as medium-term fair value
      4. Consolidated: Weighted blend of all methods
    
    Returns fair_value, gap_to_fair_pct, and method breakdown.
    """
    n = len(df)
    closes = df["Close"].values
    current_price = float(closes[-1])
    
    if n < 60:
        return {
            "fair_value": round(current_price, 2),
            "gap_to_fair_pct": 0.0,
            "is_undervalued": False,
            "method": "insufficient_data",
            "breakdown": {},
        }
    
    cs = pd.Series(closes)
    
    # Method 1: 200-day SMA (long-term value anchor)
    sma_200 = float(cs.rolling(window=200).mean().iloc[-1]) if n >= 200 else float(cs.rolling(window=n).mean().iloc[-1])
    
    # Method 2: 50-day EMA (medium-term trend value)
    ema_50 = float(cs.ewm(span=50, adjust=False).mean().iloc[-1])
    
    # Method 3: 20-day SMA (short-term fair value)
    sma_20 = float(cs.rolling(window=20).mean().iloc[-1]) if n >= 20 else sma_200
    
    # Method 4: Linear regression channel midpoint (trend-consistent FV)
    x = np.arange(n)
    y = closes
    slope, intercept = np.polyfit(x, y, 1)
    regression_now = float(slope * (n - 1) + intercept)
    
    # Weighted blend: more weight to longer-term methods (they're more stable)
    # SMA200: 35%, EMA50: 30%, SMA20: 15%, Regression: 20%
    fair_value = sma_200 * 0.35 + ema_50 * 0.30 + sma_20 * 0.15 + regression_now * 0.20
    fair_value = round(fair_value, 2)
    
    gap_pct = round((current_price - fair_value) / fair_value * 100, 2)
    is_undervalued = gap_pct < -2  # More than 2% below FV = undervalued
    is_overvalued = gap_pct > 2   # More than 2% above FV = overvalued
    
    return {
        "fair_value": fair_value,
        "gap_to_fair_pct": gap_pct,
        "is_undervalued": is_undervalued,
        "is_overvalued": is_overvalued,
        "method": "consolidated",
        "breakdown": {
            "sma_200": round(sma_200, 2),
            "ema_50": round(ema_50, 2),
            "sma_20": round(sma_20, 2),
            "regression_trend": round(regression_now, 2),
            "weights": {"sma_200": 0.35, "ema_50": 0.30, "sma_20": 0.15, "regression": 0.20},
        },
    }