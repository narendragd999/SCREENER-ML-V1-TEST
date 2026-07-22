"""
ML Feature Engineering Pipeline
================================
Extracts 80+ quantitative features from OHLCV data for TensorFlow models.
Designed to work with the same yfinance data format used by ema9_router.py.

Feature Categories:
  1. Price-Based (15): Returns, momentum, distance to EMAs/SMAs
  2. Volume-Based (10): Volume ratios, OBV, volume profile
  3. Volatility-Based (12): ATR, Bollinger, Keltner, historical vol
  4. Trend-Based (12): EMA/SMA crossovers, ADX-like, linear regression slope
  5. Pattern-Based (15): Candlestick patterns, fractal pivots, HH/LL structure
  6. Support/Resistance-Based (10): Distance to detected levels, touch counts
  7. Macro/Momentum (10): RSI, MACD, Stochastic, CCI, Williams %R
  8. Target-Proximity (6): Distance to target, historical win rate context

All features are normalized to [0, 1] or z-scored for neural network input.
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("ml_features")


# ═══════════════════════════════════════════════════════════════════════════
#  CATEGORY 1: PRICE-BASED FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def _price_features(df: pd.DataFrame) -> Dict[str, float]:
    """Returns, momentum, and distance-to-indicator features."""
    closes = df["Close"].values
    n = len(closes)
    feats = {}

    # Multi-period returns
    for period in [1, 3, 5, 10, 20, 60]:
        if n > period:
            ret = (closes[-1] / closes[-period - 1] - 1) * 100
            feats[f"return_{period}d"] = ret

    # Price momentum (rate of change acceleration)
    if n > 20:
        r5 = (closes[-1] / closes[-6] - 1) * 100
        r20 = (closes[-1] / closes[-21] - 1) * 100
        feats["momentum_accel"] = r5 - r20  # positive = accelerating

    # Distance to EMA9, SMA20, SMA50, SMA200
    for period, name in [(9, "ema9"), (20, "sma20"), (50, "sma50"), (200, "sma200")]:
        if period == 9:
            series = pd.Series(closes).ewm(span=9, adjust=False).mean().values
        else:
            series = pd.Series(closes).rolling(period).mean().values
        if n > period and not np.isnan(series[-1]) and series[-1] > 0:
            dist_pct = (closes[-1] / series[-1] - 1) * 100
            feats[f"dist_{name}_pct"] = dist_pct

    # Price position in recent range (0 = at low, 1 = at high)
    for window in [20, 60]:
        if n > window:
            recent = closes[-window:]
            low, high = recent.min(), recent.max()
            if high > low:
                feats[f"price_position_{window}d"] = (closes[-1] - low) / (high - low)

    # Gap features
    if n >= 2:
        feats["gap_up_pct"] = max(0, (closes[-1] / closes[-2] - 1) * 100)
        feats["gap_down_pct"] = max(0, (closes[-2] / closes[-1] - 1) * 100)

    return feats


# ═══════════════════════════════════════════════════════════════════════════
#  CATEGORY 2: VOLUME-BASED FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def _volume_features(df: pd.DataFrame) -> Dict[str, float]:
    """Volume ratios, OBV, and volume profile features."""
    volumes = df["Volume"].values.astype(float)
    closes = df["Close"].values.astype(float)
    n = len(volumes)
    feats = {}

    if n < 2:
        return feats

    # Volume ratios vs moving averages
    for period in [5, 10, 20, 60]:
        if n > period:
            avg_vol = np.mean(volumes[-period:])
            if avg_vol > 0:
                feats[f"vol_ratio_{period}d"] = volumes[-1] / avg_vol

    # Volume trend (linear regression slope of log-volume)
    if n > 20:
        log_vols = np.log(volumes[-20:] + 1)
        x = np.arange(20, dtype=float)
        slope = np.polyfit(x, log_vols, 1)[0]
        feats["vol_trend_20d"] = slope

    # On-Balance Volume (OBV) trend
    if n > 20:
        obv = np.zeros(n)
        obv[0] = volumes[0]
        for i in range(1, n):
            if closes[i] > closes[i - 1]:
                obv[i] = obv[i - 1] + volumes[i]
            elif closes[i] < closes[i - 1]:
                obv[i] = obv[i - 1] - volumes[i]
            else:
                obv[i] = obv[i - 1]
        obv_recent = obv[-20:]
        if obv_recent[-1] != 0:
            feats["obv_trend_20d"] = (obv_recent[-1] - obv_recent[0]) / (abs(obv_recent[0]) + 1)

    # Volume-price confirmation (price up + volume up = bullish)
    if n > 5:
        price_up = closes[-1] > closes[-6]
        vol_up = np.mean(volumes[-5:]) > np.mean(volumes[-10:-5]) if n > 10 else False
        feats["vol_price_confirm"] = 1.0 if (price_up and vol_up) else (0.0 if (not price_up and not vol_up) else 0.5)

    return feats


# ═══════════════════════════════════════════════════════════════════════════
#  CATEGORY 3: VOLATILITY-BASED FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def _volatility_features(df: pd.DataFrame) -> Dict[str, float]:
    """ATR, Bollinger Band, and historical volatility features."""
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    n = len(closes)
    feats = {}

    if n < 15:
        return feats

    # ATR (Wilder's smoothing)
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(
        np.maximum(highs - lows, np.abs(highs - prev_close)),
        np.abs(lows - prev_close)
    )

    for period in [7, 14, 21]:
        if n > period:
            atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().values
            atr_val = atr[-1]
            if not np.isnan(atr_val) and closes[-1] > 0:
                feats[f"atr_{period}d_pct"] = (atr_val / closes[-1]) * 100
            # ATR expansion/contraction vs its own moving average
            atr_series = atr[-period:]
            valid = atr_series[~np.isnan(atr_series)]
            if len(valid) > 5:
                feats[f"atr_expansion_{period}d"] = atr_val / (np.mean(valid) + 1e-8)

    # Bollinger Band position & width
    for period in [20]:
        if n > period:
            sma = pd.Series(closes).rolling(period).mean().values
            std = pd.Series(closes).rolling(period).std().values
            upper = sma + 2 * std
            lower = sma - 2 * std
            if not np.isnan(sma[-1]) and sma[-1] > 0:
                band_width = (upper[-1] - lower[-1]) / sma[-1] * 100
                feats["bb_width_20d"] = band_width
                # %B: where price sits within the bands (0=lower, 1=upper)
                if upper[-1] - lower[-1] > 0:
                    feats["bb_pctb_20d"] = (closes[-1] - lower[-1]) / (upper[-1] - lower[-1])

    # Historical volatility (std of log-returns)
    for period in [10, 20, 60]:
        if n > period + 1:
            log_rets = np.diff(np.log(closes[-period - 1:]))
            feats[f"hist_vol_{period}d"] = np.std(log_rets) * math.sqrt(252) * 100

    # Keltner Channel position
    if n > 20:
        ema20 = pd.Series(closes).ewm(span=20, adjust=False).mean().values[-1]
        atr14 = pd.Series(tr).ewm(alpha=1.0 / 14, adjust=False, min_periods=14).mean().values[-1]
        if not np.isnan(ema20) and not np.isnan(atr14) and atr14 > 0:
            feats["keltner_position"] = (closes[-1] - ema20) / (atr14 * 2)

    return feats


# ═══════════════════════════════════════════════════════════════════════════
#  CATEGORY 4: TREND-BASED FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def _trend_features(df: pd.DataFrame) -> Dict[str, float]:
    """EMA/SMA alignment, ADX-like, and linear regression features."""
    closes = df["Close"].values.astype(float)
    n = len(closes)
    feats = {}

    if n < 60:
        return feats

    # EMA/SMA hierarchy (golden alignment score)
    ema9 = pd.Series(closes).ewm(span=9, adjust=False).mean().values
    sma20 = pd.Series(closes).rolling(20).mean().values
    sma50 = pd.Series(closes).rolling(50).mean().values
    sma200 = pd.Series(closes).rolling(200).mean().values

    # Trend alignment score: how many of {Close > EMA9, EMA9 > SMA20, SMA20 > SMA50, SMA50 > SMA200}
    alignment = 0
    if not np.isnan(ema9[-1]) and closes[-1] > ema9[-1]: alignment += 1
    if not np.isnan(ema9[-1]) and not np.isnan(sma20[-1]) and ema9[-1] > sma20[-1]: alignment += 1
    if not np.isnan(sma20[-1]) and not np.isnan(sma50[-1]) and sma20[-1] > sma50[-1]: alignment += 1
    if not np.isnan(sma50[-1]) and not np.isnan(sma200[-1]) and sma50[-1] > sma200[-1]: alignment += 1
    feats["trend_alignment"] = alignment / 4.0

    # Linear regression slope (normalized)
    for window in [20, 60]:
        if n > window:
            y = closes[-window:]
            x = np.arange(window, dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            r_squared = 1 - np.sum((y - (slope * x + intercept)) ** 2) / np.sum((y - y.mean()) ** 2)
            feats[f"lr_slope_{window}d"] = (slope / closes[-1]) * 100 if closes[-1] > 0 else 0
            feats[f"lr_r2_{window}d"] = r_squared

    # EMA9 crossover recency (bars since last crossover)
    if not np.isnan(ema9[-1]):
        cross_ago = 0
        for i in range(n - 2, max(n - 30, 0) - 1, -1):
            if i > 0 and not np.isnan(ema9[i]) and not np.isnan(ema9[i - 1]):
                if (closes[i] > ema9[i] and closes[i - 1] <= ema9[i - 1]):
                    cross_ago = n - 1 - i
                    break
        feats["ema9_cross_recency"] = cross_ago / 30.0  # normalized

    # Higher-High / Higher-Low count in recent window
    if n > 10:
        hh_count = 0
        hl_count = 0
        lookback = min(20, n - 1)
        for i in range(n - lookback, n):
            if i >= 2:
                # Higher high: current high > high of 5 bars ago
                window_start = max(0, i - 5)
                if df["High"].values[i] > np.max(df["High"].values[window_start:i]):
                    hh_count += 1
                # Higher low: current low > low of 5 bars ago
                if df["Low"].values[i] > np.min(df["Low"].values[window_start:i]):
                    hl_count += 1
        feats["hh_count_20d"] = hh_count / lookback
        feats["hl_count_20d"] = hl_count / lookback

    return feats


# ═══════════════════════════════════════════════════════════════════════════
#  CATEGORY 5: CANDLESTICK PATTERN FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def _pattern_features(df: pd.DataFrame) -> Dict[str, float]:
    """Candlestick pattern recognition features."""
    opens = df["Open"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    n = len(closes)
    feats = {}

    if n < 5:
        return feats

    # Candle body and wick features for last 5 candles
    for i in range(1, min(6, n)):
        idx = n - i
        body = abs(closes[idx] - opens[idx])
        total_range = highs[idx] - lows[idx]
        upper_wick = highs[idx] - max(closes[idx], opens[idx])
        lower_wick = min(closes[idx], opens[idx]) - lows[idx]

        if total_range > 0:
            feats[f"body_ratio_{i}"] = body / total_range
            feats[f"upper_wick_{i}"] = upper_wick / total_range
            feats[f"lower_wick_{i}"] = lower_wick / total_range
            feats[f"is_green_{i}"] = 1.0 if closes[idx] > opens[idx] else 0.0

    # Doji detection (body < 10% of range)
    if n >= 1:
        body = abs(closes[-1] - opens[-1])
        total = highs[-1] - lows[-1]
        feats["is_doji"] = 1.0 if total > 0 and body / total < 0.1 else 0.0

    # Hammer/Hanging Man (small body at top, long lower wick)
    if n >= 1 and (highs[-1] - lows[-1]) > 0:
        body = abs(closes[-1] - opens[-1])
        lower_wick = min(closes[-1], opens[-1]) - lows[-1]
        upper_wick = highs[-1] - max(closes[-1], opens[-1])
        total = highs[-1] - lows[-1]
        feats["is_hammer"] = 1.0 if (lower_wick > body * 2 and upper_wick < body * 0.5) else 0.0
        feats["is_shooting_star"] = 1.0 if (upper_wick > body * 2 and lower_wick < body * 0.5) else 0.0

    # Engulfing pattern (current candle engulfs previous)
    if n >= 2:
        prev_body = abs(closes[-2] - opens[-2])
        curr_body = abs(closes[-1] - opens[-1])
        if prev_body > 0 and curr_body > 0:
            bullish_engulf = (closes[-2] < opens[-2] and closes[-1] > opens[-1]
                              and closes[-1] > opens[-2] and opens[-1] < closes[-2])
            bearish_engulf = (closes[-2] > opens[-2] and closes[-1] < opens[-1]
                              and closes[-1] < opens[-2] and opens[-1] > closes[-2])
            feats["bullish_engulf"] = 1.0 if bullish_engulf else 0.0
            feats["bearish_engulf"] = 1.0 if bearish_engulf else 0.0

    # Consecutive green/red candle count
    green_streak = 0
    red_streak = 0
    for i in range(n - 1, max(n - 10, 0) - 1, -1):
        if closes[i] > opens[i]:
            if red_streak == 0:
                green_streak += 1
            else:
                break
        elif closes[i] < opens[i]:
            if green_streak == 0:
                red_streak += 1
            else:
                break
    feats["green_streak"] = green_streak / 5.0
    feats["red_streak"] = red_streak / 5.0

    return feats


# ═══════════════════════════════════════════════════════════════════════════
#  CATEGORY 6: OSCILLATOR / MOMENTUM FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def _oscillator_features(df: pd.DataFrame) -> Dict[str, float]:
    """RSI, MACD, Stochastic, CCI, Williams %R."""
    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    n = len(closes)
    feats = {}

    if n < 30:
        return feats

    # RSI (14-period)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
    avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
    if avg_loss > 0:
        rs = avg_gain / avg_loss
        feats["rsi_14"] = 100 - (100 / (1 + rs))
    else:
        feats["rsi_14"] = 100.0 if avg_gain > 0 else 50.0

    # MACD
    ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean().values
    if not np.isnan(ema12[-1]) and not np.isnan(ema26[-1]):
        macd_line = ema12 - ema26
        signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
        if not np.isnan(signal_line[-1]):
            feats["macd_hist"] = macd_line[-1] - signal_line[-1]
            if signal_line[-1] != 0:
                feats["macd_hist_pct"] = ((macd_line[-1] - signal_line[-1]) / abs(signal_line[-1])) * 100
            # MACD crossover direction
            if len(signal_line) >= 2 and not np.isnan(signal_line[-2]):
                feats["macd_cross"] = 1.0 if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2] else (
                    -1.0 if macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2] else 0.0)

    # Stochastic Oscillator (%K and %D)
    if n >= 14:
        highest_high = np.max(highs[-14:])
        lowest_low = np.min(lows[-14:])
        if highest_high > lowest_low:
            feats["stoch_k"] = (closes[-1] - lowest_low) / (highest_high - lowest_low) * 100
            # %D = 3-period SMA of %K
            k_values = []
            for j in range(n - 14, n):
                hh = np.max(highs[j - 13:j + 1])
                ll = np.min(lows[j - 13:j + 1])
                if hh > ll:
                    k_values.append((closes[j] - ll) / (hh - ll) * 100)
            if len(k_values) >= 3:
                feats["stoch_d"] = np.mean(k_values[-3:])

    # CCI (Commodity Channel Index, 20-period)
    if n > 20:
        typical_price = (highs[-20:] + lows[-20:] + closes[-20:]) / 3
        tp_mean = np.mean(typical_price)
        tp_mad = np.mean(np.abs(typical_price - tp_mean))
        if tp_mad > 0:
            feats["cci_20"] = (typical_price[-1] - tp_mean) / (0.015 * tp_mad)

    # Williams %R
    if n >= 14:
        hh = np.max(highs[-14:])
        ll = np.min(lows[-14:])
        if hh > ll:
            feats["williams_r"] = (hh - closes[-1]) / (hh - ll) * -100

    return feats


# ═══════════════════════════════════════════════════════════════════════════
#  CATEGORY 7: SUPPORT/RESISTANCE FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def _support_resistance_features(df: pd.DataFrame) -> Dict[str, float]:
    """Distance to nearest detected support/resistance levels."""
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    n = len(closes)
    feats = {}

    if n < 30:
        return feats

    # Detect support levels using fractal pivots (same algorithm as support_detector.py)
    def find_pivot_lows(series, fractal=2):
        pivots = []
        for i in range(fractal, len(series) - fractal):
            window = series[i - fractal: i + fractal + 1]
            center = series[i]
            if np.isnan(center):
                continue
            is_min = True
            for j, v in enumerate(window):
                if j == fractal:
                    continue
                if not np.isnan(v) and v < center:
                    is_min = False
                    break
            if is_min:
                pivots.append((i, float(center)))
        return pivots

    def find_pivot_highs(series, fractal=2):
        pivots = []
        for i in range(fractal, len(series) - fractal):
            window = series[i - fractal: i + fractal + 1]
            center = series[i]
            if np.isnan(center):
                continue
            is_max = True
            for j, v in enumerate(window):
                if j == fractal:
                    continue
                if not np.isnan(v) and v > center:
                    is_max = False
                    break
            if is_max:
                pivots.append((i, float(center)))
        return pivots

    # Find support levels from recent 120 bars
    lookback = min(120, n)
    recent_lows = lows[-lookback:]
    recent_highs = highs[-lookback:]

    support_pivots = find_pivot_lows(recent_lows)
    resistance_pivots = find_pivot_highs(recent_highs)

    # Cluster supports and resistances
    def cluster_levels(pivots, threshold_pct=1.0):
        if not pivots:
            return []
        sorted_pivots = sorted(pivots, key=lambda p: p[1])
        clusters = [[sorted_pivots[0]]]
        for date_idx, price in sorted_pivots[1:]:
            ref = clusters[-1][0][1]
            if abs(price - ref) / ref * 100 <= threshold_pct:
                clusters[-1].append((date_idx, price))
            else:
                clusters.append([(date_idx, price)])
        # Return cluster averages, sorted by distance to current price
        levels = [np.mean([p[1] for p in c]) for c in clusters if len(c) >= 2]
        return sorted(levels, key=lambda l: abs(closes[-1] - l))

    support_levels = cluster_levels(support_pivots)
    resistance_levels = cluster_levels(resistance_pivots)

    # Distance to nearest support and resistance
    if support_levels:
        nearest_support = support_levels[0]
        feats["dist_to_support_pct"] = (closes[-1] - nearest_support) / nearest_support * 100
        feats["support_touches"] = min(len([p for p in support_pivots if abs(p[1] - nearest_support) / nearest_support < 0.01]), 10) / 10.0
    else:
        feats["dist_to_support_pct"] = 5.0  # no support nearby
        feats["support_touches"] = 0.0

    if resistance_levels:
        nearest_resistance = resistance_levels[0]
        feats["dist_to_resistance_pct"] = (nearest_resistance - closes[-1]) / nearest_resistance * 100
        feats["resistance_touches"] = min(len([p for p in resistance_pivots if abs(p[1] - nearest_resistance) / nearest_resistance < 0.01]), 10) / 10.0
    else:
        feats["dist_to_resistance_pct"] = 5.0
        feats["resistance_touches"] = 0.0

    # Support/Resistance risk ratio (how close to support vs resistance)
    if support_levels and resistance_levels:
        total_range = resistance_levels[0] - support_levels[0]
        if total_range > 0:
            feats["sr_position"] = (closes[-1] - support_levels[0]) / total_range
            feats["sr_reward_risk"] = feats["dist_to_resistance_pct"] / (abs(feats["dist_to_support_pct"]) + 0.1)

    return feats


# ═══════════════════════════════════════════════════════════════════════════
#  MASTER FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════

# Canonical feature name list (for consistent model input shape)
FEATURE_NAMES: List[str] = [
    # Price (15)
    "return_1d", "return_3d", "return_5d", "return_10d", "return_20d", "return_60d",
    "momentum_accel",
    "dist_ema9_pct", "dist_sma20_pct", "dist_sma50_pct", "dist_sma200_pct",
    "price_position_20d", "price_position_60d",
    "gap_up_pct", "gap_down_pct",
    # Volume (10)
    "vol_ratio_5d", "vol_ratio_10d", "vol_ratio_20d", "vol_ratio_60d",
    "vol_trend_20d", "obv_trend_20d", "vol_price_confirm",
    # Volatility (12)
    "atr_7d_pct", "atr_14d_pct", "atr_21d_pct",
    "atr_expansion_7d", "atr_expansion_14d", "atr_expansion_21d",
    "bb_width_20d", "bb_pctb_20d",
    "hist_vol_10d", "hist_vol_20d", "hist_vol_60d",
    "keltner_position",
    # Trend (12)
    "trend_alignment",
    "lr_slope_20d", "lr_r2_20d", "lr_slope_60d", "lr_r2_60d",
    "ema9_cross_recency",
    "hh_count_20d", "hl_count_20d",
    # Pattern (15)
    "body_ratio_1", "upper_wick_1", "lower_wick_1", "is_green_1",
    "body_ratio_2", "upper_wick_2", "lower_wick_2", "is_green_2",
    "body_ratio_3", "upper_wick_3", "lower_wick_3", "is_green_3",
    "is_doji", "is_hammer", "is_shooting_star",
    "bullish_engulf", "bearish_engulf",
    "green_streak", "red_streak",
    # Oscillator (10)
    "rsi_14", "macd_hist", "macd_hist_pct", "macd_cross",
    "stoch_k", "stoch_d", "cci_20", "williams_r",
    # Support/Resistance (10)
    "dist_to_support_pct", "dist_to_resistance_pct",
    "support_touches", "resistance_touches",
    "sr_position", "sr_reward_risk",
]

# Number of input features
NUM_FEATURES: int = len(FEATURE_NAMES)

# Feature normalization stats (mean, std) — updated during training
FEATURE_STATS: Dict[str, Dict[str, float]] = {}


def extract_features(df: pd.DataFrame, normalize: bool = True) -> np.ndarray:
    """
    Extract all features from an OHLCV DataFrame.
    Returns a 1D numpy array of shape (num_features,).

    Parameters
    ----------
    df : pd.DataFrame
        Must have Open, High, Low, Close, Volume columns.
    normalize : bool
        If True, z-score normalize using stored stats. If no stats yet, returns raw.
    """
    df = df.copy().dropna()
    if len(df) < 15:
        # Return zeros for insufficient data
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    # Extract all feature categories
    all_feats = {}
    all_feats.update(_price_features(df))
    all_feats.update(_volume_features(df))
    all_feats.update(_volatility_features(df))
    all_feats.update(_trend_features(df))
    all_feats.update(_pattern_features(df))
    all_feats.update(_oscillator_features(df))
    all_feats.update(_support_resistance_features(df))

    # Build the canonical feature vector
    vector = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    for i, name in enumerate(FEATURE_NAMES):
        val = all_feats.get(name, 0.0)
        if isinstance(val, (int, float)) and not np.isnan(val) and not np.isinf(val):
            vector[i] = float(val)
        else:
            vector[i] = 0.0

    # Z-score normalize if stats are available
    if normalize and FEATURE_STATS:
        for i, name in enumerate(FEATURE_NAMES):
            if name in FEATURE_STATS:
                stats = FEATURE_STATS[name]
                if stats["std"] > 1e-8:
                    vector[i] = (vector[i] - stats["mean"]) / stats["std"]
                else:
                    vector[i] = 0.0

    return vector


def update_feature_stats(feature_matrix: np.ndarray) -> None:
    """
    Compute and store normalization statistics from a batch of feature vectors.
    Called after collecting enough training data.
    """
    global FEATURE_STATS
    n_features = feature_matrix.shape[1]
    for i in range(min(n_features, len(FEATURE_NAMES))):
        col = feature_matrix[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) > 10:
            FEATURE_STATS[FEATURE_NAMES[i]] = {
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid)),
                "min": float(np.min(valid)),
                "max": float(np.max(valid)),
            }


def extract_features_for_training(
    df: pd.DataFrame,
    lookahead_days: int = 5,
    target_pct: float = 3.0,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Extract features AND label from historical data for supervised learning.
    The label is the maximum gain % achieved within `lookahead_days` bars
    after the current bar.

    Returns (features_array, label) or (None, None) if insufficient data.
    """
    df = df.copy().dropna()
    n = len(df)

    # Need at least 200 bars for all indicators + lookahead
    min_required = max(200, 60) + lookahead_days
    if n < min_required:
        return None, None

    # Walk through history and create (features, label) pairs
    # We take the last 60 valid starting points (not overlapping with lookahead)
    features_list = []
    labels_list = []

    start_idx = 200  # skip warmup period
    end_idx = n - lookahead_days

    for i in range(start_idx, end_idx, 3):  # step by 3 to reduce correlation
        # Extract features using data up to bar i
        window_df = df.iloc[:i + 1]
        feats = extract_features(window_df, normalize=False)

        # Label: max gain % in next `lookahead_days` bars
        future_highs = df["High"].values[i + 1: i + 1 + lookahead_days]
        future_lows = df["Low"].values[i + 1: i + 1 + lookahead_days]
        entry_close = df["Close"].values[i]

        if entry_close <= 0 or len(future_highs) == 0:
            continue

        max_gain = (np.max(future_highs) / entry_close - 1) * 100
        max_loss = (np.min(future_lows) / entry_close - 1) * 100
        exit_close = df["Close"].values[min(i + lookahead_days, n - 1)]
        actual_return = (exit_close / entry_close - 1) * 100

        # Multi-label:
        # 1. Will it hit +target_pct%? (binary classification)
        # 2. What's the expected return? (regression)
        # 3. What's the max drawdown? (risk estimation)
        features_list.append(feats)
        labels_list.append({
            "hit_target": 1.0 if max_gain >= target_pct else 0.0,
            "return_pct": actual_return,
            "max_gain_pct": max_gain,
            "max_loss_pct": max_loss,
        })

    if not features_list:
        return None, None

    feature_matrix = np.stack(features_list)
    return feature_matrix, labels_list