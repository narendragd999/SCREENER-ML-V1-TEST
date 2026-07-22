"""
Per-Stock Deep Analyzer with TensorFlow
========================================
Downloads full history for each stock, analyzes individual behavior,
recognizes patterns, and suggests the best strategy to hit 3% target.

Analysis includes:
  1. Historical win rate for 3% target (overall + conditional)
  2. Pattern frequency analysis (candlestick + structural)
  3. Best entry condition discovery (what features predict 3% hit)
  4. Time-to-target distribution
  5. Support/Resistance level detection
  6. Volatility & drawdown profiling
  7. Per-stock strategy recommendation
"""

from __future__ import annotations
import math
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger("stock_analyzer")


# ═══════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PatternStats:
    name: str
    count: int
    win_rate: float       # % times 3% hit after this pattern
    avg_days_to_target: float
    avg_return: float     # average 5-day return after pattern
    sample_size: int


@dataclass
class EntryCondition:
    name: str
    description: str
    occurrences: int
    win_rate: float
    avg_return: float
    avg_days_to_target: float
    avg_max_drawdown: float
    score: float          # composite score for ranking


@dataclass 
class StockReport:
    ticker: str
    data_points: int
    years_of_data: float

    # Overall 3% target stats
    overall_win_rate: float
    avg_days_to_target: float
    median_days_to_target: float
    avg_return_5d: float
    max_drawdown_avg: float

    # Current state
    current_price: float
    current_regime: str
    current_rsi: float
    current_trend_alignment: float
    current_volatility_pct: float

    # Top patterns
    top_patterns: List[Dict]

    # Best entry conditions (ranked)
    best_entries: List[Dict]

    # Support / Resistance
    support_levels: List[float]
    resistance_levels: List[float]

    # Strategy recommendation
    recommended_strategy: str
    recommended_entry: str
    recommended_stop: str
    recommended_target: str
    confidence: float

    # Risk profile
    risk_level: str
    avg_daily_move_pct: float
    max_intraday_move_pct: float
    best_day_of_week: str
    worst_day_of_week: str

    # Pattern recognition summary
    recognized_patterns: List[str]
    pattern_follow_through: Dict[str, float]  # pattern_name -> win_rate


# ═══════════════════════════════════════════════════════════════════════════
#  CORE ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def analyze_stock(df: pd.DataFrame, ticker: str, target_pct: float = 3.0) -> StockReport:
    """
    Perform deep per-stock analysis.
    
    Args:
        df: OHLCV DataFrame with columns [Open, High, Low, Close, Volume]
        ticker: Stock symbol
        target_pct: Target return % (default 3%)
    
    Returns:
        StockReport with complete analysis
    """
    if df is None or len(df) < 200:
        return _empty_report(ticker)

    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    opens = df["Open"].values.astype(float)
    volumes = df["Volume"].values.astype(float)
    n = len(closes)

    # ── 1. Historical 3% target analysis ──
    win_rates, days_to_target, returns_5d, max_drawdowns = [], [], [], []
    lookahead = 10  # max days to look ahead for target

    for i in range(200, n - lookahead):
        entry = closes[i]
        future_highs = highs[i+1:i+1+lookahead]
        future_lows = lows[i+1:i+1+lookahead]
        future_close = closes[i+1:i+1+lookahead]

        # Did it hit target?
        max_gain = (np.max(future_highs) / entry - 1) * 100
        max_loss = (np.min(future_lows) / entry - 1) * 100
        actual_return = (future_close[-1] / entry - 1) * 100

        hit_target = 1.0 if max_gain >= target_pct else 0.0
        # Find days to target
        days = 0
        if max_gain >= target_pct:
            for j in range(lookahead):
                if (highs[i+1+j] / entry - 1) * 100 >= target_pct:
                    days = j + 1
                    break

        win_rates.append(hit_target)
        days_to_target.append(days)
        returns_5d.append(actual_return)
        max_drawdowns.append(max_loss)

    overall_win_rate = np.mean(win_rates) * 100 if win_rates else 0
    hit_days = [d for d in days_to_target if d > 0]
    avg_days = np.mean(hit_days) if hit_days else 0
    median_days = np.median(hit_days) if hit_days else 0
    avg_return = np.mean(returns_5d) if returns_5d else 0
    avg_dd = np.mean(max_drawdowns) if max_drawdowns else 0

    # ── 2. Pattern Detection ──
    patterns = _detect_all_patterns(df)
    pattern_stats = _analyze_patterns(df, patterns, target_pct, lookahead)

    # ── 3. Best Entry Conditions ──
    entries = _find_best_entries(df, target_pct, lookahead)

    # ── 4. Support / Resistance ──
    supports, resistances = _find_support_resistance(df)

    # ── 5. Current State ──
    ema9 = pd.Series(closes).ewm(span=9, adjust=False).mean().values
    sma20 = pd.Series(closes).rolling(20).mean().values
    sma50 = pd.Series(closes).rolling(50).mean().values
    sma200 = pd.Series(closes).rolling(200).mean().values

    # RSI
    rsi = _compute_rsi(closes, 14)

    # Trend alignment
    alignment = 0
    if not np.isnan(ema9[-1]) and closes[-1] > ema9[-1]: alignment += 1
    if not np.isnan(ema9[-1]) and not np.isnan(sma20[-1]) and ema9[-1] > sma20[-1]: alignment += 1
    if not np.isnan(sma20[-1]) and not np.isnan(sma50[-1]) and sma20[-1] > sma50[-1]: alignment += 1
    if not np.isnan(sma50[-1]) and not np.isnan(sma200[-1]) and sma50[-1] > sma200[-1]: alignment += 1
    alignment_score = alignment / 4.0

    if alignment_score >= 0.75:
        regime = "STRONG UPTREND"
    elif alignment_score >= 0.5:
        regime = "UPTREND"
    elif alignment_score >= 0.25:
        regime = "SIDEWAYS"
    else:
        regime = "DOWNTREND"

    # Volatility
    log_rets = np.diff(np.log(closes[-21:]))
    vol_pct = np.std(log_rets) * math.sqrt(252) * 100 if len(log_rets) > 1 else 0

    # ── 6. Day-of-week analysis ──
    if hasattr(df.index, 'dayofweek'):
        dow_returns = {}
        for dow in range(5):
            mask = df.index.dayofweek == dow
            if mask.sum() > 10:
                daily_rets = df.loc[mask, "Close"].pct_change().dropna()
                dow_returns[dow] = daily_rets.mean() * 100
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        best_dow = max(dow_returns, key=dow_returns.get) if dow_returns else "N/A"
        worst_dow = min(dow_returns, key=dow_returns.get) if dow_returns else "N/A"
    else:
        best_dow = "N/A"
        worst_dow = "N/A"

    # ── 7. Risk profiling ──
    daily_returns = pd.Series(closes).pct_change().dropna().values
    avg_daily_move = np.mean(np.abs(daily_returns)) * 100 if len(daily_returns) > 0 else 0
    intraday_moves = ((highs - lows) / closes * 100)
    max_intraday = np.mean(intraday_moves[-60:]) if len(intraday_moves) > 0 else 0

    if avg_daily_move > 3.0:
        risk = "VERY HIGH"
    elif avg_daily_move > 2.0:
        risk = "HIGH"
    elif avg_daily_move > 1.0:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # ── 8. Recognized current patterns ──
    current_patterns = _detect_current_patterns(df)
    pattern_follow = {p.name: p.win_rate for p in pattern_stats[:10]}

    # ── 9. Strategy Recommendation ──
    strategy, entry, stop, target, conf = _recommend_strategy(
        overall_win_rate, entries, regime, vol_pct, alignment_score,
        rsi[-1] if not np.isnan(rsi[-1]) else 50, supports, resistances,
        current_patterns, avg_dd
    )

    # Years of data
    if hasattr(df.index, 'to_period'):
        years = (df.index[-1] - df.index[0]).days / 365.25
    else:
        years = n / 252

    report = StockReport(
        ticker=ticker,
        data_points=n,
        years_of_data=round(years, 1),
        overall_win_rate=round(overall_win_rate, 1),
        avg_days_to_target=round(avg_days, 1),
        median_days_to_target=round(median_days, 1),
        avg_return_5d=round(avg_return, 2),
        max_drawdown_avg=round(avg_dd, 2),
        current_price=round(closes[-1], 2),
        current_regime=regime,
        current_rsi=round(float(rsi[-1]), 1) if not np.isnan(rsi[-1]) else 0,
        current_trend_alignment=round(alignment_score, 2),
        current_volatility_pct=round(vol_pct, 2),
        top_patterns=[asdict(p) for p in pattern_stats[:8]],
        best_entries=[asdict(e) for e in entries[:6]],
        support_levels=[round(s, 2) for s in supports[:5]],
        resistance_levels=[round(r, 2) for r in resistances[:5]],
        recommended_strategy=strategy,
        recommended_entry=entry,
        recommended_stop=stop,
        recommended_target=target,
        confidence=round(conf, 1),
        risk_level=risk,
        avg_daily_move_pct=round(avg_daily_move, 2),
        max_intraday_move_pct=round(max_intraday, 2),
        best_day_of_week=best_dow,
        worst_day_of_week=worst_dow,
        recognized_patterns=current_patterns,
        pattern_follow_through=pattern_follow,
    )

    return report


# ═══════════════════════════════════════════════════════════════════════════
#  PATTERN DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def _compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI."""
    n = len(closes)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, n):
        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100.0
    return rsi


def _detect_candlestick_pattern(opens, highs, lows, closes, idx) -> str:
    """Detect candlestick pattern at index idx."""
    if idx < 2:
        return ""
    
    o, h, l, c = opens[idx], highs[idx], lows[idx], closes[idx]
    po, ph, pl, pc = opens[idx-1], highs[idx-1], lows[idx-1], closes[idx-1]
    body = abs(c - o)
    total_range = h - l
    upper_wick = h - max(c, o)
    lower_wick = min(c, o) - l
    
    if total_range == 0:
        return ""
    
    body_ratio = body / total_range
    is_green = c > o
    prev_green = pc > po
    prev_body = abs(pc - po)
    prev_range = ph - pl
    
    # Doji
    if body_ratio < 0.1:
        return "Doji"
    
    # Hammer (bullish reversal)
    if lower_wick > body * 2 and upper_wick < body * 0.5 and not is_green:
        return "Hammer"
    
    # Shooting Star (bearish reversal)
    if upper_wick > body * 2 and lower_wick < body * 0.5 and is_green:
        return "Shooting Star"
    
    # Bullish Engulfing
    if prev_green == False and is_green and c > po and o < pc and body > prev_body:
        return "Bullish Engulfing"
    
    # Bearish Engulfing
    if prev_green and not is_green and o > pc and c < po and body > prev_body:
        return "Bearish Engulfing"
    
    # Morning Star (3-candle)
    if idx >= 2:
        ppo, pph, ppl, ppc = opens[idx-2], highs[idx-2], lows[idx-2], closes[idx-2]
        if (ppc < ppo and abs(ppc - ppo) / max(pph - ppl, 1) > 0.6 and  # big red
            body_ratio < 0.3 and  # small body middle
            is_green and c > (ppo + ppc) / 2 and body > prev_body * 0.5):  # big green
            return "Morning Star"
    
    # Evening Star (3-candle)
    if idx >= 2:
        ppo, pph, ppl, ppc = opens[idx-2], highs[idx-2], lows[idx-2], closes[idx-2]
        if (ppc > ppo and abs(ppc - ppo) / max(pph - ppl, 1) > 0.6 and
            body_ratio < 0.3 and
            not is_green and c < (ppo + ppc) / 2 and body > prev_body * 0.5):
            return "Evening Star"
    
    # Three White Soldiers
    if idx >= 2:
        if (closes[idx] > closes[idx-1] > closes[idx-2] and
            opens[idx] > opens[idx-1] > opens[idx-2] and
            is_green and prev_green and closes[idx-2] > opens[idx-2]):
            return "Three White Soldiers"
    
    # Three Black Crows
    if idx >= 2:
        if (closes[idx] < closes[idx-1] < closes[idx-2] and
            opens[idx] < opens[idx-1] < opens[idx-2] and
            not is_green and not prev_green and closes[idx-2] < opens[idx-2]):
            return "Three Black Crows"
    
    if is_green and body_ratio > 0.7:
        return "Bullish Marubozu"
    if not is_green and body_ratio > 0.7:
        return "Bearish Marubozu"
    
    return ""


def _detect_all_patterns(df: pd.DataFrame) -> Dict[int, str]:
    """Detect candlestick patterns for entire history. Returns {index: pattern_name}."""
    opens = df["Open"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    n = len(closes)
    
    patterns = {}
    for i in range(2, n):
        p = _detect_candlestick_pattern(opens, highs, lows, closes, i)
        if p:
            patterns[i] = p
    return patterns


def _detect_current_patterns(df: pd.DataFrame) -> List[str]:
    """Detect patterns on the most recent candles."""
    opens = df["Open"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    n = len(closes)
    
    found = []
    for i in range(max(0, n-5), n):
        p = _detect_candlestick_pattern(opens, highs, lows, closes, i)
        if p and p not in found:
            found.append(p)
    return found


def _analyze_patterns(df: pd.DataFrame, patterns: Dict[int, str],
                       target_pct: float, lookahead: int) -> List[PatternStats]:
    """Analyze win rate for each pattern type."""
    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)
    n = len(closes)

    # Group by pattern name
    pattern_data: Dict[str, List] = {}
    for idx, name in patterns.items():
        if idx + lookahead < n:
            if name not in pattern_data:
                pattern_data[name] = []
            entry = closes[idx]
            future_highs = highs[idx+1:idx+1+lookahead]
            future_close = closes[idx+1:idx+1+lookahead]
            max_gain = (np.max(future_highs) / entry - 1) * 100
            actual_return = (future_close[-1] / entry - 1) * 100
            hit = 1.0 if max_gain >= target_pct else 0.0
            days = 0
            if max_gain >= target_pct:
                for j in range(lookahead):
                    if (highs[idx+1+j] / entry - 1) * 100 >= target_pct:
                        days = j + 1
                        break
            pattern_data[name].append((hit, days, actual_return))

    stats = []
    for name, data in sorted(pattern_data.items(), key=lambda x: -len(x[1])):
        if len(data) < 5:
            continue
        hits = [d[0] for d in data]
        days = [d[1] for d in data if d[1] > 0]
        rets = [d[2] for d in data]
        stats.append(PatternStats(
            name=name,
            count=len(data),
            win_rate=round(np.mean(hits) * 100, 1),
            avg_days_to_target=round(np.mean(days), 1) if days else 0,
            avg_return=round(np.mean(rets), 2),
            sample_size=len(data),
        ))
    
    return sorted(stats, key=lambda x: -x.win_rate)


# ═══════════════════════════════════════════════════════════════════════════
#  BEST ENTRY CONDITIONS
# ═══════════════════════════════════════════════════════════════════════════

def _find_best_entries(df: pd.DataFrame, target_pct: float, lookahead: int) -> List[EntryCondition]:
    """Find which market conditions lead to the highest 3% win rate."""
    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    volumes = df["Volume"].values.astype(float)
    n = len(closes)

    ema9 = pd.Series(closes).ewm(span=9, adjust=False).mean().values
    sma20 = pd.Series(closes).rolling(20).mean().values
    sma50 = pd.Series(closes).rolling(50).mean().values
    rsi = _compute_rsi(closes, 14)
    atr = _compute_atr(highs, lows, closes, 14)

    conditions_def = [
        ("EMA9 Crossover Bullish", "Price crosses above 9-EMA",
         lambda i: i > 0 and not np.isnan(ema9[i]) and not np.isnan(ema9[i-1])
         and closes[i] > ema9[i] and closes[i-1] <= ema9[i-1]),

        ("Near Support Bounce", "Price within 1% of detected support level",
         lambda i: _near_support_check(closes, lows, i)),

        ("RSI Oversold Bounce", "RSI < 35 and next day closes higher",
         lambda i: rsi[i] < 35 and i + 1 < n and closes[i+1] > closes[i]),

        ("Volume Surge + Green", "Volume > 2x avg and bullish candle",
         lambda i: i > 20 and volumes[i] > np.mean(volumes[i-20:i]) * 2
         and closes[i] > df["Open"].values[i]),

        ("Pullback to EMA9", "Price pulls back to within 0.5% of 9-EMA in uptrend",
         lambda i: not np.isnan(ema9[i]) and not np.isnan(sma50[i])
         and ema9[i] > sma50[i]  # uptrend
         and abs(closes[i] - ema9[i]) / ema9[i] < 0.005),  # near EMA9

        ("Bollinger Band Lower Touch", "Price touches lower Bollinger Band",
         lambda i: i >= 20 and not np.isnan(sma20[i])
         and abs(closes[i] - (sma20[i] - 2 * pd.Series(closes[i-20:i+1]).std())) / closes[i] < 0.01),

        ("Consolidation Breakout", "Low volatility period followed by big green candle",
         lambda i: i >= 20 and volumes[i] > np.mean(volumes[i-20:i]) * 1.5
         and (highs[i] - lows[i]) / closes[i] > np.mean([(highs[j]-lows[j])/closes[j] for j in range(i-10, i)]) * 1.8),

        ("Hammer Candle", "Hammer pattern at support",
         lambda i: _detect_candlestick_pattern(
             df["Open"].values, highs, lows, closes, i) == "Hammer"),

        ("Bullish Engulfing", "Bullish engulfing pattern",
         lambda i: _detect_candlestick_pattern(
             df["Open"].values, highs, lows, closes, i) == "Bullish Engulfing"),

        ("SMA20 Golden Cross", "Price crosses above SMA20",
         lambda i: i > 0 and not np.isnan(sma20[i]) and not np.isnan(sma20[i-1])
         and closes[i] > sma20[i] and closes[i-1] <= sma20[i-1]),

        ("Low ATR Entry", "ATR is below average (quiet period entry)",
         lambda i: not np.isnan(atr[i]) and i > 20
         and atr[i] < np.mean(atr[max(0,i-20):i]) * 0.7
         and closes[i] > ema9[i] if not np.isnan(ema9[i]) else False),

        ("Gap Up + Volume", "Gap up opening with high volume",
         lambda i: i > 0 and closes[i] > highs[i-1]
         and volumes[i] > np.mean(volumes[max(0,i-20):i]) * 1.3),
    ]

    results = []
    for name, desc, condition_fn in conditions_def:
        hit_count = 0
        total_count = 0
        total_return = 0
        total_days = 0
        total_dd = 0

        for i in range(200, n - lookahead):
            try:
                if condition_fn(i):
                    entry = closes[i]
                    future_highs = highs[i+1:i+1+lookahead]
                    future_lows = lows[i+1:i+1+lookahead]
                    future_close = closes[i+1:idx+1+lookahead] if False else closes[i+1:i+1+lookahead]

                    max_gain = (np.max(future_highs) / entry - 1) * 100
                    max_loss = (np.min(future_lows) / entry - 1) * 100
                    actual_return = (future_close[-1] / entry - 1) * 100 if len(future_close) > 0 else 0

                    hit = 1.0 if max_gain >= target_pct else 0.0
                    days = 0
                    if max_gain >= target_pct:
                        for j in range(lookahead):
                            if (highs[i+1+j] / entry - 1) * 100 >= target_pct:
                                days = j + 1
                                break

                    total_count += 1
                    hit_count += hit
                    total_return += actual_return
                    total_dd += max_loss
                    if days > 0:
                        total_days += days
            except Exception:
                continue

        if total_count >= 5:
            win_rate = (hit_count / total_count) * 100
            avg_ret = total_return / total_count
            avg_d = total_days / hit_count if hit_count > 0 else 0
            avg_mdd = total_dd / total_count
            
            # Composite score: win_rate * 0.4 + (avg_ret/target) * 30 + (1/avg_days if avg_days>0) * 20
            score = (win_rate / 100) * 40
            score += min(avg_ret / target_pct, 1.5) * 30
            if avg_d > 0:
                score += min(5.0 / avg_d, 1.0) * 20
            score += max(0, 1 - abs(avg_mdd) / (target_pct * 2)) * 10

            results.append(EntryCondition(
                name=name,
                description=desc,
                occurrences=total_count,
                win_rate=round(win_rate, 1),
                avg_return=round(avg_ret, 2),
                avg_days_to_target=round(avg_d, 1),
                avg_max_drawdown=round(avg_mdd, 2),
                score=round(score, 1),
            ))

    return sorted(results, key=lambda x: -x.score)


def _compute_atr(highs, lows, closes, period=14):
    """Compute ATR."""
    n = len(closes)
    atr = np.full(n, np.nan)
    if n < period + 1:
        return atr
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(
        np.maximum(highs - lows, np.abs(highs - prev_close)),
        np.abs(lows - prev_close))
    atr[period:] = pd.Series(tr).ewm(alpha=1.0/period, adjust=False, min_periods=period).mean().values[period:]
    return atr


def _near_support_check(closes, lows, idx) -> bool:
    """Check if price is near a support level."""
    if idx < 20:
        return False
    # Simple support: recent swing low within 1%
    recent_lows = lows[idx-20:idx]
    swing_low = np.min(recent_lows)
    return abs(closes[idx] - swing_low) / swing_low < 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  SUPPORT / RESISTANCE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def _find_support_resistance(df: pd.DataFrame, 
                              touch_threshold: float = 0.015,
                              min_touches: int = 3) -> Tuple[List[float], List[float]]:
    """Find horizontal support and resistance levels using pivot clustering."""
    closes = df["Close"].values.astype(float)
    lows = df["Low"].values.astype(float)
    highs = df["High"].values.astype(float)
    n = len(closes)
    
    # Find fractal pivot lows and highs
    pivot_lows = []
    pivot_highs = []
    for i in range(2, n - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            pivot_lows.append((i, lows[i]))
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            pivot_highs.append((i, highs[i]))
    
    # Cluster nearby levels
    supports = _cluster_levels([p[1] for p in pivot_lows], touch_threshold, min_touches)
    resistances = _cluster_levels([p[1] for p in pivot_highs], touch_threshold, min_touches)
    
    # Filter: only levels below current price are supports, above are resistance
    current = closes[-1]
    supports = [s for s in supports if s < current * 1.01]
    resistances = [r for r in resistances if r > current * 0.99]
    
    return sorted(supports, reverse=True)[:5], sorted(resistances)[:5]


def _cluster_levels(levels: List[float], threshold: float, min_touches: int) -> List[float]:
    """Cluster nearby price levels together."""
    if not levels:
        return []
    
    levels = sorted(levels)
    clusters = []
    current_cluster = [levels[0]]
    
    for i in range(1, len(levels)):
        if abs(levels[i] - current_cluster[0]) / current_cluster[0] < threshold:
            current_cluster.append(levels[i])
        else:
            if len(current_cluster) >= min_touches:
                clusters.append(np.mean(current_cluster))
            current_cluster = [levels[i]]
    
    if len(current_cluster) >= min_touches:
        clusters.append(np.mean(current_cluster))
    
    return clusters


# ═══════════════════════════════════════════════════════════════════════════
#  STRATEGY RECOMMENDATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def _recommend_strategy(win_rate, entries, regime, vol, alignment, rsi,
                         supports, resistances, patterns, avg_dd) -> Tuple[str, str, str, str, float]:
    """Generate strategy recommendation based on analysis."""
    confidence = 50.0
    strategy_parts = []
    entry_part = ""
    stop_part = ""
    target_part = "3.0% (fixed)"

    # Base strategy on best entry condition
    if entries:
        best = entries[0]
        entry_part = best.description
        confidence = min(95, confidence + best.win_rate * 0.3)
        
        if best.win_rate > 60:
            strategy_parts.append(f"Use '{best.name}' entry (win rate: {best.win_rate}%)")
        elif best.win_rate > 45:
            strategy_parts.append(f"'{best.name}' shows edge ({best.win_rate}%)")
        else:
            strategy_parts.append(f"No strong edge found — best is '{best.name}' ({best.win_rate}%)")
    else:
        entry_part = "No clear entry signal detected"
    
    # Adjust for regime
    if "UPTREND" in regime:
        strategy_parts.append("Favor longs in current uptrend")
        confidence += 10
        entry_part += " + Confirm with uptrend alignment"
    elif "DOWNTREND" in regime:
        strategy_parts.append("CAUTION: Downtrend — reduce position size or wait")
        confidence -= 15
        entry_part += " + Wait for trend reversal confirmation"
    
    # Adjust for volatility
    if vol > 40:
        strategy_parts.append(f"High volatility ({vol:.0f}%) — use wider stops")
        stop_part = "ATR-based: 1.5x ATR below entry"
    elif vol > 25:
        stop_part = "ATR-based: 1.2x ATR below entry"
    else:
        stop_part = "Below recent swing low or 1.5% below entry"
    
    # Adjust for RSI
    if rsi < 30:
        strategy_parts.append("RSI oversold — potential bounce zone")
        confidence += 5
    elif rsi > 70:
        strategy_parts.append("RSI overbought — avoid fresh longs")
        confidence -= 10
    
    # Support/resistance guidance
    if supports:
        nearest_support = supports[0] if supports else None
        if nearest_support:
            stop_part = f"Below {nearest_support:.0f} (nearest support)"
    
    if resistances:
        nearest_resist = resistances[0] if resistances else None
        if nearest_resist:
            target_part = f"3.0% or below {nearest_resist:.0f} (nearest resistance)"
    
    # Pattern-based suggestions
    bullish_patterns = [p for p in patterns if "Bullish" in p or "Hammer" in p or "Morning" in p]
    bearish_patterns = [p for p in patterns if "Bearish" in p or "Shooting" in p or "Evening" in p]
    
    if bullish_patterns:
        strategy_parts.append(f"Bullish pattern detected: {', '.join(bullish_patterns)}")
        confidence += 8
    if bearish_patterns:
        strategy_parts.append(f"BEARISH pattern: {', '.join(bearish_patterns)} — be cautious")
        confidence -= 10
    
    # Drawdown awareness
    if abs(avg_dd) > 5:
        strategy_parts.append(f"Avg drawdown is high ({avg_dd:.1f}%) — use strict risk management")
    
    # Confidence bounds
    confidence = max(10, min(95, confidence))
    
    # Build final strategy text
    if not strategy_parts:
        strategy = "Insufficient data for strategy recommendation"
    else:
        strategy = " | ".join(strategy_parts)
    
    if not entry_part:
        entry_part = "Wait for confirmed signal"
    if not stop_part:
        stop_part = "1.5% below entry or below 9-EMA"
    
    return strategy, entry_part, stop_part, target_part, confidence


def _empty_report(ticker: str) -> StockReport:
    """Return empty report for stocks with insufficient data."""
    return StockReport(
        ticker=ticker, data_points=0, years_of_data=0,
        overall_win_rate=0, avg_days_to_target=0, median_days_to_target=0,
        avg_return_5d=0, max_drawdown_avg=0,
        current_price=0, current_regime="N/A", current_rsi=0,
        current_trend_alignment=0, current_volatility_pct=0,
        top_patterns=[], best_entries=[],
        support_levels=[], resistance_levels=[],
        recommended_strategy="Insufficient data",
        recommended_entry="N/A", recommended_stop="N/A",
        recommended_target="N/A", confidence=0,
        risk_level="N/A", avg_daily_move_pct=0,
        max_intraday_move_pct=0, best_day_of_week="N/A",
        worst_day_of_week="N/A", recognized_patterns=[],
        pattern_follow_through={},
    )