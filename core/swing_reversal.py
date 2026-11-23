"""Confirmed support bases, staged resistance targets, and swing reversal evidence.
Targets are chart levels, not promised returns; proximity alone is not a buy.
"""
import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from core.data_fetcher import clean_ohlcv, fetch_batch_ohlcv, fetch_ticker_ohlcv, load_ticker_registry, normalize_ticker, market_now
from core.fair_value import compute_fair_value

logger = logging.getLogger('swing_reversal')
Pivot = Tuple[int, str, float]


def _date_str(value: Any) -> str:
    return str(value.date()) if hasattr(value, 'date') else str(value)[:10]


def _find_pivot_lows(lows: np.ndarray, dates: Any, fractal: int = 2) -> List[Pivot]:
    """Require right-hand confirmation; represent a flat bottom by one pivot."""
    if fractal < 1:
        raise ValueError('fractal must be positive')
    lows = np.asarray(lows, dtype=float)
    pivots = []
    for i in range(fractal, len(lows) - fractal):
        window = lows[i - fractal:i + fractal + 1]
        if not np.isfinite(window).all() or window.min() <= 0:
            continue
        if lows[i] <= window[:fractal].min() and lows[i] < window[fractal + 1:].min():
            pivots.append((i, _date_str(dates[i]), float(lows[i])))
    return pivots


def _find_pivot_highs(highs: np.ndarray, dates: Any, fractal: int = 2) -> List[Pivot]:
    """Find confirmed highs, representing a flat top by its last bar."""
    if fractal < 1:
        raise ValueError('fractal must be positive')
    highs = np.asarray(highs, dtype=float)
    pivots = []
    for i in range(fractal, len(highs) - fractal):
        window = highs[i - fractal:i + fractal + 1]
        if not np.isfinite(window).all() or window.min() <= 0:
            continue
        if highs[i] >= window[:fractal].max() and highs[i] > window[fractal + 1:].max():
            pivots.append((i, _date_str(dates[i]), float(highs[i])))
    return pivots


def _independent_touches(pivots: List[Pivot], highs: np.ndarray, atr: float) -> List[Pivot]:
    """Separate retests by time and a material intervening rebound."""
    touches = []
    for pivot in sorted(pivots):
        if not touches:
            touches.append(pivot)
            continue
        previous = touches[-1]
        if pivot[0] - previous[0] < 7:
            continue
        rebound = float(np.max(highs[previous[0]:pivot[0] + 1])) - max(previous[2], pivot[2])
        if rebound >= max(1.5 * atr, pivot[2] * .025):
            touches.append(pivot)
    return touches


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError):
        return None


def analyze_stock_swing_reversal(ticker: str, df: Optional[pd.DataFrame] = None,
                                min_upside_pct: float = 15.0,
                                max_dist_to_floor_pct: float = 14.0,
                                lookback_days: int = 250,
                                include_fair_value: bool = True) -> Optional[Dict[str, Any]]:
    """Only return setups whose FIRST resistance meets the requested upside.
    Isolated major lows may be watch candidates. Actionability additionally
    needs independent support retests, recent reversal, liquidity and 2:1 R/R.
    """
    if (not math.isfinite(min_upside_pct) or not math.isfinite(max_dist_to_floor_pct)
            or min_upside_pct < 0 or max_dist_to_floor_pct < 0 or lookback_days < 80):
        raise ValueError('Use nonnegative distance/upside filters and at least 80 bars')
    sym = normalize_ticker(ticker)
    if not sym:
        return None
    if df is None:
        df = fetch_ticker_ohlcv(sym, interval='1d', lookback_days=max(365, lookback_days))
    if df is None or df.empty:
        return None
    df = clean_ohlcv(df, interval='1d', completed_only=True)
    if len(df) < 80:
        return None
    # Warm up indicators before limiting pattern history.
    ema = df['Close'].ewm(span=9, adjust=False).mean()
    sma50 = df['Close'].rolling(50).mean()
    prev = df['Close'].shift(1)
    tr = pd.concat([df['High'] - df['Low'], (df['High'] - prev).abs(), (df['Low'] - prev).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().iloc[-1])
    if not math.isfinite(atr) or atr <= 0:
        return None
    recent = df.tail(lookback_days)
    ema_values = ema.loc[recent.index].to_numpy(dtype=float)
    closes = recent['Close'].to_numpy(dtype=float)
    lows = recent['Low'].to_numpy(dtype=float)
    highs = recent['High'].to_numpy(dtype=float)
    close, n = float(closes[-1]), len(recent)
    pivot_lows = [p for p in _find_pivot_lows(lows, recent.index) if p[0] >= n - 180]
    pivot_highs = [p for p in _find_pivot_highs(highs, recent.index) if p[0] >= n - 150]
    if not pivot_lows or not pivot_highs:
        return None
    candidates, seen = [], set()
    for anchor in pivot_lows:
        tolerance = max(.015, min(.028, .75 * atr / anchor[2]))
        nearby = [p for p in pivot_lows if abs(p[2] - anchor[2]) / anchor[2] <= tolerance]
        touches = _independent_touches(nearby, highs, atr)
        identity = tuple(p[0] for p in touches)
        if not touches or identity in seen:
            continue
        seen.add(identity)
        floor = min(p[2] for p in touches)
        if (max(p[2] for p in touches) - floor) / floor > .028:
            continue
        distance = (close - floor) / floor * 100
        if distance < 0 or distance > max_dist_to_floor_pct or n - 1 - touches[-1][0] > 45:
            continue
        # A broken shelf cannot be recycled as intact support.
        invalidation_buffer = max(floor * .005, atr * .35)
        if np.any(closes[touches[0][0] + 1:] < floor - invalidation_buffer):
            continue
        prior_peak = float(np.max(highs[max(0, touches[0][0] - 60):touches[0][0] + 1]))
        correction = (prior_peak - floor) / prior_peak * 100
        if correction < (8.0 if len(touches) >= 2 else 12.0):
            continue
        overhead = sorted({p[2] for p in pivot_highs if p[2] > close + max(.5 * atr, close * .005)})
        if not overhead:
            continue
        first_target = overhead[0]
        if (first_target - close) / close * 100 + 1e-9 < min_upside_pct:
            continue
        candidates.append((touches, floor, distance, correction, overhead))
    if not candidates:
        return None
    touches, floor, distance, correction, overhead = min(candidates, key=lambda c: (-len(c[0]), c[2], n-1-c[0][-1][0]))
    target = overhead[0]
    target_gain = (target - close) / close * 100
    stop = floor - max(floor * .01, atr * .75)
    if stop <= 0 or stop >= close:
        return None
    risk = (close - stop) / close * 100
    reward_risk = target_gain / risk
    above_ema = close > ema_values[-1]
    ema_rising = ema_values[-1] > ema_values[-3]
    confirmation = None
    for i in range(n - 2, max(0, n - 7), -1):
        if (closes[i-1] <= ema_values[i-1] and closes[i] > ema_values[i]
                and closes[i+1] > highs[i] and closes[i+1] > ema_values[i+1]
                and np.all(closes[i+1:] > ema_values[i+1:])):
            confirmation = i + 1
            break
    reversal_confirmed = confirmation is not None and n-1-confirmation <= 3 and ema_rising
    fresh_crossover = closes[-2] <= ema_values[-2] and above_ema
    falling = bool(not above_ema and closes[-1] < closes[-2] < closes[-3] and not ema_rising)
    volume = recent['Volume'] if 'Volume' in recent else pd.Series(dtype=float)
    prior_volume = volume.iloc[-21:-1]
    mean_volume = float(prior_volume.mean()) if len(prior_volume) == 20 else 0.0
    turnover = float((recent['Close'] * volume).iloc[-21:-1].median()) if len(volume) else 0.0
    volume_ok = bool(df.attrs.get('volume_available', True) and len(prior_volume) == 20
                     and (prior_volume > 0).sum() >= 15 and math.isfinite(turnover)
                     and turnover >= 10_000_000 and mean_volume > 0 and float(volume.iloc[-1]) > 0)
    volume_ratio = float(volume.iloc[-1] / mean_volume) if mean_volume > 0 else None
    confirmation_ratio = None
    if confirmation is not None and confirmation >= 20:
        confirmation_average = float(volume.iloc[confirmation-20:confirmation].mean())
        if confirmation_average > 0:
            confirmation_ratio = float(volume.iloc[confirmation] / confirmation_average)
    stale_days = max(0, (market_now().date() - recent.index[-1].date()).days)
    warnings = []
    if len(touches) < 2:
        warnings.append('Only one independently confirmed low; support needs another retest')
    if not reversal_confirmed:
        warnings.append('No recent two-candle EMA reversal confirmation; wait for confirmation')
    if falling:
        warnings.append('Price and short-term momentum are still falling')
    if reward_risk < 2:
        warnings.append('Reward to the first resistance is less than twice structural risk')
    if not volume_ok:
        warnings.append('Low or unavailable traded-value evidence; liquidity is insufficient for actionability')
    if reversal_confirmed and (confirmation_ratio is None or confirmation_ratio < 1):
        warnings.append('Confirmation volume does not reach its prior 20-session average')
    if stale_days > 5:
        warnings.append('Daily price history is stale')
    if df.attrs.get('invalid_bars_removed', 0):
        warnings.append('Invalid or duplicate price bars were removed')
    actionable_blockers = list(warnings)
    is_actionable = not actionable_blockers
    if reversal_confirmed:
        status, label, badge = 'BREAKOUT_CONFIRMED', 'Confirmed EMA reversal', 'badge-primary'
    elif distance <= 3.5 and not falling:
        status, label, badge = 'ACCUMULATION_ZONE', 'Support base watch', 'badge-success'
    elif distance <= 7 or falling:
        status, label, badge = 'APPROACHING_SUPPORT', 'Await support stabilization', 'badge-warning'
    else:
        status, label, badge = 'SWING_IN_PROGRESS', 'Recovery watch', 'badge-purple'
    if is_actionable:
        action = f'Confirmed setup; first resistance ₹{target:.2f}, structural stop ₹{stop:.2f}. Recheck the entry price.'
    else:
        action = f"Watch only. {warnings[0] if warnings else 'Await stronger reversal evidence'}."
    try:
        fv_data = compute_fair_value(sym, current_price=close) if include_fair_value else {}
    except Exception as exc:
        logger.warning('Fair value unavailable for %s: %s', sym, exc)
        fv_data = {}
    fair_price = _number(fv_data.get('composite_fair_price'))
    fair_confidence = fv_data.get('valuation_confidence', 'unavailable')
    fv_gain = round((fair_price - close) / close * 100, 2) if fair_price else None
    fv_confluence = bool(fair_price and fair_price > close and fair_confidence in {'high', 'medium'})
    if fair_price and not fv_confluence:
        warnings.append('Fair value is not sufficiently reliable to count as confluence')
    neckline = float(np.max(highs[touches[0][0]:touches[-1][0]+1])) if len(touches) >= 2 else None
    double_bottom = bool(len(touches) == 2 and touches[1][0] - touches[0][0] >= 10 and neckline is not None and neckline >= floor * 1.05)
    if double_bottom:
        pattern_type = 'Double Bottom (W-Pattern)' if close > neckline else 'Double Bottom Candidate'
    elif len(touches) >= 3:
        pattern_type = f'Support Shelf ({len(touches)} Independent Touches)'
    elif len(touches) == 2:
        pattern_type = 'Multi-Touch Base (2 Touches)'
    else:
        pattern_type = 'Isolated Major Swing Low'
    score = min(100, 15 * min(len(touches), 3) + 20 * int(reversal_confirmed) + 15 * int(reward_risk >= 2) + 10 * int(volume_ok) + 10 * int(fv_confluence))
    if stale_days > 5 or df.attrs.get('invalid_bars_removed', 0):
        score = min(score, 35)
    return {
        'ticker': sym, 'current_price': round(close, 2), 'as_of': _date_str(recent.index[-1]),
        'completed_candles_only': True, 'history_bars': len(df), 'data_age_days': stale_days,
        'floor_price': round(floor, 2), 'floor_zone_upper': round(max(p[2] for p in touches), 2),
        'floor_age_bars': n-1-touches[-1][0], 'dist_to_floor_pct': round(distance, 2),
        'correction_from_prior_peak_pct': round(correction, 2),
        'target_price': round(target, 2), 'target_source': 'nearest_confirmed_resistance',
        'target_gain_pct': round(target_gain, 2), 'secondary_targets': [round(p, 2) for p in overhead[1:4]],
        'pattern_potential_pct': round((target-floor)/floor*100, 2),
        'stop_loss_price': round(stop, 2), 'stop_basis': 'support floor minus max(1% of floor, 0.75 ATR14)',
        'atr14': round(atr, 4), 'risk_pct': round(risk, 2), 'risk_reward': f'1:{reward_risk:.1f}',
        'risk_reward_ratio': round(reward_risk, 2), 'pattern_type': pattern_type,
        'is_double_bottom': double_bottom, 'double_bottom_confirmed': bool(double_bottom and close > neckline),
        'neckline_price': round(neckline, 2) if neckline else None,
        'shelf_touches': len(touches), 'touch_dates': [p[1] for p in touches],
        'touch_prices': [round(p[2], 2) for p in touches],
        'current_ema9': round(float(ema_values[-1]), 2), 'current_sma50': round(float(sma50.iloc[-1]), 2),
        'is_above_ema9': bool(above_ema), 'fresh_ema9_crossover': bool(fresh_crossover),
        'reversal_confirmed': bool(reversal_confirmed),
        'confirmation_date': _date_str(recent.index[confirmation]) if confirmation is not None else None,
        'median_turnover_20d': round(turnover, 2) if math.isfinite(turnover) else None,
        'volume_ratio': round(volume_ratio, 2) if volume_ratio is not None else None,
        'confirmation_volume_ratio': round(confirmation_ratio, 2) if confirmation_ratio is not None else None,
        'composite_fair_price': fair_price, 'fv_gain_pct': fv_gain,
        'fv_bucket': fv_data.get('valuation_bucket', 'N/A'), 'fv_confluence': fv_confluence,
        'valuation_confidence': fair_confidence, 'is_actionable': is_actionable,
        'setup_score': score, 'score_is_probability': False,
        'evidence': [f'{len(touches)} independent confirmed support touches',
                     f'First resistance offers {target_gain:.2f}% chart upside', f'Structural reward/risk {reward_risk:.2f}:1'],
        'warnings': warnings, 'actionable_blockers': actionable_blockers, 'alert_status': status, 'alert_label': label,
        'alert_badge': badge, 'action_recommendation': action,
    }


def scan_swing_reversals(tickers: Optional[List[str]] = None, min_upside_pct: float = 15.0,
                        max_dist_to_floor_pct: float = 14.0, source: str = 'fno', progress=None) -> List[Dict[str, Any]]:
    """Scan the selected registry in a batch; rank evidence before upside."""
    if tickers is None:
        registry = load_ticker_registry(source)
        tickers = registry['symbol'].tolist() if not registry.empty else []
    tickers = list(dict.fromkeys(normalize_ticker(t) for t in tickers if str(t).strip()))
    if not tickers:
        return []
    def download_progress(info):
        if progress:
            progress({**info, 'percent': 40 * info['completed'] / max(1, info['total']),
                      'message': 'Fetching completed daily bars'})
    kwargs = {'progress': download_progress} if progress else {}
    frames = fetch_batch_ohlcv(tickers, interval='1d', lookback_days=400, **kwargs)
    results = []
    failed_count = 0
    for completed, ticker in enumerate(tickers, 1):
        if progress:
            progress({'phase': 'analysis', 'completed': completed - 1, 'total': len(tickers),
                      'ticker': ticker, 'percent': 40 + 59 * (completed - 1) / len(tickers),
                      'failed_count': failed_count, 'message': 'Checking support retests, reversal and resistance'})
        df = frames.get(ticker)
        if df is None or df.empty:
            failed_count += 1
            continue
        try:
            result = analyze_stock_swing_reversal(ticker, df=df, min_upside_pct=min_upside_pct,
                                                 max_dist_to_floor_pct=max_dist_to_floor_pct)
            if result:
                results.append(result)
        except Exception as exc:
            failed_count += 1
            logger.warning('Error scanning swing reversal for %s: %s', ticker, exc)
    if progress:
        progress({'phase': 'analysis', 'completed': len(tickers), 'total': len(tickers),
                  'ticker': None, 'percent': 99, 'failed_count': failed_count, 'message': 'Preparing support setups'})
    results.sort(key=lambda r: (r['is_actionable'], r['setup_score'], r['risk_reward_ratio']), reverse=True)
    return results
