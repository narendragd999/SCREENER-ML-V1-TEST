"""
screener.py — Opportunity Scanner with Primary Focus on Structural Low Support Price
=====================================================================================
Identifies high-conviction fundamental stocks:
1. PRIME FOCUS: Sitting ON or rebounding from Major Structural Low Support Prices
   (e.g. HCLTECH at ₹1,210.2 / ₹1,367.8, FINEORG at ₹4,020.4, SBIN at ₹940 / ₹1,038)
2. Retesting institutional fortress floors that previously launched +15% to +50%+ rallies
3. 9-EMA Momentum Breakouts from structural demand floors
4. Deep intrinsic Fair Value anchors under a strict No-Stop policy.
"""
import asyncio
import math
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from core.data_fetcher import (
    load_ticker_registry,
    fetch_ticker_ohlcv_async,
    bare_symbol,
)
from core.strategy import indicator_frame, confirmed_breakout_mask, compute_trend
from core.fair_value import compute_stock_fair_value
from core.no_stop_engine import simulate_no_stop_trade
from core.support_detector import (
    detect_structural_support,
    get_all_structural_low_floors,
    detect_horizontal_support,
    get_all_support_floors,
    detect_powerful_macro_floor,
)


def analyze_stock_opportunity(
    ticker: str,
    df: pd.DataFrame,
    target_pct: float = 3.0,
    max_candles_ago: int = 15,
    max_support_proximity_pct: float = 6.0,
) -> Optional[Dict[str, Any]]:
    """
    Analyzes a stock with PRIME FOCUS on STRUCTURAL LOW SUPPORT PRICE:
    - Major Structural Low Floor identification & distance %
    - Historical rally/impulse generated from this floor
    - Multi-touch horizontal support confluence
    - 9-EMA confirmation
    - Fair Value upside & No-Stop target path
    """
    if df is None or len(df) < 40:
        return None

    bars = indicator_frame(df)
    n = len(bars)
    row = bars.iloc[-1]
    current_price = float(row.Close)
    ema9 = float(row.ema9)
    atr = float(row.atr14) if pd.notna(row.atr14) else 0.0

    # Trend details
    trend_info = compute_trend(bars)
    regime = trend_info["trend_regime"]

    # =========================================================================
    # 0. POWERFUL MACRO STRUCTURAL LOW (ROCK-BOTTOM CYCLE FLOOR)
    # Highest Gain Probability & Lowest Structural Downside Risk
    # =========================================================================
    macro_floor = detect_powerful_macro_floor(bars, lookback_bars=750)
    macro_price = macro_floor["macro_floor_price"] if macro_floor else None
    macro_prox = macro_floor["prox_to_macro_floor_pct"] if macro_floor else None
    macro_touches = macro_floor["touches"] if macro_floor else 0
    macro_dates = macro_floor["touch_dates"] if macro_floor else []
    macro_rally = macro_floor["max_historical_rally_pct"] if macro_floor else 0.0
    range_percentile = macro_floor["range_percentile"] if macro_floor else 50.0
    cycle_low = macro_floor["cycle_low"] if macro_floor else current_price
    cycle_high = macro_floor["cycle_high"] if macro_floor else current_price
    upside_to_high = macro_floor["upside_to_cycle_high"] if macro_floor else 0.0
    gain_prob_rating = macro_floor["gain_probability_rating"] if macro_floor else "MODERATE"
    is_at_macro_floor = macro_floor["is_at_macro_floor"] if macro_floor else False
    is_rebounding_from_macro = macro_floor["is_rebounding_from_macro"] if macro_floor else False

    # =========================================================================
    # 1. PRIME FOCUS: IMMEDIATE STRUCTURAL LOW SUPPORT PRICE DETECTION
    # =========================================================================
    struct_data = detect_structural_support(bars, lookback_bars=750, max_proximity_pct=max_support_proximity_pct)
    has_structural_low = struct_data is not None
    struct_price = struct_data["structural_low_price"] if struct_data else None
    struct_prox = struct_data["structural_proximity_pct"] if struct_data else None
    struct_status = struct_data["structural_status"] if struct_data else "NONE"
    struct_rally = struct_data["max_historical_rally_pct"] if struct_data else 0.0
    struct_touches = struct_data["structural_touches"] if struct_data else 0
    struct_dates = struct_data["structural_dates"] if struct_data else []
    is_at_structural_low = struct_data["is_at_structural_low"] if struct_data else False

    # 2. General Multi-Touch Horizontal Support Floor
    support_data = detect_horizontal_support(bars, lookback_bars=600, min_touches=2, cluster_pct=2.5, max_proximity_pct=max_support_proximity_pct)
    is_on_support = support_data is not None
    support_level = support_data["support_level"] if support_data else None
    support_touches = support_data["touches"] if support_data else 0
    support_proximity_pct = support_data["proximity_pct"] if support_data else None
    support_dates = support_data["touch_dates"] if support_data else []
    support_status = support_data["status"] if support_data else "NO_SUPPORT"

    # 3. 9-EMA Breakout Confirmation
    confirm_indices = np.flatnonzero(confirmed_breakout_mask(bars).to_numpy())
    recent_confirms = [int(i) for i in confirm_indices if n - 1 - i <= max_candles_ago]

    # MUST qualify for at least one:
    # A) Is sitting ON or rebounding from a Major Structural Low Support Price (PRIME FOCUS)
    # B) Is sitting ON or rebounding from the Powerful Macro Rock-Bottom Cycle Floor
    # C) Is sitting on a verified multi-touch support floor
    # D) Has a recent confirmed 9-EMA breakout
    if not has_structural_low and not is_on_support and not recent_confirms and not is_at_macro_floor and not is_rebounding_from_macro:
        return None

    # Setup Classification
    if is_at_structural_low:
        # HIGHEST CONVICTION: EXACT STRUCTURAL LOW TEST (e.g. HCLTECH at 1210 / 1367, FINEORG at 4020, SBIN at 940)
        setup_type = "AT_STRUCTURAL_LOW"
        confirm_idx = n - 1
        candles_ago = 0
        entry_price = current_price
        target_price = entry_price * (1.0 + target_pct / 100.0)
        status = "AT_STRUCTURAL_LOW"
        hold_sessions = 0
        mae_pct = 0.0
        confirm_date = struct_dates[-1] if struct_dates else str(row.name.date())
        breakout_date = struct_dates[0] if struct_dates else str(row.name.date())
        vol_ratio = float(row.volume_ratio) if hasattr(row, "volume_ratio") and pd.notna(row.volume_ratio) else 1.0

    elif struct_status == "REBOUNDING_FROM_STRUCTURAL_LOW":
        setup_type = "REBOUNDING_STRUCTURAL_LOW"
        confirm_idx = n - 1
        candles_ago = 0
        entry_price = current_price
        target_price = entry_price * (1.0 + target_pct / 100.0)
        status = "REBOUNDING"
        hold_sessions = 0
        mae_pct = 0.0
        confirm_date = struct_dates[-1] if struct_dates else str(row.name.date())
        breakout_date = struct_dates[0] if struct_dates else str(row.name.date())
        vol_ratio = float(row.volume_ratio) if hasattr(row, "volume_ratio") and pd.notna(row.volume_ratio) else 1.0

    elif is_on_support and recent_confirms:
        setup_type = "EMA_BREAKOUT_AT_SUPPORT"
        confirm_idx = recent_confirms[-1]
        candles_ago = n - 1 - confirm_idx
        conf_row = bars.iloc[confirm_idx]
        entry_idx = confirm_idx + 1
        entered = entry_idx < n
        entry_price = float(bars.Open.iloc[entry_idx]) if entered else float(conf_row.Close)
        target_price = entry_price * (1.0 + target_pct / 100.0)

        sim = simulate_no_stop_trade(bars, entry_idx if entered else confirm_idx, entry_price, target_pct=target_pct)
        status = sim.get("status", "FRESH")
        hold_sessions = sim.get("hold_sessions", 0)
        mae_pct = sim.get("mae_pct", 0.0)
        if status == "OPEN":
            if candles_ago <= 1:
                status = "FRESH"
            elif current_price >= entry_price:
                status = "IN_PROFIT"
            else:
                status = "IN_DRAWDOWN"
        confirm_date = str(conf_row.name.date())
        breakout_date = str(bars.index[confirm_idx - 1].date())
        vol_ratio = float(conf_row.volume_ratio) if pd.notna(conf_row.volume_ratio) else 1.0

    elif is_on_support and not recent_confirms:
        setup_type = support_status
        confirm_idx = n - 1
        candles_ago = 0
        entry_price = current_price
        target_price = entry_price * (1.0 + target_pct / 100.0)
        status = "AT_SUPPORT" if support_status == "AT_SUPPORT_FLOOR" else "REBOUNDING"
        hold_sessions = 0
        mae_pct = 0.0
        confirm_date = support_data.get("last_touch_date", str(row.name.date()))
        breakout_date = support_data.get("first_touch_date", str(row.name.date()))
        vol_ratio = float(row.volume_ratio) if hasattr(row, "volume_ratio") and pd.notna(row.volume_ratio) else 1.0

    elif is_at_macro_floor and not recent_confirms:
        setup_type = "AT_MACRO_FLOOR"
        confirm_idx = n - 1
        candles_ago = 0
        entry_price = current_price
        target_price = entry_price * (1.0 + target_pct / 100.0)
        status = "AT_MACRO_FLOOR"
        hold_sessions = 0
        mae_pct = 0.0
        confirm_date = macro_dates[-1] if macro_dates else str(row.name.date())
        breakout_date = macro_dates[0] if macro_dates else str(row.name.date())
        vol_ratio = float(row.volume_ratio) if hasattr(row, "volume_ratio") and pd.notna(row.volume_ratio) else 1.0

    elif is_rebounding_from_macro and not recent_confirms:
        setup_type = "REBOUNDING_MACRO_FLOOR"
        confirm_idx = n - 1
        candles_ago = 0
        entry_price = current_price
        target_price = entry_price * (1.0 + target_pct / 100.0)
        status = "REBOUNDING"
        hold_sessions = 0
        mae_pct = 0.0
        confirm_date = macro_dates[-1] if macro_dates else str(row.name.date())
        breakout_date = macro_dates[0] if macro_dates else str(row.name.date())
        vol_ratio = float(row.volume_ratio) if hasattr(row, "volume_ratio") and pd.notna(row.volume_ratio) else 1.0

    else:
        setup_type = "EMA_BREAKOUT"
        confirm_idx = recent_confirms[-1]
        candles_ago = n - 1 - confirm_idx
        conf_row = bars.iloc[confirm_idx]
        entry_idx = confirm_idx + 1
        entered = entry_idx < n
        entry_price = float(bars.Open.iloc[entry_idx]) if entered else float(conf_row.Close)
        target_price = entry_price * (1.0 + target_pct / 100.0)

        sim = simulate_no_stop_trade(bars, entry_idx if entered else confirm_idx, entry_price, target_pct=target_pct)
        status = sim.get("status", "FRESH")
        hold_sessions = sim.get("hold_sessions", 0)
        mae_pct = sim.get("mae_pct", 0.0)
        if status == "OPEN":
            if candles_ago <= 1:
                status = "FRESH"
            elif current_price >= entry_price:
                status = "IN_PROFIT"
            else:
                status = "IN_DRAWDOWN"
        confirm_date = str(conf_row.name.date())
        breakout_date = str(bars.index[confirm_idx - 1].date())
        vol_ratio = float(conf_row.volume_ratio) if pd.notna(conf_row.volume_ratio) else 1.0

    turnover = float(row.turnover20) if pd.notna(row.turnover20) else 0.0
    extension_atr = ((current_price - ema9) / atr) if atr > 0 else 0.0
    distance_to_target_pct = ((target_price / current_price) - 1.0) * 100.0

    # Intrinsic Fair Value
    fv_data = compute_stock_fair_value(ticker, current_price)
    fair_gap = fv_data.get("fair_gap_pct")

    # Score calculation (0 to 100)
    score = 0
    reasons = []

    # Factor 0: Powerful Macro Rock-Bottom Floor (High Gain Probability & Multi-Year Cycle Anchor)
    if is_at_macro_floor:
        score += 25
        reasons.append(f"⚡ AT ROCK-BOTTOM CYCLE FLOOR ₹{macro_price} ({macro_touches} tests, prev surge: +{macro_rally}%, upside to peak: +{upside_to_high}%, Range %ile: {range_percentile}%)")
    elif is_rebounding_from_macro:
        score += 15
        reasons.append(f"⚡ Rebound off Rock-Bottom Floor ₹{macro_price} (+{macro_prox}%, upside to peak: +{upside_to_high}%)")
    elif range_percentile <= 25.0:
        score += 10
        reasons.append(f"Cycle Value Accumulation Zone (bottom {range_percentile}% of 3-year range, +{upside_to_high}% upside to peak)")

    # Factor 1: STRUCTURAL LOW SUPPORT (Up to 40 pts — Primary Core Focus)
    if is_at_structural_low:
        score += 35
        reasons.append(f"★ AT STRUCTURAL LOW SUPPORT ₹{struct_price} ({struct_prox:+0.2f}% away, historical impulse: +{struct_rally}%)")
        if struct_touches >= 2:
            score += 5
            reasons.append(f"Multi-touch structural base ({struct_touches} macro tests)")
        if struct_rally >= 20.0:
            score += 5
            reasons.append(f"High-conviction demand floor (historical rally +{struct_rally}%)")
    elif struct_status == "REBOUNDING_FROM_STRUCTURAL_LOW":
        score += 25
        reasons.append(f"Rebounding off Structural Low ₹{struct_price} (+{struct_prox}%, historical impulse: +{struct_rally}%)")
    elif is_on_support:
        score += 20
        reasons.append(f"At horizontal support floor ₹{support_level} ({support_touches} touches)")

    # Factor 2: Trend Regime & Long-Term Structure (Up to 20 pts)
    if regime == "UPTREND":
        score += 20
        reasons.append("Primary institutional uptrend (50 & 200 SMA)")
    elif regime == "SIDEWAYS":
        score += 15
        reasons.append("Multi-month structural accumulation base")

    # Factor 3: Fair Value Headroom (Up to 20 pts)
    if fair_gap is not None:
        if fair_gap >= 10.0:
            score += 20
            reasons.append(f"Deep fundamental value (+{fair_gap}% to Fair Value)")
        elif fair_gap >= 3.0:
            score += 15
            reasons.append(f"Positive valuation headroom (+{fair_gap}% to Fair Value)")
        elif fair_gap >= -5.0:
            score += 10
            reasons.append("Fairly valued intrinsic anchor")

    # Factor 4: Volume Surge / Momentum (Up to 10 pts)
    if vol_ratio >= 1.5:
        score += 10
        reasons.append(f"Institutional volume surge ({vol_ratio:.1f}x 20d avg)")
    elif vol_ratio >= 1.1:
        score += 6
        reasons.append("Above-average volume")

    # Factor 5: Controlled Extension (Up to 10 pts)
    if candles_ago <= 2 and extension_atr <= 1.2:
        score += 10
        reasons.append("Low extension, optimal risk-free entry window")

    # High Conviction Prime Setup:
    is_prime = (
        is_at_macro_floor
        or is_at_structural_low
        or (is_rebounding_from_macro and regime in ("UPTREND", "SIDEWAYS"))
        or (struct_status == "REBOUNDING_FROM_STRUCTURAL_LOW" and regime == "UPTREND")
        or (is_on_support and regime == "UPTREND")
        or (regime == "UPTREND" and candles_ago <= 3 and extension_atr <= 1.5)
    )

    sym = bare_symbol(ticker)
    new_entry_target = round(current_price * (1.0 + target_pct / 100.0), 2)
    return {
        "symbol": sym,
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "ema9": round(ema9, 2),
        "entry_price": round(entry_price, 2),
        "target_price": round(target_price, 2),
        "new_entry_target": new_entry_target,
        "target_pct": target_pct,
        "distance_to_target_pct": round(distance_to_target_pct, 2),
        "breakout_date": breakout_date,
        "confirm_date": confirm_date,
        "candles_ago": candles_ago,
        "status": status,
        "setup_type": setup_type,
        "hold_sessions": hold_sessions,
        "mae_pct": round(mae_pct, 2),
        "trend_regime": regime,
        # Powerful Macro Rock-Bottom Floor & Gain Probability (High Conviction Focus)
        "macro_floor_price": macro_price,
        "macro_touches": macro_touches,
        "macro_dates": macro_dates,
        "macro_rally_pct": macro_rally,
        "macro_proximity_pct": macro_prox,
        "range_percentile": range_percentile,
        "cycle_low": cycle_low,
        "cycle_high": cycle_high,
        "upside_to_cycle_high": upside_to_high,
        "gain_probability_rating": gain_prob_rating,
        "is_at_macro_floor": is_at_macro_floor,
        "is_rebounding_from_macro": is_rebounding_from_macro,
        # Structural Low Support Specifics (User Primary Focus)
        "has_structural_low": has_structural_low,
        "structural_low_price": struct_price,
        "structural_proximity_pct": struct_prox,
        "structural_status": struct_status,
        "structural_rally_pct": struct_rally,
        "structural_touches": struct_touches,
        "structural_dates": struct_dates,
        "is_at_structural_low": is_at_structural_low,
        # General Support
        "is_on_support": is_on_support or has_structural_low or is_at_macro_floor,
        "support_level": struct_price if has_structural_low else (macro_price if is_at_macro_floor else support_level),
        "support_touches": struct_touches if has_structural_low else (macro_touches if is_at_macro_floor else support_touches),
        "support_proximity_pct": struct_prox if has_structural_low else (macro_prox if is_at_macro_floor else support_proximity_pct),
        "support_dates": struct_dates if has_structural_low else (macro_dates if is_at_macro_floor else support_dates),
        "support_status": struct_status if has_structural_low else ("AT_MACRO_FLOOR" if is_at_macro_floor else support_status),
        # Fundamental & Indicators
        "volume_ratio": round(vol_ratio, 2),
        "turnover_cr": round(turnover / 10_000_000.0, 2),
        "atr_pct": round((atr / current_price) * 100.0, 2) if current_price > 0 else 0.0,
        "extension_atr": round(extension_atr, 2),
        "rsi14": round(float(row.rsi14), 1) if pd.notna(row.rsi14) else None,
        "composite_fair_price": fv_data.get("composite_fair_price"),
        "fair_gap_pct": fair_gap,
        "valuation_bucket": fv_data.get("valuation_bucket", "N/A"),
        "fundamental_score": fv_data.get("fundamental_score", 0),
        "fundamental_verdict": fv_data.get("fundamental_verdict", "N/A"),
        "score": min(score, 100),
        "reasons": reasons,
        "is_prime": is_prime,
        "as_of": str(bars.index[-1].date()),
    }


async def scan_opportunities(
    limit: int = 60,
    target_pct: float = 3.0,
    max_candles_ago: int = 15,
    uptrend_only: bool = False,
    support_only: bool = False,
    pending_only: bool = False,
    structural_only: bool = False,
    macro_floor_only: bool = False,
    min_score: int = 35,
) -> Dict[str, Any]:
    """Scan the 307 vetted healthy stocks registry for active high-conviction opportunities."""
    registry = load_ticker_registry()
    if not registry:
        return {"error": "Registry empty or tickers.csv not found", "opportunities": []}

    target_tickers = registry[:limit] if limit > 0 else registry
    sem = asyncio.Semaphore(10)
    opportunities = []

    async def _scan_single(rec):
        ticker = rec["ticker"]
        async with sem:
            df = await fetch_ticker_ohlcv_async(ticker, period="3y")
            opp = analyze_stock_opportunity(ticker, df, target_pct=target_pct, max_candles_ago=max_candles_ago)
            if opp:
                opp["security"] = rec["security"]
                # Filter gates
                if macro_floor_only and not (opp.get("is_at_macro_floor") or opp.get("is_rebounding_from_macro") or opp.get("range_percentile", 100) <= 30.0):
                    return
                if structural_only and not opp["is_at_structural_low"]:
                    return
                if support_only and not opp["is_on_support"]:
                    return
                if uptrend_only and opp["trend_regime"] != "UPTREND":
                    return
                if pending_only and opp["status"] == "TARGET_HIT":
                    return
                if opp["score"] < min_score:
                    return
                opportunities.append(opp)

    tasks = [_scan_single(r) for r in target_tickers]
    await asyncio.gather(*tasks, return_exceptions=True)

    # Sort opportunities: AT_MACRO_FLOOR first (Highest Conviction), then AT_STRUCTURAL_LOW, then REBOUNDING, then score descending
    def sort_key(x):
        is_macro_at = 4 if x.get("is_at_macro_floor") else 0
        is_struct_at = 3 if x.get("is_at_structural_low") else (2 if (x.get("is_rebounding_from_macro") or x.get("structural_status") == "REBOUNDING_FROM_STRUCTURAL_LOW") else (1 if x.get("is_on_support") else 0))
        is_prime = 1 if x.get("is_prime") else 0
        return (is_macro_at, is_struct_at, is_prime, x.get("score", 0))

    opportunities.sort(key=sort_key, reverse=True)

    macro_at_list = [o for o in opportunities if o.get("is_at_macro_floor")]
    macro_rebound_list = [o for o in opportunities if o.get("is_rebounding_from_macro")]
    high_gain_prob_list = [o for o in opportunities if str(o.get("gain_probability_rating", "")).startswith("HIGH") or str(o.get("gain_probability_rating", "")).startswith("VERY HIGH")]
    structural_at_list = [o for o in opportunities if o.get("is_at_structural_low")]
    structural_rebound_list = [o for o in opportunities if o.get("structural_status") == "REBOUNDING_FROM_STRUCTURAL_LOW"]
    support_list = [o for o in opportunities if o["is_on_support"]]
    prime_list = [o for o in opportunities if o["is_prime"]]
    active_pending = [o for o in opportunities if o["status"] in ("FRESH", "OPEN", "IN_PROFIT", "IN_DRAWDOWN", "AT_STRUCTURAL_LOW", "AT_SUPPORT", "AT_MACRO_FLOOR", "REBOUNDING")]
    completed_hits = [o for o in opportunities if o["status"] == "TARGET_HIT"]

    return {
        "scanned_count": len(target_tickers),
        "total_registry": len(registry),
        "opportunities_count": len(opportunities),
        "macro_at_count": len(macro_at_list),
        "macro_rebound_count": len(macro_rebound_list),
        "high_gain_prob_count": len(high_gain_prob_list),
        "structural_at_count": len(structural_at_list),
        "structural_rebound_count": len(structural_rebound_list),
        "support_count": len(support_list),
        "prime_count": len(prime_list),
        "active_pending_count": len(active_pending),
        "completed_hits_count": len(completed_hits),
        "opportunities": opportunities,
    }
