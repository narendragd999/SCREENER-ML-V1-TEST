"""
TensorFlow ML Stock Intelligence Service — INTEGRATED EDITION
=============================================================
This is the complete, integrated ml_service.py that combines:

  1. Original functionality (port 8001):
     • Chartink/EMA9 signal ingestion
     • ML predictions (TF models + intelligent_scoring fallback)
     • Per-stock deep analysis
     • Paper trading with order lifecycle
     • Multi-stock scanner

  2. Continuous Learning Pipeline (NEW):
     • SQLite DB replacing all JSON files
     • Outcome tracking — every prediction is verified 5 days later
     • Walk-forward validation (no data leakage)
     • Versioned model registry with promotion gate + rollback
     • Incremental fine-tune every 6 hours
     • Weekly full retrain (Sunday 2 AM IST)
     • Daily outcome resolution (4 PM IST)
     • Hourly drift detection (concept + feature + data staleness)
     • 17 new monitoring endpoints under /api/ml-monitor/*
     • Delta data downloads (30x faster than full re-download)
     • Feature caching per (ticker, date)

Run:
    python ml_service.py

Endpoints (in addition to original):
    /api/ml-monitor/dashboard
    /api/ml-monitor/accuracy?days=7
    /api/ml-monitor/accuracy/trend?days=30
    /api/ml-monitor/predictions/recent
    /api/ml-monitor/predictions/pending
    /api/ml-monitor/models
    /api/ml-monitor/models/production
    /api/ml-monitor/models/{version}/promote
    /api/ml-monitor/models/rollback/{model_type}
    /api/ml-monitor/training/history
    /api/ml-monitor/training/trigger-fine-tune
    /api/ml-monitor/training/trigger-full-retrain
    /api/ml-monitor/alerts
    /api/ml-monitor/drift/check
    /api/ml-monitor/outcomes/resolve
    /api/ml-monitor/outcomes/stats
    /api/ml-monitor/data/staleness
    /api/bootstrap-training-data  (call once after install)
"""

import os
import sys
import json
import time
import math
import logging
import threading
import asyncio
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from apscheduler.schedulers.background import BackgroundScheduler
import pytz

# ── Ensure service modules are importable ──
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

# ── Optional ML model imports (graceful fallback to intelligent scoring) ──
HAS_ML_MODELS = False
try:
    from ml_features import extract_features, FEATURE_NAMES, FEATURE_STATS
    from ml_models import model_manager
    HAS_ML_MODELS = True
except ImportError as e:
    import logging as _logging
    _log = _logging.getLogger("ml_service")
    _log.warning(f"[Startup] ML modules not found ({e}), using Intelligent Scoring fallback.")
    extract_features = None
    FEATURE_NAMES: List[str] = []
    FEATURE_STATS: Dict = {}

    class _StubModelManager:
        """Fallback that delegates to intelligent scoring when TF models are absent."""
        def __init__(self):
            self.models = {"signal_quality": None, "trend_regime": None,
                           "return_forecaster": None, "risk_estimator": None}
            self.is_training = False
            self.total_training_samples = 0

        def predict(self, df):
            try:
                from intelligent_scoring import compute_intelligent_score
                result = compute_intelligent_score(df)
            except Exception:
                result = {"signal_quality": 0.5, "trend_regime": "SIDEWAYS",
                          "expected_return": 0, "max_drawdown": -5,
                          "risk_reward_ratio": 1.0, "ml_score": 50,
                          "components": {}, "scoring_method": "intelligent_v2_fallback"}
            return result

        def predict_batch(self, ticker_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
            try:
                from intelligent_scoring import compute_intelligent_score
            except Exception:
                return {}
            results = {}
            for ticker, df in ticker_dfs.items():
                try:
                    results[ticker] = compute_intelligent_score(df)
                except Exception:
                    pass
            return results

        def get_status(self):
            return {"models_trained": 0, "total_samples": 0,
                    "scoring_method": "intelligent_v2 (no TF models)"}

        def train_initial(self, *args, **kwargs):
            pass

        def fine_tune(self, *args, **kwargs):
            pass

        def _save_models(self):
            pass

        def _build_missing_models(self):
            pass

    model_manager = _StubModelManager()

from ml_stock_analyzer import analyze_stock, scan_stocks_for_3pct

# ═══════════════════════════════════════════════════════════════════════════
#  CONTINUOUS LEARNING PIPELINE (NEW)
# ═══════════════════════════════════════════════════════════════════════════
HAS_CONTINUOUS_LEARNING = False
try:
    from db import db, now_ist, IST_OFFSET_HOURS, Database
    from outcome_tracker import OutcomeTracker, get_tracker
    from continuous_learner import ContinuousLearner
    from drift_detector import DriftDetector
    from delta_downloader import get_downloader, DeltaDownloader
    from feature_store import FeatureStore, get_feature_store
    from model_registry import get_registry, ModelRegistry
    from monitoring_api import create_monitoring_router
    HAS_CONTINUOUS_LEARNING = True
except ImportError as e:
    import logging as _logging
    _log = _logging.getLogger("ml_service")
    _log.warning(f"[Startup] Continuous learning modules not found ({e}) — "
                 "running in legacy mode. Install the ml_continuous_learning package "
                 "to enable outcome tracking, drift detection, and continuous training.")

# Singletons for continuous learning (lazily initialized)
_continuous_learner: Optional[ContinuousLearner] = None
_outcome_tracker: Optional[OutcomeTracker] = None
_drift_detector: Optional[DriftDetector] = None
_delta_downloader: Optional[DeltaDownloader] = None
_feature_store: Optional[FeatureStore] = None
_model_registry: Optional[ModelRegistry] = None

# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

IST = pytz.timezone("Asia/Kolkata")

CONFIG_FILE = BASE_DIR / "ml_config.json"
DEFAULT_CONFIG = {
    "fine_tune_interval_min": 30,
    "data_download_interval_min": 60,
    "target_tickers": [],
    "main_app_url": "http://localhost:8000",
    "target_pct": 3.0,
    "lookahead_days": 5,
    "min_data_bars": 200,
    "max_training_tickers": 0,
    "auto_train_on_startup": True,
    "chartink_auto_ingest": True,
    "chartink_poll_interval_min": 5,
    # NEW: continuous learning settings
    "continuous_learning_enabled": True,
    "cl_fine_tune_interval_hours": 6,
    "cl_full_retrain_day_of_week": 6,    # Sunday
    "cl_full_retrain_hour": 2,           # 2 AM IST
    "cl_outcome_resolve_hour": 16,       # 4 PM IST
    "cl_drift_check_interval_min": 60,
    "cl_bootstrap_max_tickers": 50,
}

def load_config() -> dict:
    cfg = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg

def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

config = load_config()

# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ml_service")

if HAS_ML_MODELS:
    logger.info("[Startup] TensorFlow ML models loaded successfully.")
if HAS_CONTINUOUS_LEARNING:
    logger.info("[Startup] Continuous learning pipeline loaded successfully.")

# ═══════════════════════════════════════════════════════════════════════════
#  IN-MEMORY STORE (legacy — being phased out by SQLite)
# ═══════════════════════════════════════════════════════════════════════════
_prediction_cache: Dict[str, Dict] = {}
_chartink_signals: List[Dict] = []
_chartink_lock = threading.Lock()
_ml_decisions: List[Dict] = []
_training_log: List[Dict] = []
_decisions_lock = threading.Lock()

DECISIONS_FILE = DATA_DIR / "ml_decisions.json"
SIGNALS_FILE = DATA_DIR / "chartink_signals.json"


def _save_decisions():
    with _decisions_lock:
        with open(DECISIONS_FILE, "w") as f:
            json.dump(_ml_decisions[-500:], f, indent=2, default=str)


def _save_signals():
    with _chartink_lock:
        with open(SIGNALS_FILE, "w") as f:
            json.dump(_chartink_signals[-500:], f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
#  DATA DOWNLOAD — uses DeltaDownloader if available (30x faster)
# ═══════════════════════════════════════════════════════════════════════════
_yf_cache: Dict[str, Dict] = {}
_yf_lock = threading.Lock()
YF_CACHE_TTL = 4 * 3600


def download_ticker_data(ticker: str, period: str = "10y") -> Optional[pd.DataFrame]:
    """Download OHLCV for a single ticker.

    Uses DeltaDownloader (SQLite-backed) if continuous learning is available,
    falling back to the legacy yfinance-with-memory-cache approach otherwise.
    """
    # NEW: prefer delta downloader
    if HAS_CONTINUOUS_LEARNING and _delta_downloader is not None:
        try:
            df = _delta_downloader.update_ticker(ticker)
            if df is not None and len(df) >= 50:
                return df
        except Exception as e:
            logger.debug(f"[Delta] {ticker} failed, falling back to legacy: {e}")

    # LEGACY fallback
    cache_key = f"{ticker}:{period}"
    with _yf_lock:
        entry = _yf_cache.get(cache_key)
        if entry and (time.time() - entry["ts"]) < YF_CACHE_TTL:
            return entry["df"].copy()

    for suffix in [".NS", ".BO"]:
        try:
            t = yf.Ticker(f"{ticker}{suffix}")
            hist = t.history(period=period, auto_adjust=True)
            if hist is not None and not hist.empty and len(hist) >= 100:
                df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
                df = df.dropna(how="all")
                with _yf_lock:
                    _yf_cache[cache_key] = {"df": df, "ts": time.time()}
                return df.copy()
        except Exception:
            continue
    return None


def download_tickers_batch(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """Download data for multiple tickers. Uses concurrent delta downloader if available."""
    if HAS_CONTINUOUS_LEARNING and _delta_downloader is not None:
        try:
            return _delta_downloader.update_tickers_batch(tickers, max_workers=10)
        except Exception as e:
            logger.warning(f"[Delta] batch failed, falling back: {e}")

    # LEGACY fallback (sequential)
    result = {}
    for i, ticker in enumerate(tickers):
        df = download_ticker_data(ticker)
        if df is not None:
            result[ticker] = df
        if i < len(tickers) - 1:
            time.sleep(0.15)
    logger.info(f"[Data] Downloaded {len(result)}/{len(tickers)} tickers")
    return result


def get_fno_tickers() -> List[str]:
    """Load F&O ticker list."""
    csv_path = os.path.join(os.path.dirname(BASE_DIR), "data", "tickers-healthy-strict-4.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(BASE_DIR), "tickers-healthy-strict-4.csv")

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, dtype=str)
            col = "SYMBOL" if "SYMBOL" in df.columns else "symbol"
            if col in df.columns:
                tickers = [s.strip().upper() for s in df[col].dropna() if s.strip()]
                return tickers[:200]
        except Exception:
            pass

    return [
        "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "HINDUNILVR",
        "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK", "LT", "AXISBANK",
        "ASIANPAINT", "BAJFINANCE", "MARUTI", "TITAN", "SUNPHARMA",
        "TATAMOTORS", "WIPRO", "HCLTECH", "ULTRACEMCO", "NESTLEIND",
        "NTPC", "POWERGRID", "ADANIENT", "ADANIPORTS", "TECHM",
        "ONGC", "COALINDIA", "JSWSTEEL", "TATASTEEL", "HINDALCO",
    ]


# ═══════════════════════════════════════════════════════════════════════════
#  CHARTINK INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════
async def _fetch_chartink_from_main() -> List[Dict]:
    import httpx
    url = f"{config['main_app_url']}/api/chartink-screener/today-primes"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("primes", data.get("results", {}).get("prime_targets", []))
    except Exception as e:
        logger.warning(f"[Chartink] Failed to fetch from main app: {e}")
    return []


async def _fetch_ema9_signals_from_main() -> List[Dict]:
    import httpx
    url = f"{config['main_app_url']}/api/ema9/quick-scan"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("prime_targets", [])
    except Exception as e:
        logger.warning(f"[EMA9] Failed to fetch from main app: {e}")
    return []


# ═══════════════════════════════════════════════════════════════════════════
#  PREDICTION LOGGING (NEW — for outcome tracking)
# ═══════════════════════════════════════════════════════════════════════════
def _get_current_model_version() -> str:
    """Get the version string of the current production model."""
    if HAS_CONTINUOUS_LEARNING and _model_registry is not None:
        try:
            v = _model_registry.get_production_version("signal_quality")
            if v:
                return v
        except Exception:
            pass
    return "intelligent_v2" if not HAS_ML_MODELS else "legacy"


def _log_prediction_to_db(ticker: str,
                           prediction: Dict,
                           source: str = "manual") -> Optional[int]:
    """Log a prediction to the database for outcome tracking.

    Returns the prediction ID, or None if logging failed.
    """
    if not HAS_CONTINUOUS_LEARNING:
        return None
    try:
        model_version = _get_current_model_version()
        prediction_id = db.log_prediction(
            ticker=ticker,
            prediction_ts=now_ist(),
            model_version=model_version,
            source=source,
            signal_quality=prediction.get("signal_quality"),
            trend_regime=prediction.get("trend_regime"),
            regime_confidence=prediction.get("regime_confidence"),
            expected_return=prediction.get("expected_return"),
            max_drawdown=prediction.get("max_drawdown"),
            risk_reward_ratio=prediction.get("risk_reward_ratio"),
            ml_score=prediction.get("ml_score"),
            actionable=prediction.get("actionable"),
            entry_price=prediction.get("entry_price") or prediction.get("current_price"),
            target_price=prediction.get("target_price"),
            stop_loss=prediction.get("stop_loss"),
            payload=prediction,
        )
        prediction["prediction_id"] = prediction_id
        prediction["model_version"] = model_version
        return prediction_id
    except Exception as e:
        logger.warning(f"Failed to log prediction for {ticker}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  INTELLIGENT DECISION ENGINE
# ═══════════════════════════════════════════════════════════════════════════
def _generate_decisions(signals: List[Dict], predictions: Dict[str, Dict]) -> List[Dict]:
    """Combine Chartink/EMA9 signals with ML predictions to generate trading decisions."""
    decisions = []

    for signal in signals:
        ticker = signal.get("ticker", "")
        if not ticker:
            continue

        ml_pred = predictions.get(ticker, {})
        ml_score = ml_pred.get("ml_score", 0)
        signal_quality = ml_pred.get("signal_quality", 0.5)
        regime = ml_pred.get("trend_regime", "SIDEWAYS")
        expected_return = ml_pred.get("expected_return", 0)
        max_drawdown = ml_pred.get("max_drawdown", 0)
        risk_reward = ml_pred.get("risk_reward_ratio", 0)

        entry_price = signal.get("current_price") or signal.get("entry_price") or signal.get("confirm_close")
        target_price = signal.get("target_3pct") or signal.get("target_price")
        near_support = signal.get("near_support", False)
        support_level = signal.get("near_support_level")
        support_touches = signal.get("near_support_touches", 0)
        fv_gap = signal.get("gap_to_fair_pct", signal.get("fair_gap_pct"))
        is_undervalued = fv_gap is not None and fv_gap > 0
        is_pullback = signal.get("is_pullback", False)
        pullback_dist = signal.get("pullback_dist_pct", 0)

        reasons = []
        confidence = ml_score / 100.0
        action = "HOLD"

        if signal_quality > 0.6 and regime == "UPTREND":
            action = "BUY"
            confidence = min(0.95, confidence * 1.2)
            reasons.append(f"High signal quality ({signal_quality:.0%}) in {regime}")

        if near_support and support_touches >= 3:
            confidence = min(0.95, confidence * 1.1)
            reasons.append(f"Near {support_touches}-touch support at {support_level}")

        if is_undervalued:
            confidence = min(0.95, confidence * 1.1)
            reasons.append(f"Undervalued by {fv_gap:.1f}%")

        if is_pullback and pullback_dist >= 10:
            confidence = min(0.95, confidence * 1.15)
            reasons.append(f"Pullback entry ({pullback_dist:.1f}% from prior high)")

        if regime == "DOWNTREND":
            if action == "BUY":
                action = "HOLD"
            reasons.append(f"WARNING: DOWNTREND regime — signal likely to fail")

        if signal_quality < 0.4:
            action = "SKIP"
            reasons.append(f"Low signal quality ({signal_quality:.0%}) — skip to preserve capital")

        if risk_reward < 0.8:
            if action == "BUY":
                action = "HOLD"
            reasons.append(f"Poor risk:reward ({risk_reward:.1f}x)")

        if max_drawdown < -5:
            action = "SKIP"
            reasons.append(f"High expected drawdown ({max_drawdown:.1f}%) — risk too high")

        if action == "BUY" and confidence > 0.75 and len(reasons) >= 3:
            action = "STRONG_BUY"

        suggested_target_pct = config.get("target_pct", 3.0)
        if expected_return > 0 and expected_return < suggested_target_pct:
            suggested_target_pct = max(2.0, round(expected_return * 0.8, 1))
            reasons.append(f"ML suggests adjusting target to {suggested_target_pct}% (expected return: {expected_return:.1f}%)")

        if risk_reward > 0 and confidence > 0:
            position_score = confidence * min(risk_reward, 3.0) / 3.0
            if position_score > 0.6:
                suggested_size = "LARGE (3-5% of capital)"
            elif position_score > 0.35:
                suggested_size = "MEDIUM (2-3% of capital)"
            else:
                suggested_size = "SMALL (1% of capital)"
        else:
            suggested_size = "SKIP"

        decision = {
            "ticker": ticker,
            "action": action,
            "confidence": round(confidence * 100, 1),
            "ml_score": ml_score,
            "signal_quality": signal_quality,
            "trend_regime": regime,
            "expected_return_pct": expected_return,
            "max_drawdown_pct": max_drawdown,
            "risk_reward_ratio": risk_reward,
            "entry_price": entry_price,
            "target_price": target_price,
            "suggested_target_pct": suggested_target_pct,
            "suggested_position_size": suggested_size,
            "reasons": reasons,
            "near_support": near_support,
            "support_level": support_level,
            "is_undervalued": is_undervalued,
            "is_pullback": is_pullback,
            "timestamp": dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
        }

        # NEW: log decision's prediction to DB for outcome tracking
        if HAS_CONTINUOUS_LEARNING:
            _log_prediction_to_db(ticker, {
                **ml_pred,
                "current_price": entry_price,
                "target_price": target_price,
                "actionable": action,
            }, source="decision_engine")

        decisions.append(decision)

    decisions.sort(key=lambda d: d["confidence"], reverse=True)
    return decisions


# ═══════════════════════════════════════════════════════════════════════════
#  BACKGROUND SCHEDULER TASKS
# ═══════════════════════════════════════════════════════════════════════════
def _is_market_open() -> bool:
    now = dt.datetime.now(IST)
    if now.weekday() >= 5:
        return False
    o = now.replace(hour=9, minute=15, second=0, microsecond=0)
    c = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return o <= now <= c


def _check_and_execute_orders():
    """Background job: monitor PENDING and EXECUTED orders against live prices."""
    if not _is_market_open():
        return

    with _orders_lock:
        orders = _load_orders()

    active = [o for o in orders if o["status"] in ("PENDING", "EXECUTED")]
    if not active:
        return

    tickers_needed = list(set(o["ticker"] for o in active))
    live_prices: Dict[str, float] = {}
    for t in tickers_needed:
        try:
            tk = yf.Ticker(f"{t}.NS")
            hist = tk.history(period="1d", interval="1m", auto_adjust=True)
            if not hist.empty:
                live_prices[t] = round(float(hist["Close"].iloc[-1]), 2)
        except Exception:
            pass

    if not live_prices:
        return

    now_str = dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M")
    changed = False

    for o in orders:
        ticker = o["ticker"]
        if ticker not in live_prices:
            continue

        price = live_prices[ticker]
        entry = o["entry_price"]
        target = o.get("target_price")
        stop_loss = o.get("stop_loss")
        action = o["action"]

        if o["status"] == "PENDING":
            filled = False
            if action == "BUY" and price <= entry:
                filled = True
            elif action == "SELL" and price >= entry:
                filled = True
            elif action == "BUY" and price >= entry:
                filled = True
            elif action == "SELL" and price <= entry:
                filled = True

            if filled:
                o["status"] = "EXECUTED"
                o["executed_at"] = now_str
                o["fill_price"] = price
                logger.info(f"[OrderMonitor] PENDING→EXECUTED {action} {ticker} @ Rs.{price}")
                changed = True

        elif o["status"] == "EXECUTED":
            close_reason = None
            exit_price = price

            if target is not None:
                if action == "BUY" and price >= target:
                    close_reason = "TARGET HIT"
                elif action == "SELL" and price <= target:
                    close_reason = "TARGET HIT"

            if close_reason is None and stop_loss is not None:
                if action == "BUY" and price <= stop_loss:
                    close_reason = "STOP LOSS HIT"
                elif action == "SELL" and price >= stop_loss:
                    close_reason = "STOP LOSS HIT"

            if close_reason:
                o["status"] = "CLOSED"
                o["exit_price"] = exit_price
                o["exit_date"] = now_str
                if action == "BUY":
                    o["pnl"] = round((exit_price - entry) * o["quantity"], 2)
                    o["pnl_pct"] = round((exit_price - entry) / entry * 100, 2)
                else:
                    o["pnl"] = round((entry - exit_price) * o["quantity"], 2)
                    o["pnl_pct"] = round((entry - exit_price) / entry * 100, 2)
                o["close_reason"] = close_reason
                logger.info(f"[OrderMonitor] EXECUTED→CLOSED {action} {ticker} @ Rs.{exit_price} ({close_reason}) P&L={o['pnl']}")
                changed = True

    if changed:
        with _orders_lock:
            _save_orders(orders)


def _background_fine_tune_legacy():
    """Legacy fine-tune task (every 30 min). Kept for backward compatibility.
    The continuous_learner handles the proper fine-tune every 6 hours.
    """
    if model_manager.is_training:
        return

    if HAS_CONTINUOUS_LEARNING and _continuous_learner is not None:
        # Skip legacy fine-tune — continuous learner handles it
        return

    tickers = get_fno_tickers()[:config.get("max_training_tickers", 100)]
    if not tickers:
        return

    ticker_dfs = download_tickers_batch(tickers)
    if ticker_dfs:
        start = time.time()
        model_manager.fine_tune(ticker_dfs, target_pct=config["target_pct"], epochs=3)
        elapsed = time.time() - start
        _training_log.append({
            "timestamp": dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
            "type": "fine_tune",
            "tickers_used": len(ticker_dfs),
            "elapsed_seconds": round(elapsed, 1),
            "total_samples": model_manager.total_training_samples,
        })
        logger.info(f"[Scheduler] Legacy fine-tune complete: {len(ticker_dfs)} tickers, {elapsed:.1f}s")


async def _background_chartink_ingest():
    """Fetch Chartink signals from main app and run ML predictions."""
    if not config.get("chartink_auto_ingest", True):
        return

    chartink_primes = await _fetch_chartink_from_main()
    ema9_signals = await _fetch_ema9_signals_from_main()

    all_signals = {}
    for s in chartink_primes:
        t = s.get("ticker", "")
        if t:
            all_signals[t] = s
    for s in ema9_signals:
        t = s.get("ticker", "")
        if t:
            all_signals[t] = s

    if not all_signals:
        return

    tickers = list(all_signals.keys())
    ticker_dfs = download_tickers_batch(tickers)
    predictions = model_manager.predict_batch(ticker_dfs)

    for ticker, pred in predictions.items():
        _prediction_cache[ticker] = pred
        # NEW: log each prediction to DB
        if HAS_CONTINUOUS_LEARNING:
            _log_prediction_to_db(ticker, pred, source="chartink_ingest")

    signal_list = list(all_signals.values())
    decisions = _generate_decisions(signal_list, predictions)

    with _decisions_lock:
        _ml_decisions.extend(decisions)
        if len(_ml_decisions) > 500:
            _ml_decisions[:] = _ml_decisions[-500:]

    with _chartink_lock:
        _chartink_signals[:] = signal_list[-200:]

    _save_decisions()
    _save_signals()

    logger.info(f"[Ingest] Processed {len(signal_list)} signals, generated {len(decisions)} decisions")


# ═══════════════════════════════════════════════════════════════════════════
#  CONTINUOUS LEARNING INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════
def _init_continuous_learning():
    """Initialize all continuous learning singletons. Called once at startup."""
    global _continuous_learner, _outcome_tracker, _drift_detector
    global _delta_downloader, _feature_store, _model_registry

    if not HAS_CONTINUOUS_LEARNING:
        logger.warning("[CL] Continuous learning modules not available — skipping init")
        return False

    try:
        # Initialize delta downloader
        _delta_downloader = get_downloader()
        logger.info("[CL] DeltaDownloader initialized")

        # Initialize model registry
        _model_registry = get_registry()
        logger.info("[CL] ModelRegistry initialized")

        # Initialize feature store (requires feature extractor)
        if HAS_ML_MODELS and extract_features is not None:
            _feature_store = get_feature_store(feature_extractor=extract_features)
            logger.info("[CL] FeatureStore initialized")
        else:
            logger.info("[CL] FeatureStore skipped (no TF feature extractor)")

        # Initialize outcome tracker
        _outcome_tracker = OutcomeTracker(
            lookahead_days=config.get("lookahead_days", 5),
            target_pct=config.get("target_pct", 3.0),
        )
        logger.info("[CL] OutcomeTracker initialized")

        # Initialize continuous learner (wraps existing model_manager)
        if HAS_ML_MODELS and extract_features is not None:
            _continuous_learner = ContinuousLearner(
                model_manager=model_manager,
                feature_extractor=extract_features,
                target_pct=config.get("target_pct", 3.0),
                lookahead_days=config.get("lookahead_days", 5),
            )
            # Register as singleton in continuous_learner module
            import continuous_learner as cl_module
            cl_module._learner = _continuous_learner
            logger.info("[CL] ContinuousLearner initialized")
        else:
            logger.info("[CL] ContinuousLearner skipped (no TF models)")

        # Initialize drift detector
        _drift_detector = DriftDetector(
            feature_store=_feature_store,
            on_drift_callback=_on_drift_alert,
            on_critical_callback=_on_critical_drift_alert,
        )
        import drift_detector as dd_module
        dd_module._detector = _drift_detector
        logger.info("[CL] DriftDetector initialized")

        return True
    except Exception as e:
        logger.error(f"[CL] Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _on_drift_alert(alert: Dict):
    """Callback for non-critical drift alerts."""
    logger.warning(f"[DRIFT ALERT] {alert['alert_type']} ({alert['severity']}): "
                   f"{alert.get('details', {}).get('message', '')}")


def _on_critical_drift_alert(alert: Dict):
    """Callback for critical drift alerts — triggers immediate retrain."""
    logger.critical(f"[CRITICAL DRIFT] {alert['alert_type']}: "
                    f"{alert.get('details', {}).get('message', '')}")
    if _continuous_learner is not None:
        import threading
        t = threading.Thread(
            target=lambda: _continuous_learner.trigger_full_retrain(
                reason=f"drift_alert_{alert['alert_type']}"
            ),
            daemon=True,
        )
        t.start()


# ═══════════════════════════════════════════════════════════════════════════
#  FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="TensorFlow ML Stock Intelligence — Integrated Edition",
    description="ML-powered stock analysis with continuous learning pipeline",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

scheduler = BackgroundScheduler()


# ─────────────────────────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 70)
    logger.info("  ML Stock Intelligence Service starting (Integrated Edition v2.0)")
    logger.info(f"  Port: 8001 | Main app: {config['main_app_url']}")
    logger.info(f"  TF Models: {'YES' if HAS_ML_MODELS else 'NO (using intelligent_scoring)'}")
    logger.info(f"  Continuous Learning: {'YES' if HAS_CONTINUOUS_LEARNING else 'NO'}")
    logger.info("=" * 70)

    # ── Initialize continuous learning pipeline ──
    cl_ok = False
    if config.get("continuous_learning_enabled", True) and HAS_CONTINUOUS_LEARNING:
        cl_ok = _init_continuous_learning()
    else:
        logger.info("[Startup] Continuous learning disabled in config")

    # ── Existing scheduler jobs ──
    scheduler.add_job(
        _background_fine_tune_legacy,
        trigger="interval",
        minutes=config.get("fine_tune_interval_min", 30),
        id="fine_tune_legacy",
        replace_existing=True,
    )
    scheduler.add_job(
        lambda: asyncio.run(_background_chartink_ingest()),
        trigger="interval",
        minutes=config.get("chartink_poll_interval_min", 5),
        id="chartink_ingest",
        replace_existing=True,
    )
    scheduler.add_job(
        _check_and_execute_orders,
        trigger="interval",
        minutes=1,
        id="order_monitor",
        replace_existing=True,
    )

    # ── NEW: Continuous learning scheduler jobs ──
    if cl_ok:
        try:
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.triggers.interval import IntervalTrigger

            # 1. Fine-tune every 6 hours
            if _continuous_learner is not None:
                _continuous_learner.start_scheduler(scheduler)
                logger.info(f"[Scheduler] Continuous fine-tune: every {config.get('cl_fine_tune_interval_hours', 6)}h")

            # 2. Outcome resolution — daily at 4 PM IST
            if _outcome_tracker is not None:
                fe = extract_features if HAS_ML_MODELS else None
                scheduler.add_job(
                    lambda: _outcome_tracker.resolve_all_pending(feature_extractor=fe),
                    trigger=CronTrigger(
                        hour=config.get("cl_outcome_resolve_hour", 16),
                        minute=0,
                        timezone="Asia/Kolkata",
                    ),
                    id="outcome_resolution",
                    replace_existing=True,
                )
                logger.info(f"[Scheduler] Outcome resolution: daily at "
                            f"{config.get('cl_outcome_resolve_hour', 16)}:00 IST")

            # 3. Drift detection — hourly
            if _drift_detector is not None:
                _drift_detector.start_scheduler(
                    scheduler,
                    interval_minutes=config.get("cl_drift_check_interval_min", 60),
                )
                logger.info(f"[Scheduler] Drift detection: every "
                            f"{config.get('cl_drift_check_interval_min', 60)} min")

            # 4. Mount monitoring API
            app.include_router(create_monitoring_router())
            logger.info("[Startup] Monitoring API mounted at /api/ml-monitor/*")

        except Exception as e:
            logger.error(f"[Startup] CL scheduler setup failed: {e}")
            import traceback
            traceback.print_exc()

    scheduler.start()
    logger.info(f"[Scheduler] Started {len(scheduler.get_jobs())} jobs")

    # Auto-train on startup if no models exist (legacy behavior)
    if HAS_ML_MODELS and config.get("auto_train_on_startup", True):
        any_trained = any(m is not None for m in model_manager.models.values())
        if not any_trained:
            logger.info("[Startup] No trained models found — triggering initial training...")
            tickers = get_fno_tickers()[:config.get("max_training_tickers", 100)]
            ticker_dfs = download_tickers_batch(tickers)
            if ticker_dfs:
                model_manager.train_initial(ticker_dfs, target_pct=config["target_pct"])
    elif not HAS_ML_MODELS:
        logger.info("[Startup] Running in Intelligent Scoring mode — skipping ML model training.")


@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown(wait=False)
    if HAS_ML_MODELS:
        model_manager._save_models()
    logger.info("[Shutdown] Scheduler stopped, models saved.")


# ═══════════════════════════════════════════════════════════════════════════
#  API ENDPOINTS — REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════
class PredictRequest(BaseModel):
    tickers: List[str]
    enrich_with_strategy: bool = True


class IngestRequest(BaseModel):
    signals: List[Dict]
    source: str = "chartink"


class TrainRequest(BaseModel):
    tickers: Optional[List[str]] = None
    epochs: int = 10
    target_pct: float = 3.0


# ─────────────────────────────────────────────────────────────────
#  PREDICTION ENDPOINTS (NOW LOGS EVERY PREDICTION TO DB)
# ─────────────────────────────────────────────────────────────────
@app.get("/api/predict/{ticker}")
async def predict_ticker(ticker: str, source: str = "manual"):
    """Get ML predictions for a single ticker.
    NEW: Every prediction is logged to the database for outcome tracking.
    """
    ticker = ticker.strip().upper()
    df = download_ticker_data(ticker)
    if df is None or len(df) < 100:
        raise HTTPException(404, f"No sufficient data for {ticker}")

    prediction = model_manager.predict(df)
    prediction["ticker"] = ticker

    # NEW: Log prediction for outcome tracking
    if HAS_CONTINUOUS_LEARNING:
        _log_prediction_to_db(ticker, prediction, source=source)

    return prediction


@app.post("/api/predict/batch")
async def predict_batch(req: PredictRequest):
    """Get ML predictions for multiple tickers.
    NEW: Every prediction is logged to the database.
    """
    tickers = [t.strip().upper() for t in req.tickers if t.strip()][:200]
    if not tickers:
        raise HTTPException(400, "No tickers provided")

    ticker_dfs = download_tickers_batch(tickers)
    predictions = model_manager.predict_batch(ticker_dfs)

    for ticker, pred in predictions.items():
        _prediction_cache[ticker] = pred
        # NEW: log each prediction
        if HAS_CONTINUOUS_LEARNING:
            _log_prediction_to_db(ticker, pred, source="batch")

    return {
        "predictions": predictions,
        "count": len(predictions),
        "model_status": model_manager.get_status()["models_trained"],
    }


# ─────────────────────────────────────────────────────────────────
#  INTELLIGENT DECISIONS
# ─────────────────────────────────────────────────────────────────
@app.get("/api/decisions")
async def get_decisions(limit: int = 20):
    """Get the latest ML-generated intelligent trading decisions."""
    with _decisions_lock:
        decisions = _ml_decisions[-limit:]
    return {
        "decisions": decisions,
        "total_stored": len(_ml_decisions),
        "showing": len(decisions),
    }


@app.get("/api/decisions/summary")
async def get_decisions_summary():
    """Summary of current decisions: counts by action, top picks, etc."""
    with _decisions_lock:
        recent = _ml_decisions[-100:] if _ml_decisions else []

    if not recent:
        return {"message": "No decisions yet. Waiting for Chartink/EMA9 signals..."}

    action_counts = {}
    for d in recent:
        action = d["action"]
        action_counts[action] = action_counts.get(action, 0) + 1

    strong_buys = [d for d in recent if d["action"] == "STRONG_BUY"]
    buys = [d for d in recent if d["action"] == "BUY"]

    return {
        "total_decisions": len(recent),
        "action_distribution": action_counts,
        "strong_buys": strong_buys[:5],
        "top_buys": (strong_buys + buys)[:10],
        "avg_ml_score": round(np.mean([d["ml_score"] for d in recent]), 1) if recent else 0,
        "avg_signal_quality": round(np.mean([d["signal_quality"] for d in recent]), 3) if recent else 0,
    }


# ─────────────────────────────────────────────────────────────────
#  CHARTINK INGESTION
# ─────────────────────────────────────────────────────────────────
@app.post("/api/ingest/chartink")
async def ingest_chartink_signals(req: IngestRequest):
    """Manually push Chartink/EMA9 signals into the ML service."""
    signals = req.signals
    if not signals:
        raise HTTPException(400, "No signals provided")

    tickers = list({s.get("ticker", "") for s in signals if s.get("ticker")})
    if not tickers:
        raise HTTPException(400, "No valid tickers in signals")

    ticker_dfs = download_tickers_batch(tickers)
    predictions = model_manager.predict_batch(ticker_dfs)

    for ticker, pred in predictions.items():
        _prediction_cache[ticker] = pred
        # NEW: log each prediction
        if HAS_CONTINUOUS_LEARNING:
            _log_prediction_to_db(ticker, pred, source=req.source)

    decisions = _generate_decisions(signals, predictions)

    with _decisions_lock:
        _ml_decisions.extend(decisions)
        if len(_ml_decisions) > 500:
            _ml_decisions[:] = _ml_decisions[-500:]

    with _chartink_lock:
        _chartink_signals.extend(signals)
        if len(_chartink_signals) > 200:
            _chartink_signals[:] = _chartink_signals[-200:]

    _save_decisions()
    _save_signals()

    return {
        "ingested": len(signals),
        "predictions_generated": len(predictions),
        "decisions_generated": len(decisions),
        "source": req.source,
    }


@app.get("/api/ingest/chartink/auto-poll")
async def trigger_chartink_poll():
    """Manually trigger the Chartink ingestion cycle."""
    await _background_chartink_ingest()
    return {"status": "ok", "message": "Chartink poll triggered"}


# ─────────────────────────────────────────────────────────────────
#  TRAINING ENDPOINTS
# ─────────────────────────────────────────────────────────────────
@app.post("/api/train/trigger")
async def trigger_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """Trigger model training (legacy — uses model_manager directly)."""
    if model_manager.is_training:
        return {"status": "already_training", "message": "Training is in progress..."}

    def _do_train():
        tickers = req.tickers or get_fno_tickers()[:config.get("max_training_tickers", 100)]
        ticker_dfs = download_tickers_batch(tickers)
        if ticker_dfs:
            model_manager.train_initial(ticker_dfs, target_pct=req.target_pct)
            model_manager.fine_tune(ticker_dfs, target_pct=req.target_pct, epochs=req.epochs)

    background_tasks.add_task(_do_train)
    return {
        "status": "training_started",
        "tickers": req.tickers or f"auto ({config.get('max_training_tickers', 100)} F&O stocks)",
        "target_pct": req.target_pct,
    }


@app.get("/api/train/status")
async def training_status():
    """Get model training status and statistics."""
    return model_manager.get_status()


@app.get("/api/train/history")
async def training_history(limit: int = 20):
    """Get recent training log entries."""
    return {
        "history": _training_log[-limit:],
        "total": len(_training_log),
    }


# ─────────────────────────────────────────────────────────────────
#  FEATURE INSIGHTS
# ─────────────────────────────────────────────────────────────────
@app.get("/api/features/{ticker}")
async def get_features(ticker: str):
    """Get the raw feature vector for a ticker."""
    ticker = ticker.strip().upper()
    df = download_ticker_data(ticker)
    if df is None or len(df) < 100:
        raise HTTPException(404, f"No data for {ticker}")

    if not HAS_ML_MODELS or extract_features is None:
        raise HTTPException(503, "Feature extraction requires ml_features module")

    features = extract_features(df, normalize=False)
    feature_dict = {name: round(float(val), 4) for name, val in zip(FEATURE_NAMES, features)}
    return {
        "ticker": ticker,
        "features": feature_dict,
        "num_features": len(feature_dict),
        "data_bars": len(df),
    }


# ─────────────────────────────────────────────────────────────────
#  MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────
@app.get("/api/config")
async def get_ml_config():
    """Get current ML service configuration."""
    return {
        "config": config,
        "feature_stats_available": len(FEATURE_STATS) if HAS_ML_MODELS else 0,
        "continuous_learning_enabled": HAS_CONTINUOUS_LEARNING and config.get("continuous_learning_enabled", True),
    }


@app.put("/api/config")
async def update_ml_config(new_config: dict):
    """Update ML service configuration."""
    global config
    config.update(new_config)
    save_config(config)
    return {"status": "updated", "config": config}


# ─────────────────────────────────────────────────────────────────
#  HEALTH / STATUS
# ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "tensorflow-ml-stock-intelligence",
        "version": "2.0.0",
        "port": 8001,
        "scoring_mode": "tensorflow_ml" if HAS_ML_MODELS else "intelligent_scoring_fallback",
        "continuous_learning": HAS_CONTINUOUS_LEARNING,
        "models_loaded": {k: v is not None for k, v in model_manager.models.items()},
        "prediction_cache_size": len(_prediction_cache),
        "decisions_stored": len(_ml_decisions),
        "chartink_signals": len(_chartink_signals),
        "scheduler_running": scheduler.running,
    }


# ─────────────────────────────────────────────────────────────────
#  PER-STOCK DEEP ANALYSIS
# ─────────────────────────────────────────────────────────────────
@app.get("/api/stock-analysis/{ticker}")
async def stock_deep_analysis(ticker: str):
    """Full per-stock deep analysis with TensorFlow ML."""
    ticker = ticker.strip().upper()
    df = download_ticker_data(ticker)
    if df is None or len(df) < 100:
        raise HTTPException(404, f"No sufficient data for {ticker} — need at least 100 bars")

    ml_prediction = model_manager.predict(df)
    ml_prediction["ticker"] = ticker

    # NEW: log this prediction
    if HAS_CONTINUOUS_LEARNING:
        _log_prediction_to_db(ticker, ml_prediction, source="stock_analysis")

    report = analyze_stock(ticker, df, ml_prediction=ml_prediction)
    return report


@app.post("/api/stock-analysis/batch")
async def stock_analysis_batch(background_tasks: BackgroundTasks):
    """Scan ALL F&O stocks with ML + pattern recognition for 3% target."""
    tickers = get_fno_tickers()
    if not tickers:
        raise HTTPException(500, "No tickers available")

    ticker_dfs = download_tickers_batch(tickers)
    if not ticker_dfs:
        raise HTTPException(500, "Failed to download data for any ticker")

    ml_predictions = model_manager.predict_batch(ticker_dfs)

    # NEW: log all predictions
    if HAS_CONTINUOUS_LEARNING:
        for ticker, pred in ml_predictions.items():
            _log_prediction_to_db(ticker, pred, source="batch_scanner")

    results = scan_stocks_for_3pct(ticker_dfs, ml_predictions)

    return {
        "total_scanned": len(tickers),
        "data_available": len(ticker_dfs),
        "results": results,
        "strong_buys": len([r for r in results if r["strategy_action"] == "STRONG_BUY"]),
        "buys": len([r for r in results if r["strategy_action"] == "BUY"]),
        "watch": len([r for r in results if r["strategy_action"] == "WATCH"]),
        "wait": len([r for r in results if r["strategy_action"] == "WAIT"]),
        "scan_time": dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/api/stock-analysis/quick/{ticker}")
async def stock_quick_analysis(ticker: str):
    """Quick analysis — just ML prediction + basic strategy."""
    ticker = ticker.strip().upper()

    if ticker in _prediction_cache:
        ml_pred = _prediction_cache[ticker]
    else:
        df = download_ticker_data(ticker)
        if df is None or len(df) < 100:
            raise HTTPException(404, f"No data for {ticker}")
        ml_pred = model_manager.predict(df)
        _prediction_cache[ticker] = ml_pred

    ml_pred["ticker"] = ticker

    # NEW: log prediction
    if HAS_CONTINUOUS_LEARNING:
        _log_prediction_to_db(ticker, ml_pred, source="quick_analysis")

    return ml_pred


@app.get("/api/available-tickers")
async def available_tickers():
    """Get list of all available F&O tickers."""
    tickers = get_fno_tickers()
    return {"tickers": tickers, "count": len(tickers)}


# ═══════════════════════════════════════════════════════════════════════════
#  PAPER TRADING — Order Management (JSON file persistence — legacy)
# ═══════════════════════════════════════════════════════════════════════════
ORDERS_FILE = DATA_DIR / "paper_trades.json"
_orders_lock = threading.Lock()


def _load_orders() -> List[Dict]:
    if ORDERS_FILE.exists():
        try:
            with open(ORDERS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _save_orders(orders: List[Dict]):
    with open(ORDERS_FILE, "w") as f:
        json.dump(orders, f, indent=2, default=str)


class OrderRequest(BaseModel):
    ticker: str
    action: str
    quantity: int
    price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    notes: Optional[str] = ""
    order_type: Optional[str] = "LIMIT"


class OrderUpdateRequest(BaseModel):
    status: Optional[str] = None
    exit_price: Optional[float] = None
    exit_date: Optional[str] = None
    pnl: Optional[float] = None
    notes: Optional[str] = None


@app.post("/api/orders")
async def add_order(req: OrderRequest):
    """Add a new paper trade order."""
    ticker = req.ticker.strip().upper()
    if req.action.upper() not in ("BUY", "SELL"):
        raise HTTPException(400, "Action must be BUY or SELL")

    order_type = (req.order_type or "LIMIT").upper()
    if order_type not in ("LIMIT", "MARKET"):
        order_type = "LIMIT"

    if order_type == "MARKET":
        initial_status = "EXECUTED"
        executed_at = dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M")
    else:
        initial_status = "PENDING"
        executed_at = None

    order = {
        "id": f"{ticker}-{int(time.time() * 1000)}",
        "ticker": ticker,
        "action": req.action.upper(),
        "quantity": req.quantity,
        "entry_price": req.price,
        "target_price": req.target_price,
        "stop_loss": req.stop_loss,
        "notes": req.notes or "",
        "status": initial_status,
        "order_type": order_type,
        "entry_date": dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M"),
        "executed_at": executed_at,
        "exit_date": None,
        "exit_price": None,
        "pnl": None,
        "pnl_pct": None,
        "strategy_action": None,
        "confidence": None,
    }

    with _orders_lock:
        orders = _load_orders()
        orders.append(order)
        _save_orders(orders)

    return {"status": "ok", "order": order}


@app.get("/api/orders")
async def list_orders(ticker: Optional[str] = None, status: Optional[str] = None):
    """List all paper trade orders, optionally filtered."""
    with _orders_lock:
        orders = _load_orders()

    if ticker:
        ticker = ticker.strip().upper()
        orders = [o for o in orders if o["ticker"] == ticker]
    if status:
        orders = [o for o in orders if o.get("status") == status.upper()]

    orders.sort(key=lambda o: o.get("entry_date", ""), reverse=True)

    open_orders = [o for o in orders if o["status"] in ("PENDING", "EXECUTED")]
    closed_orders = [o for o in orders if o["status"] == "CLOSED"]
    total_pnl = sum(o.get("pnl") or 0 for o in closed_orders)
    winning_trades = sum(1 for o in closed_orders if (o.get("pnl") or 0) > 0)

    tickers_seen = set(o["ticker"] for o in orders)
    ticker_summary = {}
    for t in tickers_seen:
        t_orders = [o for o in orders if o["ticker"] == t]
        t_closed = [o for o in t_orders if o["status"] == "CLOSED"]
        t_wins = sum(1 for o in t_closed if (o.get("pnl") or 0) > 0)
        ticker_summary[t] = {
            "total_orders": len(t_orders),
            "open": sum(1 for o in t_orders if o["status"] in ("PENDING", "EXECUTED")),
            "pending": sum(1 for o in t_orders if o["status"] == "PENDING"),
            "executed": sum(1 for o in t_orders if o["status"] == "EXECUTED"),
            "closed_pnl": round(sum(o.get("pnl") or 0 for o in t_closed), 2),
            "win_rate": round(t_wins / len(t_closed) * 100, 1) if t_closed else 0,
        }

    return {
        "orders": orders,
        "total": len(orders),
        "pending_count": sum(1 for o in orders if o["status"] == "PENDING"),
        "executed_count": sum(1 for o in orders if o["status"] == "EXECUTED"),
        "open_count": len(open_orders),
        "closed_count": len(closed_orders),
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(winning_trades / len(closed_orders) * 100, 1) if closed_orders else 0,
        "ticker_summary": ticker_summary,
    }


@app.put("/api/orders/{order_id}")
async def update_order(order_id: str, req: OrderUpdateRequest):
    """Update an order (close, cancel, add exit details)."""
    with _orders_lock:
        orders = _load_orders()
        found = False
        for o in orders:
            if o["id"] == order_id:
                found = True
                if req.status is not None:
                    o["status"] = req.status.upper()
                if req.exit_price is not None:
                    o["exit_price"] = req.exit_price
                if req.exit_date is not None:
                    o["exit_date"] = req.exit_date
                elif req.status and req.status.upper() in ("CLOSED", "CANCELLED"):
                    o["exit_date"] = dt.datetime.now(IST).strftime("%Y-%m-%d %H:%M")
                if req.notes is not None:
                    o["notes"] = req.notes
                if o["status"] == "CLOSED" and o.get("exit_price") and o.get("entry_price"):
                    if o["action"] == "BUY":
                        o["pnl"] = round((o["exit_price"] - o["entry_price"]) * o["quantity"], 2)
                        o["pnl_pct"] = round((o["exit_price"] - o["entry_price"]) / o["entry_price"] * 100, 2)
                    else:
                        o["pnl"] = round((o["entry_price"] - o["exit_price"]) * o["quantity"], 2)
                        o["pnl_pct"] = round((o["entry_price"] - o["exit_price"]) / o["entry_price"] * 100, 2)
                break
        if not found:
            raise HTTPException(404, f"Order {order_id} not found")
        _save_orders(orders)
        return {"status": "ok", "order": o}


@app.get("/api/orders/live-prices")
async def get_orders_live_prices():
    """Fetch current live prices for all active orders."""
    with _orders_lock:
        orders = _load_orders()
    active = [o for o in orders if o["status"] in ("PENDING", "EXECUTED")]
    if not active:
        return {"prices": {}}

    tickers_needed = list(set(o["ticker"] for o in active))
    prices = {}
    for t in tickers_needed:
        try:
            tk = yf.Ticker(f"{t}.NS")
            hist = tk.history(period="1d", interval="1m", auto_adjust=True)
            if not hist.empty:
                prices[t] = round(float(hist["Close"].iloc[-1]), 2)
        except Exception:
            pass
    return {"prices": prices, "timestamp": dt.datetime.now(IST).strftime("%H:%M:%S")}


@app.delete("/api/orders/{order_id}")
async def delete_order(order_id: str):
    """Delete a paper trade order."""
    with _orders_lock:
        orders = _load_orders()
        new_orders = [o for o in orders if o["id"] != order_id]
        if len(new_orders) == len(orders):
            raise HTTPException(404, f"Order {order_id} not found")
        _save_orders(new_orders)
    return {"status": "ok", "deleted": order_id}


@app.get("/api/orders/summary")
async def orders_summary():
    """Get paper trading portfolio summary."""
    with _orders_lock:
        orders = _load_orders()

    active_orders = [o for o in orders if o["status"] in ("PENDING", "EXECUTED")]
    closed_orders = [o for o in orders if o["status"] == "CLOSED"]
    cancelled = [o for o in orders if o["status"] == "CANCELLED"]

    total_invested = sum(o["entry_price"] * o["quantity"] for o in active_orders
                         if o["action"] == "BUY" and o["status"] == "EXECUTED")
    total_pnl = sum(o.get("pnl") or 0 for o in closed_orders)
    wins = sum(1 for o in closed_orders if (o.get("pnl") or 0) > 0)
    losses = len(closed_orders) - wins

    open_positions = {}
    for o in active_orders:
        if o["status"] != "EXECUTED":
            continue
        t = o["ticker"]
        if t not in open_positions:
            open_positions[t] = {"qty": 0, "invested": 0, "orders": []}
        if o["action"] == "BUY":
            open_positions[t]["qty"] += o["quantity"]
            open_positions[t]["invested"] += o["entry_price"] * o["quantity"]
        else:
            open_positions[t]["qty"] -= o["quantity"]
            open_positions[t]["invested"] -= o["entry_price"] * o["quantity"]
        open_positions[t]["orders"].append(o["id"])

    return {
        "total_orders": len(orders),
        "pending_orders": sum(1 for o in orders if o["status"] == "PENDING"),
        "executed_orders": sum(1 for o in orders if o["status"] == "EXECUTED"),
        "open_orders": len(active_orders),
        "closed_orders": len(closed_orders),
        "cancelled_orders": len(cancelled),
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(wins / len(closed_orders) * 100, 1) if closed_orders else 0,
        "wins": wins,
        "losses": losses,
        "total_invested_open": round(total_invested, 2),
        "open_positions": open_positions,
        "unique_stocks": len(set(o["ticker"] for o in orders)),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  BOOTSTRAP TRAINING DATA (NEW — call once after install)
# ═══════════════════════════════════════════════════════════════════════════
@app.post("/api/bootstrap-training-data")
async def bootstrap_training_data(background_tasks: BackgroundTasks,
                                   max_tickers: Optional[int] = None):
    """Bootstrap the training queue with historical labels.

    Call this ONCE after installing the continuous learning pipeline.
    It walks through historical OHLCV data and creates labeled training
    samples for every bar where we have 5 days of future outcome data.

    This populates the training_queue so the first fine-tune cycle has
    data to work with.
    """
    if not HAS_CONTINUOUS_LEARNING:
        raise HTTPException(503, "Continuous learning modules not available")
    if not HAS_ML_MODELS or extract_features is None:
        raise HTTPException(503, "ML feature extractor not available — install ml_features.py")

    n_tickers = max_tickers or config.get("cl_bootstrap_max_tickers", 50)

    def _do_bootstrap():
        try:
            logger.info(f"[Bootstrap] Starting with {n_tickers} tickers...")
            tracker = OutcomeTracker(
                lookahead_days=config.get("lookahead_days", 5),
                target_pct=config.get("target_pct", 3.0),
            )

            tickers = get_fno_tickers()[:n_tickers]
            ticker_dfs = download_tickers_batch(tickers)

            total = tracker.generate_historical_labels(
                ticker_dfs=ticker_dfs,
                feature_extractor=extract_features,
                step=3,
            )

            logger.info(f"[Bootstrap] Done — added {total} historical training samples")
        except Exception as e:
            logger.error(f"[Bootstrap] Failed: {e}")
            import traceback
            traceback.print_exc()

    background_tasks.add_task(_do_bootstrap)
    return {
        "status": "bootstrap_started",
        "tickers": n_tickers,
        "message": "Running in background. Check /api/ml-monitor/training/history for progress.",
    }


# ═══════════════════════════════════════════════════════════════════════════
#  DASHBOARD (serve HTML)
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the ML Intelligence Dashboard."""
    dashboard_path = BASE_DIR / "ml_dashboard.html"
    if dashboard_path.exists():
        return dashboard_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>ML Service Running on Port 8001</h1><p>Dashboard file not found.</p>")


@app.get("/stock-analysis", response_class=HTMLResponse)
async def stock_analysis_page():
    """Serve the Per-Stock ML Analysis Dashboard."""
    analysis_path = BASE_DIR / "ml_stock_analysis.html"
    if analysis_path.exists():
        try:
            return analysis_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return analysis_path.read_text(encoding="latin-1")
    return HTMLResponse("<h1>Stock Analysis page not found.</h1>")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
