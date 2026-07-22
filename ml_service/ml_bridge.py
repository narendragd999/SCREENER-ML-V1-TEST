"""
ML Bridge — Connects Main App (8000) to ML Service (8001)
============================================================
This module is imported by your main app's main.py to:
  1. Push Chartink/EMA9 signals to the ML service automatically
  2. Fetch ML predictions to enrich API responses
  3. Add a /api/ml/* proxy route to the main app's FastAPI instance

Usage in main.py (add these lines):
    from ml_bridge import create_ml_bridge
    ml_bridge_router = create_ml_bridge()
    app.include_router(ml_bridge_router)

The ML service runs independently on port 8001. This bridge
just makes HTTP calls to it. If the ML service is down, all
calls fail gracefully (return None / empty results).
"""

import os
import sys
import logging
import threading
from typing import Dict, List, Optional, Any

import httpx

logger = logging.getLogger("ml_bridge")

ML_SERVICE_URL = os.environ.get("ML_SERVICE_URL", "http://localhost:8001")

# Timeout for ML service calls (generous — training can be slow)
_TIMEOUT = 30.0

# In-memory prediction cache (avoids calling ML service on every page load)
_pred_cache: Dict[str, Dict] = {}
_pred_cache_lock = threading.Lock()
_PRED_CACHE_TTL = 300  # 5 minutes


def is_ml_service_available() -> bool:
    """Quick health check — is the ML service running?"""
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{ML_SERVICE_URL}/api/health")
            return resp.status_code == 200
    except Exception:
        return False


def push_signals_to_ml(signals: List[Dict], source: str = "chartink") -> bool:
    """
    Push Chartink/EMA9 signals to the ML service for prediction.
    Called automatically by the chartink_watcher scheduler.

    Returns True if the push succeeded.
    """
    if not signals:
        return False
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(
                f"{ML_SERVICE_URL}/api/ingest/chartink",
                json={"signals": signals, "source": source},
            )
            if resp.status_code == 200:
                result = resp.json()
                logger.info(f"[ML Bridge] Pushed {result.get('ingested', 0)} signals, "
                           f"got {result.get('decisions_generated', 0)} decisions")
                return True
            else:
                logger.warning(f"[ML Bridge] Push failed: {resp.status_code}")
                return False
    except Exception as e:
        logger.debug(f"[ML Bridge] Push error (ML service may be down): {e}")
        return False


def get_ml_prediction(ticker: str) -> Optional[Dict]:
    """Get ML prediction for a single ticker (with caching)."""
    import time
    ticker = ticker.strip().upper()

    with _pred_cache_lock:
        cached = _pred_cache.get(ticker)
        if cached and (time.time() - cached["ts"]) < _PRED_CACHE_TTL:
            return cached["data"]

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{ML_SERVICE_URL}/api/predict/{ticker}")
            if resp.status_code == 200:
                data = resp.json()
                with _pred_cache_lock:
                    _pred_cache[ticker] = {"data": data, "ts": time.time()}
                return data
    except Exception:
        pass
    return None


def get_ml_predictions_batch(tickers: List[str]) -> Dict[str, Dict]:
    """Get ML predictions for multiple tickers."""
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{ML_SERVICE_URL}/api/predict/batch",
                json={"tickers": tickers},
            )
            if resp.status_code == 200:
                return resp.json().get("predictions", {})
    except Exception:
        pass
    return {}


def get_ml_decisions(limit: int = 20) -> List[Dict]:
    """Get latest ML intelligent decisions."""
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{ML_SERVICE_URL}/api/decisions?limit={limit}")
            if resp.status_code == 200:
                return resp.json().get("decisions", [])
    except Exception:
        pass
    return []


def get_ml_status() -> Optional[Dict]:
    """Get ML service status."""
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{ML_SERVICE_URL}/api/train/status")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return None


def trigger_ml_training(tickers: Optional[List[str]] = None) -> bool:
    """Trigger ML model training."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{ML_SERVICE_URL}/api/train/trigger",
                json={"tickers": tickers or [], "epochs": 10, "target_pct": 3.0},
            )
            return resp.status_code == 200
    except Exception:
        return False


def enrich_signal_with_ml(signal: Dict) -> Dict:
    """
    Enrich a single signal dict (from EMA9/Chartink pipeline) with ML predictions.
    This is the KEY integration point — call this on every signal before
    returning it to the frontend.

    Usage:
        for signal in prime_targets:
            signal = enrich_signal_with_ml(signal)
    """
    ticker = signal.get("ticker", "")
    if not ticker:
        return signal

    ml_pred = get_ml_prediction(ticker)
    if ml_pred is None:
        signal["ml_score"] = None
        signal["ml_actionable"] = None
        signal["ml_status"] = "ml_service_offline"
        return signal

    signal["ml_score"] = ml_pred.get("ml_score")
    signal["ml_signal_quality"] = ml_pred.get("signal_quality")
    signal["ml_trend_regime"] = ml_pred.get("trend_regime")
    signal["ml_regime_confidence"] = ml_pred.get("regime_confidence")
    signal["ml_expected_return_pct"] = ml_pred.get("expected_return")
    signal["ml_max_drawdown_pct"] = ml_pred.get("max_drawdown")
    signal["ml_risk_reward_ratio"] = ml_pred.get("risk_reward_ratio")
    signal["ml_actionable"] = ml_pred.get("actionable")
    signal["ml_status"] = "enriched"

    return signal


# ═══════════════════════════════════════════════════════════════════════════
#  FASTAPI ROUTER (optional — mount in main.py if you want /api/ml/* routes)
# ═══════════════════════════════════════════════════════════════════════════

def create_ml_bridge() -> "APIRouter":
    """
    Create a FastAPI router that proxies ML service endpoints.
    Mount in main.py:  app.include_router(create_ml_bridge())

    This allows the frontend to call /api/ml/* on port 8000,
    which are proxied to the ML service on port 8001.
    """
    from fastapi import APIRouter, HTTPException

    router = APIRouter(prefix="/api/ml", tags=["ML Intelligence"])

    @router.get("/health")
    async def ml_health():
        """Check if ML service is running."""
        available = is_ml_service_available()
        status = get_ml_status() if available else None
        return {
            "ml_service_running": available,
            "status": status,
            "url": ML_SERVICE_URL,
        }

    @router.get("/predict/{ticker}")
    async def ml_predict(ticker: str):
        """Get ML prediction for a ticker (proxied from ML service)."""
        pred = get_ml_prediction(ticker)
        if pred is None:
            raise HTTPException(503, "ML service unavailable or no data for ticker")
        return pred

    @router.get("/decisions")
    async def ml_decisions(limit: int = 20):
        """Get latest ML intelligent decisions (proxied)."""
        decisions = get_ml_decisions(limit)
        return {"decisions": decisions, "count": len(decisions)}

    @router.get("/decisions/summary")
    async def ml_decisions_summary():
        """Get decision summary (proxied)."""
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{ML_SERVICE_URL}/api/decisions/summary")
                if resp.status_code == 200:
                    return resp.json()
        except Exception:
            pass
        return {"message": "ML service unavailable"}

    @router.post("/train/trigger")
    async def ml_trigger_train(tickers: Optional[List[str]] = None):
        """Trigger ML training (proxied)."""
        ok = trigger_ml_training(tickers)
        return {"status": "triggered" if ok else "failed"}

    return router