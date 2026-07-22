"""
Monitoring API — FastAPI router with new endpoints for the continuous
learning pipeline.

Mount this in your existing ml_service.py:
    from monitoring_api import create_monitoring_router
    app.include_router(create_monitoring_router())
"""

from __future__ import annotations
import logging
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

try:
    from .db import db, now_ist, IST_OFFSET_HOURS
except ImportError:
    from db import db, now_ist, IST_OFFSET_HOURS

logger = logging.getLogger("monitoring_api")


def create_monitoring_router() -> APIRouter:
    """Create the monitoring router. Mount in ml_service.py."""

    router = APIRouter(prefix="/api/ml-monitor", tags=["ML Monitoring"])

    # ─────────────────────────────────────────────────────────────────────
    #  ACCURACY ENDPOINTS
    # ─────────────────────────────────────────────────────────────────────

    @router.get("/accuracy")
    async def get_accuracy(days: int = Query(7, ge=1, le=90)):
        """Get rolling accuracy stats for the last N days."""
        stats = db.get_outcome_stats(days=days)
        return {
            "window_days": days,
            "stats": stats,
            "timestamp": now_ist(),
        }

    @router.get("/accuracy/trend")
    async def get_accuracy_trend(days: int = Query(30, ge=1, le=180)):
        """Get daily accuracy metrics for trend charts."""
        try:
            from outcome_tracker import get_tracker
            tracker = get_tracker()
            trend = tracker.get_accuracy_trend(days=days)
            return {
                "trend": trend,
                "days": len(trend),
                "timestamp": now_ist(),
            }
        except Exception as e:
            raise HTTPException(500, f"Failed to get trend: {e}")

    @router.get("/predictions/recent")
    async def get_recent_predictions(limit: int = Query(50, le=500)):
        """Get recent predictions (most recent first)."""
        rows = db.query(
            """SELECT p.*,
                  CASE WHEN o.prediction_id IS NOT NULL THEN 1 ELSE 0 END as has_outcome,
                  o.actual_return, o.actual_max_gain, o.actual_max_loss,
                  o.hit_target, o.return_error
               FROM predictions p
               LEFT JOIN outcomes o ON o.prediction_id = p.id
               ORDER BY p.prediction_ts DESC
               LIMIT ?""",
            (limit,)
        )
        return {
            "predictions": [dict(r) for r in rows],
            "count": len(rows),
        }

    @router.get("/predictions/pending")
    async def get_pending_predictions():
        """Get predictions that haven't been resolved yet (awaiting 5-day outcome)."""
        rows = db.get_pending_outcomes(lookahead_days=5)
        return {
            "pending_count": len(rows),
            "pending": [dict(r) for r in rows[:100]],  # cap to 100 for response size
        }

    # ─────────────────────────────────────────────────────────────────────
    #  MODEL REGISTRY ENDPOINTS
    # ─────────────────────────────────────────────────────────────────────

    @router.get("/models")
    async def list_models(model_type: Optional[str] = None):
        """List all model versions."""
        try:
            from model_registry import get_registry
            registry = get_registry()
            models = registry.list_versions(model_type)
            return {
                "models": models,
                "count": len(models),
            }
        except Exception as e:
            raise HTTPException(500, f"Failed to list models: {e}")

    @router.get("/models/production")
    async def get_production_models():
        """Get current production model for each type."""
        result = {}
        for mt in ["signal_quality", "trend_regime", "return_forecaster", "risk_estimator"]:
            row = db.get_production_model(mt)
            if row:
                result[mt] = dict(row)
            else:
                result[mt] = None
        return result

    @router.post("/models/{version}/promote")
    async def promote_model(version: str):
        """Manually promote a model version to production."""
        try:
            from model_registry import get_registry
            registry = get_registry()
            registry.promote(version, notes="manual promotion via API")
            return {"status": "ok", "promoted": version}
        except Exception as e:
            raise HTTPException(500, f"Failed to promote: {e}")

    @router.post("/models/rollback/{model_type}")
    async def rollback_model(model_type: str, reason: str = "manual"):
        """Roll back to the previous production model."""
        try:
            from model_registry import get_registry
            registry = get_registry()
            prev = registry.rollback(model_type, reason=reason)
            if prev is None:
                raise HTTPException(404, "No previous model to roll back to")
            return {"status": "ok", "rolled_back_to": prev}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Failed to rollback: {e}")

    # ─────────────────────────────────────────────────────────────────────
    #  TRAINING LOG ENDPOINTS
    # ─────────────────────────────────────────────────────────────────────

    @router.get("/training/history")
    async def get_training_history(limit: int = Query(50, le=500)):
        """Get recent training runs."""
        rows = db.get_training_history(limit=limit)
        return {
            "history": [dict(r) for r in rows],
            "count": len(rows),
        }

    @router.post("/training/trigger-fine-tune")
    async def trigger_fine_tune():
        """Manually trigger a fine-tune cycle."""
        try:
            from continuous_learner import get_learner  # may not exist yet
            learner = get_learner()
            if learner.is_training:
                return {"status": "already_training"}
            # Run async — don't block the request
            import threading
            t = threading.Thread(target=learner.run_fine_tune_cycle, daemon=True)
            t.start()
            return {"status": "triggered"}
        except Exception as e:
            raise HTTPException(500, f"Failed to trigger: {e}")

    @router.post("/training/trigger-full-retrain")
    async def trigger_full_retrain(reason: str = "manual"):
        """Manually trigger a full retrain."""
        try:
            from continuous_learner import get_learner
            learner = get_learner()
            if learner.is_training:
                return {"status": "already_training"}
            import threading
            t = threading.Thread(
                target=lambda: learner.trigger_full_retrain(reason=reason),
                daemon=True
            )
            t.start()
            return {"status": "triggered", "reason": reason}
        except Exception as e:
            raise HTTPException(500, f"Failed to trigger: {e}")

    # ─────────────────────────────────────────────────────────────────────
    #  DRIFT ALERTS
    # ─────────────────────────────────────────────────────────────────────

    @router.get("/alerts")
    async def get_alerts(hours: int = Query(24, le=720),
                          severity: Optional[str] = None):
        """Get drift alerts from the last N hours."""
        rows = db.get_recent_alerts(hours=hours, severity=severity)
        return {
            "alerts": [dict(r) for r in rows],
            "count": len(rows),
        }

    @router.post("/drift/check")
    async def trigger_drift_check():
        """Manually trigger a drift check."""
        try:
            from drift_detector import get_detector
            detector = get_detector()
            alerts = detector.check_all()
            return {
                "alerts_raised": len(alerts),
                "alerts": alerts,
            }
        except Exception as e:
            raise HTTPException(500, f"Failed to check drift: {e}")

    # ─────────────────────────────────────────────────────────────────────
    #  OUTCOME TRACKING
    # ─────────────────────────────────────────────────────────────────────

    @router.post("/outcomes/resolve")
    async def resolve_outcomes():
        """Manually trigger outcome resolution."""
        try:
            from outcome_tracker import get_tracker
            tracker = get_tracker()
            result = tracker.resolve_all_pending()
            return result
        except Exception as e:
            raise HTTPException(500, f"Failed to resolve outcomes: {e}")

    @router.get("/outcomes/stats")
    async def get_outcome_stats(days: int = Query(7, le=90)):
        """Get outcome stats for the last N days."""
        stats = db.get_outcome_stats(days=days)
        return {
            "window_days": days,
            "stats": stats,
        }

    # ─────────────────────────────────────────────────────────────────────
    #  DATA FRESHNESS
    # ─────────────────────────────────────────────────────────────────────

    @router.get("/data/staleness")
    async def get_data_staleness(tickers: Optional[str] = None):
        """Get staleness report for tickers."""
        try:
            from delta_downloader import get_downloader
            dd = get_downloader()
            ticker_list = tickers.split(",") if tickers else None
            if ticker_list is None:
                # Get all tickers from DB
                rows = db.query("SELECT DISTINCT ticker FROM ohlcv")
                ticker_list = [r["ticker"] for r in rows]
            report = dd.get_staleness_report(ticker_list)
            return {
                "report": report,
                "total": len(report),
                "needs_update": sum(1 for r in report if r["needs_update"]),
            }
        except Exception as e:
            raise HTTPException(500, f"Failed to get staleness: {e}")

    # ─────────────────────────────────────────────────────────────────────
    #  DASHBOARD SUMMARY
    # ─────────────────────────────────────────────────────────────────────

    @router.get("/dashboard")
    async def get_dashboard():
        """One-shot dashboard summary — all key metrics in one call."""
        # Accuracy (7-day)
        acc_7d = db.get_outcome_stats(days=7)
        acc_30d = db.get_outcome_stats(days=30)

        # Prediction counts
        total_predictions = db.query_one("SELECT COUNT(*) as c FROM predictions")["c"]
        resolved = db.query_one("SELECT COUNT(*) as c FROM outcomes")["c"]
        pending = total_predictions - resolved

        # Training queue size
        queue_size = db.query_one(
            "SELECT COUNT(*) as c FROM training_queue WHERE used_in_training IS NULL"
        )["c"]

        # Recent alerts
        recent_alerts = db.get_recent_alerts(hours=24)
        critical_alerts = [a for a in recent_alerts if a["severity"] == "CRITICAL"]

        # Production models
        production_models = {}
        for mt in ["signal_quality", "trend_regime", "return_forecaster", "risk_estimator"]:
            row = db.get_production_model(mt)
            if row:
                production_models[mt] = {
                    "version": row["version"],
                    "trained_at": row["trained_at"],
                    "val_accuracy": row["val_accuracy"],
                    "walk_forward_score": row["walk_forward_score"],
                }

        # Recent training runs
        recent_training = db.get_training_history(limit=5)
        last_fine_tune = None
        last_full_retrain = None
        for r in recent_training:
            if r["run_type"] == "fine_tune" and last_fine_tune is None:
                last_fine_tune = r["run_ts"]
            elif r["run_type"] == "full_retrain" and last_full_retrain is None:
                last_full_retrain = r["run_ts"]
            if last_fine_tune and last_full_retrain:
                break

        return {
            "timestamp": now_ist(),
            "accuracy": {
                "last_7_days": acc_7d,
                "last_30_days": acc_30d,
            },
            "predictions": {
                "total": total_predictions,
                "resolved": resolved,
                "pending": pending,
                "resolution_rate": round(resolved / max(total_predictions, 1) * 100, 1),
            },
            "training_queue": {
                "unused_samples": queue_size,
            },
            "alerts": {
                "last_24h": len(recent_alerts),
                "critical_count": len(critical_alerts),
                "latest": dict(recent_alerts[0]) if recent_alerts else None,
            },
            "models": production_models,
            "training": {
                "last_fine_tune": last_fine_tune,
                "last_full_retrain": last_full_retrain,
                "recent_runs": [dict(r) for r in recent_training],
            },
        }

    return router
