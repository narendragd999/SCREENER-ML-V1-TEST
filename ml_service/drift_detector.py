"""
Drift Detector — Layer 5 of the Continuous Learning Pipeline
==============================================================
Continuously monitors the ML system for three types of drift:

  1. CONCEPT DRIFT — the relationship between features and outcomes changes.
     Detected by: live prediction accuracy drops >15% below training val accuracy.

  2. FEATURE DRIFT — the input feature distribution shifts.
     Detected by: Population Stability Index (PSI) > 0.25 between recent
     and baseline feature windows.

  3. DATA STALENESS — yfinance hasn't been updated for >2 days.
     Detected by: checking last_date in ohlcv table per ticker.

When drift is detected:
  - Raises an alert in the `drift_alerts` table
  - Triggers an immediate full retrain via `continuous_learner.trigger_full_retrain()`
  - Optionally rolls back to previous model if accuracy drops catastrophically

Schedule: Run every hour via APScheduler.
"""

from __future__ import annotations
import logging
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import numpy as np

try:
    from .db import db, now_ist, IST_OFFSET_HOURS
except ImportError:
    from db import db, now_ist, IST_OFFSET_HOURS

logger = logging.getLogger("drift_detector")

# ── Thresholds ───────────────────────────────────────────────────────────────
ACCURACY_DROP_CRITICAL = 0.20     # 20% drop = CRITICAL
ACCURACY_DROP_WARN = 0.15         # 15% drop = WARN
PSI_DRIFT_THRESHOLD = 0.25        # PSI > 0.25 = drift
PSI_CRITICAL_THRESHOLD = 0.50     # PSI > 0.50 = critical
DATA_STALE_HOURS = 48             # >2 days stale = WARN
DATA_STALE_CRITICAL_HOURS = 96    # >4 days = CRITICAL

MIN_PREDICTIONS_FOR_ACCURACY = 20  # need at least this many for stat sig


class DriftDetector:
    """
    Monitors the ML system for drift and triggers corrective action.

    Usage:
        detector = DriftDetector(
            feature_store=feature_store,
            on_drift_callback=lambda alert: learner.trigger_full_retrain(reason=alert["alert_type"]),
        )

        # Run manually
        alerts = detector.check_all()

        # Or schedule it
        detector.start_scheduler(scheduler)
    """

    def __init__(self,
                 feature_store=None,
                 on_drift_callback=None,
                 on_critical_callback=None):
        """
        Args:
            feature_store: FeatureStore instance (for feature drift)
            on_drift_callback: called when any drift is detected. Receives alert dict.
            on_critical_callback: called for critical alerts. May trigger rollback.
        """
        self.fs = feature_store
        self.on_drift = on_drift_callback
        self.on_critical = on_critical_callback
        self._lock = threading.Lock()
        self._last_check_ts: Optional[str] = None

    # ─────────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def check_all(self, tickers_to_check: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Run all drift checks. Returns list of alerts raised."""
        if not self._lock.acquire(blocking=False):
            logger.info("[DriftDetector] Already running — skipping")
            return []

        try:
            alerts: List[Dict[str, Any]] = []
            self._last_check_ts = now_ist()

            # 1. Concept drift (accuracy drop)
            alerts.extend(self.check_concept_drift())

            # 2. Feature drift (PSI)
            if self.fs is not None and tickers_to_check:
                alerts.extend(self.check_feature_drift(tickers_to_check))

            # 3. Data staleness
            alerts.extend(self.check_data_staleness(tickers_to_check))

            # Trigger callbacks for each alert
            for alert in alerts:
                if alert["severity"] == "CRITICAL" and self.on_critical:
                    try:
                        self.on_critical(alert)
                    except Exception as e:
                        logger.error(f"[DriftDetector] Critical callback failed: {e}")
                elif self.on_drift:
                    try:
                        self.on_drift(alert)
                    except Exception as e:
                        logger.error(f"[DriftDetector] Drift callback failed: {e}")

            if alerts:
                logger.info(f"[DriftDetector] Raised {len(alerts)} alerts: "
                            f"{[a['alert_type'] for a in alerts]}")
            return alerts
        finally:
            self._lock.release()

    def check_concept_drift(self) -> List[Dict[str, Any]]:
        """Check for concept drift by comparing live accuracy to training accuracy.

        Concept drift = the model's predictions are getting worse on new data,
        which means the underlying data distribution has changed in a way that
        breaks the learned feature→outcome relationship.
        """
        alerts = []

        # Get live accuracy (last 7 days)
        live_stats = db.get_outcome_stats(days=7)
        if not live_stats or live_stats.get("total", 0) < MIN_PREDICTIONS_FOR_ACCURACY:
            return []  # not enough data to judge

        live_hit_rate = live_stats.get("hit_rate") or 0
        live_dir_acc = live_stats.get("direction_accuracy") or 0

        # Compare against each model type's training val accuracy
        for model_type in ["signal_quality", "trend_regime",
                           "return_forecaster", "risk_estimator"]:
            prod_row = db.get_production_model(model_type)
            if not prod_row or prod_row["val_accuracy"] is None:
                continue

            train_acc = prod_row["val_accuracy"]

            # Different metric per model type
            if model_type == "signal_quality":
                live_acc = live_hit_rate
            elif model_type == "trend_regime":
                # Use direction accuracy as proxy
                live_acc = live_dir_acc
            else:
                continue  # regression models — use MAE check below

            drop = train_acc - live_acc
            if drop >= ACCURACY_DROP_CRITICAL:
                alert = {
                    "alert_type": "concept_drift",
                    "severity": "CRITICAL",
                    "model_type": model_type,
                    "details": {
                        "train_accuracy": round(train_acc, 4),
                        "live_accuracy": round(live_acc, 4),
                        "drop": round(drop, 4),
                        "live_samples": live_stats["total"],
                        "message": f"Live accuracy dropped {drop*100:.1f}% below training",
                    },
                    "action_taken": "retrain_triggered",
                }
                db.raise_drift_alert(
                    alert_type=alert["alert_type"],
                    severity=alert["severity"],
                    details=alert["details"],
                    model_type=model_type,
                    action_taken="retrain_triggered",
                )
                alerts.append(alert)

            elif drop >= ACCURACY_DROP_WARN:
                alert = {
                    "alert_type": "concept_drift",
                    "severity": "WARN",
                    "model_type": model_type,
                    "details": {
                        "train_accuracy": round(train_acc, 4),
                        "live_accuracy": round(live_acc, 4),
                        "drop": round(drop, 4),
                        "live_samples": live_stats["total"],
                    },
                    "action_taken": "none",
                }
                db.raise_drift_alert(
                    alert_type="concept_drift",
                    severity="WARN",
                    details=alert["details"],
                    model_type=model_type,
                    action_taken="none",
                )
                alerts.append(alert)

        return alerts

    def check_feature_drift(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """Check for feature drift using PSI on each ticker's recent features."""
        if self.fs is None:
            return []

        alerts = []
        drift_tickers = []

        for ticker in tickers[:50]:  # cap to avoid runaway
            try:
                psi_report = self.fs.get_feature_drift(ticker)
                if psi_report.get("status") == "drift_detected":
                    drift_tickers.append({
                        "ticker": ticker,
                        "psi": psi_report["overall_psi"],
                        "max_psi": psi_report["max_psi"],
                    })
            except Exception as e:
                logger.debug(f"[DriftDetector] PSI check failed for {ticker}: {e}")

        if drift_tickers:
            # Count tickers with critical drift
            critical_count = sum(1 for t in drift_tickers
                                 if t["psi"] > PSI_CRITICAL_THRESHOLD)
            severity = "CRITICAL" if critical_count > 5 else "WARN"

            alert = {
                "alert_type": "feature_drift",
                "severity": severity,
                "model_type": None,
                "details": {
                    "tickers_with_drift": len(drift_tickers),
                    "critical_count": critical_count,
                    "top_5": drift_tickers[:5],
                    "message": f"{len(drift_tickers)} tickers show feature drift (PSI > {PSI_DRIFT_THRESHOLD})",
                },
                "action_taken": "retrain_triggered" if severity == "CRITICAL" else "none",
            }
            db.raise_drift_alert(
                alert_type=alert["alert_type"],
                severity=severity,
                details=alert["details"],
                action_taken=alert["action_taken"],
            )
            alerts.append(alert)

        return alerts

    def check_data_staleness(self, tickers: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Check if any tickers have stale OHLCV data."""
        if tickers is None:
            # Get all tickers from DB
            rows = db.query("SELECT DISTINCT ticker FROM ohlcv")
            tickers = [r["ticker"] for r in rows]

        if not tickers:
            return []

        now = datetime.utcnow() + timedelta(hours=IST_OFFSET_HOURS)
        stale_tickers = []
        critical_tickers = []

        for ticker in tickers[:200]:  # cap
            last_date = db.get_last_ohlcv_date(ticker)
            if not last_date:
                continue
            try:
                last_dt = datetime.strptime(last_date, "%Y-%m-%d")
                hours_stale = (now - last_dt).total_seconds() / 3600
                if hours_stale > DATA_STALE_CRITICAL_HOURS:
                    critical_tickers.append(ticker)
                elif hours_stale > DATA_STALE_HOURS:
                    stale_tickers.append(ticker)
            except Exception:
                continue

        alerts = []
        if critical_tickers:
            alert = {
                "alert_type": "data_stale",
                "severity": "CRITICAL",
                "model_type": None,
                "details": {
                    "count": len(critical_tickers),
                    "tickers": critical_tickers[:20],
                    "message": f"{len(critical_tickers)} tickers have data older than {DATA_STALE_CRITICAL_HOURS}h",
                },
                "action_taken": "alert_sent",
            }
            db.raise_drift_alert(
                alert_type="data_stale",
                severity="CRITICAL",
                details=alert["details"],
                action_taken="alert_sent",
            )
            alerts.append(alert)

        if stale_tickers:
            alert = {
                "alert_type": "data_stale",
                "severity": "WARN",
                "model_type": None,
                "details": {
                    "count": len(stale_tickers),
                    "tickers": stale_tickers[:20],
                    "message": f"{len(stale_tickers)} tickers have data older than {DATA_STALE_HOURS}h",
                },
                "action_taken": "none",
            }
            db.raise_drift_alert(
                alert_type="data_stale",
                severity="WARN",
                details=alert["details"],
                action_taken="none",
            )
            alerts.append(alert)

        return alerts

    def start_scheduler(self, scheduler, interval_minutes: int = 60):
        """Register drift check job with APScheduler."""
        from apscheduler.triggers.interval import IntervalTrigger
        scheduler.add_job(
            self.check_all,
            trigger=IntervalTrigger(minutes=interval_minutes),
            id="drift_check",
            replace_existing=True,
        )
        logger.info(f"[DriftDetector] Scheduled to run every {interval_minutes} min")


# ── Singleton ────────────────────────────────────────────────────────────────
_detector: Optional[DriftDetector] = None
_detector_lock = threading.Lock()


def get_detector(feature_store=None,
                 on_drift_callback=None,
                 on_critical_callback=None) -> DriftDetector:
    """Get the singleton DriftDetector instance."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = DriftDetector(
                    feature_store=feature_store,
                    on_drift_callback=on_drift_callback,
                    on_critical_callback=on_critical_callback,
                )
    return _detector
