"""
Continuous Learner — Layer 4 of the Continuous Learning Pipeline
=================================================================
This is the orchestration layer that ties everything together:

    ┌──────────────────────────────────────────────────────────┐
    │  Every 6 hours (configurable):                            │
    │  1. Pull latest labeled samples from training_queue       │
    │  2. Walk-forward split (train: 180d, val: 30d)            │
    │  3. Apply purge + embargo (3-day gap, prevents leakage)   │
    │  4. Fine-tune current production model (LR 5e-5, 5 epochs)│
    │  5. Evaluate against current production on val set        │
    │  6. PROMOTION GATE:                                        │
    │     • >+0.5% improvement → promote new model              │
    │     • <-2% degradation → reject, keep current             │
    │     • Otherwise → marginal, keep current                  │
    │  7. If promoted: invalidate prediction cache              │
    │  8. Log everything to training_log table                  │
    └──────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────┐
    │  Weekly (Sunday 2 AM):                                    │
    │  1. Pull ALL historical training data (5y)                │
    │  2. Full retrain from scratch (30 epochs)                 │
    │  3. Same promotion gate                                   │
    │  4. Reset feature normalization stats from fresh data     │
    └──────────────────────────────────────────────────────────┘

    ┌──────────────────────────────────────────────────────────┐
    │  On drift alert (triggered by drift_detector):            │
    │  1. Immediate full retrain                                │
    │  2. If new model is worse than current → rollback         │
    └──────────────────────────────────────────────────────────┘
"""

from __future__ import annotations
import os
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# Lazy imports for TF — heavy and may not always be needed
try:
    from .db import db, now_ist, IST_OFFSET_HOURS
except ImportError:
    from db import db, now_ist, IST_OFFSET_HOURS

try:
    from .model_registry import get_registry, ModelRegistry
except ImportError:
    from model_registry import get_registry, ModelRegistry

try:
    from .walk_forward import simple_time_split, purge_and_embargo
except ImportError:
    from walk_forward import simple_time_split, purge_and_embargo

logger = logging.getLogger("continuous_learner")

# ── Configuration ────────────────────────────────────────────────────────────
FINE_TUNE_INTERVAL_HOURS = 6         # how often to fine-tune
FULL_RETRAIN_DAY_OF_WEEK = 6         # 6 = Sunday
FULL_RETRAIN_HOUR = 2                # 2 AM IST

FINE_TUNE_LEARNING_RATE = 5e-5       # very small to avoid catastrophic forgetting
FINE_TUNE_EPOCHS = 5
FINE_TUNE_BATCH_SIZE = 128
FINE_TUNE_MIN_SAMPLES = 200          # skip if fewer samples available

FULL_RETRAIN_LEARNING_RATE = 1e-3
FULL_RETRAIN_EPOCHS = 30
FULL_RETRAIN_BATCH_SIZE = 256

TRAIN_LOOKBACK_DAYS = 180            # train window for fine-tune
VAL_LOOKAHEAD_DAYS = 30              # val window for fine-tune
EMBARGO_DAYS = 3                     # gap between train and val


class ContinuousLearner:
    """
    Always-on continuous learning orchestrator.

    Wraps the existing `ModelManager` from ml_models.py and adds:
      - Walk-forward validation (no data leakage)
      - Promotion gate (only promote if actually better)
      - Model versioning (no overwriting)
      - Drift-triggered retraining
      - Full audit log

    Usage:
        learner = ContinuousLearner(
            model_manager=model_manager,        # from ml_models.py
            feature_extractor=extract_features, # from ml_features.py
        )

        # Run once (manual trigger)
        result = learner.run_fine_tune_cycle()

        # Or schedule it (in FastAPI startup)
        learner.start_scheduler(scheduler)

        # Drift was detected? Trigger immediate retrain
        learner.trigger_full_retrain(reason="concept_drift_detected")
    """

    def __init__(self,
                 model_manager,
                 feature_extractor: Callable,
                 registry: Optional[ModelRegistry] = None,
                 target_pct: float = 3.0,
                 lookahead_days: int = 5):
        """
        Args:
            model_manager: an instance of ml_models.ModelManager (the existing one)
            feature_extractor: callable(df, normalize=False) -> np.ndarray
            registry: ModelRegistry instance (defaults to singleton)
            target_pct: target gain % for hit_target label
            lookahead_days: days ahead to compute labels
        """
        self.mm = model_manager
        self.feature_extractor = feature_extractor
        self.registry = registry or get_registry()
        self.target_pct = target_pct
        self.lookahead_days = lookahead_days
        self._lock = threading.RLock()           # allow nested locking
        self._is_training = False
        self._last_fine_tune_ts: Optional[str] = None
        self._last_full_retrain_ts: Optional[str] = None

    # ─────────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_training(self) -> bool:
        return self._is_training

    def run_fine_tune_cycle(self,
                            model_types: Optional[List[str]] = None,
                            max_samples: int = 5000) -> Dict[str, Any]:
        """Run one fine-tune cycle. Called every 6 hours by scheduler.

        Steps:
          1. Pull recent training samples from training_queue
          2. Walk-forward split
          3. Fine-tune each model
          4. Promotion gate
          5. Log everything

        Returns:
            Summary dict with results per model type.
        """
        if not self._lock.acquire(blocking=False):
            logger.info("[ContinuousLearner] Already training — skipping fine-tune cycle")
            return {"status": "skipped", "reason": "already_training"}

        self._is_training = True
        start_ts = time.time()
        try:
            logger.info("=" * 60)
            logger.info("[ContinuousLearner] Starting fine-tune cycle")
            logger.info("=" * 60)

            # Pull training samples (most recent N)
            X, y = db.get_training_samples(n=max_samples)
            if len(X) < FINE_TUNE_MIN_SAMPLES:
                logger.info(f"[ContinuousLearner] Only {len(X)} samples available "
                           f"(min {FINE_TUNE_MIN_SAMPLES}) — skipping")
                return {"status": "skipped", "reason": "insufficient_samples",
                        "samples_available": len(X)}

            # Get corresponding dates and tickers for walk-forward split
            # We need to query training_queue again to get metadata
            sample_meta = db.query(
                """SELECT id, ticker, feature_date FROM training_queue
                   WHERE used_in_training IS NULL
                   ORDER BY queued_at DESC LIMIT ?""",
                (max_samples,)
            )
            sample_ids = [r["id"] for r in sample_meta]
            dates = [r["feature_date"] for r in sample_meta]
            tickers = [r["ticker"] for r in sample_meta]

            # Build a DataFrame for walk-forward splitting
            import pandas as pd
            samples_df = pd.DataFrame({
                "date": dates,
                "ticker": tickers,
                "sample_id": sample_ids,
            })
            # Attach feature arrays and labels as columns
            samples_df["features"] = list(X)
            samples_df["label_hit_target"] = y["hit_target"]
            samples_df["label_return_pct"] = y["return_pct"]
            samples_df["label_max_gain_pct"] = y["max_gain_pct"]
            samples_df["label_max_loss_pct"] = y["max_loss_pct"]
            samples_df["label_regime"] = y["regime"]

            # Walk-forward split
            train_df, val_df = simple_time_split(
                samples_df,
                date_col="date",
                train_days=TRAIN_LOOKBACK_DAYS,
                val_days=VAL_LOOKAHEAD_DAYS,
            )

            if len(train_df) < 50 or len(val_df) < 20:
                logger.warning(f"[ContinuousLearner] Insufficient data after split: "
                              f"train={len(train_df)}, val={len(val_df)}")
                return {"status": "skipped", "reason": "insufficient_after_split",
                        "train_size": len(train_df), "val_size": len(val_df)}

            # Apply purge + embargo
            train_df, val_df = purge_and_embargo(
                train_df, val_df, date_col="date", embargo_days=EMBARGO_DAYS
            )

            # Extract arrays
            X_train = np.stack(train_df["features"].values).astype(np.float32)
            X_val = np.stack(val_df["features"].values).astype(np.float32)

            results: Dict[str, Any] = {
                "status": "ok",
                "started_at": now_ist(),
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "models": {},
            }

            # Determine which model types to train
            all_types = ["signal_quality", "trend_regime", "return_forecaster", "risk_estimator"]
            model_types = model_types or all_types

            # Update feature normalization stats from training data
            self._update_feature_stats(X_train)

            # Normalize train and val using new stats
            X_train_norm = self._normalize(X_train)
            X_val_norm = self._normalize(X_val)

            # Train each model type
            for model_type in model_types:
                try:
                    result = self._fine_tune_one_model(
                        model_type=model_type,
                        X_train=X_train_norm, X_val=X_val_norm,
                        y_train=self._extract_labels(train_df, model_type),
                        y_val=self._extract_labels(val_df, model_type),
                        train_sample_ids=train_df["sample_id"].tolist(),
                    )
                    results["models"][model_type] = result
                except Exception as e:
                    logger.error(f"[ContinuousLearner] {model_type} fine-tune failed: {e}")
                    results["models"][model_type] = {
                        "status": "FAILED",
                        "error": str(e),
                    }

            results["elapsed_seconds"] = round(time.time() - start_ts, 2)
            results["finished_at"] = now_ist()
            self._last_fine_tune_ts = now_ist()

            logger.info(f"[ContinuousLearner] Fine-tune cycle complete in "
                        f"{results['elapsed_seconds']}s")
            return results

        finally:
            self._is_training = False
            self._lock.release()

    def trigger_full_retrain(self, reason: str = "scheduled") -> Dict[str, Any]:
        """Trigger a full retrain from scratch. Called weekly or on drift alert.

        WARNING: This is expensive (1-3 hours). Run on a separate thread.
        """
        if not self._lock.acquire(blocking=False):
            logger.info("[ContinuousLearner] Already training — skipping full retrain")
            return {"status": "skipped", "reason": "already_training"}

        self._is_training = True
        start_ts = time.time()
        try:
            logger.info("=" * 60)
            logger.info(f"[ContinuousLearner] Starting FULL RETRAIN. Reason: {reason}")
            logger.info("=" * 60)

            # Pull ALL training samples (no limit)
            X, y = db.get_training_samples(n=100000)
            if len(X) < 500:
                return {"status": "skipped", "reason": "insufficient_samples",
                        "samples_available": len(X)}

            # Get metadata for splitting
            sample_meta = db.query(
                """SELECT id, ticker, feature_date FROM training_queue
                   ORDER BY queued_at ASC""",
                ()
            )
            sample_ids = [r["id"] for r in sample_meta]
            dates = [r["feature_date"] for r in sample_meta]
            tickers = [r["ticker"] for r in sample_meta]

            import pandas as pd
            samples_df = pd.DataFrame({
                "date": dates,
                "ticker": tickers,
                "sample_id": sample_ids,
                "features": list(X),
                "label_hit_target": y["hit_target"],
                "label_return_pct": y["return_pct"],
                "label_max_gain_pct": y["max_gain_pct"],
                "label_max_loss_pct": y["max_loss_pct"],
                "label_regime": y["regime"],
            })

            # Walk-forward split with larger windows
            train_df, val_df = simple_time_split(
                samples_df,
                date_col="date",
                train_days=365,  # 1 year for full retrain
                val_days=60,
            )
            train_df, val_df = purge_and_embargo(
                train_df, val_df, date_col="date", embargo_days=EMBARGO_DAYS
            )

            # Recompute normalization stats from scratch
            X_train = np.stack(train_df["features"].values).astype(np.float32)
            self._update_feature_stats(X_train, reset=True)
            X_train_norm = self._normalize(X_train)
            X_val_norm = self._normalize(np.stack(val_df["features"].values).astype(np.float32))

            results: Dict[str, Any] = {
                "status": "ok",
                "reason": reason,
                "started_at": now_ist(),
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "models": {},
            }

            # Build fresh models from scratch
            self.mm._build_missing_models()  # ensures builders are called

            # Train each model type with full retrain settings
            for model_type in ["signal_quality", "trend_regime",
                               "return_forecaster", "risk_estimator"]:
                try:
                    # Reset the model to a fresh state
                    builder_map = {
                        "signal_quality": "build_signal_quality_model",
                        "trend_regime": "build_trend_regime_model",
                        "return_forecaster": "build_return_forecaster_model",
                        "risk_estimator": "build_risk_estimator_model",
                    }
                    import ml_models
                    builder = getattr(ml_models, builder_map[model_type])
                    self.mm.models[model_type] = builder()

                    # Train with full settings
                    result = self._train_model_full(
                        model_type=model_type,
                        X_train=X_train_norm,
                        X_val=X_val_norm,
                        y_train=self._extract_labels(train_df, model_type),
                        y_val=self._extract_labels(val_df, model_type),
                        train_sample_ids=train_df["sample_id"].tolist(),
                    )
                    results["models"][model_type] = result
                except Exception as e:
                    logger.error(f"[ContinuousLearner] {model_type} full retrain failed: {e}")
                    results["models"][model_type] = {"status": "FAILED", "error": str(e)}

            results["elapsed_seconds"] = round(time.time() - start_ts, 2)
            results["finished_at"] = now_ist()
            self._last_full_retrain_ts = now_ist()

            logger.info(f"[ContinuousLearner] Full retrain complete in "
                        f"{results['elapsed_seconds']}s")
            return results

        finally:
            self._is_training = False
            self._lock.release()

    def start_scheduler(self, scheduler):
        """Register fine-tune and full-retrain jobs with APScheduler."""
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.interval import IntervalTrigger
        from apscheduler.triggers.cron import CronTrigger

        # Fine-tune every N hours
        scheduler.add_job(
            self.run_fine_tune_cycle,
            trigger=IntervalTrigger(hours=FINE_TUNE_INTERVAL_HOURS),
            id="continuous_fine_tune",
            replace_existing=True,
        )

        # Full retrain weekly (Sunday 2 AM IST)
        scheduler.add_job(
            lambda: self.trigger_full_retrain(reason="scheduled_weekly"),
            trigger=CronTrigger(
                day_of_week=FULL_RETRAIN_DAY_OF_WEEK,
                hour=FULL_RETRAIN_HOUR,
                minute=0,
                timezone="Asia/Kolkata",
            ),
            id="continuous_full_retrain",
            replace_existing=True,
        )

        logger.info(f"[ContinuousLearner] Scheduler started: "
                    f"fine-tune every {FINE_TUNE_INTERVAL_HOURS}h, "
                    f"full retrain weekly on Sunday {FULL_RETRAIN_HOUR} AM IST")

    # ─────────────────────────────────────────────────────────────────────────
    #  INTERNALS
    # ─────────────────────────────────────────────────────────────────────────

    def _fine_tune_one_model(self,
                             model_type: str,
                             X_train: np.ndarray,
                             X_val: np.ndarray,
                             y_train: Dict[str, np.ndarray],
                             y_val: Dict[str, np.ndarray],
                             train_sample_ids: List[int]) -> Dict[str, Any]:
        """Fine-tune a single model type and apply promotion gate."""

        # Get current production model
        prod_model = self.registry.load_production_model(model_type)
        if prod_model is None:
            # No production model — use whatever is in model_manager
            prod_model = self.mm.models.get(model_type)
            if prod_model is None:
                self.mm._build_missing_models()
                prod_model = self.mm.models.get(model_type)

        # Lower learning rate for fine-tuning
        import tensorflow as tf
        keras_backend = tf.keras.backend
        keras_backend.set_value(prod_model.optimizer.learning_rate, FINE_TUNE_LEARNING_RATE)

        # Train
        from tensorflow import keras
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=3,
                restore_best_weights=True, verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=0
            ),
        ]

        # Build training data per model type
        train_data, val_data = self._prepare_model_io(
            model_type, X_train, X_val, y_train, y_val
        )

        start = time.time()
        history = prod_model.fit(
            *train_data,
            validation_data=val_data,
            epochs=FINE_TUNE_EPOCHS,
            batch_size=FINE_TUNE_BATCH_SIZE,
            callbacks=callbacks_list,
            verbose=0,
        )
        elapsed = time.time() - start

        # Evaluate on val set
        val_metrics = self._evaluate_model(model_type, prod_model, X_val, y_val)

        # Save as new version
        version = self.registry.save_version(
            model=prod_model,
            model_type=model_type,
            training_samples=len(X_train),
            metrics={
                "training_loss": float(history.history.get("loss", [None])[-1]),
                "val_loss": val_metrics.get("val_loss"),
                "val_accuracy": val_metrics.get("val_accuracy"),
                "val_return_mae": val_metrics.get("val_return_mae"),
            },
            walk_forward_score=val_metrics.get("walk_forward_score"),
            notes=f"Fine-tune cycle {now_ist()}",
        )

        # Promotion gate
        decision = self.registry.evaluate_for_promotion(model_type, version)

        if decision["should_promote"]:
            self.registry.promote(version, notes=decision["reason"])
            # Mark samples as used by this version
            db.mark_samples_used(train_sample_ids, version)
            # Update model_manager's in-memory model
            self.mm.models[model_type] = prod_model
            status = "PROMOTED"
        else:
            # Don't mark samples as used — they can be re-used next cycle
            status = decision["decision"]

        # Log to training_log
        db.log_training_run(
            run_type="fine_tune",
            model_type=model_type,
            status=status,
            model_version=version,
            samples_used=len(X_train),
            epochs_run=FINE_TUNE_EPOCHS,
            final_loss=float(history.history.get("loss", [None])[-1]),
            val_loss=val_metrics.get("val_loss"),
            val_accuracy=val_metrics.get("val_accuracy"),
            duration_seconds=round(elapsed, 2),
            notes=decision["reason"],
        )

        return {
            "status": status,
            "version": version,
            "decision": decision["decision"],
            "reason": decision["reason"],
            "val_metrics": val_metrics,
            "elapsed_seconds": round(elapsed, 2),
        }

    def _train_model_full(self,
                          model_type: str,
                          X_train: np.ndarray,
                          X_val: np.ndarray,
                          y_train: Dict[str, np.ndarray],
                          y_val: Dict[str, np.ndarray],
                          train_sample_ids: List[int]) -> Dict[str, Any]:
        """Full retrain for a single model type."""

        model = self.mm.models.get(model_type)
        if model is None:
            self.mm._build_missing_models()
            model = self.mm.models.get(model_type)

        # Reset learning rate
        import tensorflow as tf
        keras_backend = tf.keras.backend
        keras_backend.set_value(model.optimizer.learning_rate, FULL_RETRAIN_LEARNING_RATE)

        from tensorflow import keras
        callbacks_list = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5,
                restore_best_weights=True, verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0
            ),
        ]

        train_data, val_data = self._prepare_model_io(
            model_type, X_train, X_val, y_train, y_val
        )

        start = time.time()
        history = model.fit(
            *train_data,
            validation_data=val_data,
            epochs=FULL_RETRAIN_EPOCHS,
            batch_size=FULL_RETRAIN_BATCH_SIZE,
            callbacks=callbacks_list,
            verbose=0,
        )
        elapsed = time.time() - start

        val_metrics = self._evaluate_model(model_type, model, X_val, y_val)

        version = self.registry.save_version(
            model=model,
            model_type=model_type,
            training_samples=len(X_train),
            metrics={
                "training_loss": float(history.history.get("loss", [None])[-1]),
                "val_loss": val_metrics.get("val_loss"),
                "val_accuracy": val_metrics.get("val_accuracy"),
                "val_return_mae": val_metrics.get("val_return_mae"),
            },
            walk_forward_score=val_metrics.get("walk_forward_score"),
            notes=f"Full retrain {now_ist()}",
        )

        decision = self.registry.evaluate_for_promotion(model_type, version)
        if decision["should_promote"]:
            self.registry.promote(version, notes=decision["reason"])
            db.mark_samples_used(train_sample_ids, version)
            self.mm.models[model_type] = model
            status = "PROMOTED"
        else:
            status = decision["decision"]

        db.log_training_run(
            run_type="full_retrain",
            model_type=model_type,
            status=status,
            model_version=version,
            samples_used=len(X_train),
            epochs_run=FULL_RETRAIN_EPOCHS,
            final_loss=float(history.history.get("loss", [None])[-1]),
            val_loss=val_metrics.get("val_loss"),
            val_accuracy=val_metrics.get("val_accuracy"),
            duration_seconds=round(elapsed, 2),
            notes=f"{reason if 'reason' in dir() else 'weekly'}: {decision['reason']}",
        )

        return {
            "status": status,
            "version": version,
            "decision": decision["decision"],
            "reason": decision["reason"],
            "val_metrics": val_metrics,
            "elapsed_seconds": round(elapsed, 2),
        }

    def _prepare_model_io(self,
                          model_type: str,
                          X_train: np.ndarray,
                          X_val: np.ndarray,
                          y_train: Dict[str, np.ndarray],
                          y_val: Dict[str, np.ndarray]):
        """Prepare (train_data, val_data) tuples specific to each model type."""
        if model_type == "signal_quality":
            return (
                (X_train, y_train["hit_target"]),
                (X_val, y_val["hit_target"]),
            )
        elif model_type == "trend_regime":
            return (
                (X_train, y_train["regime"]),
                (X_val, y_val["regime"]),
            )
        elif model_type == "return_forecaster":
            # Multi-output: return_pct, max_gain_pct, max_loss_pct
            return (
                (X_train, {
                    "return_pct": y_train["return_pct"].reshape(-1, 1),
                    "max_gain_pct": y_train["max_gain_pct"].reshape(-1, 1),
                    "max_loss_pct": y_train["max_loss_pct"].reshape(-1, 1),
                }),
                (X_val, {
                    "return_pct": y_val["return_pct"].reshape(-1, 1),
                    "max_gain_pct": y_val["max_gain_pct"].reshape(-1, 1),
                    "max_loss_pct": y_val["max_loss_pct"].reshape(-1, 1),
                }),
            )
        elif model_type == "risk_estimator":
            return (
                (X_train, y_train["max_loss_pct"].reshape(-1, 1)),
                (X_val, y_val["max_loss_pct"].reshape(-1, 1)),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _evaluate_model(self,
                        model_type: str,
                        model,
                        X_val: np.ndarray,
                        y_val: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate model on validation set, returning a metrics dict."""
        try:
            if model_type == "signal_quality":
                preds = model.predict(X_val, verbose=0).flatten()
                loss = float(model.evaluate(X_val, y_val["hit_target"], verbose=0, return_dict=True).get("loss", 0))
                binary_preds = (preds > 0.5).astype(int)
                accuracy = float(np.mean(binary_preds == y_val["hit_target"]))
                # Walk-forward score: composite accuracy + AUC-like measure
                from sklearn.metrics import roc_auc_score
                try:
                    auc = float(roc_auc_score(y_val["hit_target"], preds))
                except Exception:
                    auc = 0.5
                wf_score = (accuracy * 60 + auc * 40)
                return {"val_loss": loss, "val_accuracy": accuracy,
                        "val_auc": auc, "walk_forward_score": round(wf_score, 2)}

            elif model_type == "trend_regime":
                preds = model.predict(X_val, verbose=0)
                pred_classes = np.argmax(preds, axis=1)
                accuracy = float(np.mean(pred_classes == y_val["regime"]))
                loss = float(model.evaluate(X_val, y_val["regime"], verbose=0, return_dict=True).get("loss", 0))
                return {"val_loss": loss, "val_accuracy": accuracy,
                        "walk_forward_score": round(accuracy * 100, 2)}

            elif model_type == "return_forecaster":
                preds = model.predict(X_val, verbose=0)
                # preds is a list of 3 arrays
                pred_return = preds[0].flatten()
                true_return = y_val["return_pct"]
                mae = float(np.mean(np.abs(pred_return - true_return)))
                # Direction accuracy
                direction_acc = float(np.mean(np.sign(pred_return) == np.sign(true_return)))
                loss = float(model.evaluate(X_val, {
                    "return_pct": y_val["return_pct"].reshape(-1, 1),
                    "max_gain_pct": y_val["max_gain_pct"].reshape(-1, 1),
                    "max_loss_pct": y_val["max_loss_pct"].reshape(-1, 1),
                }, verbose=0, return_dict=True).get("loss", 0))
                wf_score = (direction_acc * 50 + max(0, 50 - mae * 10))
                return {"val_loss": loss, "val_return_mae": mae,
                        "val_direction_accuracy": direction_acc,
                        "walk_forward_score": round(wf_score, 2)}

            elif model_type == "risk_estimator":
                preds = model.predict(X_val, verbose=0).flatten()
                true_loss = y_val["max_loss_pct"]
                mae = float(np.mean(np.abs(preds - true_loss)))
                loss = float(model.evaluate(X_val, true_loss.reshape(-1, 1),
                                           verbose=0, return_dict=True).get("loss", 0))
                wf_score = max(0, 100 - mae * 20)
                return {"val_loss": loss, "val_return_mae": mae,
                        "walk_forward_score": round(wf_score, 2)}

        except Exception as e:
            logger.error(f"[ContinuousLearner] Evaluation failed for {model_type}: {e}")
            return {"val_loss": 1.0, "val_accuracy": 0.0, "walk_forward_score": 0.0}

    def _extract_labels(self, df, model_type: str) -> Dict[str, np.ndarray]:
        """Extract the relevant label arrays for a given model type."""
        if model_type == "signal_quality":
            return {"hit_target": df["label_hit_target"].values.astype(np.float32)}
        elif model_type == "trend_regime":
            return {"regime": df["label_regime"].values.astype(np.int32)}
        elif model_type == "return_forecaster":
            return {
                "return_pct": df["label_return_pct"].values.astype(np.float32),
                "max_gain_pct": df["label_max_gain_pct"].values.astype(np.float32),
                "max_loss_pct": df["label_max_loss_pct"].values.astype(np.float32),
            }
        elif model_type == "risk_estimator":
            return {"max_loss_pct": df["label_max_loss_pct"].values.astype(np.float32)}
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _update_feature_stats(self, X: np.ndarray, reset: bool = False) -> None:
        """Update the global FEATURE_STATS used for normalization."""
        try:
            from ml_features import FEATURE_NAMES, update_feature_stats
            if reset:
                import ml_features
                ml_features.FEATURE_STATS = {}
            update_feature_stats(X)
        except ImportError:
            logger.warning("[ContinuousLearner] ml_features not available — skipping stats update")

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Z-score normalize using current FEATURE_STATS."""
        try:
            from ml_features import FEATURE_NAMES, FEATURE_STATS
            if not FEATURE_STATS:
                return X
            X_norm = X.copy()
            for i, name in enumerate(FEATURE_NAMES):
                if name in FEATURE_STATS:
                    stats = FEATURE_STATS[name]
                    if stats["std"] > 1e-8:
                        X_norm[:, i] = (X[:, i] - stats["mean"]) / stats["std"]
            return X_norm
        except ImportError:
            return X
