"""
Model Registry — Versioned Model Storage with Promotion Gate
==============================================================
Replaces the original code which OVERWRITES models at fixed paths like
`models/signal_quality.keras`. That approach has no rollback capability.

This module provides:
  1. Versioned model storage: each training run saves a new file
     named like `signal_quality_v012_2026-01-15_acc0.65.keras`
  2. Promotion gate: a new model is only promoted to production if it
     beats the current production model on walk-forward validation
  3. Rollback: instantly revert to previous production model
  4. History: every model version is preserved for audit / comparison

Models live in `models/{model_type}/v{N}_{date}_{metric}.keras`.
Production pointer is in the `model_registry` DB table.
"""

from __future__ import annotations
import os
import shutil
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger("model_registry")

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MODEL_ROOT = BASE_DIR / "models"

# Promotion gate thresholds
PROMOTION_MIN_IMPROVEMENT = 0.005     # +0.5% accuracy required
ROLLBACK_DEGRADATION_THRESHOLD = 0.02 # -2% triggers rollback
MARGINAL_THRESHOLD = 0.002            # <0.2% change is "marginal"


class ModelRegistry:
    """
    Versioned model storage with promotion gate and rollback support.

    Usage:
        registry = ModelRegistry()

        # After training a new model
        version = registry.save_version(
            model=model,
            model_type="signal_quality",
            training_samples=5000,
            metrics={"val_accuracy": 0.65, "val_loss": 0.42},
            walk_forward_score=68.5,
        )

        # Promote if good enough (auto-checks against current production)
        decision = registry.evaluate_for_promotion(
            model_type="signal_quality",
            new_version=version,
        )
        if decision["should_promote"]:
            registry.promote(version)

        # Rollback if production model starts failing
        registry.rollback("signal_quality", reason="live_accuracy_drop")
    """

    def __init__(self, model_root: Optional[Path] = None):
        self.model_root = model_root or MODEL_ROOT
        self.model_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # In-memory cache of currently-loaded production models
        self._loaded_models: Dict[str, Any] = {}
        # Avoid circular import — import db lazily
        self._db = None

    @property
    def db(self):
        if self._db is None:
            try:
                from .db import db as db_instance
            except ImportError:
                from db import db as db_instance
            self._db = db_instance
        return self._db

    def _get_model_dir(self, model_type: str) -> Path:
        """Get the directory for a specific model type."""
        d = self.model_root / model_type
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _next_version_number(self, model_type: str) -> int:
        """Get the next version number for a model type (starts at 1)."""
        models = self.db.list_models(model_type, include_rolled_back=True)
        if not models:
            return 1
        max_n = 0
        for m in models:
            # Parse version like 'signal_quality_v012_...'
            try:
                n_str = m["version"].split("_v")[1].split("_")[0]
                n = int(n_str)
                max_n = max(max_n, n)
            except (IndexError, ValueError):
                continue
        return max_n + 1

    def save_version(self,
                     model,
                     model_type: str,
                     training_samples: int,
                     metrics: Dict[str, float],
                     walk_forward_score: Optional[float] = None,
                     notes: Optional[str] = None) -> str:
        """Save a newly trained model as a new version.

        Args:
            model: the trained Keras model object
            model_type: 'signal_quality' | 'trend_regime' | 'return_forecaster' | 'risk_estimator'
            training_samples: number of samples used for training
            metrics: dict with keys like 'val_accuracy', 'val_loss', 'training_loss'
            walk_forward_score: composite score on walk-forward validation (0-100)
            notes: free-text notes

        Returns:
            The version string (e.g. 'signal_quality_v012_2026-01-15_acc0.65')
        """
        with self._lock:
            n = self._next_version_number(model_type)
            date_str = datetime.utcnow().strftime("%Y-%m-%d")

            # Build version string with key metric for at-a-glance comparison
            metric_str = ""
            if "val_accuracy" in metrics:
                metric_str = f"_acc{metrics['val_accuracy']:.3f}"
            elif "val_loss" in metrics:
                metric_str = f"_loss{metrics['val_loss']:.3f}"

            version = f"{model_type}_v{n:03d}_{date_str}{metric_str}"
            model_path = self._get_model_dir(model_type) / f"{version}.keras"

            # Save the model
            try:
                model.save(str(model_path))
                logger.info(f"[ModelRegistry] Saved {version} → {model_path}")
            except Exception as e:
                logger.error(f"[ModelRegistry] Failed to save model: {e}")
                raise

            # Register in DB
            self.db.register_model(
                version=version,
                model_type=model_type,
                model_path=str(model_path),
                training_samples=training_samples,
                training_loss=metrics.get("training_loss"),
                val_accuracy=metrics.get("val_accuracy"),
                val_loss=metrics.get("val_loss"),
                val_return_mae=metrics.get("val_return_mae"),
                walk_forward_score=walk_forward_score,
                notes=notes,
            )

            # Log the training run
            self.db.log_training_run(
                run_type="fine_tune",
                model_type=model_type,
                status="SAVED",
                model_version=version,
                samples_used=training_samples,
                final_loss=metrics.get("training_loss"),
                val_loss=metrics.get("val_loss"),
                val_accuracy=metrics.get("val_accuracy"),
                notes=f"Walk-forward score: {walk_forward_score}",
            )

            return version

    def evaluate_for_promotion(self,
                                model_type: str,
                                new_version: str) -> Dict[str, Any]:
        """Compare a new model version against the current production model.

        Returns a dict with:
          - should_promote: bool
          - should_rollback: bool
          - decision: 'PROMOTE' | 'REJECT' | 'MARGINAL' | 'NO_BASELINE'
          - improvement: float (new_val_acc - prod_val_acc)
          - reason: human-readable explanation
        """
        new_model = self.db.query_one(
            "SELECT * FROM model_registry WHERE version = ?",
            (new_version,)
        )
        if not new_model:
            return {"should_promote": False, "decision": "ERROR",
                    "reason": f"Version {new_version} not found"}

        prod_model = self.db.get_production_model(model_type)

        # No production model yet → always promote the first one
        if not prod_model:
            return {
                "should_promote": True,
                "should_rollback": False,
                "decision": "PROMOTE",
                "improvement": None,
                "reason": "No production model yet — promoting first version",
            }

        # Compare metrics
        new_acc = new_model["val_accuracy"] or 0
        prod_acc = prod_model["val_accuracy"] or 0
        improvement = new_acc - prod_acc

        # Also consider walk_forward_score if available
        new_wf = new_model["walk_forward_score"] or 0
        prod_wf = prod_model["walk_forward_score"] or 0
        wf_improvement = new_wf - prod_wf

        # Composite improvement: weighted average of accuracy + walk-forward
        composite = (improvement * 0.6) + (wf_improvement / 100.0 * 0.4)

        if composite >= PROMOTION_MIN_IMPROVEMENT:
            return {
                "should_promote": True,
                "should_rollback": False,
                "decision": "PROMOTE",
                "improvement": round(improvement, 4),
                "walk_forward_improvement": round(wf_improvement, 2),
                "composite_improvement": round(composite, 4),
                "reason": f"New model beats production by {improvement*100:.2f}% accuracy "
                         f"({new_acc:.3f} vs {prod_acc:.3f})",
            }
        elif composite <= -ROLLBACK_DEGRADATION_THRESHOLD:
            # Note: this is for the NEW model — we don't promote it, but we
            # also don't rollback the existing production model. Rollback
            # is triggered separately when LIVE production model degrades.
            return {
                "should_promote": False,
                "should_rollback": False,
                "decision": "REJECT",
                "improvement": round(improvement, 4),
                "reason": f"New model is {abs(improvement)*100:.2f}% WORSE than production — rejecting",
            }
        elif abs(composite) < MARGINAL_THRESHOLD:
            return {
                "should_promote": False,
                "should_rollback": False,
                "decision": "MARGINAL",
                "improvement": round(improvement, 4),
                "reason": f"Change is marginal ({improvement*100:+.2f}%) — keeping current production",
            }
        else:
            return {
                "should_promote": False,
                "should_rollback": False,
                "decision": "REJECT",
                "improvement": round(improvement, 4),
                "reason": f"New model not significantly better ({improvement*100:+.2f}%) — keeping current production",
            }

    def promote(self, version: str, notes: Optional[str] = None) -> None:
        """Promote a model version to production."""
        self.db.promote_model(version, notes)
        # Invalidate the in-memory cache for this model type
        row = self.db.query_one(
            "SELECT model_type FROM model_registry WHERE version = ?",
            (version,)
        )
        if row:
            self._loaded_models.pop(row["model_type"], None)

    def rollback(self, model_type: str, reason: str) -> Optional[str]:
        """Rollback to the previous production model."""
        prev_version = self.db.rollback_model(model_type, reason)
        if prev_version:
            self._loaded_models.pop(model_type, None)
            # Log the rollback
            self.db.log_training_run(
                run_type="rollback",
                model_type=model_type,
                status="ROLLED_BACK",
                model_version=prev_version,
                notes=reason,
            )
            # Raise an alert
            self.db.raise_drift_alert(
                alert_type="model_rollback",
                severity="WARN",
                details={"model_type": model_type, "reason": reason,
                        "rolled_back_to": prev_version},
                model_type=model_type,
                action_taken="rollback",
            )
        return prev_version

    def load_production_model(self, model_type: str):
        """Load the current production model for a given type.

        Uses an in-memory cache to avoid reloading on every prediction.
        Returns None if no production model exists.
        """
        if model_type in self._loaded_models:
            return self._loaded_models[model_type]

        row = self.db.get_production_model(model_type)
        if not row:
            return None

        model_path = row["model_path"]
        if not os.path.exists(model_path):
            logger.error(f"[ModelRegistry] Production model file missing: {model_path}")
            return None

        try:
            # Lazy import to avoid TF dependency at module load
            from tensorflow import keras
            model = keras.models.load_model(model_path)
            self._loaded_models[model_type] = model
            logger.info(f"[ModelRegistry] Loaded production {model_type}: {row['version']}")
            return model
        except Exception as e:
            logger.error(f"[ModelRegistry] Failed to load {model_path}: {e}")
            return None

    def get_production_version(self, model_type: str) -> Optional[str]:
        """Get the version string of the current production model."""
        row = self.db.get_production_model(model_type)
        return row["version"] if row else None

    def list_versions(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all model versions with their metadata."""
        rows = self.db.list_models(model_type, include_rolled_back=False)
        return [dict(r) for r in rows]

    def cleanup_old_versions(self, model_type: str, keep: int = 5) -> int:
        """Delete old non-production model files (keep the N most recent).

        Returns the number of files deleted.
        """
        models = self.db.list_models(model_type, include_rolled_back=True)
        # Sort by trained_at desc
        models = sorted(models, key=lambda m: m["trained_at"], reverse=True)

        deleted = 0
        for m in models[keep:]:
            if m["is_production"]:
                continue  # never delete production
            try:
                p = Path(m["model_path"])
                if p.exists():
                    p.unlink()
                    deleted += 1
            except Exception as e:
                logger.warning(f"Failed to delete {m['model_path']}: {e}")
        return deleted


# ── Singleton ────────────────────────────────────────────────────────────────
_registry: Optional[ModelRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> ModelRegistry:
    """Get the singleton ModelRegistry instance."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry()
    return _registry
