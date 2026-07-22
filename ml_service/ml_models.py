"""
TensorFlow Model Suite for Stock Intelligence
===============================================
Four specialized models that learn from the same feature vector:

  1. SignalQualityModel     — P(hit 3% target | current features) → [0, 1]
  2. TrendRegimeClassifier  — UPTREND / DOWNTREND / SIDEWAYS probabilities
  3. ReturnForecaster        — Expected return % in next 5 days (regression)
  4. RiskEstimator          — Max drawdown % in next 5 days (regression)

Training strategy:
  - Initial training on historical OHLCV data (yfinance 10y)
  - Continuous retraining: every 30 min during market hours, collect new
    (features, outcome) pairs from the latest bars and fine-tune.
  - Models are saved/loaded from disk so they survive restarts.
  - Early stopping + learning rate scheduling prevent overfitting.
"""

from __future__ import annotations
import os
import json
import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, regularizers

try:
    from .ml_features import extract_features_for_training, update_feature_stats, FEATURE_NAMES, FEATURE_STATS, extract_features
except ImportError:
    from ml_features import extract_features_for_training, update_feature_stats, FEATURE_NAMES, FEATURE_STATS, extract_features



logger = logging.getLogger("ml_models")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Import feature names (deferred to avoid circular imports at module level)
def _get_num_features():
    from ml_features import FEATURE_NAMES
    return len(FEATURE_NAMES)

# All model builders use a callable to get num features (resolved at build time, not import time)
_NUM_FEATS = None

def _nf():
    global _NUM_FEATS
    if _NUM_FEATS is None:
        from ml_features import FEATURE_NAMES
        _NUM_FEATS = len(FEATURE_NAMES)
    return _NUM_FEATS


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 1: SIGNAL QUALITY PREDICTOR
#  Binary classification: Will this stock hit +3% within 5 days?
# ═══════════════════════════════════════════════════════════════════════════

def build_signal_quality_model() -> keras.Model:
    """
    Deep neural network for predicting signal quality.
    Architecture: Dense → BatchNorm → Dropout → Dense → ... → Sigmoid

    Uses:
    - BatchNorm for stable training
    - Dropout for regularization
    - Residual connections for gradient flow
    - Class weights to handle imbalanced data (few hits vs many misses)
    """
    nf = _nf()
    inputs = keras.Input(shape=(nf,), name="features")

    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Residual block 1 (128 -> 128)
    residual = x
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Add()([x, residual])
    x = layers.Activation("relu")(x)

    # Residual block 2 (128 -> 64 with projection)
    residual = layers.Dense(64)(x)  # projection to match dimensions
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Add()([x, residual])
    x = layers.Activation("relu")(x)

    # Output head
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    output = layers.Dense(1, activation="sigmoid", name="hit_probability")(x)

    model = keras.Model(inputs=inputs, outputs=output, name="SignalQualityPredictor")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 2: TREND REGIME CLASSIFIER
#  3-class: UPTREND (0) / SIDEWAYS (1) / DOWNTREND (2)
# ═══════════════════════════════════════════════════════════════════════════

def build_trend_regime_model() -> keras.Model:
    """
    Classifies the current market regime based on features.
    Uses the same feature vector but predicts a categorical outcome.
    """
    nf = _nf()
    inputs = keras.Input(shape=(nf,), name="features")

    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(96, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(32, activation="relu")(x)
    output = layers.Dense(3, activation="softmax", name="regime_probs")(x)

    model = keras.Model(inputs=inputs, outputs=output, name="TrendRegimeClassifier")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 3: RETURN FORECASTER
#  Regression: Expected return % in next 5 days
# ═══════════════════════════════════════════════════════════════════════════

def build_return_forecaster_model() -> keras.Model:
    """
    Predicts the expected return percentage over the next N days.
    Multi-output: return_pct + max_gain_pct + max_loss_pct
    """
    nf = _nf()
    inputs = keras.Input(shape=(nf,), name="features")

    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(96, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)

    # Three output heads
    return_out = layers.Dense(1, name="return_pct")(x)
    max_gain_out = layers.Dense(1, name="max_gain_pct")(x)
    max_loss_out = layers.Dense(1, name="max_loss_pct")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=[return_out, max_gain_out, max_loss_out],
        name="ReturnForecaster",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            "return_pct": "mse",
            "max_gain_pct": "mse",
            "max_loss_pct": "mse",
        },
        loss_weights={
            "return_pct": 1.0,
            "max_gain_pct": 0.5,
            "max_loss_pct": 0.5,
        },
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL 4: RISK ESTIMATOR
#  Regression: Max drawdown % expected in next 5 days
# ═══════════════════════════════════════════════════════════════════════════

def build_risk_estimator_model() -> keras.Model:
    """
    Predicts the maximum drawdown (worst-case loss) expected in the next N days.
    Separate from ReturnForecaster because risk and return need different
    feature emphasis (risk focuses more on volatility and support proximity).
    """
    nf = _nf()
    inputs = keras.Input(shape=(nf,), name="features")

    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(96, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Use leaky relu for the final layers (drawdowns are non-positive)
    x = layers.Dense(32, activation="leaky_relu")(x)
    output = layers.Dense(1, name="max_drawdown_pct")(x)

    model = keras.Model(inputs=inputs, outputs=output, name="RiskEstimator")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL MANAGER — Orchestrates all 4 models
# ═══════════════════════════════════════════════════════════════════════════

class ModelManager:
    """
    Manages the lifecycle of all 4 TensorFlow models:
    - Build / load from disk
    - Initial training on historical data
    - Continuous fine-tuning with new data
    - Prediction API
    - Save / load state
    """

    MODEL_PATHS = {
        "signal_quality": os.path.join(MODEL_DIR, "signal_quality.keras"),
        "trend_regime": os.path.join(MODEL_DIR, "trend_regime.keras"),
        "return_forecaster": os.path.join(MODEL_DIR, "return_forecaster.keras"),
        "risk_estimator": os.path.join(MODEL_DIR, "risk_estimator.keras"),
    }

    def __init__(self):
        self.models: Dict[str, Optional[keras.Model]] = {
            "signal_quality": None,
            "trend_regime": None,
            "return_forecaster": None,
            "risk_estimator": None,
        }
        self.training_history: Dict[str, List] = {
            "signal_quality": [],
            "trend_regime": [],
            "return_forecaster": [],
            "risk_estimator": [],
        }
        self.is_training: bool = False
        self.last_trained: float = 0.0
        self.total_training_samples: int = 0
        self._load_models()

    def _load_models(self):
        """Load saved models from disk (if they exist)."""
        for name, path in self.MODEL_PATHS.items():
            if os.path.exists(path):
                try:
                    self.models[name] = keras.models.load_model(path)
                    logger.info(f"[ModelManager] Loaded {name} from {path}")
                except Exception as e:
                    logger.warning(f"[ModelManager] Failed to load {name}: {e}")
            else:
                logger.info(f"[ModelManager] No saved model for {name} — will train from scratch")

    def _save_models(self):
        """Save all trained models to disk."""
        for name, model in self.models.items():
            if model is not None:
                try:
                    model.save(self.MODEL_PATHS[name])
                    logger.info(f"[ModelManager] Saved {name}")
                except Exception as e:
                    logger.error(f"[ModelManager] Failed to save {name}: {e}")

    def _build_missing_models(self):
        """Build any models that haven't been loaded or created yet."""
        builders = {
            "signal_quality": build_signal_quality_model,
            "trend_regime": build_trend_regime_model,
            "return_forecaster": build_return_forecaster_model,
            "risk_estimator": build_risk_estimator_model,
        }
        for name, builder in builders.items():
            if self.models[name] is None:
                self.models[name] = builder()
                logger.info(f"[ModelManager] Built new {name}")

    # ─────────────────────────────────────────────────────────────────
    #  INITIAL TRAINING (on historical data)
    # ─────────────────────────────────────────────────────────────────

    def train_initial(self, ticker_dfs: Dict[str, pd.DataFrame], target_pct: float = 3.0):
        """
        Initial full training on historical OHLCV data for multiple tickers.
        This is the "cold start" — run once when the service starts with no
        saved models, or when the user explicitly triggers retraining.

        Parameters
        ----------
        ticker_dfs : dict of {ticker: DataFrame}
            Historical OHLCV data from yfinance (same format as ema9_router).
        target_pct : float
            The target gain % for the signal quality label (default 3%).
        """
        import pandas as pd

        self.is_training = True
        logger.info(f"[ModelManager] Starting initial training on {len(ticker_dfs)} tickers...")

        # Collect all (features, labels) pairs across all tickers
        all_features = []
        all_hit_target = []
        all_regimes = []
        all_returns = []
        all_max_gains = []
        all_max_losses = []

        processed = 0
        for ticker, df in ticker_dfs.items():
            if df is None or len(df) < 200:
                continue

            feature_matrix, labels = extract_features_for_training(
                df, lookahead_days=5, target_pct=target_pct
            )
            if feature_matrix is None or labels is None:
                continue

            all_features.append(feature_matrix)
            for label_dict in labels:
                all_hit_target.append(label_dict["hit_target"])
                all_returns.append(label_dict["return_pct"])
                all_max_gains.append(label_dict["max_gain_pct"])
                all_max_losses.append(label_dict["max_loss_pct"])

                # Compute regime label from the label data
                if label_dict["return_pct"] > 1.0:
                    all_regimes.append(0)  # UPTREND
                elif label_dict["return_pct"] < -1.0:
                    all_regimes.append(2)  # DOWNTREND
                else:
                    all_regimes.append(1)  # SIDEWAYS

            processed += 1
            if processed % 20 == 0:
                logger.info(f"  Processed {processed}/{len(ticker_dfs)} tickers...")

        if not all_features:
            logger.warning("[ModelManager] No training data collected. Skipping training.")
            self.is_training = False
            return

        # Stack into arrays
        X = np.concatenate(all_features, axis=0).astype(np.float32)
        y_hit = np.array(all_hit_target, dtype=np.float32)
        y_regime = np.array(all_regimes, dtype=np.int32)
        y_return = np.array(all_returns, dtype=np.float32).reshape(-1, 1)
        y_max_gain = np.array(all_max_gains, dtype=np.float32).reshape(-1, 1)
        y_max_loss = np.array(all_max_losses, dtype=np.float32).reshape(-1, 1)

        # Update normalization stats from training data
        update_feature_stats(X)

        # Re-extract with normalization (or normalize in-place)
        from ml_features import FEATURE_STATS as fs
        for i in range(X.shape[1]):
            name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat_{i}"
            if name in fs and fs[name]["std"] > 1e-8:
                X[:, i] = (X[:, i] - fs[name]["mean"]) / fs[name]["std"]

        self.total_training_samples = len(X)
        logger.info(f"[ModelManager] Training data: {X.shape[0]} samples x {X.shape[1]} features")
        logger.info(f"  Hit target rate: {y_hit.mean() * 100:.1f}%")
        logger.info(f"  Regime distribution: UPTREND={(y_regime == 0).sum()}, SIDEWAYS={(y_regime == 1).sum()}, DOWNTREND={(y_regime == 2).sum()}")

        # Train each model
        self._build_missing_models()

        # ── Signal Quality ──
        self._train_model(
            "signal_quality",
            X, y_hit,
            epochs=30, batch_size=256,
            class_weight={0: 1.0, 1: max(1.0, (1 - y_hit.mean()) / (y_hit.mean() + 1e-8) * 0.5)},
        )

        # ── Trend Regime ──
        self._train_model(
            "trend_regime",
            X, y_regime,
            epochs=30, batch_size=256,
        )

        # ── Return Forecaster ──
        if self.models["return_forecaster"] is not None:
            es = callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
            lr_sched = callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=1e-6)
            history = self.models["return_forecaster"].fit(
                X,
                {"return_pct": y_return, "max_gain_pct": y_max_gain, "max_loss_pct": y_max_loss},
                epochs=30, batch_size=256, verbose=0,
                validation_split=0.1,
                callbacks=[es, lr_sched],
            )
            self.training_history["return_forecaster"] = history.history.get("loss", [])

        # ── Risk Estimator ──
        self._train_model(
            "risk_estimator",
            X, y_max_loss,
            epochs=30, batch_size=256,
        )

        self._save_models()
        self.last_trained = time.time()
        self.is_training = False
        logger.info(f"[ModelManager] Initial training complete. Models saved.")

    def _train_model(self, name: str, X: np.ndarray, y: np.ndarray,
                     epochs: int = 30, batch_size: int = 256, **kwargs):
        """Train a single model with early stopping and LR scheduling."""
        model = self.models.get(name)
        if model is None:
            return

        es = callbacks.EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        lr_sched = callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=1e-6)

        try:
            history = model.fit(
                X, y,
                epochs=epochs, batch_size=batch_size, verbose=0,
                validation_split=0.1,
                callbacks=[es, lr_sched],
                **kwargs,
            )
            self.training_history[name] = history.history.get("loss", [])
            logger.info(f"[ModelManager] {name}: final loss = {history.history['loss'][-1]:.4f}")
        except Exception as e:
            logger.error(f"[ModelManager] {name} training failed: {e}")

    # ─────────────────────────────────────────────────────────────────
    #  CONTINUOUS FINE-TUNING (incremental learning)
    # ─────────────────────────────────────────────────────────────────

    def fine_tune(self, ticker_dfs: Dict[str, pd.DataFrame], target_pct: float = 3.0,
                  epochs: int = 5):
        """
        Fine-tune existing models with new data. Called periodically by
        the background scheduler. Uses a small learning rate to avoid
        catastrophic forgetting.
        """
        if self.is_training:
            logger.info("[ModelManager] Skipping fine-tune — already training")
            return

        any_trained = any(m is not None for m in self.models.values())
        if not any_trained:
            logger.info("[ModelManager] No trained models yet — running initial training instead")
            self.train_initial(ticker_dfs, target_pct)
            return

        self.is_training = True
        logger.info(f"[ModelManager] Fine-tuning on {len(ticker_dfs)} tickers...")

        # Collect data (same as train_initial but fewer samples)
        all_features = []
        all_hit_target = []
        all_regimes = []
        all_returns = []
        all_max_gains = []
        all_max_losses = []

        for ticker, df in ticker_dfs.items():
            if df is None or len(df) < 200:
                continue
            feature_matrix, labels = extract_features_for_training(
                df, lookahead_days=5, target_pct=target_pct
            )
            if feature_matrix is None or labels is None:
                continue
            all_features.append(feature_matrix)
            for label_dict in labels:
                all_hit_target.append(label_dict["hit_target"])
                all_returns.append(label_dict["return_pct"])
                all_max_gains.append(label_dict["max_gain_pct"])
                all_max_losses.append(label_dict["max_loss_pct"])
                if label_dict["return_pct"] > 1.0:
                    all_regimes.append(0)
                elif label_dict["return_pct"] < -1.0:
                    all_regimes.append(2)
                else:
                    all_regimes.append(1)

        if not all_features:
            self.is_training = False
            return

        X = np.concatenate(all_features, axis=0).astype(np.float32)
        y_hit = np.array(all_hit_target, dtype=np.float32)
        y_regime = np.array(all_regimes, dtype=np.int32)
        y_return = np.array(all_returns, dtype=np.float32).reshape(-1, 1)
        y_max_gain = np.array(all_max_gains, dtype=np.float32).reshape(-1, 1)
        y_max_loss = np.array(all_max_losses, dtype=np.float32).reshape(-1, 1)

        # Normalize using existing stats
        from ml_features import FEATURE_STATS as fs
        for i in range(X.shape[1]):
            name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat_{i}"
            if name in fs and fs[name]["std"] > 1e-8:
                X[:, i] = (X[:, i] - fs[name]["mean"]) / fs[name]["std"]

        self.total_training_samples += len(X)

        # Fine-tune with lower learning rate
        for name, model in self.models.items():
            if model is None:
                continue
            keras.backend.set_value(model.optimizer.learning_rate, 1e-4)

        self._train_model("signal_quality", X, y_hit, epochs=epochs, batch_size=128,
                          class_weight={0: 1.0, 1: max(1.0, (1 - y_hit.mean()) / (y_hit.mean() + 1e-8) * 0.5)})
        self._train_model("trend_regime", X, y_regime, epochs=epochs, batch_size=128)

        if self.models["return_forecaster"] is not None:
            self.models["return_forecaster"].fit(
                X,
                {"return_pct": y_return, "max_gain_pct": y_max_gain, "max_loss_pct": y_max_loss},
                epochs=epochs, batch_size=128, verbose=0,
                validation_split=0.1,
            )

        self._train_model("risk_estimator", X, y_max_loss, epochs=epochs, batch_size=128)

        self._save_models()
        self.last_trained = time.time()
        self.is_training = False
        logger.info(f"[ModelManager] Fine-tuning complete.")

    # ─────────────────────────────────────────────────────────────────
    #  PREDICTION API
    # ─────────────────────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> Dict:
        """
        Run all 4 models on a single ticker's current data.
        Returns a dict with predictions from each model.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data (at least 200 rows for reliable features).

        Returns
        -------
        dict with keys:
            signal_quality    : float [0-1] — probability of hitting +3% target
            trend_regime      : str — "UPTREND" / "SIDEWAYS" / "DOWNTREND"
            regime_confidence : float [0-1] — confidence in regime prediction
            expected_return   : float — expected return % in next 5 days
            max_gain_expected : float — max expected gain % in next 5 days
            max_drawdown      : float — max expected drawdown % in next 5 days
            risk_reward_ratio : float — expected_return / abs(max_drawdown)
            ml_score          : float [0-100] — composite ML confidence score
            actionable        : str — "STRONG_BUY" / "BUY" / "HOLD" / "SELL" / "STRONG_SELL"
            model_status      : str — "trained" / "untrained"
        """
        any_trained = any(m is not None for m in self.models.values())
        if not any_trained:
            return {
                "signal_quality": 0.5, "trend_regime": "SIDEWAYS",
                "regime_confidence": 0.0, "expected_return": 0.0,
                "max_gain_expected": 0.0, "max_drawdown": 0.0,
                "risk_reward_ratio": 0.0, "ml_score": 0.0,
                "actionable": "HOLD", "model_status": "untrained",
            }

        features = extract_features(df, normalize=True)
        X = features.reshape(1, -1).astype(np.float32)

        result = {"model_status": "trained"}

        # Model 1: Signal Quality
        if self.models["signal_quality"] is not None:
            try:
                prob = float(self.models["signal_quality"].predict(X, verbose=0)[0][0])
                result["signal_quality"] = round(prob, 4)
            except Exception as e:
                logger.error(f"Signal quality prediction failed: {e}")
                result["signal_quality"] = 0.5
        else:
            result["signal_quality"] = 0.5

        # Model 2: Trend Regime
        regime_names = ["UPTREND", "SIDEWAYS", "DOWNTREND"]
        if self.models["trend_regime"] is not None:
            try:
                probs = self.models["trend_regime"].predict(X, verbose=0)[0]
                regime_idx = int(np.argmax(probs))
                result["trend_regime"] = regime_names[regime_idx]
                result["regime_confidence"] = round(float(probs[regime_idx]), 4)
                result["regime_probs"] = {
                    name: round(float(p), 4) for name, p in zip(regime_names, probs)
                }
            except Exception as e:
                logger.error(f"Trend regime prediction failed: {e}")
                result["trend_regime"] = "SIDEWAYS"
                result["regime_confidence"] = 0.0
        else:
            result["trend_regime"] = "SIDEWAYS"
            result["regime_confidence"] = 0.0

        # Model 3: Return Forecaster
        if self.models["return_forecaster"] is not None:
            try:
                preds = self.models["return_forecaster"].predict(X, verbose=0)
                # predict() returns list of arrays (one per output head), each shape (batch, 1)
                result["expected_return"] = round(float(np.squeeze(preds[0])), 2)
                result["max_gain_expected"] = round(float(np.squeeze(preds[1])), 2)
            except Exception as e:
                logger.error(f"Return forecaster prediction failed: {e}")
                result["expected_return"] = 0.0
                result["max_gain_expected"] = 0.0
        else:
            result["expected_return"] = 0.0
            result["max_gain_expected"] = 0.0

        # Model 4: Risk Estimator
        if self.models["risk_estimator"] is not None:
            try:
                dd = float(np.squeeze(self.models["risk_estimator"].predict(X, verbose=0)))
                result["max_drawdown"] = round(dd, 2)
            except Exception as e:
                logger.error(f"Risk estimator prediction failed: {e}")
                result["max_drawdown"] = 0.0
        else:
            result["max_drawdown"] = 0.0

        # ── Composite ML Score ──
        # Weighted combination: signal_quality × 40 + regime × 25 + return × 20 + risk × 15
        sq = result["signal_quality"]
        rc = result["regime_confidence"]
        er = result["expected_return"]
        dd = result["max_drawdown"]

        # Regime score: uptrend=1, sideways=0.5, downtrend=0
        regime_score = {"UPTREND": 1.0, "SIDEWAYS": 0.5, "DOWNTREND": 0.0}.get(result["trend_regime"], 0.5)

        # Return score: normalize to [0, 1] using sigmoid
        return_score = 1.0 / (1.0 + np.exp(-er))  # sigmoid

        # Risk score: lower drawdown = higher score
        risk_score = 1.0 / (1.0 + np.exp(dd + 2))  # shifted so 0% dd ≈ 0.88

        ml_score = (sq * 0.35 + regime_score * rc * 0.25 +
                    return_score * 0.25 + risk_score * 0.15) * 100
        result["ml_score"] = round(ml_score, 1)

        # Risk/Reward ratio
        if abs(dd) > 0.1:
            result["risk_reward_ratio"] = round(er / abs(dd), 2)
        else:
            result["risk_reward_ratio"] = 99.9 if er > 0 else 0.0

        # ── Actionable Decision ──
        # Based on your EMA9 strategy context:
        #   - Strong Buy: ML score > 70, signal quality > 0.6, UPTREND, R:R > 1.5
        #   - Buy: ML score > 55, signal quality > 0.45
        #   - Hold: ML score 40-55
        #   - Sell: ML score < 40, DOWNTREND
        #   - Strong Sell: ML score < 25, signal quality < 0.3, DOWNTREND
        if ml_score > 70 and sq > 0.6 and result["trend_regime"] == "UPTREND":
            result["actionable"] = "STRONG_BUY"
        elif ml_score > 55 and sq > 0.45:
            result["actionable"] = "BUY"
        elif ml_score >= 40:
            result["actionable"] = "HOLD"
        elif ml_score < 25 and sq < 0.3 and result["trend_regime"] == "DOWNTREND":
            result["actionable"] = "STRONG_SELL"
        elif ml_score < 40:
            result["actionable"] = "SELL"
        else:
            result["actionable"] = "HOLD"

        return result

    def predict_batch(self, ticker_dfs: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Predict for multiple tickers. Returns {ticker: prediction_dict}."""
        results = {}
        for ticker, df in ticker_dfs.items():
            if df is not None and len(df) >= 200:
                try:
                    results[ticker] = self.predict(df)
                except Exception as e:
                    logger.error(f"Batch predict failed for {ticker}: {e}")
        return results

    # ─────────────────────────────────────────────────────────────────
    #  STATUS
    # ─────────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Return training status and model info."""
        from ml_features import FEATURE_NAMES
        return {
            "models_trained": {name: (model is not None) for name, model in self.models.items()},
            "is_training": self.is_training,
            "last_trained": self.last_trained,
            "total_training_samples": self.total_training_samples,
            "num_features": _nf(),
            "feature_names_count": len(FEATURE_NAMES),
            "training_history_samples": {
                name: len(h) for name, h in self.training_history.items()
            },
        }


# ── Global singleton ──
model_manager = ModelManager()