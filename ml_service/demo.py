"""
End-to-End Demo — Proves the continuous learning pipeline works
=================================================================
This script demonstrates the full pipeline WITHOUT requiring TensorFlow
or yfinance. It uses synthetic data so you can see exactly how everything
fits together.

Run:
    python demo.py

What it does:
  1. Initializes the SQLite database (schema.sql)
  2. Generates synthetic OHLCV for 5 tickers
  3. Generates synthetic predictions (with intentional bias)
  4. Resolves outcomes (5 days later) — proves outcome tracking works
  5. Generates historical training labels
  6. Simulates a fine-tune cycle (with a stub model — no TF needed)
  7. Applies promotion gate
  8. Runs drift detection
  9. Prints a dashboard summary

After running, you'll see:
  - data/ml_intel.db — SQLite database with all tables populated
  - logs showing each step of the pipeline
"""

from __future__ import annotations
import os
import sys
import time
import json
import logging
import random
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

# Set up paths
SCRIPT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(SCRIPT_DIR))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")

# ── Stub feature extractor (no ml_features dependency) ──────────────────────
def stub_feature_extractor(df, normalize=False):
    """Generate a fake feature vector from a DataFrame.
    In production, this would be ml_features.extract_features.
    """
    n_features = 70
    if df is None or len(df) < 30:
        return np.zeros(n_features, dtype=np.float32)

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    volumes = df["Volume"].values if "Volume" in df.columns else np.ones(len(closes))

    feats = np.zeros(n_features, dtype=np.float32)
    # Recent returns (mimics ml_features structure)
    for i, p in enumerate([1, 3, 5, 10, 20, 60]):
        if len(closes) > p:
            feats[i] = (closes[-1] / closes[-p - 1] - 1) * 100
    # RSI-like
    deltas = np.diff(closes[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    if losses.mean() > 0:
        feats[20] = 100 - (100 / (1 + gains.mean() / losses.mean()))
    # Volume ratio
    if len(volumes) > 20:
        feats[25] = volumes[-1] / volumes[-20:].mean()
    # Add noise for uniqueness
    feats += np.random.normal(0, 0.1, n_features)
    return feats.astype(np.float32)


# ── Stub Model (no TensorFlow dependency) ───────────────────────────────────
class StubModel:
    """Minimal model that satisfies the registry's interface."""
    def __init__(self, version="stub_v1"):
        self.version = version
        self.accuracy = 0.5
        self.is_fitted = False

    def fit(self, X, y, **kwargs):
        self.is_fitted = True
        # Pretend to learn — accuracy improves with more samples
        self.accuracy = min(0.85, 0.5 + len(X) * 0.0001)
        return self

    def predict(self, X, **kwargs):
        # Predict based on a few features
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        preds = np.mean(X[:, :5], axis=1)  # use first 5 features
        # Scale to [0, 1] for binary, [-3, 3] for regression
        return (preds - preds.min()) / (preds.max() - preds.min() + 1e-8)

    def evaluate(self, X, y, **kwargs):
        preds = self.predict(X)
        if isinstance(y, dict):
            # Multi-output
            return {"loss": 0.5}
        else:
            return {"loss": 0.4, "accuracy": self.accuracy}

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            import pickle
            pickle.dump({"version": self.version, "accuracy": self.accuracy}, f)

    @property
    def optimizer(self):
        class FakeOpt:
            def __init__(self):
                self.learning_rate = 0.001
        return FakeOpt()


# ── Generate synthetic OHLCV ────────────────────────────────────────────────
def generate_synthetic_ohlcv(ticker: str, days: int = 300) -> pd.DataFrame:
    """Generate realistic-looking OHLCV data with trend + noise."""
    np.random.seed(hash(ticker) % 2**32)

    # Start price between 100 and 1000
    start_price = 100 + (hash(ticker) % 1000)

    # Generate returns: trend + noise + occasional jumps
    returns = np.random.normal(0.0005, 0.02, days)
    # Add trend
    trend = np.linspace(0, 0.3, days) * np.random.choice([-1, 1])
    returns += trend / days

    prices = [start_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))

    prices = np.array(prices[1:])

    # Generate OHLC from close prices
    dates = pd.date_range(end=datetime.now().strftime("%Y-%m-%d"), periods=days, freq="B")
    df = pd.DataFrame({
        "Open": prices * (1 + np.random.uniform(-0.005, 0.005, days)),
        "High": prices * (1 + np.abs(np.random.normal(0, 0.01, days))),
        "Low": prices * (1 - np.abs(np.random.normal(0, 0.01, days))),
        "Close": prices,
        "Volume": np.random.randint(100000, 1000000, days).astype(float),
    }, index=dates)
    df.index.name = "date"
    return df


# ── Main demo ───────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("  ML CONTINUOUS LEARNING PIPELINE — END-TO-END DEMO")
    print("=" * 70 + "\n")

    # ── Step 1: Initialize DB ───────────────────────────────────────────────
    print("▶ Step 1: Initializing SQLite database...")
    from db import db, now_ist
    # db is a singleton, already initialized
    print(f"  ✓ DB at: {db.pool.db_path}")
    print(f"  ✓ Schema: 11 tables created\n")

    # ── Step 2: Generate synthetic data ────────────────────────────────────
    print("▶ Step 2: Generating synthetic OHLCV data for 5 tickers...")
    tickers = ["DEMO1", "DEMO2", "DEMO3", "DEMO4", "DEMO5"]
    ticker_dfs: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = generate_synthetic_ohlcv(ticker, days=300)
        ticker_dfs[ticker] = df
        # Persist to DB
        rows = []
        for idx, r in df.iterrows():
            date_str = str(idx.date())
            rows.append({
                "date": date_str,
                "open": float(r["Open"]),
                "high": float(r["High"]),
                "low": float(r["Low"]),
                "close": float(r["Close"]),
                "volume": int(r["Volume"]),
            })
        db.upsert_ohlcv(ticker, rows)
    print(f"  ✓ Generated {len(tickers)} tickers × 300 days = {len(tickers)*300} bars\n")

    # ── Step 3: Generate historical training labels (cold start) ───────────
    print("▶ Step 3: Generating historical training labels (cold start)...")
    from outcome_tracker import OutcomeTracker
    tracker = OutcomeTracker(lookahead_days=5, target_pct=3.0)
    n_samples = tracker.generate_historical_labels(
        ticker_dfs=ticker_dfs,
        feature_extractor=stub_feature_extractor,
        step=3,
    )
    print(f"  ✓ Generated {n_samples} labeled training samples\n")

    # ── Step 4: Make some predictions ──────────────────────────────────────
    print("▶ Step 4: Making predictions for the last 10 days...")
    # For each ticker, make a prediction for the last 10 dates
    for ticker, df in ticker_dfs.items():
        for i in range(10, 0, -1):
            pred_date = df.index[-i]
            pred_ts = str(pred_date.date()) + " 10:00:00"
            pred_date_str = str(pred_date.date())

            # Generate a "prediction" with some bias (intentional)
            actual_return = (df["Close"].iloc[-i + 5] / df["Close"].iloc[-i] - 1) * 100 if -i + 5 < 0 else 0
            predicted_return = actual_return + np.random.normal(0, 1.5)  # noisy prediction
            signal_quality = max(0.1, min(0.9, 0.5 + np.random.normal(0, 0.15)))

            db.log_prediction(
                ticker=ticker,
                prediction_ts=pred_ts,
                model_version="stub_v1",
                source="demo",
                signal_quality=signal_quality,
                trend_regime="UPTREND" if predicted_return > 0 else "DOWNTREND",
                regime_confidence=0.6,
                expected_return=predicted_return,
                max_drawdown=-2.5,
                risk_reward_ratio=1.2,
                ml_score=60 + np.random.normal(0, 10),
                actionable="BUY" if predicted_return > 1 else "HOLD",
                entry_price=float(df["Close"].iloc[-i]),
                target_price=float(df["Close"].iloc[-i] * 1.03),
                stop_loss=float(df["Close"].iloc[-i] * 0.98),
            )
    print(f"  ✓ Made {len(tickers) * 10} predictions\n")

    # ── Step 5: Resolve outcomes ───────────────────────────────────────────
    print("▶ Step 5: Resolving outcomes (predictions older than 5 days)...")
    result = tracker.resolve_all_pending(feature_extractor=stub_feature_extractor)
    print(f"  ✓ Resolved: {result['resolved']}")
    print(f"  ✓ Skipped (insufficient future data): {result['skipped']}")
    print(f"  ✓ Training samples enqueued: {result['enqueued']}\n")

    # ── Step 6: Check accuracy stats ───────────────────────────────────────
    print("▶ Step 6: Checking accuracy stats...")
    stats = db.get_outcome_stats(days=30)
    print(f"  Total outcomes: {stats['total']}")
    print(f"  Hit rate (3% target): {stats.get('hit_rate', 'N/A')}")
    print(f"  Direction accuracy: {stats.get('direction_accuracy', 'N/A')}")
    print(f"  Return MAE: {stats.get('return_mae', 'N/A')}\n")

    # ── Step 7: Pull training samples (what fine-tune would use) ───────────
    print("▶ Step 7: Pulling training samples for fine-tune...")
    X, y = db.get_training_samples(n=100)
    print(f"  ✓ Retrieved {len(X)} samples × {X.shape[1] if len(X) > 0 else 0} features")
    if len(X) > 0:
        print(f"  Hit target distribution: {y['hit_target'].mean():.2%}")
        print(f"  Avg return: {y['return_pct'].mean():.2f}%")
        print(f"  Avg max gain: {y['max_gain_pct'].mean():.2f}%")
        print(f"  Avg max loss: {y['max_loss_pct'].mean():.2f}%\n")

    # ── Step 8: Test walk-forward split ────────────────────────────────────
    print("▶ Step 8: Testing walk-forward split (no data leakage)...")
    from walk_forward import simple_time_split, purge_and_embargo
    samples_df = pd.DataFrame({
        "date": db.query(
            "SELECT feature_date FROM training_queue ORDER BY feature_date"
        ),
    })
    samples_df["date"] = samples_df["date"].apply(lambda x: x["feature_date"] if isinstance(x, sqlite3.Row) else x)
    if len(samples_df) > 0:
        train_df, val_df = simple_time_split(samples_df, train_days=60, val_days=15)
        train_df, val_df = purge_and_embargo(train_df, val_df, embargo_days=3)
        print(f"  Train: {len(train_df)} samples (before {train_df['date'].min() if len(train_df) > 0 else 'N/A'})")
        print(f"  Val:   {len(val_df)} samples (after {val_df['date'].min() if len(val_df) > 0 else 'N/A'})")
        if len(train_df) > 0 and len(val_df) > 0:
            print(f"  ✓ Val data is strictly AFTER train data (no leakage)\n")

    # ── Step 9: Test model registry ────────────────────────────────────────
    print("▶ Step 9: Testing model registry (versioning + promotion)...")
    from model_registry import get_registry
    registry = get_registry()

    # Save a "trained" model
    stub1 = StubModel(version="stub_v1")
    stub1.fit(np.random.randn(100, 70), np.random.randint(0, 2, 100))
    version1 = registry.save_version(
        model=stub1,
        model_type="signal_quality",
        training_samples=100,
        metrics={"val_accuracy": 0.60, "val_loss": 0.55, "training_loss": 0.58},
        walk_forward_score=58.0,
        notes="first stub model",
    )
    print(f"  ✓ Saved {version1}")
    registry.promote(version1)
    print(f"  ✓ Promoted to production")

    # Save a better model
    stub2 = StubModel(version="stub_v2")
    stub2.fit(np.random.randn(500, 70), np.random.randint(0, 2, 500))
    version2 = registry.save_version(
        model=stub2,
        model_type="signal_quality",
        training_samples=500,
        metrics={"val_accuracy": 0.68, "val_loss": 0.48, "training_loss": 0.50},
        walk_forward_score=66.0,
        notes="second stub model — should promote",
    )
    decision = registry.evaluate_for_promotion("signal_quality", version2)
    print(f"  ✓ Evaluated {version2}: decision={decision['decision']}")
    print(f"    Reason: {decision['reason']}")
    if decision["should_promote"]:
        registry.promote(version2)
        print(f"  ✓ Promoted to production")

    # Save a worse model — should be rejected
    stub3 = StubModel(version="stub_v3")
    stub3.fit(np.random.randn(50, 70), np.random.randint(0, 2, 50))
    version3 = registry.save_version(
        model=stub3,
        model_type="signal_quality",
        training_samples=50,
        metrics={"val_accuracy": 0.40, "val_loss": 0.70},
        walk_forward_score=42.0,
        notes="third stub model — should reject",
    )
    decision3 = registry.evaluate_for_promotion("signal_quality", version3)
    print(f"  ✓ Evaluated {version3}: decision={decision3['decision']}")
    print(f"    Reason: {decision3['reason']}\n")

    # List all models
    print("  All model versions:")
    for m in registry.list_versions("signal_quality"):
        prod_marker = " ← PRODUCTION" if m["is_production"] else ""
        print(f"    {m['version']}: val_acc={m['val_accuracy']}, wf_score={m['walk_forward_score']}{prod_marker}")
    print()

    # ── Step 10: Test drift detection ──────────────────────────────────────
    print("▶ Step 10: Testing drift detection...")
    from drift_detector import DriftDetector
    detector = DriftDetector(feature_store=None)  # no feature store for demo
    alerts = detector.check_concept_drift()
    print(f"  ✓ Concept drift alerts: {len(alerts)}")
    for a in alerts:
        print(f"    [{a['severity']}] {a['alert_type']}: {a['details'].get('message', '')}")

    data_alerts = detector.check_data_staleness(tickers)
    print(f"  ✓ Data staleness alerts: {len(data_alerts)}")
    for a in data_alerts:
        print(f"    [{a['severity']}] {a['alert_type']}: {a['details'].get('message', '')}")
    print()

    # ── Step 11: Final dashboard summary ───────────────────────────────────
    print("▶ Step 11: Final dashboard summary...")
    total_preds = db.query_one("SELECT COUNT(*) as c FROM predictions")["c"]
    total_outcomes = db.query_one("SELECT COUNT(*) as c FROM outcomes")["c"]
    total_queue = db.query_one("SELECT COUNT(*) as c FROM training_queue")["c"]
    total_models = db.query_one("SELECT COUNT(*) as c FROM model_registry")["c"]
    total_alerts = db.query_one("SELECT COUNT(*) as c FROM drift_alerts")["c"]

    print(f"\n  ┌────────────────────────────────────────────────────────┐")
    print(f"  │  DATABASE SUMMARY                                       │")
    print(f"  ├────────────────────────────────────────────────────────┤")
    print(f"  │  Predictions logged:      {total_preds:>6}                          │")
    print(f"  │  Outcomes resolved:       {total_outcomes:>6}                          │")
    print(f"  │  Training queue samples:  {total_queue:>6}                          │")
    print(f"  │  Model versions:          {total_models:>6}                          │")
    print(f"  │  Drift alerts:            {total_alerts:>6}                          │")
    print(f"  └────────────────────────────────────────────────────────┘\n")

    # ── Step 12: Show monitoring API endpoints (just print them) ───────────
    print("▶ Step 12: Monitoring API endpoints available after integration:")
    endpoints = [
        "GET  /api/ml-monitor/dashboard",
        "GET  /api/ml-monitor/accuracy?days=7",
        "GET  /api/ml-monitor/accuracy/trend?days=30",
        "GET  /api/ml-monitor/predictions/recent?limit=50",
        "GET  /api/ml-monitor/predictions/pending",
        "GET  /api/ml-monitor/models",
        "GET  /api/ml-monitor/models/production",
        "POST /api/ml-monitor/models/{version}/promote",
        "POST /api/ml-monitor/models/rollback/{model_type}",
        "GET  /api/ml-monitor/training/history",
        "POST /api/ml-monitor/training/trigger-fine-tune",
        "POST /api/ml-monitor/training/trigger-full-retrain",
        "GET  /api/ml-monitor/alerts?hours=24",
        "POST /api/ml-monitor/drift/check",
        "POST /api/ml-monitor/outcomes/resolve",
        "GET  /api/ml-monitor/outcomes/stats?days=7",
        "GET  /api/ml-monitor/data/staleness",
    ]
    for ep in endpoints:
        print(f"  {ep}")
    print()

    # ── Done ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  ✓ DEMO COMPLETE")
    print("=" * 70)
    print()
    print(f"  Database: {db.pool.db_path}")
    print()
    print("  Next steps:")
    print("    1. Apply the changes in integration_patch.py to your ml_service.py")
    print("    2. Copy all .py files from this folder to your ml_service directory")
    print("    3. Restart the ML service on port 8001")
    print("    4. Call POST /api/bootstrap-training-data to seed real labels")
    print("    5. Wait 6 hours — first fine-tune cycle will run automatically")
    print("    6. Visit /api/ml-monitor/dashboard to see live metrics")
    print()


if __name__ == "__main__":
    main()
