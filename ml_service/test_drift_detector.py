"""
Tests for drift_detector.py.

Run: python -m pytest tests/test_drift_detector.py -v
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    """Use a fresh temp DB for each test."""
    from db import _db_lock
    import db as db_module

    with _db_lock:
        new_db = db_module.Database(tmp_path / "test.db")
        db_module._db_singleton = new_db
        db_module._pool = new_db.pool
        db_module.db = new_db  # critical: update module-level `db` reference

    # Update references in modules that imported `db` at module level
    import drift_detector as dd_module
    monkeypatch.setattr(dd_module, "db", new_db)
    monkeypatch.setattr(dd_module, "_detector", None)

    yield

    with _db_lock:
        db_module._db_singleton = None
        db_module._pool = None
        db_module.db = None


def test_drift_detector_initialization():
    """Detector can be created."""
    from drift_detector import DriftDetector
    detector = DriftDetector()
    assert detector.fs is None
    assert detector.on_drift is None


def test_concept_drift_alert_when_accuracy_drops():
    """CRITICAL alert raised when live accuracy drops >20% below training."""
    from db import db
    from drift_detector import DriftDetector

    # Register a production model with high accuracy
    db.register_model(
        version="signal_quality_v1",
        model_type="signal_quality",
        model_path="/tmp/fake.keras",
        training_samples=1000,
        val_accuracy=0.80,
    )
    db.promote_model("signal_quality_v1")

    # Add 25 resolved predictions with poor hit rate (40%)
    for i in range(25):
        pred_id = db.log_prediction(
            ticker=f"T{i}",
            prediction_ts=f"2025-01-{i+1:02d} 10:00:00",
            model_version="signal_quality_v1",
            source="test",
            signal_quality=0.5,
            expected_return=2.0,
        )
        db.resolve_outcome(
            prediction_id=pred_id,
            actual_return=1.0,
            actual_max_gain=2.5,
            actual_max_loss=-1.5,
            hit_target=False,  # 0% hit rate
            days_to_target=None,
            predicted_return=2.0,
            predicted_drawdown=-1.5,
        )

    detector = DriftDetector()
    alerts = detector.check_concept_drift()

    # Should have raised at least one CRITICAL alert
    critical = [a for a in alerts if a["severity"] == "CRITICAL"]
    assert len(critical) >= 1


def test_concept_drift_no_alert_when_accuracy_ok():
    """No alert when live accuracy matches training accuracy."""
    from db import db
    from drift_detector import DriftDetector

    db.register_model(
        version="signal_quality_v2",
        model_type="signal_quality",
        model_path="/tmp/fake.keras",
        training_samples=1000,
        val_accuracy=0.50,
    )
    db.promote_model("signal_quality_v2")

    # Add 25 predictions with 50% hit rate (matches training)
    for i in range(25):
        pred_id = db.log_prediction(
            ticker=f"T{i}",
            prediction_ts=f"2025-01-{i+1:02d} 10:00:00",
            model_version="signal_quality_v2",
            source="test",
            expected_return=2.0,
        )
        db.resolve_outcome(
            prediction_id=pred_id,
            actual_return=1.0,
            actual_max_gain=3.5 if i % 2 == 0 else 2.0,  # 50% hit rate
            actual_max_loss=-1.5,
            hit_target=(i % 2 == 0),
            days_to_target=3 if i % 2 == 0 else None,
            predicted_return=2.0,
            predicted_drawdown=-1.5,
        )

    detector = DriftDetector()
    alerts = detector.check_concept_drift()

    # Should NOT have any critical alerts (accuracy matches)
    critical = [a for a in alerts if a["severity"] == "CRITICAL"]
    assert len(critical) == 0


def test_no_alert_when_insufficient_data():
    """No alert when fewer than MIN_PREDICTIONS_FOR_ACCURACY outcomes."""
    from db import db
    from drift_detector import DriftDetector

    db.register_model(
        version="signal_quality_v3",
        model_type="signal_quality",
        model_path="/tmp/fake.keras",
        training_samples=1000,
        val_accuracy=0.90,  # very high — would trigger if we had enough data
    )
    db.promote_model("signal_quality_v3")

    # Only 5 predictions (below threshold of 20)
    for i in range(5):
        pred_id = db.log_prediction(
            ticker=f"T{i}",
            prediction_ts=f"2025-01-{i+1:02d} 10:00:00",
            model_version="signal_quality_v3",
            source="test",
        )
        db.resolve_outcome(
            prediction_id=pred_id,
            actual_return=0,
            actual_max_gain=0,
            actual_max_loss=0,
            hit_target=False,
            days_to_target=None,
            predicted_return=None,
            predicted_drawdown=None,
        )

    detector = DriftDetector()
    alerts = detector.check_concept_drift()
    assert len(alerts) == 0


def test_data_staleness_alert():
    """WARN alert raised when ticker data is older than 48h."""
    from db import db
    from drift_detector import DriftDetector

    # Insert a ticker with old data (5 days ago)
    old_date = (datetime.utcnow() - timedelta(days=5)).strftime("%Y-%m-%d")
    db.upsert_ohlcv("STALE1", [{
        "date": old_date,
        "open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000,
    }])

    detector = DriftDetector()
    alerts = detector.check_data_staleness(["STALE1"])

    # Should have raised at least one stale alert
    stale_alerts = [a for a in alerts if a["alert_type"] == "data_stale"]
    assert len(stale_alerts) >= 1
    assert stale_alerts[0]["severity"] in ("WARN", "CRITICAL")


def test_callback_triggered_on_alert():
    """on_drift callback is called when an alert is raised."""
    from db import db
    from drift_detector import DriftDetector

    callback_calls = []

    db.register_model(
        version="signal_quality_v4",
        model_type="signal_quality",
        model_path="/tmp/fake.keras",
        training_samples=1000,
        val_accuracy=0.95,
    )
    db.promote_model("signal_quality_v4")

    for i in range(25):
        pred_id = db.log_prediction(
            ticker=f"T{i}",
            prediction_ts=f"2025-01-{i+1:02d} 10:00:00",
            model_version="signal_quality_v4",
            source="test",
        )
        db.resolve_outcome(
            prediction_id=pred_id,
            actual_return=0,
            actual_max_gain=0,
            actual_max_loss=0,
            hit_target=False,
            days_to_target=None,
            predicted_return=None,
            predicted_drawdown=None,
        )

    detector = DriftDetector(
        on_drift_callback=lambda alert: callback_calls.append(alert),
        on_critical_callback=lambda alert: callback_calls.append(alert),
    )
    detector.check_all()

    assert len(callback_calls) > 0
