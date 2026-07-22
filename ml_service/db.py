"""
Database Layer for ML Continuous Learning
==========================================
Replaces JSON file storage (paper_trades.json, ml_decisions.json,
chartink_signals.json) with SQLite. Provides:

  1. Thread-safe connection pooling (FastAPI runs in threadpool)
  2. Schema migration on first use
  3. High-level helpers for predictions/outcomes/training queue
  4. JSON serialization helpers for numpy arrays

Usage:
    from db import db
    db.init()                          # creates tables if missing
    db.log_prediction(...)             # records a prediction
    db.resolve_outcomes()              # resolves 5-day-old predictions
    db.get_training_samples(n=5000)    # pulls labeled samples for fine-tuning
"""

from __future__ import annotations
import os
import json
import pickle
import sqlite3
import threading
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple, Iterable
from contextlib import contextmanager

import numpy as np

logger = logging.getLogger("ml_db")

# ── Configuration ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DEFAULT_DB_PATH = BASE_DIR / "data" / "ml_intel.db"
SCHEMA_PATH = BASE_DIR / "schema.sql"

# IST timezone (no external dep — used only for display strings)
IST_OFFSET_HOURS = 5.5


def now_ist() -> str:
    """Return current timestamp in IST as ISO-8601 string."""
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=IST_OFFSET_HOURS)
    return ist_now.strftime("%Y-%m-%d %H:%M:%S")


def today_ist() -> str:
    """Return today's date in IST as 'YYYY-MM-DD'."""
    return (datetime.utcnow() + timedelta(hours=IST_OFFSET_HOURS)).strftime("%Y-%m-%d")


# ── Connection Pooling ───────────────────────────────────────────────────────
# SQLite with check_same_thread=False + per-thread connections is the safest
# pattern for FastAPI. We use WAL mode for concurrent read/write.

class _ConnectionPool:
    """Per-thread SQLite connection pool with WAL mode."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialized = False

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0,           # wait up to 30s on locks
                isolation_level=None,   # autocommit mode; we manage txns explicitly
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")      # 5x faster, still safe
            conn.execute("PRAGMA cache_size=-64000")        # 64MB cache
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA mmap_size=268435456")     # 256MB memory-mapped I/O
            self._local.conn = conn
        return self._local.conn

    def init_schema(self):
        """Create tables if not exist. Safe to call multiple times."""
        with self._init_lock:
            if self._initialized:
                return
            schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
            conn = self._get_conn()
            conn.executescript(schema_sql)
            self._initialized = True
            logger.info(f"[DB] Schema initialized at {self.db_path}")

    @contextmanager
    def transaction(self):
        """Context manager for explicit transactions."""
        conn = self._get_conn()
        conn.execute("BEGIN")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def execute(self, sql: str, params: Tuple = ()):
        """Execute a single statement (autocommit)."""
        conn = self._get_conn()
        return conn.execute(sql, params)

    def executemany(self, sql: str, params_iter: Iterable[Tuple]):
        """Execute multiple statements in a single transaction."""
        conn = self._get_conn()
        conn.execute("BEGIN")
        try:
            conn.executemany(sql, list(params_iter))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def query(self, sql: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Run a SELECT and return all rows."""
        conn = self._get_conn()
        return conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Tuple = ()) -> Optional[sqlite3.Row]:
        """Run a SELECT and return one row."""
        conn = self._get_conn()
        return conn.execute(sql, params).fetchone()


# ── Singleton ────────────────────────────────────────────────────────────────
_pool: Optional[_ConnectionPool] = None
_pool_lock = threading.Lock()


def get_pool(db_path: Optional[Path] = None) -> _ConnectionPool:
    """Get or create the connection pool singleton."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = _ConnectionPool(db_path or DEFAULT_DB_PATH)
    return _pool


# ── High-level Database API ──────────────────────────────────────────────────
class Database:
    """High-level helpers for predictions, outcomes, training queue, etc."""

    def __init__(self, db_path: Optional[Path] = None):
        # If a specific db_path is given, create a fresh pool for it
        # (don't reuse the singleton pool — that would ignore the path).
        if db_path is not None:
            self.pool = _ConnectionPool(db_path)
            self.pool.init_schema()
        else:
            self.pool = get_pool()
            self.pool.init_schema()

    # ── OHLCV ────────────────────────────────────────────────────────────────
    def upsert_ohlcv(self, ticker: str, rows: List[Dict[str, Any]]) -> int:
        """Insert or update OHLCV rows. `rows` should have keys:
        date, open, high, low, close, volume."""
        if not rows:
            return 0
        ticker = ticker.strip().upper()
        ts = now_ist()
        params = [
            (ticker, r["date"], r["open"], r["high"], r["low"], r["close"],
             int(r.get("volume", 0)), ts)
            for r in rows
        ]
        self.pool.executemany(
            """INSERT OR REPLACE INTO ohlcv
               (ticker, date, open, high, low, close, volume, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            params
        )
        return len(params)

    def get_ohlcv(self, ticker: str, start_date: Optional[str] = None,
                  end_date: Optional[str] = None) -> pd.DataFrame:
        """Load OHLCV from DB into a DataFrame. Returns empty DF if no data."""
        import pandas as pd
        ticker = ticker.strip().upper()
        sql = "SELECT date, open, high, low, close, volume FROM ohlcv WHERE ticker = ?"
        params: List = [ticker]
        if start_date:
            sql += " AND date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND date <= ?"
            params.append(end_date)
        sql += " ORDER BY date ASC"
        rows = self.pool.query(sql, tuple(params))
        if not rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        df = pd.DataFrame([dict(r) for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                "close": "Close", "volume": "Volume"})
        return df

    def get_last_ohlcv_date(self, ticker: str) -> Optional[str]:
        """Get the most recent OHLCV date we have for a ticker."""
        ticker = ticker.strip().upper()
        row = self.pool.query_one(
            "SELECT MAX(date) as d FROM ohlcv WHERE ticker = ?", (ticker,)
        )
        return row["d"] if row and row["d"] else None

    # ── Features ────────────────────────────────────────────────────────────
    def cache_features(self, ticker: str, date_str: str,
                       feature_vector: np.ndarray) -> str:
        """Cache a computed feature vector. Returns the feature hash."""
        ticker = ticker.strip().upper()
        blob = pickle.dumps(feature_vector.astype(np.float32))
        feature_hash = hashlib.sha1(blob).hexdigest()[:16]
        self.pool.execute(
            """INSERT OR REPLACE INTO features
               (ticker, date, feature_vector, feature_hash, computed_at)
               VALUES (?, ?, ?, ?, ?)""",
            (ticker, date_str, blob, feature_hash, now_ist())
        )
        return feature_hash

    def get_cached_features(self, ticker: str, date_str: str) -> Optional[np.ndarray]:
        """Retrieve a cached feature vector, or None if not cached."""
        ticker = ticker.strip().upper()
        row = self.pool.query_one(
            "SELECT feature_vector FROM features WHERE ticker = ? AND date = ?",
            (ticker, date_str)
        )
        if row is None:
            return None
        return pickle.loads(row["feature_vector"])

    # ── Predictions ──────────────────────────────────────────────────────────
    def log_prediction(self,
                       ticker: str,
                       prediction_ts: str,
                       model_version: str,
                       source: str,
                       signal_quality: Optional[float] = None,
                       trend_regime: Optional[str] = None,
                       regime_confidence: Optional[float] = None,
                       expected_return: Optional[float] = None,
                       max_drawdown: Optional[float] = None,
                       risk_reward_ratio: Optional[float] = None,
                       ml_score: Optional[float] = None,
                       actionable: Optional[str] = None,
                       entry_price: Optional[float] = None,
                       target_price: Optional[float] = None,
                       stop_loss: Optional[float] = None,
                       payload: Optional[Dict] = None) -> int:
        """Insert a prediction. Returns the prediction ID.

        If a prediction already exists for (ticker, prediction_ts), it is
        updated instead of duplicated. This makes the call idempotent.
        """
        ticker = ticker.strip().upper()
        # Extract date portion (first 10 chars of ISO timestamp)
        prediction_date = prediction_ts[:10]

        payload_json = json.dumps(payload, default=str) if payload else None

        with self.pool.transaction() as conn:
            # Try update first
            cur = conn.execute(
                """UPDATE predictions SET
                       model_version = ?, source = ?,
                       signal_quality = ?, trend_regime = ?, regime_confidence = ?,
                       expected_return = ?, max_drawdown = ?, risk_reward_ratio = ?,
                       ml_score = ?, actionable = ?,
                       entry_price = ?, target_price = ?, stop_loss = ?,
                       payload_json = ?
                   WHERE ticker = ? AND prediction_ts = ?""",
                (model_version, source,
                 signal_quality, trend_regime, regime_confidence,
                 expected_return, max_drawdown, risk_reward_ratio,
                 ml_score, actionable,
                 entry_price, target_price, stop_loss,
                 payload_json,
                 ticker, prediction_ts)
            )
            if cur.rowcount > 0:
                row = conn.execute(
                    "SELECT id FROM predictions WHERE ticker = ? AND prediction_ts = ?",
                    (ticker, prediction_ts)
                ).fetchone()
                return row["id"]

            # Insert new
            cur = conn.execute(
                """INSERT INTO predictions
                   (ticker, prediction_ts, prediction_date, model_version, source,
                    signal_quality, trend_regime, regime_confidence,
                    expected_return, max_drawdown, risk_reward_ratio,
                    ml_score, actionable,
                    entry_price, target_price, stop_loss,
                    payload_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ticker, prediction_ts, prediction_date, model_version, source,
                 signal_quality, trend_regime, regime_confidence,
                 expected_return, max_drawdown, risk_reward_ratio,
                 ml_score, actionable,
                 entry_price, target_price, stop_loss,
                 payload_json)
            )
            return cur.lastrowid

    def get_pending_outcomes(self, lookahead_days: int = 5) -> List[sqlite3.Row]:
        """Get predictions that are older than `lookahead_days` but have no
        resolved outcome yet. These need to be resolved."""
        cutoff = (datetime.utcnow() + timedelta(hours=IST_OFFSET_HOURS)
                  - timedelta(days=lookahead_days)).strftime("%Y-%m-%d %H:%M:%S")
        return self.pool.query(
            """SELECT p.* FROM predictions p
               LEFT JOIN outcomes o ON o.prediction_id = p.id
               WHERE o.prediction_id IS NULL
                 AND p.prediction_ts <= ?
               ORDER BY p.prediction_ts ASC""",
            (cutoff,)
        )

    # ── Outcomes ─────────────────────────────────────────────────────────────
    def resolve_outcome(self,
                        prediction_id: int,
                        actual_return: float,
                        actual_max_gain: float,
                        actual_max_loss: float,
                        hit_target: bool,
                        days_to_target: Optional[int],
                        predicted_return: Optional[float],
                        predicted_drawdown: Optional[float]) -> None:
        """Insert a resolved outcome and compute error metrics."""
        return_error = None
        if predicted_return is not None:
            return_error = round(predicted_return - actual_return, 2)

        drawdown_error = None
        if predicted_drawdown is not None:
            drawdown_error = round(predicted_drawdown - actual_max_loss, 2)

        # Direction correct: sign of predicted return matches sign of actual
        direction_correct = None
        if predicted_return is not None:
            if (predicted_return > 0 and actual_return > 0) or \
               (predicted_return < 0 and actual_return < 0) or \
               (predicted_return == 0 and actual_return == 0):
                direction_correct = 1
            else:
                direction_correct = 0

        resolved_ts = now_ist()
        resolved_date = resolved_ts[:10]

        self.pool.execute(
            """INSERT OR REPLACE INTO outcomes
               (prediction_id, resolved_at, resolved_date,
                actual_return, actual_max_gain, actual_max_loss,
                hit_target, days_to_target,
                return_error, drawdown_error, direction_correct)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (prediction_id, resolved_ts, resolved_date,
             actual_return, actual_max_gain, actual_max_loss,
             1 if hit_target else 0, days_to_target,
             return_error, drawdown_error, direction_correct)
        )

    def get_outcome_stats(self, days: int = 7,
                          model_type: Optional[str] = None) -> Dict[str, Any]:
        """Get rolling accuracy stats for the last N days."""
        cutoff = (datetime.utcnow() + timedelta(hours=IST_OFFSET_HOURS)
                  - timedelta(days=days)).strftime("%Y-%m-%d")
        sql = """
            SELECT
                COUNT(*) as total,
                AVG(CASE WHEN o.hit_target = 1 THEN 1.0 ELSE 0.0 END) as hit_rate,
                AVG(CASE WHEN o.direction_correct = 1 THEN 1.0 ELSE 0.0 END) as dir_acc,
                AVG(ABS(o.return_error)) as return_mae,
                AVG(ABS(o.drawdown_error)) as drawdown_mae,
                AVG(ABS(o.actual_return)) as avg_actual_move
            FROM outcomes o
            JOIN predictions p ON p.id = o.prediction_id
            WHERE o.resolved_date >= ?
        """
        params: Tuple = (cutoff,)
        if model_type:
            sql += " AND p.model_version LIKE ?"
            params = (cutoff, f"{model_type}_%")

        row = self.pool.query_one(sql, params)
        if not row or row["total"] == 0:
            return {"total": 0, "hit_rate": None, "direction_accuracy": None,
                    "return_mae": None, "drawdown_mae": None}

        return {
            "total": row["total"],
            "hit_rate": round(row["hit_rate"], 4) if row["hit_rate"] else None,
            "direction_accuracy": round(row["dir_acc"], 4) if row["dir_acc"] else None,
            "return_mae": round(row["return_mae"], 3) if row["return_mae"] else None,
            "drawdown_mae": round(row["drawdown_mae"], 3) if row["drawdown_mae"] else None,
            "avg_actual_move": round(row["avg_actual_move"], 3) if row["avg_actual_move"] else None,
        }

    # ── Training Queue ───────────────────────────────────────────────────────
    def enqueue_training_sample(self,
                                ticker: str,
                                feature_date: str,
                                feature_vector: np.ndarray,
                                hit_target: bool,
                                return_pct: float,
                                max_gain_pct: float,
                                max_loss_pct: float,
                                regime: int,
                                source_prediction_id: Optional[int] = None) -> None:
        """Add a labeled sample to the training queue."""
        ticker = ticker.strip().upper()
        blob = pickle.dumps(feature_vector.astype(np.float32))
        self.pool.execute(
            """INSERT OR REPLACE INTO training_queue
               (ticker, feature_date, feature_vector,
                label_hit_target, label_return_pct, label_max_gain_pct,
                label_max_loss_pct, label_regime,
                source_prediction_id, queued_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, feature_date, blob,
             1 if hit_target else 0,
             return_pct, max_gain_pct, max_loss_pct, regime,
             source_prediction_id, now_ist())
        )

    def get_training_samples(self, n: int = 5000,
                             since_date: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Pull N most recent unused training samples.

        Returns:
            X: np.ndarray of shape (N, num_features)
            y: dict with keys 'hit_target', 'return_pct', 'max_gain_pct',
               'max_loss_pct', 'regime' — each a np.ndarray of shape (N,)
        """
        sql = """
            SELECT id, feature_vector,
                   label_hit_target, label_return_pct, label_max_gain_pct,
                   label_max_loss_pct, label_regime
            FROM training_queue
            WHERE used_in_training IS NULL
        """
        params: Tuple = ()
        if since_date:
            sql += " AND feature_date >= ?"
            params = (since_date,)
        sql += " ORDER BY queued_at DESC LIMIT ?"
        params = params + (n,)

        rows = self.pool.query(sql, params)
        if not rows:
            return np.zeros((0, 0)), {}

        X = np.stack([pickle.loads(r["feature_vector"]) for r in rows])
        y = {
            "hit_target": np.array([r["label_hit_target"] for r in rows], dtype=np.float32),
            "return_pct": np.array([r["label_return_pct"] for r in rows], dtype=np.float32),
            "max_gain_pct": np.array([r["label_max_gain_pct"] for r in rows], dtype=np.float32),
            "max_loss_pct": np.array([r["label_max_loss_pct"] for r in rows], dtype=np.float32),
            "regime": np.array([r["label_regime"] for r in rows], dtype=np.int32),
        }
        return X, y

    def mark_samples_used(self, sample_ids: List[int], model_version: str) -> None:
        """Mark training samples as consumed by a specific model version."""
        if not sample_ids:
            return
        # SQLite doesn't support IN with bind parameters elegantly; use chunking
        chunk_size = 500
        for i in range(0, len(sample_ids), chunk_size):
            chunk = sample_ids[i:i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            self.pool.execute(
                f"""UPDATE training_queue
                    SET used_in_training = ?
                    WHERE id IN ({placeholders})""",
                (model_version, *chunk)
            )

    # ── Model Registry ───────────────────────────────────────────────────────
    def register_model(self,
                       version: str,
                       model_type: str,
                       model_path: str,
                       training_samples: int,
                       training_loss: Optional[float] = None,
                       val_accuracy: Optional[float] = None,
                       val_loss: Optional[float] = None,
                       val_return_mae: Optional[float] = None,
                       walk_forward_score: Optional[float] = None,
                       notes: Optional[str] = None) -> None:
        """Register a new model version (does NOT auto-promote)."""
        self.pool.execute(
            """INSERT OR REPLACE INTO model_registry
               (version, model_type, model_path, trained_at,
                training_samples, training_loss, val_accuracy, val_loss,
                val_return_mae, walk_forward_score, is_production, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)""",
            (version, model_type, model_path, now_ist(),
             training_samples, training_loss, val_accuracy, val_loss,
             val_return_mae, walk_forward_score, notes)
        )

    def promote_model(self, version: str, notes: Optional[str] = None) -> None:
        """Promote a model version to production (demotes others of same type)."""
        with self.pool.transaction() as conn:
            # Get the model type
            row = conn.execute(
                "SELECT model_type FROM model_registry WHERE version = ?",
                (version,)
            ).fetchone()
            if not row:
                raise ValueError(f"Model version {version} not found")
            model_type = row["model_type"]

            # Demote all current production models of this type
            conn.execute(
                """UPDATE model_registry
                   SET is_production = 0
                   WHERE model_type = ? AND is_production = 1""",
                (model_type,)
            )
            # Promote the new one
            conn.execute(
                """UPDATE model_registry
                   SET is_production = 1, promoted_at = ?, notes = COALESCE(?, notes)
                   WHERE version = ?""",
                (now_ist(), notes, version)
            )
            logger.info(f"[DB] Promoted {version} to production for {model_type}")

    def rollback_model(self, model_type: str, reason: str) -> Optional[str]:
        """Roll back to the previous production model. Returns the version
        rolled back to, or None if no previous version exists."""
        with self.pool.transaction() as conn:
            # Current production
            current = conn.execute(
                """SELECT version FROM model_registry
                   WHERE model_type = ? AND is_production = 1""",
                (model_type,)
            ).fetchone()
            if not current:
                return None
            current_version = current["version"]

            # Mark current as rolled back
            conn.execute(
                """UPDATE model_registry
                   SET is_production = 0, rolled_back_at = ?, rollback_reason = ?
                   WHERE version = ?""",
                (now_ist(), reason, current_version)
            )

            # Find the most recent previously-promoted version
            prev = conn.execute(
                """SELECT version FROM model_registry
                   WHERE model_type = ?
                     AND version != ?
                     AND rolled_back_at IS NULL
                   ORDER BY promoted_at DESC
                   LIMIT 1""",
                (model_type, current_version)
            ).fetchone()
            if not prev:
                logger.warning(f"[DB] No previous model to roll back to for {model_type}")
                return None

            # Promote the previous one
            conn.execute(
                """UPDATE model_registry
                   SET is_production = 1, promoted_at = ?
                   WHERE version = ?""",
                (now_ist(), prev["version"])
            )
            logger.info(f"[DB] Rolled back {model_type}: {current_version} → {prev['version']}")
            return prev["version"]

    def get_production_model(self, model_type: str) -> Optional[sqlite3.Row]:
        """Get the current production model for a given type."""
        return self.pool.query_one(
            """SELECT * FROM model_registry
               WHERE model_type = ? AND is_production = 1""",
            (model_type,)
        )

    def list_models(self, model_type: Optional[str] = None,
                    include_rolled_back: bool = False) -> List[sqlite3.Row]:
        """List all registered models, optionally filtered by type."""
        sql = "SELECT * FROM model_registry"
        params: Tuple = ()
        conditions = []
        if model_type:
            conditions.append("model_type = ?")
            params = (model_type,)
        if not include_rolled_back:
            conditions.append("rolled_back_at IS NULL")
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY trained_at DESC"
        return self.pool.query(sql, params)

    # ── Training Log ─────────────────────────────────────────────────────────
    def log_training_run(self,
                         run_type: str,
                         model_type: str,
                         status: str,
                         model_version: Optional[str] = None,
                         samples_used: Optional[int] = None,
                         epochs_run: Optional[int] = None,
                         final_loss: Optional[float] = None,
                         val_loss: Optional[float] = None,
                         val_accuracy: Optional[float] = None,
                         duration_seconds: Optional[float] = None,
                         notes: Optional[str] = None) -> int:
        """Log a training run. Returns the log entry ID."""
        cur = self.pool.execute(
            """INSERT INTO training_log
               (run_ts, run_type, model_type, model_version,
                samples_used, epochs_run, final_loss,
                val_loss, val_accuracy, duration_seconds, status, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (now_ist(), run_type, model_type, model_version,
             samples_used, epochs_run, final_loss,
             val_loss, val_accuracy, duration_seconds, status, notes)
        )
        return cur.lastrowid

    def get_training_history(self, limit: int = 50) -> List[sqlite3.Row]:
        """Get recent training runs."""
        return self.pool.query(
            "SELECT * FROM training_log ORDER BY run_ts DESC LIMIT ?",
            (limit,)
        )

    # ── Drift Alerts ─────────────────────────────────────────────────────────
    def raise_drift_alert(self,
                          alert_type: str,
                          severity: str,
                          details: Dict[str, Any],
                          model_type: Optional[str] = None,
                          action_taken: str = "none") -> int:
        """Raise a drift alert. Returns the alert ID."""
        cur = self.pool.execute(
            """INSERT INTO drift_alerts
               (alert_ts, alert_type, severity, model_type, details, action_taken)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (now_ist(), alert_type, severity, model_type,
             json.dumps(details, default=str), action_taken)
        )
        return cur.lastrowid

    def get_recent_alerts(self, hours: int = 24,
                          severity: Optional[str] = None) -> List[sqlite3.Row]:
        """Get drift alerts from the last N hours."""
        cutoff = (datetime.utcnow() + timedelta(hours=IST_OFFSET_HOURS)
                  - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        sql = "SELECT * FROM drift_alerts WHERE alert_ts >= ?"
        params: Tuple = (cutoff,)
        if severity:
            sql += " AND severity = ?"
            params = (cutoff, severity)
        sql += " ORDER BY alert_ts DESC LIMIT 100"
        return self.pool.query(sql, params)

    # ── Pass-through helpers (delegate to pool) ────────────────────────────
    def query(self, sql: str, params: Tuple = ()) -> List[sqlite3.Row]:
        """Run a SELECT and return all rows. Delegates to pool."""
        return self.pool.query(sql, params)

    def query_one(self, sql: str, params: Tuple = ()) -> Optional[sqlite3.Row]:
        """Run a SELECT and return one row. Delegates to pool."""
        return self.pool.query_one(sql, params)

    def execute(self, sql: str, params: Tuple = ()):
        """Execute a single statement. Delegates to pool."""
        return self.pool.execute(sql, params)


# ── Singleton accessor ───────────────────────────────────────────────────────
_db_singleton: Optional[Database] = None
_db_lock = threading.Lock()


def get_db(db_path: Optional[Path] = None) -> Database:
    """Get the singleton Database instance."""
    global _db_singleton
    if _db_singleton is None:
        with _db_lock:
            if _db_singleton is None:
                _db_singleton = Database(db_path)
    return _db_singleton


# Convenience alias
db = get_db()
