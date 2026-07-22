"""
Outcome Tracker — Layer 3 of the Continuous Learning Pipeline
==============================================================
The most critical module in the system. Without this, predictions are
made but never verified — the ML system cannot learn.

What it does:
  1. Finds all predictions older than `lookahead_days` (default 5)
  2. Loads OHLCV data for each predicted ticker
  3. Computes what ACTUALLY happened after the prediction:
     - actual_return: close-to-close return over 5 days
     - actual_max_gain: peak % gain in the 5-day window
     - actual_max_loss: worst % loss in the 5-day window
     - hit_target: did the stock hit +3% (or whatever target_pct)?
     - days_to_target: how many days until +3% hit (None if never)
  4. Stores the outcome in the `outcomes` table
  5. Creates a labeled training sample in `training_queue`:
     (features_at_prediction_time, actual_outcomes)

This creates the closed loop:
    predict → wait 5 days → measure → label → retrain → predict better

Schedule: Run daily at 4:00 PM IST (after market close) via APScheduler.
"""

from __future__ import annotations
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Relative imports (work both as package and standalone)
try:
    from .db import db, now_ist, IST_OFFSET_HOURS
except ImportError:
    from db import db, now_ist, IST_OFFSET_HOURS

logger = logging.getLogger("outcome_tracker")


class OutcomeTracker:
    """
    Resolves pending predictions into labeled training samples.

    Usage:
        tracker = OutcomeTracker(lookahead_days=5, target_pct=3.0)
        resolved = tracker.resolve_all_pending()
        # Also: generate labels from raw OHLCV history (cold start)
        samples = tracker.generate_historical_labels(
            ticker_dfs={"RELIANCE": df, "INFY": df2},
            feature_extractor=extract_features,
        )
    """

    def __init__(self,
                 lookahead_days: int = 5,
                 target_pct: float = 3.0,
                 min_bars_required: int = 200):
        self.lookahead_days = lookahead_days
        self.target_pct = target_pct
        self.min_bars_required = min_bars_required
        self._lock = threading.Lock()  # prevent concurrent resolve runs

    # ─────────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def resolve_all_pending(self,
                            feature_extractor=None,
                            max_per_run: int = 500) -> Dict[str, Any]:
        """Resolve all pending predictions older than `lookahead_days`.

        Args:
            feature_extractor: callable(df) -> np.ndarray. If provided,
                the features at prediction time are cached in the features
                table and the labeled sample is added to training_queue.
            max_per_run: cap on predictions to resolve per call (prevents
                runaway jobs when backlog is large).

        Returns:
            Summary dict with counts and timing.
        """
        if not self._lock.acquire(blocking=False):
            logger.info("[OutcomeTracker] Already running — skipping")
            return {"status": "skipped", "reason": "already_running"}

        try:
            start = time.time()
            pending = db.get_pending_outcomes(self.lookahead_days)
            if not pending:
                logger.info("[OutcomeTracker] No pending outcomes to resolve")
                return {"status": "ok", "resolved": 0, "skipped": 0}

            pending = pending[:max_per_run]
            logger.info(f"[OutcomeTracker] Resolving {len(pending)} pending predictions...")

            # Group by ticker to avoid repeated DB loads
            by_ticker: Dict[str, List[sqlite3.Row]] = {}
            for row in pending:
                by_ticker.setdefault(row["ticker"], []).append(row)

            resolved_count = 0
            skipped_count = 0
            enqueued_count = 0

            for ticker, preds in by_ticker.items():
                # Load OHLCV for this ticker (from DB cache or yfinance fallback)
                df = self._load_ticker_data(ticker)
                if df is None or len(df) < self.min_bars_required:
                    skipped_count += len(preds)
                    continue

                for pred in preds:
                    success = self._resolve_one(pred, df, feature_extractor)
                    if success:
                        resolved_count += 1
                        if feature_extractor is not None:
                            enqueued_count += 1
                    else:
                        skipped_count += 1

            elapsed = time.time() - start
            logger.info(
                f"[OutcomeTracker] Done: {resolved_count} resolved, "
                f"{skipped_count} skipped, {enqueued_count} training samples enqueued "
                f"in {elapsed:.1f}s"
            )

            return {
                "status": "ok",
                "resolved": resolved_count,
                "skipped": skipped_count,
                "enqueued": enqueued_count,
                "elapsed_seconds": round(elapsed, 2),
            }
        finally:
            self._lock.release()

    def generate_historical_labels(self,
                                    ticker_dfs: Dict[str, pd.DataFrame],
                                    feature_extractor,
                                    step: int = 3,
                                    start_offset: int = 200) -> int:
        """Cold-start helper: walk through history and create labeled samples
        for every bar where we have enough future data.

        This is what you call once on initial training data to bootstrap the
        training_queue before the continuous learner has any real predictions
        to learn from.

        Args:
            ticker_dfs: dict of {ticker: OHLCV DataFrame}
            feature_extractor: callable(df) -> np.ndarray (raw, unnormalized)
            step: stride through history (3 = every 3rd bar, reduces correlation)
            start_offset: skip first N bars (warmup for indicators)

        Returns:
            Number of training samples added to the queue.
        """
        total_added = 0
        for ticker, df in ticker_dfs.items():
            if df is None or len(df) < start_offset + self.lookahead_days + 10:
                continue

            n = len(df)
            closes = df["Close"].values
            highs = df["High"].values
            lows = df["Low"].values
            dates = df.index if isinstance(df.index, pd.DatetimeIndex) else None

            # Walk through history
            for i in range(start_offset, n - self.lookahead_days, step):
                # Extract features using data up to bar i
                window_df = df.iloc[:i + 1]
                try:
                    features = feature_extractor(window_df, normalize=False)
                except Exception as e:
                    logger.debug(f"Feature extraction failed at {ticker} bar {i}: {e}")
                    continue

                if features is None or np.isnan(features).all():
                    continue

                # Compute labels from future bars
                entry_close = float(closes[i])
                if entry_close <= 0:
                    continue

                future_highs = highs[i + 1: i + 1 + self.lookahead_days]
                future_lows = lows[i + 1: i + 1 + self.lookahead_days]
                exit_close = float(closes[min(i + self.lookahead_days, n - 1)])

                if len(future_highs) == 0:
                    continue

                max_gain = (np.max(future_highs) / entry_close - 1) * 100
                max_loss = (np.min(future_lows) / entry_close - 1) * 100
                actual_return = (exit_close / entry_close - 1) * 100
                hit_target = max_gain >= self.target_pct

                # Days to target
                days_to_target = None
                for j, h in enumerate(future_highs, start=1):
                    if (h / entry_close - 1) * 100 >= self.target_pct:
                        days_to_target = j
                        break

                # Regime label (matches ml_models.py logic)
                if actual_return > 1.0:
                    regime = 0  # UPTREND
                elif actual_return < -1.0:
                    regime = 2  # DOWNTREND
                else:
                    regime = 1  # SIDEWAYS

                # Get the date string for this bar
                if dates is not None:
                    date_str = str(dates[i])[:10]
                else:
                    date_str = f"hist_{i}"

                # Cache features and enqueue training sample
                db.cache_features(ticker, date_str, features)
                db.enqueue_training_sample(
                    ticker=ticker,
                    feature_date=date_str,
                    feature_vector=features,
                    hit_target=hit_target,
                    return_pct=round(actual_return, 3),
                    max_gain_pct=round(max_gain, 3),
                    max_loss_pct=round(max_loss, 3),
                    regime=regime,
                    source_prediction_id=None,  # historical, no source prediction
                )
                total_added += 1

            logger.info(f"[OutcomeTracker] {ticker}: added historical training samples")

        logger.info(f"[OutcomeTracker] Historical label generation complete: "
                    f"{total_added} samples added")
        return total_added

    def get_accuracy_report(self, days: int = 7) -> Dict[str, Any]:
        """Get a rolling accuracy report for the last N days."""
        return db.get_outcome_stats(days=days)

    def get_accuracy_trend(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get daily accuracy metrics for the last N days, for trend charts."""
        cutoff = (datetime.utcnow() + timedelta(hours=IST_OFFSET_HOURS)
                  - timedelta(days=days)).strftime("%Y-%m-%d")
        rows = db.query(
            """SELECT
                   o.resolved_date as date,
                   COUNT(*) as total,
                   AVG(CASE WHEN o.hit_target = 1 THEN 1.0 ELSE 0.0 END) as hit_rate,
                   AVG(CASE WHEN o.direction_correct = 1 THEN 1.0 ELSE 0.0 END) as dir_acc,
                   AVG(ABS(o.return_error)) as return_mae
               FROM outcomes o
               WHERE o.resolved_date >= ?
               GROUP BY o.resolved_date
               ORDER BY o.resolved_date ASC""",
            (cutoff,)
        )
        return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────────────
    #  INTERNALS
    # ─────────────────────────────────────────────────────────────────────────

    def _load_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load OHLCV from DB. Falls back to yfinance if not in DB yet."""
        df = db.get_ohlcv(ticker)
        if df is not None and len(df) >= self.min_bars_required:
            return df

        # Fallback: try yfinance (this also populates the DB for future use)
        try:
            import yfinance as yf
            for suffix in [".NS", ".BO"]:
                try:
                    t = yf.Ticker(f"{ticker}{suffix}")
                    hist = t.history(period="2y", auto_adjust=True)
                    if hist is not None and not hist.empty:
                        df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
                        df = df.dropna(how="all")
                        # Persist to DB for future use
                        rows = []
                        for idx, r in df.iterrows():
                            rows.append({
                                "date": str(idx.date()) if hasattr(idx, "date") else str(idx)[:10],
                                "open": float(r["Open"]),
                                "high": float(r["High"]),
                                "low": float(r["Low"]),
                                "close": float(r["Close"]),
                                "volume": int(r["Volume"]),
                            })
                        db.upsert_ohlcv(ticker, rows)
                        return df
                except Exception:
                    continue
        except ImportError:
            logger.warning("[OutcomeTracker] yfinance not available for fallback")
        return None

    def _resolve_one(self,
                     prediction: sqlite3.Row,
                     df: pd.DataFrame,
                     feature_extractor=None) -> bool:
        """Resolve a single prediction. Returns True on success."""
        try:
            pred_ts_str = prediction["prediction_ts"]
            pred_date_str = prediction["prediction_date"]

            # Find the bar in df that matches (or is closest to) the prediction date
            pred_date = pd.to_datetime(pred_date_str).date()

            # df index should be datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                # Try to convert
                try:
                    df.index = pd.to_datetime(df.index)
                except Exception:
                    logger.debug(f"Cannot parse dates for {prediction['ticker']}")
                    return False

            # Find the index of the prediction bar
            try:
                bar_idx = df.index.get_loc(pd.Timestamp(pred_date))
            except KeyError:
                # Find the closest bar on or before the prediction date
                mask = df.index.date <= pred_date
                if not mask.any():
                    return False
                bar_idx = df.index.get_loc(df.index[mask][-1])

            if isinstance(bar_idx, slice):
                bar_idx = bar_idx.start

            # Need lookahead_days of future data
            if bar_idx + self.lookahead_days >= len(df):
                # Not enough future data yet — wait longer
                return False

            closes = df["Close"].values
            highs = df["High"].values
            lows = df["Low"].values

            entry_close = float(closes[bar_idx])
            if entry_close <= 0:
                return False

            future_highs = highs[bar_idx + 1: bar_idx + 1 + self.lookahead_days]
            future_lows = lows[bar_idx + 1: bar_idx + 1 + self.lookahead_days]
            exit_close = float(closes[bar_idx + self.lookahead_days])

            if len(future_highs) == 0:
                return False

            actual_return = (exit_close / entry_close - 1) * 100
            actual_max_gain = (np.max(future_highs) / entry_close - 1) * 100
            actual_max_loss = (np.min(future_lows) / entry_close - 1) * 100
            hit_target = actual_max_gain >= self.target_pct

            # Days to target
            days_to_target = None
            for j, h in enumerate(future_highs, start=1):
                if (h / entry_close - 1) * 100 >= self.target_pct:
                    days_to_target = j
                    break

            # Get predicted values (for error computation)
            predicted_return = prediction["expected_return"]
            predicted_drawdown = prediction["max_drawdown"]

            db.resolve_outcome(
                prediction_id=prediction["id"],
                actual_return=round(actual_return, 3),
                actual_max_gain=round(actual_max_gain, 3),
                actual_max_loss=round(actual_max_loss, 3),
                hit_target=hit_target,
                days_to_target=days_to_target,
                predicted_return=predicted_return,
                predicted_drawdown=predicted_drawdown,
            )

            # If we have a feature extractor, also create a training sample
            if feature_extractor is not None:
                try:
                    # Extract features at prediction time
                    window_df = df.iloc[:bar_idx + 1]
                    features = feature_extractor(window_df, normalize=False)
                    if features is not None and not np.isnan(features).all():
                        date_str = pred_date_str

                        # Cache features
                        db.cache_features(prediction["ticker"], date_str, features)

                        # Determine regime
                        if actual_return > 1.0:
                            regime = 0
                        elif actual_return < -1.0:
                            regime = 2
                        else:
                            regime = 1

                        db.enqueue_training_sample(
                            ticker=prediction["ticker"],
                            feature_date=date_str,
                            feature_vector=features,
                            hit_target=hit_target,
                            return_pct=round(actual_return, 3),
                            max_gain_pct=round(actual_max_gain, 3),
                            max_loss_pct=round(actual_max_loss, 3),
                            regime=regime,
                            source_prediction_id=prediction["id"],
                        )
                except Exception as e:
                    logger.debug(f"Feature extraction at resolve time failed: {e}")

            return True

        except Exception as e:
            logger.warning(f"[OutcomeTracker] Failed to resolve prediction "
                          f"{prediction['id']} for {prediction['ticker']}: {e}")
            return False


# ── Singleton ────────────────────────────────────────────────────────────────
_tracker: Optional[OutcomeTracker] = None


def get_tracker(lookahead_days: int = 5, target_pct: float = 3.0) -> OutcomeTracker:
    """Get the singleton OutcomeTracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = OutcomeTracker(lookahead_days=lookahead_days, target_pct=target_pct)
    return _tracker
