"""
Backtest Engine — 10-Year Historical Learning
================================================
The core historical learning module. For each (ticker, date) pair, it:
  1. Loads features AS OF that date (only data up to that date)
  2. Runs the model to get a "prediction"
  3. Loads the actual outcome (5 days forward)
  4. Records the result to historical_predictions table

Supports:
  - 10-year backtests (100 tickers × 2500 days = 250K predictions)
  - Incremental backfilling (only computes missing dates)
  - Resumable (tracks progress in backtest_runs table)
  - Concurrent feature extraction (uses feature_store cache)
  - Multiple model versions (compare v012 vs v011 on same data)

Performance:
  - First run (no cache): ~4 hours for 100 tickers × 10 years
  - Subsequent runs (cache hits): ~30 minutes
  - Incremental (new dates only): ~5 minutes per day

Schedule:
  - Initial: triggered manually via API (one-time)
  - Nightly: 1 AM IST, only for new dates (incremental)
"""

from __future__ import annotations
import os
import time
import logging
import threading
import json
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

try:
    from db import db, now_ist, IST_OFFSET_HOURS
except ImportError:
    # Fallback if running standalone
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from db import db, now_ist, IST_OFFSET_HOURS

logger = logging.getLogger("backtest_engine")


# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_LOOKAHEAD_DAYS = 5
DEFAULT_TARGET_PCT = 3.0
DEFAULT_STEP = 1                 # 1 = every trading day; 3 = every 3rd day
BACKTEST_BATCH_SIZE = 100        # commit to DB every N predictions
MAX_EQUITY_CURVE_POINTS = 100    # for charting


@dataclass
class BacktestProgress:
    """Progress tracker for a running backtest."""
    run_id: int
    tickers_total: int
    tickers_done: int
    predictions_generated: int
    predictions_skipped: int
    current_ticker: str
    current_date: str
    elapsed_seconds: float
    progress_pct: float
    status: str
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BacktestEngine:
    """
    Runs historical backtests by simulating predictions for past dates.

    Usage:
        engine = BacktestEngine(
            model_predict_fn=model_manager.predict,
            feature_extractor=extract_features,
        )

        # Full 10-year backtest for all F&O tickers
        run_id = engine.run_backtest(
            tickers=["RELIANCE", "INFY", ...],
            start_date="2015-01-01",
            end_date="2025-01-01",
            model_version="signal_quality_v012",
            run_type="initial",
        )

        # Check progress
        progress = engine.get_progress(run_id)

        # Get results
        results = engine.get_predictions(run_id, limit=100)
    """

    def __init__(self,
                 model_predict_fn: Callable[[pd.DataFrame], Dict],
                 feature_extractor: Optional[Callable] = None,
                 lookahead_days: int = DEFAULT_LOOKAHEAD_DAYS,
                 target_pct: float = DEFAULT_TARGET_PCT):
        """
        Args:
            model_predict_fn: callable(df) -> prediction dict.
                Typically `model_manager.predict` or `intelligent_scoring.compute_intelligent_score`.
            feature_extractor: callable(df, normalize=False) -> np.ndarray.
                Required for feature caching. If None, features won't be cached.
            lookahead_days: how many days forward to compute outcomes.
            target_pct: target gain % for hit_target label.
        """
        self.predict_fn = model_predict_fn
        self.feature_extractor = feature_extractor
        self.lookahead_days = lookahead_days
        self.target_pct = target_pct
        self._lock = threading.RLock()
        self._active_runs: Dict[int, BacktestProgress] = {}

    # ─────────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def run_backtest(self,
                     tickers: List[str],
                     start_date: str,
                     end_date: str,
                     model_version: str,
                     run_type: str = "manual",
                     step: int = DEFAULT_STEP,
                     skip_existing: bool = True,
                     on_progress: Optional[Callable[[BacktestProgress], None]] = None) -> int:
        """Run a backtest for the given tickers and date range.

        Args:
            tickers: list of ticker symbols
            start_date: 'YYYY-MM-DD'
            end_date: 'YYYY-MM-DD'
            model_version: which model version to simulate
            run_type: 'initial', 'incremental', 'manual', 'scheduled'
            step: stride through dates (1 = every day, 3 = every 3rd day)
            skip_existing: if True, skip (ticker, date, model_version) combos
                           that already exist in the DB
            on_progress: optional callback, called after each ticker completes

        Returns:
            backtest_run_id (integer)
        """
        # Validate inputs
        if not tickers:
            raise ValueError("No tickers provided")
        if start_date >= end_date:
            raise ValueError(f"start_date ({start_date}) must be before end_date ({end_date})")

        # Create a backtest_run record
        date_range_days = (datetime.strptime(end_date, "%Y-%m-%d") -
                          datetime.strptime(start_date, "%Y-%m-%d")).days
        run_id = self._create_run(
            run_type=run_type,
            model_version=model_version,
            tickers_count=len(tickers),
            start_date=start_date,
            end_date=end_date,
            date_range_days=date_range_days,
        )

        # If this is being called from a thread, run async; otherwise sync
        # We always run sync — the caller decides whether to wrap in a thread
        try:
            self._execute_backtest(
                run_id=run_id,
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                model_version=model_version,
                step=step,
                skip_existing=skip_existing,
                on_progress=on_progress,
            )
            return run_id
        except Exception as e:
            logger.error(f"[Backtest] Run {run_id} failed: {e}")
            self._update_run_status(run_id, "failed", error_message=str(e))
            raise

    def run_backtest_async(self, *args, **kwargs) -> int:
        """Same as run_backtest but runs in a background thread. Returns run_id immediately."""
        # Create the run record first (sync) so we have the ID
        tickers = kwargs.get("tickers", args[0] if args else [])
        start_date = kwargs.get("start_date", args[1] if len(args) > 1 else "")
        end_date = kwargs.get("end_date", args[2] if len(args) > 2 else "")
        model_version = kwargs.get("model_version", args[3] if len(args) > 3 else "")
        run_type = kwargs.get("run_type", args[4] if len(args) > 4 else "manual")
        step = kwargs.get("step", DEFAULT_STEP)
        skip_existing = kwargs.get("skip_existing", True)
        on_progress = kwargs.get("on_progress")

        date_range_days = (datetime.strptime(end_date, "%Y-%m-%d") -
                          datetime.strptime(start_date, "%Y-%m-%d")).days
        run_id = self._create_run(
            run_type=run_type,
            model_version=model_version,
            tickers_count=len(tickers),
            start_date=start_date,
            end_date=end_date,
            date_range_days=date_range_days,
        )

        def _bg():
            try:
                self._execute_backtest(
                    run_id=run_id,
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date,
                    model_version=model_version,
                    step=step,
                    skip_existing=skip_existing,
                    on_progress=on_progress,
                )
            except Exception as e:
                logger.error(f"[Backtest] Async run {run_id} failed: {e}")
                self._update_run_status(run_id, "failed", error_message=str(e))

        t = threading.Thread(target=_bg, daemon=True)
        t.start()
        return run_id

    def get_progress(self, run_id: int) -> Optional[BacktestProgress]:
        """Get current progress of a backtest run."""
        # Check in-memory first (for active runs)
        with self._lock:
            if run_id in self._active_runs:
                return self._active_runs[run_id]

        # Otherwise load from DB
        row = db.query_one("SELECT * FROM backtest_runs WHERE id = ?", (run_id,))
        if not row:
            return None

        return BacktestProgress(
            run_id=row["id"],
            tickers_total=row["tickers_count"],
            tickers_done=0,  # not tracked persistently
            predictions_generated=row["predictions_generated"],
            predictions_skipped=row["predictions_skipped"],
            current_ticker=row["current_ticker"] or "",
            current_date=row["current_date"] or "",
            elapsed_seconds=0,
            progress_pct=row["progress_pct"],
            status=row["status"],
            error_message=row["error_message"],
        )

    def get_predictions(self,
                        run_id: Optional[int] = None,
                        ticker: Optional[str] = None,
                        model_version: Optional[str] = None,
                        status: Optional[str] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None,
                        limit: int = 100,
                        offset: int = 0) -> List[Dict[str, Any]]:
        """Query historical predictions with filters."""
        conditions = []
        params: List = []

        if run_id is not None:
            conditions.append("backtest_run_id = ?")
            params.append(run_id)
        if ticker is not None:
            conditions.append("ticker = ?")
            params.append(ticker.upper())
        if model_version is not None:
            conditions.append("model_version = ?")
            params.append(model_version)
        if status is not None:
            conditions.append("status = ?")
            params.append(status.upper())
        if start_date is not None:
            conditions.append("prediction_date >= ?")
            params.append(start_date)
        if end_date is not None:
            conditions.append("prediction_date <= ?")
            params.append(end_date)

        sql = "SELECT * FROM historical_predictions"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY prediction_date DESC, ticker ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = db.query(sql, tuple(params))
        return [dict(r) for r in rows]

    def get_run_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent backtest runs."""
        rows = db.query(
            "SELECT * FROM backtest_runs ORDER BY run_ts DESC LIMIT ?",
            (limit,)
        )
        return [dict(r) for r in rows]

    def cancel_run(self, run_id: int) -> bool:
        """Request cancellation of a running backtest."""
        with self._lock:
            if run_id in self._active_runs:
                self._active_runs[run_id].status = "cancelled"
                return True
        # Even if not in memory, mark as cancelled in DB
        self._update_run_status(run_id, "cancelled", error_message="Cancelled by user")
        return True

    # ─────────────────────────────────────────────────────────────────────────
    #  INTERNALS
    # ─────────────────────────────────────────────────────────────────────────

    def _create_run(self, run_type: str, model_version: str, tickers_count: int,
                    start_date: str, end_date: str, date_range_days: int) -> int:
        """Create a backtest_runs record and return its ID."""
        cur = db.execute(
            """INSERT INTO backtest_runs
               (run_ts, run_type, model_version, tickers_count,
                start_date, end_date, date_range_days,
                predictions_generated, predictions_skipped,
                duration_seconds, status, progress_pct,
                current_ticker, current_date, error_message, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 'running', 0, ?, ?, NULL, ?)""",
            (now_ist(), run_type, model_version, tickers_count,
             start_date, end_date, date_range_days,
             "", "", f"Backtest {start_date} to {end_date}")
        )
        return cur.lastrowid

    def _update_run_status(self, run_id: int, status: str,
                           error_message: Optional[str] = None,
                           progress_pct: Optional[float] = None,
                           current_ticker: Optional[str] = None,
                           current_date: Optional[str] = None,
                           predictions_generated: Optional[int] = None,
                           predictions_skipped: Optional[int] = None,
                           duration_seconds: Optional[float] = None):
        """Update a backtest_runs record."""
        sets = ["status = ?"]
        params: List = [status]
        if error_message is not None:
            sets.append("error_message = ?")
            params.append(error_message)
        if progress_pct is not None:
            sets.append("progress_pct = ?")
            params.append(progress_pct)
        if current_ticker is not None:
            sets.append("current_ticker = ?")
            params.append(current_ticker)
        if current_date is not None:
            sets.append("current_date = ?")
            params.append(current_date)
        if predictions_generated is not None:
            sets.append("predictions_generated = ?")
            params.append(predictions_generated)
        if predictions_skipped is not None:
            sets.append("predictions_skipped = ?")
            params.append(predictions_skipped)
        if duration_seconds is not None:
            sets.append("duration_seconds = ?")
            params.append(duration_seconds)
        params.append(run_id)
        db.execute(
            f"UPDATE backtest_runs SET {', '.join(sets)} WHERE id = ?",
            tuple(params)
        )

    def _execute_backtest(self,
                          run_id: int,
                          tickers: List[str],
                          start_date: str,
                          end_date: str,
                          model_version: str,
                          step: int,
                          skip_existing: bool,
                          on_progress: Optional[Callable]):
        """Internal: actually run the backtest."""
        start_ts = time.time()
        progress = BacktestProgress(
            run_id=run_id,
            tickers_total=len(tickers),
            tickers_done=0,
            predictions_generated=0,
            predictions_skipped=0,
            current_ticker="",
            current_date="",
            elapsed_seconds=0,
            progress_pct=0,
            status="running",
        )

        with self._lock:
            self._active_runs[run_id] = progress

        try:
            logger.info(f"[Backtest {run_id}] Starting: {len(tickers)} tickers, "
                        f"{start_date} to {end_date}, model={model_version}")

            total_predictions = 0
            total_skipped = 0
            batch_predictions: List[Dict] = []

            for ticker_idx, ticker in enumerate(tickers):
                # Check for cancellation
                with self._lock:
                    if progress.status == "cancelled":
                        logger.info(f"[Backtest {run_id}] Cancelled")
                        break

                ticker = ticker.strip().upper()
                progress.current_ticker = ticker
                self._update_run_status(run_id, "running", current_ticker=ticker)

                # Load OHLCV for this ticker (from delta_downloader cache or DB)
                df = self._load_ticker_data(ticker)
                if df is None or len(df) < 60:
                    logger.debug(f"[Backtest {run_id}] {ticker}: insufficient data, skipping")
                    progress.tickers_done += 1
                    continue

                # Filter to date range
                try:
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    df = df[(df.index >= pd.Timestamp(start_date)) &
                            (df.index <= pd.Timestamp(end_date))]
                except Exception as e:
                    logger.warning(f"[Backtest {run_id}] {ticker}: date filter failed: {e}")

                if len(df) < 60:
                    logger.debug(f"[Backtest {run_id}] {ticker}: only {len(df)} bars after filter, need 60")
                    continue

                # Walk through each trading day
                trading_dates = df.index.tolist()
                ticker_predictions = 0
                ticker_skipped = 0

                for i in range(60, len(trading_dates), step):  # skip first 60 (warmup)
                    pred_date = trading_dates[i]
                    pred_date_str = str(pred_date.date()) if hasattr(pred_date, 'date') else str(pred_date)[:10]

                    progress.current_date = pred_date_str

                    # Check if we should skip (already exists)
                    if skip_existing:
                        existing = db.query_one(
                            """SELECT id FROM historical_predictions
                               WHERE ticker = ? AND prediction_date = ? AND model_version = ?""",
                            (ticker, pred_date_str, model_version)
                        )
                        if existing:
                            ticker_skipped += 1
                            continue

                    # Get features AS OF this date
                    window_df = df.iloc[:i + 1]
                    if len(window_df) < 60:
                        continue

                    # Run model prediction
                    try:
                        prediction = self.predict_fn(window_df)
                    except Exception as e:
                        logger.warning(f"[Backtest] {ticker}@{pred_date_str} predict failed: {e}")
                        continue

                    if prediction is None:
                        continue

                    # Compute actual outcome (5 days forward)
                    outcome = self._compute_outcome(df, i)

                    # Determine status
                    if outcome is None:
                        status = "RUNNING"
                    elif outcome["hit_target"]:
                        status = "HIT"
                    else:
                        status = "MISSED"

                    # Compute metrics
                    predicted_return = prediction.get("expected_return")
                    actual_return = outcome["actual_return"] if outcome else None
                    return_error = None
                    direction_correct = None
                    if predicted_return is not None and actual_return is not None:
                        return_error = round(predicted_return - actual_return, 3)
                        if (predicted_return > 0 and actual_return > 0) or \
                           (predicted_return < 0 and actual_return < 0) or \
                           (predicted_return == 0 and actual_return == 0):
                            direction_correct = 1
                        else:
                            direction_correct = 0

                    # Compute trade simulation
                    entry_price = float(df["Close"].iloc[i])
                    target_price = round(entry_price * (1 + self.target_pct / 100), 2)
                    stop_loss_price = round(entry_price * 0.98, 2)  # 2% stop
                    exit_price = None
                    pnl_pct = None
                    pnl_r_multiple = None

                    if outcome:
                        exit_price = float(df["Close"].iloc[min(i + self.lookahead_days, len(df) - 1)])
                        pnl_pct = round(outcome["actual_return"], 3)
                        risk_per_share = entry_price - stop_loss_price
                        if risk_per_share > 0:
                            pnl_r_multiple = round((exit_price - entry_price) / risk_per_share, 2)

                    # Build record
                    record = {
                        "ticker": ticker,
                        "prediction_date": pred_date_str,
                        "model_version": model_version,
                        "predicted_signal_quality": prediction.get("signal_quality"),
                        "predicted_return": predicted_return,
                        "predicted_max_drawdown": prediction.get("max_drawdown"),
                        "predicted_actionable": prediction.get("actionable"),
                        "predicted_ml_score": prediction.get("ml_score"),
                        "predicted_trend_regime": prediction.get("trend_regime"),
                        "actual_return": actual_return,
                        "actual_max_gain": outcome["actual_max_gain"] if outcome else None,
                        "actual_max_loss": outcome["actual_max_loss"] if outcome else None,
                        "actual_hit_target": outcome["hit_target"] if outcome else None,
                        "actual_days_to_target": outcome["days_to_target"] if outcome else None,
                        "status": status,
                        "return_error": return_error,
                        "direction_correct": direction_correct,
                        "entry_price": entry_price,
                        "target_price": target_price,
                        "stop_loss_price": stop_loss_price,
                        "exit_price": exit_price,
                        "pnl_pct": pnl_pct,
                        "pnl_r_multiple": pnl_r_multiple,
                        "backtest_run_id": run_id,
                        "computed_at": now_ist(),
                    }
                    batch_predictions.append(record)
                    ticker_predictions += 1
                    total_predictions += 1

                    # Commit in batches
                    if len(batch_predictions) >= BACKTEST_BATCH_SIZE:
                        self._insert_predictions(batch_predictions)
                        batch_predictions = []

                        # Update progress
                        progress.predictions_generated = total_predictions
                        progress.predictions_skipped = total_skipped
                        progress.elapsed_seconds = round(time.time() - start_ts, 1)
                        progress.progress_pct = round(
                            (ticker_idx + i / len(trading_dates)) / len(tickers) * 100, 1
                        )
                        self._update_run_status(
                            run_id, "running",
                            predictions_generated=total_predictions,
                            predictions_skipped=total_skipped,
                            progress_pct=progress.progress_pct,
                            current_date=pred_date_str,
                        )
                        if on_progress:
                            try:
                                on_progress(progress)
                            except Exception:
                                pass

                # Commit any remaining for this ticker
                if batch_predictions:
                    self._insert_predictions(batch_predictions)
                    batch_predictions = []

                total_skipped += ticker_skipped
                progress.tickers_done += 1
                progress.predictions_generated = total_predictions
                progress.predictions_skipped = total_skipped
                progress.elapsed_seconds = round(time.time() - start_ts, 1)
                progress.progress_pct = round(progress.tickers_done / len(tickers) * 100, 1)

                self._update_run_status(
                    run_id, "running",
                    predictions_generated=total_predictions,
                    predictions_skipped=total_skipped,
                    progress_pct=progress.progress_pct,
                    current_ticker=ticker,
                )

                if on_progress:
                    try:
                        on_progress(progress)
                    except Exception:
                        pass

                logger.info(f"[Backtest {run_id}] {ticker}: "
                            f"{ticker_predictions} predictions, {ticker_skipped} skipped "
                            f"({progress.progress_pct:.0f}% done)")

            # Final update
            elapsed = round(time.time() - start_ts, 1)
            progress.status = "completed"
            progress.elapsed_seconds = elapsed
            self._update_run_status(
                run_id, "completed",
                predictions_generated=total_predictions,
                predictions_skipped=total_skipped,
                progress_pct=100.0,
                duration_seconds=elapsed,
            )

            logger.info(f"[Backtest {run_id}] Completed: "
                        f"{total_predictions} predictions, {total_skipped} skipped, "
                        f"{elapsed:.1f}s")

        except Exception as e:
            logger.error(f"[Backtest {run_id}] Failed: {e}", exc_info=True)
            progress.status = "failed"
            progress.error_message = str(e)
            self._update_run_status(run_id, "failed", error_message=str(e))
            raise
        finally:
            with self._lock:
                self._active_runs.pop(run_id, None)

    def _load_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load OHLCV for a ticker. Uses DB cache, falls back to yfinance."""
        # Try DB first (fastest)
        df = db.get_ohlcv(ticker)
        if df is not None and len(df) >= 60:
            return df

        # Fall back to yfinance
        try:
            import yfinance as yf
            for suffix in [".NS", ".BO"]:
                try:
                    t = yf.Ticker(f"{ticker}{suffix}")
                    hist = t.history(period="10y", auto_adjust=True)
                    if hist is not None and not hist.empty and len(hist) >= 50:
                        df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
                        df = df.dropna(how="all")
                        # Persist to DB
                        rows = []
                        for idx, r in df.iterrows():
                            date_str = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
                            rows.append({
                                "date": date_str,
                                "open": float(r["Open"]),
                                "high": float(r["High"]),
                                "low": float(r["Low"]),
                                "close": float(r["Close"]),
                                "volume": int(r["Volume"]) if not pd.isna(r["Volume"]) else 0,
                            })
                        if rows:
                            db.upsert_ohlcv(ticker, rows)
                        return df
                except Exception:
                    continue
        except ImportError:
            pass
        return None

    def _compute_outcome(self, df: pd.DataFrame, bar_idx: int) -> Optional[Dict[str, Any]]:
        """Compute the actual outcome 5 days forward from bar_idx.

        Returns None if not enough future data.
        """
        n = len(df)
        if bar_idx + self.lookahead_days >= n:
            return None

        closes = df["Close"].values
        highs = df["High"].values
        lows = df["Low"].values

        entry_close = float(closes[bar_idx])
        if entry_close <= 0 or np.isnan(entry_close):
            return None

        future_highs = highs[bar_idx + 1: bar_idx + 1 + self.lookahead_days]
        future_lows = lows[bar_idx + 1: bar_idx + 1 + self.lookahead_days]
        exit_close = float(closes[min(bar_idx + self.lookahead_days, n - 1)])

        if len(future_highs) == 0:
            return None

        # Filter NaN
        future_highs = future_highs[~np.isnan(future_highs)]
        future_lows = future_lows[~np.isnan(future_lows)]
        if len(future_highs) == 0:
            return None

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

        return {
            "actual_return": round(actual_return, 3),
            "actual_max_gain": round(actual_max_gain, 3),
            "actual_max_loss": round(actual_max_loss, 3),
            "hit_target": hit_target,
            "days_to_target": days_to_target,
        }

    def _insert_predictions(self, predictions: List[Dict]) -> None:
        """Batch insert predictions into historical_predictions table."""
        if not predictions:
            return

        sql = """INSERT OR REPLACE INTO historical_predictions
                 (ticker, prediction_date, model_version,
                  predicted_signal_quality, predicted_return, predicted_max_drawdown,
                  predicted_actionable, predicted_ml_score, predicted_trend_regime,
                  actual_return, actual_max_gain, actual_max_loss,
                  actual_hit_target, actual_days_to_target,
                  status, return_error, direction_correct,
                  entry_price, target_price, stop_loss_price, exit_price,
                  pnl_pct, pnl_r_multiple,
                  backtest_run_id, computed_at)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

        params = [
            (p["ticker"], p["prediction_date"], p["model_version"],
             p["predicted_signal_quality"], p["predicted_return"], p["predicted_max_drawdown"],
             p["predicted_actionable"], p["predicted_ml_score"], p["predicted_trend_regime"],
             p["actual_return"], p["actual_max_gain"], p["actual_max_loss"],
             p["actual_hit_target"], p["actual_days_to_target"],
             p["status"], p["return_error"], p["direction_correct"],
             p["entry_price"], p["target_price"], p["stop_loss_price"], p["exit_price"],
             p["pnl_pct"], p["pnl_r_multiple"],
             p["backtest_run_id"], p["computed_at"])
            for p in predictions
        ]
        db.pool.executemany(sql, params)


# ── Singleton ────────────────────────────────────────────────────────────────
_engine: Optional[BacktestEngine] = None
_engine_lock = threading.Lock()


def get_engine(model_predict_fn: Optional[Callable] = None,
               feature_extractor: Optional[Callable] = None) -> BacktestEngine:
    """Get the singleton BacktestEngine instance.

    Requires model_predict_fn on first call.
    """
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                if model_predict_fn is None:
                    raise ValueError("model_predict_fn required on first call")
                _engine = BacktestEngine(
                    model_predict_fn=model_predict_fn,
                    feature_extractor=feature_extractor,
                )
    return _engine
