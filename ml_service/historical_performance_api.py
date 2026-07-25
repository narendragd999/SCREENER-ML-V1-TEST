"""
Historical Performance API — FastAPI router
=============================================
All endpoints for the historical performance dashboard.

Mount in ml_service.py:
    from historical_performance_api import create_historical_performance_router
    app.include_router(create_historical_performance_router())
"""

from __future__ import annotations
import logging
import threading
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

try:
    from db import db, now_ist, IST_OFFSET_HOURS
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from db import db, now_ist, IST_OFFSET_HOURS

logger = logging.getLogger("historical_performance_api")


# ── Request models ──────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    tickers: Optional[List[str]] = None
    start_date: Optional[str] = None  # default: 10 years ago
    end_date: Optional[str] = None    # default: today
    model_version: Optional[str] = None  # default: current production
    step: int = 1
    run_type: str = "manual"


def create_historical_performance_router() -> APIRouter:
    """Create the historical performance router. Mount in ml_service.py."""

    router = APIRouter(prefix="/api/historical-performance", tags=["Historical Performance"])

    # ─────────────────────────────────────────────────────────────────────
    #  SUMMARY
    # ─────────────────────────────────────────────────────────────────────

    @router.get("/summary")
    async def get_summary(period: str = Query("90d", regex="^(7d|30d|90d|1y|5y|10y|all)$")):
        """Get overall portfolio-level summary for the given period."""
        try:
            from accuracy_aggregator import get_aggregator
            agg = get_aggregator()
            return agg.get_summary(period=period)
        except Exception as e:
            raise HTTPException(500, f"Failed to get summary: {e}")

    # ─────────────────────────────────────────────────────────────────────
    #  PER-STOCK PERFORMANCE
    # ─────────────────────────────────────────────────────────────────────

    @router.get("/stocks")
    async def get_stocks(
        period: str = Query("90d", regex="^(7d|30d|90d|1y|5y|10y|all)$"),
        sort: str = Query("hit_rate", regex="^(hit_rate|total_predictions|avg_return|direction_accuracy|ticker|profit_factor|sharpe_ratio)$"),
        limit: int = Query(100, ge=1, le=500),
        offset: int = Query(0, ge=0),
        min_predictions: int = Query(10, ge=0, le=1000),
    ):
        """Get per-stock accuracy stats, sortable."""
        try:
            from accuracy_aggregator import get_aggregator
            agg = get_aggregator()
            return agg.get_stock_stats(
                period=period, sort_by=sort, limit=limit, offset=offset,
                min_predictions=min_predictions,
            )
        except Exception as e:
            raise HTTPException(500, f"Failed to get stocks: {e}")

    @router.get("/stocks/{ticker}")
    async def get_stock_detail(
        ticker: str,
        period: str = Query("all", regex="^(7d|30d|90d|1y|5y|10y|all)$"),
    ):
        """Get detailed stats for a single stock, including monthly breakdown."""
        try:
            from accuracy_aggregator import get_aggregator
            agg = get_aggregator()
            return agg.get_stock_detail(ticker, period=period)
        except Exception as e:
            raise HTTPException(500, f"Failed to get stock detail: {e}")

    # ─────────────────────────────────────────────────────────────────────
    #  MODEL COMPARISON
    # ─────────────────────────────────────────────────────────────────────

    @router.get("/models")
    async def get_models(
        period: str = Query("90d", regex="^(7d|30d|90d|1y|5y|10y|all)$"),
    ):
        """Compare all model versions on the same historical data."""
        try:
            from accuracy_aggregator import get_aggregator
            agg = get_aggregator()
            return agg.get_model_comparison(period=period)
        except Exception as e:
            raise HTTPException(500, f"Failed to get models: {e}")

    # ─────────────────────────────────────────────────────────────────────
    #  TRENDS
    # ─────────────────────────────────────────────────────────────────────

    @router.get("/trend")
    async def get_trend(
        bucket: str = Query("weekly", regex="^(weekly|monthly)$"),
        days: int = Query(180, ge=7, le=3650),
    ):
        """Get accuracy trend over time, bucketed by week or month."""
        try:
            from accuracy_aggregator import get_aggregator
            agg = get_aggregator()
            return agg.get_trend(bucket=bucket, days=days)
        except Exception as e:
            raise HTTPException(500, f"Failed to get trend: {e}")

    # ─────────────────────────────────────────────────────────────────────
    #  TRADE LOG
    # ─────────────────────────────────────────────────────────────────────

    @router.get("/trades")
    async def get_trades(
        ticker: Optional[str] = None,
        model_version: Optional[str] = None,
        status: Optional[str] = Query(None, regex="^(HIT|MISSED|RUNNING|NO_DATA)$"),
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = Query(100, ge=1, le=1000),
        offset: int = Query(0, ge=0),
    ):
        """Get paginated trade log with filters."""
        try:
            from accuracy_aggregator import get_aggregator
            agg = get_aggregator()
            return agg.get_trade_log(
                ticker=ticker, model_version=model_version, status=status,
                start_date=start_date, end_date=end_date,
                limit=limit, offset=offset,
            )
        except Exception as e:
            raise HTTPException(500, f"Failed to get trades: {e}")

    # ─────────────────────────────────────────────────────────────────────
    #  BACKTEST TRIGGER & STATUS
    # ─────────────────────────────────────────────────────────────────────

    @router.post("/backtest")
    async def trigger_backtest(req: BacktestRequest, background_tasks: BackgroundTasks):
        """Trigger a backtest. Returns run_id immediately, runs in background."""
        try:
            from backtest_engine import get_engine
            from model_registry import get_registry

            # Determine model version
            if req.model_version and req.model_version != "current":
                model_version = req.model_version
            else:
                model_version = get_registry().get_production_version("signal_quality")
                if not model_version:
                    model_version = "intelligent_v2"

            # Determine tickers
            if req.tickers:
                tickers = [t.strip().upper() for t in req.tickers if t.strip()]
            else:
                # Get all tickers from OHLCV table
                rows = db.query("SELECT DISTINCT ticker FROM ohlcv ORDER BY ticker")
                tickers = [r["ticker"] for r in rows]
                if not tickers:
                    raise HTTPException(400, "No tickers available. Run delta_downloader first.")

            # Determine date range (default: 10 years)
            end_date = req.end_date or datetime.now().strftime("%Y-%m-%d")
            if req.start_date:
                start_date = req.start_date
            else:
                start_date = (datetime.now() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")

            # Get model predict function
            # This needs to be passed to the engine — we use a deferred approach
            # to avoid circular imports
            def _get_predict_fn():
                try:
                    from ml_models import model_manager
                    return model_manager.predict
                except ImportError:
                    try:
                        from intelligent_scoring import compute_intelligent_score
                        return compute_intelligent_score
                    except ImportError:
                        raise HTTPException(503, "No model available for backtest")

            predict_fn = _get_predict_fn()

            # Get or create engine
            try:
                engine = get_engine()
            except ValueError:
                # First time — need to initialize
                try:
                    from ml_features import extract_features as fe
                except ImportError:
                    fe = None
                engine = get_engine(model_predict_fn=predict_fn, feature_extractor=fe)

            # Run async
            run_id = engine.run_backtest_async(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                model_version=model_version,
                run_type=req.run_type,
                step=req.step,
            )

            return {
                "status": "started",
                "run_id": run_id,
                "tickers_count": len(tickers),
                "start_date": start_date,
                "end_date": end_date,
                "model_version": model_version,
                "step": req.step,
                "message": f"Backtest started for {len(tickers)} tickers. "
                          f"Check /api/historical-performance/backtest/{run_id} for progress."
            }
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Failed to start backtest: {e}")

    @router.get("/backtest/{run_id}")
    async def get_backtest_status(run_id: int):
        """Get progress of a backtest run."""
        try:
            from backtest_engine import get_engine
            engine = get_engine()
            progress = engine.get_progress(run_id)
            if not progress:
                raise HTTPException(404, f"Run {run_id} not found")
            return progress.to_dict()
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Failed to get status: {e}")

    @router.get("/backtest")
    async def list_backtests(limit: int = Query(20, ge=1, le=100)):
        """List recent backtest runs."""
        try:
            from backtest_engine import get_engine
            engine = get_engine()
            return {"runs": engine.get_run_history(limit=limit)}
        except Exception as e:
            raise HTTPException(500, f"Failed to list backtests: {e}")

    @router.post("/backtest/{run_id}/cancel")
    async def cancel_backtest(run_id: int):
        """Cancel a running backtest."""
        try:
            from backtest_engine import get_engine
            engine = get_engine()
            success = engine.cancel_run(run_id)
            return {"status": "cancelled" if success else "not_found", "run_id": run_id}
        except Exception as e:
            raise HTTPException(500, f"Failed to cancel: {e}")

    # ─────────────────────────────────────────────────────────────────────
    #  AGGREGATION TRIGGER
    # ─────────────────────────────────────────────────────────────────────

    @router.post("/aggregate")
    async def trigger_aggregation(background_tasks: BackgroundTasks):
        """Manually trigger accuracy aggregation. Normally runs nightly."""
        def _do_aggregate():
            try:
                from accuracy_aggregator import get_aggregator
                agg = get_aggregator()
                agg.aggregate_all()
                logger.info("[API] Aggregation complete")
            except Exception as e:
                logger.error(f"[API] Aggregation failed: {e}")

        background_tasks.add_task(_do_aggregate)
        return {"status": "started", "message": "Aggregation running in background"}

    # ─────────────────────────────────────────────────────────────────────
    #  CLEAR / DELETE BACKTEST DATA
    #  (supports both DELETE and POST methods for compatibility)
    # ─────────────────────────────────────────────────────────────────────

    async def _clear_backtest_data_impl(
        scope: str = "all",
        confirm: bool = False,
    ):
        """Internal implementation — called by both DELETE and POST handlers."""
        # Count what would be deleted
        counts = {}
        if scope in ("all", "predictions"):
            row = db.query_one("SELECT COUNT(*) as c FROM historical_predictions")
            counts["historical_predictions"] = row["c"] if row else 0
        if scope in ("all", "accuracy"):
            row = db.query_one("SELECT COUNT(*) as c FROM historical_accuracy")
            counts["historical_accuracy"] = row["c"] if row else 0
        if scope in ("all", "runs"):
            row = db.query_one("SELECT COUNT(*) as c FROM backtest_runs")
            counts["backtest_runs"] = row["c"] if row else 0

        # Dry run — just return counts
        if not confirm:
            return {
                "status": "dry_run",
                "message": "Pass confirm=true to actually delete. This is what WILL be deleted:",
                "scope": scope,
                "would_delete": counts,
                "total_to_delete": sum(counts.values()),
            }

        # Actually delete
        deleted = {}
        try:
            if scope in ("all", "predictions"):
                db.execute("DELETE FROM historical_predictions")
                deleted["historical_predictions"] = counts["historical_predictions"]
            if scope in ("all", "accuracy"):
                db.execute("DELETE FROM historical_accuracy")
                deleted["historical_accuracy"] = counts["historical_accuracy"]
            if scope in ("all", "runs"):
                db.execute("DELETE FROM backtest_runs")
                deleted["backtest_runs"] = counts["backtest_runs"]

            # Also clear the autoincrement sequences (so next run starts at 1)
            if scope in ("all", "predictions"):
                try:
                    db.execute("DELETE FROM sqlite_sequence WHERE name = 'historical_predictions'")
                except Exception:
                    pass
            if scope in ("all", "runs"):
                try:
                    db.execute("DELETE FROM sqlite_sequence WHERE name = 'backtest_runs'")
                except Exception:
                    pass

            return {
                "status": "success",
                "message": f"Cleared {scope} data successfully.",
                "scope": scope,
                "deleted": deleted,
                "total_deleted": sum(deleted.values()),
            }
        except Exception as e:
            raise HTTPException(500, f"Failed to clear data: {e}")

    @router.delete("/clear")
    async def clear_backtest_data_delete(
        scope: str = Query("all", regex="^(all|predictions|accuracy|runs)$"),
        confirm: bool = Query(False),
    ):
        """Clear historical performance data (DELETE method).

        WARNING: This permanently deletes data. Use the 'confirm=true' parameter
        to actually delete. Without it, returns a preview of what would be deleted.

        Args:
            scope: What to delete
                - 'all'          → clear everything (predictions + accuracy + runs)
                - 'predictions'  → clear only historical_predictions table
                - 'accuracy'     → clear only historical_accuracy table
                - 'runs'         → clear only backtest_runs table
            confirm: Must be True to actually delete. False = dry run.
        """
        return await _clear_backtest_data_impl(scope=scope, confirm=confirm)

    @router.post("/clear")
    async def clear_backtest_data_post(
        background_tasks: BackgroundTasks,
        scope: str = Query("all", regex="^(all|predictions|accuracy|runs)$"),
        confirm: bool = Query(False),
    ):
        """Clear historical performance data (POST method — fallback for servers that block DELETE).

        Same functionality as DELETE /clear, but uses POST for compatibility.
        """
        return await _clear_backtest_data_impl(scope=scope, confirm=confirm)

    @router.get("/clear/preview")
    async def clear_preview():
        """Get a preview of what would be deleted — useful for the UI button."""
        counts = {}
        for table in ("historical_predictions", "historical_accuracy", "backtest_runs"):
            row = db.query_one(f"SELECT COUNT(*) as c FROM {table}")
            counts[table] = row["c"] if row else 0

        # Per-ticker breakdown (top 10)
        ticker_rows = db.query(
            """SELECT ticker, COUNT(*) as count
               FROM historical_predictions
               GROUP BY ticker
               ORDER BY count DESC
               LIMIT 10"""
        )
        top_tickers = [{"ticker": r["ticker"], "count": r["count"]} for r in ticker_rows]

        # Per-model breakdown
        model_rows = db.query(
            """SELECT model_version, COUNT(*) as count
               FROM historical_predictions
               GROUP BY model_version
               ORDER BY count DESC"""
        )
        models = [{"model_version": r["model_version"], "count": r["count"]} for r in model_rows]

        return {
            "tables": counts,
            "total_rows": sum(counts.values()),
            "top_tickers": top_tickers,
            "models": models,
        }

    # ─────────────────────────────────────────────────────────────────────
    #  DASHBOARD (combined data in one call for fast page load)
    # ─────────────────────────────────────────────────────────────────────

    @router.get("/dashboard")
    async def get_dashboard(period: str = Query("90d", regex="^(7d|30d|90d|1y|5y|10y|all)$")):
        """One-shot dashboard data — all key metrics in one call."""
        try:
            from accuracy_aggregator import get_aggregator
            agg = get_aggregator()

            summary = agg.get_summary(period=period)
            stocks = agg.get_stock_stats(period=period, sort_by="hit_rate",
                                         limit=10, offset=0, min_predictions=10)
            models = agg.get_model_comparison(period=period)
            trend = agg.get_trend(bucket="weekly", days=180)

            # Recent backtests
            recent_runs = db.query(
                "SELECT * FROM backtest_runs ORDER BY run_ts DESC LIMIT 5"
            )

            # Counts
            total_preds_row = db.query_one(
                "SELECT COUNT(*) as c FROM historical_predictions"
            )
            total_predictions = total_preds_row["c"] if total_preds_row else 0

            status_counts_rows = db.query(
                """SELECT status, COUNT(*) as count
                   FROM historical_predictions
                   GROUP BY status"""
            )
            status_counts = {r["status"]: r["count"] for r in status_counts_rows}

            return {
                "period": period,
                "timestamp": now_ist(),
                "summary": summary,
                "top_stocks": stocks["top_5"],
                "bottom_stocks": stocks["bottom_5"],
                "models": models["models"][:5],
                "best_model": models.get("best", {}),
                "trend": trend["buckets"][-12:] if trend["buckets"] else [],
                "recent_backtests": [dict(r) for r in recent_runs],
                "total_predictions": total_predictions,
                "status_counts": status_counts,
            }
        except Exception as e:
            raise HTTPException(500, f"Failed to get dashboard: {e}")

    return router
