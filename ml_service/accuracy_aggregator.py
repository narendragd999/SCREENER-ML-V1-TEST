"""
Accuracy Aggregator — rolls up raw predictions into aggregated stats
=====================================================================
Computes summary metrics per stock, per model version, per sector, and
overall, across multiple time periods (7d, 30d, 90d, 1y, 5y, 10y, all).

Stores results in the `historical_accuracy` table for fast dashboard queries.

Usage:
    from accuracy_aggregator import AccuracyAggregator
    agg = AccuracyAggregator()
    agg.aggregate_all()  # recompute everything (run nightly)
    # or
    agg.aggregate_stock("RELIANCE", period="90d")
"""

from __future__ import annotations
import json
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta

import numpy as np

try:
    from db import db, now_ist, IST_OFFSET_HOURS
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from db import db, now_ist, IST_OFFSET_HOURS

logger = logging.getLogger("accuracy_aggregator")


# ── Period definitions ──────────────────────────────────────────────────────
PERIODS = {
    "7d":   7,
    "30d":  30,
    "90d":  90,
    "1y":   365,
    "5y":   365 * 5,
    "10y":  365 * 10,
    "all":  99999,
}

MAX_EQUITY_CURVE_POINTS = 100


class AccuracyAggregator:
    """
    Computes aggregated accuracy stats from historical_predictions.

    Usage:
        agg = AccuracyAggregator()
        agg.aggregate_all()  # full recompute (slow, ~30s)
        # or
        agg.aggregate_stock("RELIANCE")  # one stock, all periods
    """

    def __init__(self, target_pct: float = 3.0):
        self.target_pct = target_pct
        self._lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────────
    #  PUBLIC API
    # ─────────────────────────────────────────────────────────────────────────

    def aggregate_all(self) -> Dict[str, int]:
        """Recompute ALL accuracy stats. Run nightly via scheduler.

        Returns dict with counts: {'stock': N, 'model': N, 'overall': N}
        """
        with self._lock:
            logger.info("[Aggregator] Starting full aggregation...")
            start = datetime.now()

            # Get all unique tickers
            ticker_rows = db.query(
                "SELECT DISTINCT ticker FROM historical_predictions"
            )
            tickers = [r["ticker"] for r in ticker_rows]

            # Get all unique model versions
            model_rows = db.query(
                "SELECT DISTINCT model_version FROM historical_predictions"
            )
            models = [r["model_version"] for r in model_rows]

            stock_count = 0
            for ticker in tickers:
                for period in PERIODS:
                    self._aggregate_scope("stock", ticker, period)
                stock_count += 1

            model_count = 0
            for model in models:
                for period in PERIODS:
                    self._aggregate_scope("model", model, period)
                model_count += 1

            for period in PERIODS:
                self._aggregate_scope("overall", "ALL", period)

            elapsed = (datetime.now() - start).total_seconds()
            logger.info(f"[Aggregator] Done in {elapsed:.1f}s: "
                        f"{stock_count} stocks, {model_count} models")

            return {"stock": stock_count, "model": model_count, "overall": len(PERIODS)}

    def aggregate_stock(self, ticker: str) -> int:
        """Recompute accuracy for a single stock across all periods."""
        ticker = ticker.strip().upper()
        count = 0
        for period in PERIODS:
            self._aggregate_scope("stock", ticker, period)
            count += 1
        return count

    def aggregate_model(self, model_version: str) -> int:
        """Recompute accuracy for a single model version across all periods."""
        count = 0
        for period in PERIODS:
            self._aggregate_scope("model", model_version, period)
            count += 1
        return count

    def get_summary(self, period: str = "90d") -> Dict[str, Any]:
        """Get the overall summary for a given period."""
        row = db.query_one(
            """SELECT * FROM historical_accuracy
               WHERE scope_type = 'overall' AND scope_id = 'ALL' AND period = ?""",
            (period,)
        )
        if not row:
            return self._empty_summary(period)
        result = dict(row)
        if result.get("equity_curve_json"):
            try:
                result["equity_curve"] = json.loads(result["equity_curve_json"])
            except Exception:
                result["equity_curve"] = []
        else:
            result["equity_curve"] = []
        return result

    def get_stock_stats(self, period: str = "90d", sort_by: str = "hit_rate",
                        limit: int = 100, offset: int = 0,
                        min_predictions: int = 10) -> Dict[str, Any]:
        """Get per-stock accuracy stats, sorted by the given column."""
        valid_sorts = {"hit_rate", "total_predictions", "avg_return", "direction_accuracy",
                       "ticker", "profit_factor", "sharpe_ratio"}
        if sort_by not in valid_sorts:
            sort_by = "hit_rate"
        sort_dir = "DESC" if sort_by != "ticker" else "ASC"

        # Get total count
        total_row = db.query_one(
            """SELECT COUNT(*) as c FROM historical_accuracy
               WHERE scope_type = 'stock' AND period = ?
                 AND total_predictions >= ?""",
            (period, min_predictions)
        )
        total = total_row["c"] if total_row else 0

        rows = db.query(
            f"""SELECT * FROM historical_accuracy
                WHERE scope_type = 'stock' AND period = ?
                  AND total_predictions >= ?
                ORDER BY {sort_by} {sort_dir}
                LIMIT ? OFFSET ?""",
            (period, min_predictions, limit, offset)
        )

        stocks = []
        for r in rows:
            d = dict(r)
            d["ticker"] = d["scope_id"]
            stocks.append(d)

        # Get top/bottom for quick reference
        top = sorted(stocks,
                     key=lambda x: x.get("hit_rate") or 0,
                     reverse=True)[:5]
        bottom = sorted(stocks,
                        key=lambda x: x.get("hit_rate") or 0)[:5]

        return {
            "period": period,
            "stocks": stocks,
            "total": total,
            "limit": limit,
            "offset": offset,
            "top_5": top,
            "bottom_5": bottom,
        }

    def get_stock_detail(self, ticker: str, period: str = "all") -> Dict[str, Any]:
        """Get detailed stats for a single stock, including monthly breakdown."""
        ticker = ticker.strip().upper()

        # Main stats
        row = db.query_one(
            """SELECT * FROM historical_accuracy
               WHERE scope_type = 'stock' AND scope_id = ? AND period = ?""",
            (ticker, period)
        )
        if not row:
            return {"ticker": ticker, "error": "no_data", "period": period,
                    "message": "No historical predictions for this stock yet."}

        result = dict(row)
        result["ticker"] = ticker
        if result.get("equity_curve_json"):
            try:
                result["equity_curve"] = json.loads(result["equity_curve_json"])
            except Exception:
                result["equity_curve"] = []
        result.pop("equity_curve_json", None)

        # Recent trades (last 20)
        recent = db.query(
            """SELECT * FROM historical_predictions
               WHERE ticker = ?
               ORDER BY prediction_date DESC LIMIT 20""",
            (ticker,)
        )
        result["recent_trades"] = [dict(r) for r in recent]

        # Monthly breakdown
        monthly = db.query(
            """SELECT
                   substr(prediction_date, 1, 7) as month,
                   COUNT(*) as total,
                   SUM(CASE WHEN status = 'HIT' THEN 1 ELSE 0 END) as hits,
                   SUM(CASE WHEN status = 'MISSED' THEN 1 ELSE 0 END) as misses,
                   AVG(actual_return) as avg_return,
                   AVG(predicted_return) as avg_predicted
               FROM historical_predictions
               WHERE ticker = ?
               GROUP BY substr(prediction_date, 1, 7)
               ORDER BY month DESC LIMIT 24""",
            (ticker,)
        )
        result["monthly_breakdown"] = [dict(r) for r in monthly]

        return result

    def get_model_comparison(self, period: str = "90d") -> Dict[str, Any]:
        """Compare all model versions on the same historical data."""
        rows = db.query(
            """SELECT * FROM historical_accuracy
               WHERE scope_type = 'model' AND period = ?
               ORDER BY hit_rate DESC""",
            (period,)
        )

        models = []
        for r in rows:
            d = dict(r)
            d["model_version"] = d["scope_id"]
            # Check if production
            prod = db.query_one(
                "SELECT is_production FROM model_registry WHERE version = ?",
                (d["model_version"],)
            )
            d["is_production"] = bool(prod["is_production"]) if prod else False
            models.append(d)

        # Find best by each metric
        if models:
            best = {
                "hit_rate": max(models, key=lambda x: x.get("hit_rate") or 0)["model_version"],
                "direction_accuracy": max(models, key=lambda x: x.get("direction_accuracy") or 0)["model_version"],
                "profit_factor": max(models, key=lambda x: x.get("profit_factor") or 0)["model_version"],
                "sharpe_ratio": max(models, key=lambda x: x.get("sharpe_ratio") or -999)["model_version"],
            }
        else:
            best = {}

        return {
            "period": period,
            "models": models,
            "best": best,
        }

    def get_trend(self, bucket: str = "weekly", days: int = 180) -> Dict[str, Any]:
        """Get accuracy trend over time, bucketed by week or month."""
        cutoff = (datetime.utcnow() + timedelta(hours=IST_OFFSET_HOURS)
                  - timedelta(days=days)).strftime("%Y-%m-%d")

        if bucket == "monthly":
            fmt = "%Y-%m"
        else:  # weekly
            fmt = "%Y-W%W"

        rows = db.query(
            f"""SELECT
                   strftime('{fmt}', prediction_date) as period,
                   COUNT(*) as predictions,
                   SUM(CASE WHEN status = 'HIT' THEN 1 ELSE 0 END) as hits,
                   SUM(CASE WHEN status = 'MISSED' THEN 1 ELSE 0 END) as misses,
                   AVG(actual_return) as avg_return,
                   AVG(predicted_return) as avg_predicted,
                   AVG(CASE WHEN direction_correct = 1 THEN 1.0 ELSE 0.0 END) as direction_acc,
                   AVG(ABS(return_error)) as return_mae
               FROM historical_predictions
               WHERE prediction_date >= ?
               GROUP BY strftime('{fmt}', prediction_date)
               ORDER BY period ASC""",
            (cutoff,)
        )

        buckets = []
        for r in rows:
            d = dict(r)
            hits = d["hits"] or 0
            misses = d["misses"] or 0
            resolved = hits + misses
            d["hit_rate"] = round(hits / resolved, 4) if resolved > 0 else None
            d["avg_return"] = round(d["avg_return"], 3) if d["avg_return"] else None
            d["avg_predicted"] = round(d["avg_predicted"], 3) if d["avg_predicted"] else None
            d["direction_acc"] = round(d["direction_acc"], 4) if d["direction_acc"] else None
            d["return_mae"] = round(d["return_mae"], 3) if d["return_mae"] else None
            buckets.append(d)

        return {
            "bucket": bucket,
            "days": days,
            "buckets": buckets,
        }

    def get_trade_log(self,
                      ticker: Optional[str] = None,
                      model_version: Optional[str] = None,
                      status: Optional[str] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      limit: int = 100,
                      offset: int = 0) -> Dict[str, Any]:
        """Get paginated trade log."""
        conditions = []
        params: List = []

        if ticker:
            conditions.append("ticker = ?")
            params.append(ticker.upper())
        if model_version:
            conditions.append("model_version = ?")
            params.append(model_version)
        if status:
            conditions.append("status = ?")
            params.append(status.upper())
        if start_date:
            conditions.append("prediction_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("prediction_date <= ?")
            params.append(end_date)

        sql = "SELECT * FROM historical_predictions"
        count_sql = "SELECT COUNT(*) as c FROM historical_predictions"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
            count_sql += " WHERE " + " AND ".join(conditions)

        total_row = db.query_one(count_sql, tuple(params))
        total = total_row["c"] if total_row else 0

        sql += " ORDER BY prediction_date DESC, ticker ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = db.query(sql, tuple(params))
        return {
            "trades": [dict(r) for r in rows],
            "total": total,
            "limit": limit,
            "offset": offset,
            "pages": (total + limit - 1) // limit if limit > 0 else 0,
            "page": (offset // limit) + 1 if limit > 0 else 1,
        }

    # ─────────────────────────────────────────────────────────────────────────
    #  INTERNALS
    # ─────────────────────────────────────────────────────────────────────────

    def _aggregate_scope(self, scope_type: str, scope_id: str, period: str) -> None:
        """Compute and store accuracy for a given (scope_type, scope_id, period)."""
        days = PERIODS.get(period, 90)
        cutoff = (datetime.utcnow() + timedelta(hours=IST_OFFSET_HOURS)
                  - timedelta(days=days)).strftime("%Y-%m-%d")

        if scope_type == "stock":
            where = "ticker = ?"
            params: Tuple = (scope_id,)
        elif scope_type == "model":
            where = "model_version = ?"
            params = (scope_id,)
        else:  # overall
            where = "1=1"
            params = ()

        # Apply date filter
        if period != "all":
            where += " AND prediction_date >= ?"
            params = params + (cutoff,)

        # Pull all matching predictions
        rows = db.query(
            f"""SELECT * FROM historical_predictions WHERE {where}""",
            params
        )

        if not rows:
            return  # don't store empty stats

        # Compute metrics
        stats = self._compute_stats(rows)

        # Build equity curve (sampled to N points)
        stats["equity_curve_json"] = json.dumps(self._build_equity_curve(rows))

        # Upsert into historical_accuracy
        self._upsert_accuracy(scope_type, scope_id, period, stats, len(rows))

    def _compute_stats(self, rows: List) -> Dict[str, Any]:
        """Compute aggregate stats from a list of prediction rows."""
        # Convert sqlite3.Row to dict for safe .get() access
        rows = [dict(r) for r in rows]
        total = len(rows)
        hits = sum(1 for r in rows if r.get("status") == "HIT")
        misses = sum(1 for r in rows if r.get("status") == "MISSED")
        running = sum(1 for r in rows if r.get("status") == "RUNNING")
        no_data = sum(1 for r in rows if r.get("status") == "NO_DATA")

        resolved = hits + misses
        hit_rate = hits / resolved if resolved > 0 else None

        # Direction accuracy
        dir_correct = [r.get("direction_correct") for r in rows
                       if r.get("direction_correct") is not None]
        direction_acc = (sum(dir_correct) / len(dir_correct)) if dir_correct else None

        # Returns
        actual_returns = [r.get("actual_return") for r in rows if r.get("actual_return") is not None]
        predicted_returns = [r.get("predicted_return") for r in rows if r.get("predicted_return") is not None]
        max_gains = [r.get("actual_max_gain") for r in rows if r.get("actual_max_gain") is not None]
        max_losses = [r.get("actual_max_loss") for r in rows if r.get("actual_max_loss") is not None]

        avg_return = float(np.mean(actual_returns)) if actual_returns else None
        avg_predicted_return = float(np.mean(predicted_returns)) if predicted_returns else None
        avg_max_gain = float(np.mean(max_gains)) if max_gains else None
        avg_max_loss = float(np.mean(max_losses)) if max_losses else None

        # Return error
        errors = [r.get("return_error") for r in rows if r.get("return_error") is not None]
        return_mae = float(np.mean(np.abs(errors))) if errors else None
        return_bias = float(np.mean(errors)) if errors else None

        # Days to target
        dtt = [r.get("actual_days_to_target") for r in rows
               if r.get("actual_days_to_target") is not None]
        avg_days_to_target = float(np.mean(dtt)) if dtt else None

        # Win/loss metrics
        pnls = [r.get("pnl_pct") for r in rows if r.get("pnl_pct") is not None]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        avg_win = float(np.mean(wins)) if wins else 0
        avg_loss = float(np.mean(losses)) if losses else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else None

        sum_gains = sum(wins)
        sum_losses = abs(sum(losses))
        profit_factor = sum_gains / sum_losses if sum_losses > 0 else None

        if pnls:
            std_pnl = float(np.std(pnls))
            sharpe = (float(np.mean(pnls)) / std_pnl * np.sqrt(252)
                      if std_pnl > 0 else None)
        else:
            sharpe = None

        # Expectancy: (hit_rate * avg_win) - (miss_rate * avg_loss)
        miss_rate = (1 - hit_rate) if hit_rate is not None else None
        expectancy = None
        if hit_rate is not None and miss_rate is not None:
            expectancy = (hit_rate * avg_win) - (miss_rate * abs(avg_loss))

        best = max(pnls) if pnls else None
        worst = min(pnls) if pnls else None

        return {
            "total_predictions": total,
            "hits": hits,
            "misses": misses,
            "running": running,
            "no_data": no_data,
            "hit_rate": round(hit_rate, 4) if hit_rate is not None else None,
            "direction_accuracy": round(direction_acc, 4) if direction_acc is not None else None,
            "avg_return": round(avg_return, 3) if avg_return is not None else None,
            "avg_predicted_return": round(avg_predicted_return, 3) if avg_predicted_return is not None else None,
            "avg_max_gain": round(avg_max_gain, 3) if avg_max_gain is not None else None,
            "avg_max_loss": round(avg_max_loss, 3) if avg_max_loss is not None else None,
            "return_mae": round(return_mae, 3) if return_mae is not None else None,
            "return_bias": round(return_bias, 3) if return_bias is not None else None,
            "avg_days_to_target": round(avg_days_to_target, 2) if avg_days_to_target is not None else None,
            "win_loss_ratio": round(win_loss_ratio, 3) if win_loss_ratio is not None else None,
            "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
            "sharpe_ratio": round(sharpe, 3) if sharpe is not None else None,
            "expectancy": round(expectancy, 3) if expectancy is not None else None,
            "best_trade_pct": round(best, 3) if best is not None else None,
            "worst_trade_pct": round(worst, 3) if worst is not None else None,
            "avg_hold_days": 5,  # default lookahead
        }

    def _build_equity_curve(self, rows: List, initial_capital: float = 100.0) -> List[Dict]:
        """Build a sampled equity curve from predictions (chronological)."""
        sorted_rows = sorted(rows, key=lambda r: r["prediction_date"])
        equity = initial_capital
        curve = [{"date": "start", "equity": round(equity, 2)}]

        for r in sorted_rows:
            # sqlite3.Row doesn't have .get() — use dict-like access
            try:
                pnl = r["pnl_pct"] if "pnl_pct" in r.keys() else None
            except (KeyError, IndexError):
                pnl = None
            if pnl is not None:
                # Each trade risks 1% of capital
                equity *= (1 + pnl / 100)

        # Sample to MAX_EQUITY_CURVE_POINTS if needed
        # For simplicity, just return start + end
        curve.append({
            "date": sorted_rows[-1]["prediction_date"] if sorted_rows else "end",
            "equity": round(equity, 2),
        })
        return curve

    def _upsert_accuracy(self, scope_type: str, scope_id: str, period: str,
                          stats: Dict, total: int) -> None:
        """Insert or update a historical_accuracy row."""
        ts = now_ist()
        db.execute(
            """INSERT OR REPLACE INTO historical_accuracy
               (scope_type, scope_id, period,
                total_predictions, hits, misses, running, no_data,
                hit_rate, direction_accuracy,
                avg_return, avg_predicted_return, avg_max_gain, avg_max_loss,
                return_mae, return_bias,
                avg_days_to_target, win_loss_ratio, profit_factor,
                sharpe_ratio, expectancy,
                best_trade_pct, worst_trade_pct, avg_hold_days,
                equity_curve_json, computed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (scope_type, scope_id, period,
             stats["total_predictions"], stats["hits"], stats["misses"],
             stats["running"], stats["no_data"],
             stats["hit_rate"], stats["direction_accuracy"],
             stats["avg_return"], stats["avg_predicted_return"],
             stats["avg_max_gain"], stats["avg_max_loss"],
             stats["return_mae"], stats["return_bias"],
             stats["avg_days_to_target"], stats["win_loss_ratio"],
             stats["profit_factor"], stats["sharpe_ratio"], stats["expectancy"],
             stats["best_trade_pct"], stats["worst_trade_pct"],
             stats["avg_hold_days"],
             stats.get("equity_curve_json", "[]"),
             ts)
        )

    def _empty_summary(self, period: str) -> Dict[str, Any]:
        return {
            "period": period,
            "scope_type": "overall",
            "scope_id": "ALL",
            "total_predictions": 0,
            "hits": 0,
            "misses": 0,
            "running": 0,
            "no_data": 0,
            "hit_rate": None,
            "direction_accuracy": None,
            "avg_return": None,
            "avg_predicted_return": None,
            "return_mae": None,
            "profit_factor": None,
            "sharpe_ratio": None,
            "equity_curve": [],
            "message": "No historical predictions yet. Run a backtest first.",
        }


# ── Singleton ────────────────────────────────────────────────────────────────
_aggregator: Optional[AccuracyAggregator] = None
_agg_lock = threading.Lock()


def get_aggregator() -> AccuracyAggregator:
    """Get the singleton AccuracyAggregator instance."""
    global _aggregator
    if _aggregator is None:
        with _agg_lock:
            if _aggregator is None:
                _aggregator = AccuracyAggregator()
    return _aggregator
