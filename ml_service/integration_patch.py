"""
Integration Patch — How to wire the new continuous learning modules
====================================================================
This file shows EXACTLY what to change in your existing ml_service.py
to enable the full continuous learning pipeline.

You don't need to rewrite ml_service.py from scratch — just apply the
changes below. Each section shows the BEFORE (current code) and AFTER
(new code).

For convenience, the full set of changes is also available as a unified
diff at the bottom of this file.
"""

# ============================================================================
# CHANGE 1: Add new imports at the top of ml_service.py
# ============================================================================

# --- ADD these imports near the top of ml_service.py (after existing imports) ---

INTEGRATION_IMPORTS = """
# ── Continuous Learning Pipeline (new) ──
try:
    from db import db, now_ist
    from outcome_tracker import get_tracker, OutcomeTracker
    from continuous_learner import ContinuousLearner, get_learner
    from drift_detector import get_detector, DriftDetector
    from delta_downloader import get_downloader, DeltaDownloader
    from feature_store import get_feature_store, FeatureStore
    from model_registry import get_registry, ModelRegistry
    from monitoring_api import create_monitoring_router
    HAS_CONTINUOUS_LEARNING = True
except ImportError as e:
    import logging as _logging
    _log = _logging.getLogger("ml_service")
    _log.warning(f"Continuous learning modules not found ({e}) — running in legacy mode")
    HAS_CONTINUOUS_LEARNING = False
"""


# ============================================================================
# CHANGE 2: Modify the prediction endpoint to log predictions
# ============================================================================

# BEFORE (current code in ml_service.py):
#
# @app.get("/api/predict/{ticker}")
# async def predict_ticker(ticker: str):
#     ticker = ticker.strip().upper()
#     df = download_ticker_data(ticker)
#     if df is None or len(df) < 100:
#         raise HTTPException(404, f"No sufficient data for {ticker}")
#     prediction = model_manager.predict(df)
#     prediction["ticker"] = ticker
#     return prediction

# AFTER:

INTEGRATION_PREDICT_ENDPOINT = '''
@app.get("/api/predict/{ticker}")
async def predict_ticker(ticker: str, source: str = "manual"):
    """Get ML predictions for a single ticker.
    Now logs every prediction to the database for outcome tracking.
    """
    ticker = ticker.strip().upper()

    # Use DeltaDownloader if available (much faster than re-downloading)
    if HAS_CONTINUOUS_LEARNING:
        df = get_downloader().update_ticker(ticker)
    else:
        df = download_ticker_data(ticker)

    if df is None or len(df) < 100:
        raise HTTPException(404, f"No sufficient data for {ticker}")

    # Get model version for tracking
    if HAS_CONTINUOUS_LEARNING:
        model_version = get_registry().get_production_version("signal_quality") or "intelligent_v2"
    else:
        model_version = "legacy"

    prediction = model_manager.predict(df)
    prediction["ticker"] = ticker

    # ── NEW: Log prediction for outcome tracking ──
    if HAS_CONTINUOUS_LEARNING:
        try:
            prediction_id = db.log_prediction(
                ticker=ticker,
                prediction_ts=now_ist(),
                model_version=model_version,
                source=source,
                signal_quality=prediction.get("signal_quality"),
                trend_regime=prediction.get("trend_regime"),
                regime_confidence=prediction.get("regime_confidence"),
                expected_return=prediction.get("expected_return"),
                max_drawdown=prediction.get("max_drawdown"),
                risk_reward_ratio=prediction.get("risk_reward_ratio"),
                ml_score=prediction.get("ml_score"),
                actionable=prediction.get("actionable"),
                payload=prediction,
            )
            prediction["prediction_id"] = prediction_id
            prediction["model_version"] = model_version
        except Exception as e:
            logger.warning(f"Failed to log prediction: {e}")

    return prediction
'''


# ============================================================================
# CHANGE 3: Add scheduler jobs in startup_event()
# ============================================================================

INTEGRATION_STARTUP = '''
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("  TensorFlow ML Stock Intelligence Service starting...")
    logger.info(f"  Port: 8001 | Main app: {config['main_app_url']}")
    logger.info("=" * 60)

    # ── EXISTING: existing scheduler jobs (unchanged) ──
    scheduler.add_job(
        _background_fine_tune,
        trigger="interval",
        minutes=config.get("fine_tune_interval_min", 30),
        id="fine_tune",
        replace_existing=True,
    )
    scheduler.add_job(
        lambda: asyncio.run(_background_chartink_ingest()),
        trigger="interval",
        minutes=config.get("chartink_poll_interval_min", 5),
        id="chartink_ingest",
        replace_existing=True,
    )
    scheduler.add_job(
        _check_and_execute_orders,
        trigger="interval",
        minutes=1,
        id="order_monitor",
        replace_existing=True,
    )

    # ── NEW: Continuous learning scheduler jobs ──
    if HAS_CONTINUOUS_LEARNING:
        # Initialize the continuous learner
        try:
            from ml_features import extract_features
            learner = ContinuousLearner(
                model_manager=model_manager,
                feature_extractor=extract_features,
            )
            # Register learner as singleton
            import continuous_learner as cl_module
            cl_module._learner = learner

            # Schedule fine-tune every 6 hours
            learner.start_scheduler(scheduler)

            # Schedule outcome resolution — daily at 4 PM IST
            from apscheduler.triggers.cron import CronTrigger
            scheduler.add_job(
                lambda: get_tracker().resolve_all_pending(
                    feature_extractor=extract_features
                ),
                trigger=CronTrigger(hour=16, minute=0, timezone="Asia/Kolkata"),
                id="outcome_resolution",
                replace_existing=True,
            )

            # Schedule drift detection — hourly
            detector = DriftDetector(
                feature_store=get_feature_store(extract_features),
                on_drift_callback=lambda alert: logger.warning(
                    f"DRIFT ALERT: {alert['alert_type']} ({alert['severity']})"
                ),
                on_critical_callback=lambda alert: learner.trigger_full_retrain(
                    reason=f"drift_alert_{alert['alert_type']}"
                ),
            )
            import drift_detector as dd_module
            dd_module._detector = detector
            detector.start_scheduler(scheduler, interval_minutes=60)

            # Mount the monitoring API
            app.include_router(create_monitoring_router())

            logger.info("[Startup] Continuous learning pipeline initialized")
            logger.info("  - Fine-tune: every 6 hours")
            logger.info("  - Full retrain: weekly (Sunday 2 AM IST)")
            logger.info("  - Outcome resolution: daily (4 PM IST)")
            logger.info("  - Drift detection: hourly")
            logger.info("  - Monitoring API: /api/ml-monitor/*")

        except Exception as e:
            logger.error(f"[Startup] Continuous learning init failed: {e}")
            import traceback
            traceback.print_exc()

    scheduler.start()
'''


# ============================================================================
# CHANGE 4: Initialize OutcomeTracker with feature_extractor
# ============================================================================

INTEGRATION_TRACKER_INIT = '''
# Add after `model_manager = ...` initialization

# Initialize outcome tracker (lazy — only when first needed)
_tracker_instance = None

def get_outcome_tracker():
    global _tracker_instance
    if _tracker_instance is None and HAS_CONTINUOUS_LEARNING:
        try:
            from ml_features import extract_features
            _tracker_instance = OutcomeTracker(
                lookahead_days=5,
                target_pct=3.0,
            )
            import outcome_tracker as ot_module
            ot_module._tracker = _tracker_instance
        except Exception as e:
            logger.warning(f"Failed to init outcome tracker: {e}")
    return _tracker_instance
'''


# ============================================================================
# CHANGE 5: Bootstrap historical training data on first run
# ============================================================================

INTEGRATION_BOOTSTRAP = '''
# Add as a new endpoint or call from startup_event() on first run

@app.post("/api/bootstrap-training-data")
async def bootstrap_training_data(background_tasks: BackgroundTasks,
                                   max_tickers: int = 50):
    """Bootstrap the training queue with historical labels.

    Call this ONCE after installing the continuous learning pipeline.
    It walks through historical OHLCV data and creates labeled training
    samples for every bar where we have 5 days of future outcome data.

    This populates the training_queue so the first fine-tune cycle has
    data to work with.
    """
    if not HAS_CONTINUOUS_LEARNING:
        raise HTTPException(503, "Continuous learning modules not available")

    def _do_bootstrap():
        try:
            from ml_features import extract_features
            tracker = OutcomeTracker(lookahead_days=5, target_pct=3.0)

            # Get F&O tickers
            tickers = get_fno_tickers()[:max_tickers]

            # Download data
            dd = get_downloader()
            ticker_dfs = dd.update_tickers_batch(tickers, max_workers=5)

            # Generate historical labels
            total = tracker.generate_historical_labels(
                ticker_dfs=ticker_dfs,
                feature_extractor=extract_features,
                step=3,
            )

            logger.info(f"[Bootstrap] Added {total} historical training samples")
        except Exception as e:
            logger.error(f"[Bootstrap] Failed: {e}")

    background_tasks.add_task(_do_bootstrap)
    return {
        "status": "bootstrap_started",
        "tickers": max_tickers,
        "message": "Running in background. Check /api/ml-monitor/training/history for progress."
    }
'''


# ============================================================================
# CHANGE 6: Use DeltaDownloader everywhere
# ============================================================================

INTEGRATION_DELTA = '''
# Replace `download_ticker_data` calls with delta downloader

# BEFORE:
#   df = download_ticker_data(ticker)
#
# AFTER:
#   df = get_downloader().update_ticker(ticker) if HAS_CONTINUOUS_LEARNING \\
#       else download_ticker_data(ticker)

# Apply this change in:
#   - predict_ticker endpoint
#   - predict_batch endpoint
#   - stock_deep_analysis endpoint
#   - stock_analysis_batch endpoint
#   - _background_chartink_ingest function
#   - _background_fine_tune function
'''


# ============================================================================
# FULL UNIFIED DIFF (for reference)
# ============================================================================

UNIFIED_DIFF = """
The complete integration involves:

1. Add INTEGRATION_IMPORTS block at the top of ml_service.py
2. Replace predict_ticker() with INTEGRATION_PREDICT_ENDPOINT
3. Add new scheduler jobs in startup_event() per INTEGRATION_STARTUP
4. Add get_outcome_tracker() helper per INTEGRATION_TRACKER_INIT
5. Add bootstrap endpoint per INTEGRATION_BOOTSTRAP
6. Replace download_ticker_data() with DeltaDownloader per INTEGRATION_DELTA
7. Add monitoring routes: app.include_router(create_monitoring_router())

Total lines changed: ~150 lines added, ~20 lines modified
Files modified: ml_service.py only
Files added (new): all files in this folder (db.py, outcome_tracker.py, etc.)
"""

if __name__ == "__main__":
    print("=" * 70)
    print("INTEGRATION PATCH SUMMARY")
    print("=" * 70)
    print()
    print("This file shows exactly how to integrate the continuous learning")
    print("pipeline into your existing ml_service.py.")
    print()
    print("To apply:")
    print("  1. Copy all .py files from this folder to your ml_service directory")
    print("  2. Apply the changes shown above to ml_service.py")
    print("  3. Restart the ML service (port 8001)")
    print("  4. Call POST /api/bootstrap-training-data once to seed historical labels")
    print("  5. Wait 6 hours — the first fine-tune cycle will run automatically")
    print()
    print("Verify with:")
    print("  curl http://localhost:8001/api/ml-monitor/dashboard")
    print()
    print(UNIFIED_DIFF)
