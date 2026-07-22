-- ============================================================================
-- ML Continuous Learning — SQLite Schema
-- ============================================================================
-- Replaces the JSON files (paper_trades.json, ml_decisions.json,
-- chartink_signals.json) with a proper relational store that supports:
--   1. Outcome tracking (predictions → resolved labels)
--   2. Model versioning / registry
--   3. Training logs
--   4. Drift alerts
--   5. Feature caching
--
-- All timestamps are stored as ISO-8601 strings (UTC or IST, see convention
-- in db.py). Tickers are uppercase. Dates are 'YYYY-MM-DD'.
-- ============================================================================

-- ── OHLCV price data (delta-updated hourly) ────────────────────────────────
CREATE TABLE IF NOT EXISTS ohlcv (
    ticker      TEXT    NOT NULL,
    date        TEXT    NOT NULL,            -- 'YYYY-MM-DD'
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      INTEGER NOT NULL DEFAULT 0,
    updated_at  TEXT    NOT NULL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_ohlcv_date ON ohlcv(date);
CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker ON ohlcv(ticker);

-- ── Cached feature vectors (one row per ticker per day) ────────────────────
CREATE TABLE IF NOT EXISTS features (
    ticker          TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    feature_vector  BLOB    NOT NULL,        -- pickled numpy float32 array
    feature_hash    TEXT    NOT NULL,        -- sha1 of vector, for drift detection
    computed_at     TEXT    NOT NULL,
    PRIMARY KEY (ticker, date)
);
CREATE INDEX IF NOT EXISTS idx_features_ticker ON features(ticker);

-- ── Predictions (every prediction the system makes gets logged here) ────────
CREATE TABLE IF NOT EXISTS predictions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker              TEXT    NOT NULL,
    prediction_ts       TEXT    NOT NULL,    -- full ISO timestamp
    prediction_date     TEXT    NOT NULL,    -- 'YYYY-MM-DD' (for easy joins)
    model_version       TEXT    NOT NULL,    -- e.g. 'signal_quality_v012_2026-01-15'
    source              TEXT    NOT NULL,    -- 'chartink' | 'ema9' | 'manual' | 'scanner'

    -- Model outputs
    signal_quality      REAL,                -- [0, 1]
    trend_regime        TEXT,                -- 'UPTREND' | 'SIDEWAYS' | 'DOWNTREND'
    regime_confidence   REAL,
    expected_return     REAL,                -- %
    max_drawdown        REAL,                -- % (negative)
    risk_reward_ratio   REAL,
    ml_score            REAL,                -- 0-100
    actionable          TEXT,                -- STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL | SKIP

    -- Context at prediction time
    entry_price         REAL,
    target_price        REAL,
    stop_loss           REAL,

    -- Full payload (JSON) for debugging / re-training
    payload_json        TEXT,

    -- Indexes
    UNIQUE(ticker, prediction_ts)
);
CREATE INDEX IF NOT EXISTS idx_pred_ticker ON predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(prediction_date);
-- Note: SQLite doesn't support subqueries in partial index WHERE clauses.
-- Use a separate flag column or query via LEFT JOIN instead.

-- ── Outcomes (resolved 5 days after prediction) ────────────────────────────
CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id   INTEGER PRIMARY KEY,
    resolved_at     TEXT    NOT NULL,
    resolved_date   TEXT    NOT NULL,

    -- Actual outcomes
    actual_return       REAL    NOT NULL,    -- % return over 5 days (close-to-close)
    actual_max_gain     REAL    NOT NULL,    -- % max gain in 5 days
    actual_max_loss     REAL    NOT NULL,    -- % max loss in 5 days (negative)
    hit_target          INTEGER NOT NULL,    -- 1 if actual_max_gain >= target_pct (default 3.0)
    days_to_target      INTEGER,             -- how many days until +3% hit (NULL if never)

    -- Accuracy metrics (computed at resolve time)
    return_error        REAL,                -- predicted_return - actual_return
    drawdown_error      REAL,                -- predicted_drawdown - actual_max_loss
    direction_correct   INTEGER,             -- 1 if sign(predicted_return) == sign(actual_return)

    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

-- ── Training queue (samples waiting to be used for fine-tuning) ────────────
CREATE TABLE IF NOT EXISTS training_queue (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    feature_date    TEXT    NOT NULL,
    feature_vector  BLOB    NOT NULL,
    label_hit_target    INTEGER NOT NULL,    -- 0 or 1
    label_return_pct    REAL    NOT NULL,
    label_max_gain_pct  REAL    NOT NULL,
    label_max_loss_pct  REAL    NOT NULL,
    label_regime        INTEGER NOT NULL,    -- 0=UP, 1=SIDEWAYS, 2=DOWN
    source_prediction_id INTEGER,            -- which prediction generated this label
    queued_at       TEXT    NOT NULL,
    used_in_training    TEXT,                -- model version that consumed this sample
    UNIQUE(ticker, feature_date)
);
-- Note: SQLite partial index with NULL check works, but simpler to use a regular index.
CREATE INDEX IF NOT EXISTS idx_tq_queued ON training_queue(queued_at);

-- ── Model registry (versioned models with promotion gate) ──────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    version             TEXT    PRIMARY KEY, -- e.g. 'signal_quality_v012'
    model_type          TEXT    NOT NULL,    -- 'signal_quality' | 'trend_regime' | 'return_forecaster' | 'risk_estimator'
    model_path          TEXT    NOT NULL,    -- filesystem path to .keras file
    trained_at          TEXT    NOT NULL,
    training_samples    INTEGER NOT NULL,
    training_loss       REAL,
    val_accuracy        REAL,
    val_loss            REAL,
    val_return_mae      REAL,                -- for regression models
    walk_forward_score  REAL,                -- composite 0-100 score on validation
    is_production       INTEGER NOT NULL DEFAULT 0,
    promoted_at         TEXT,
    rolled_back_at      TEXT,
    rollback_reason     TEXT,
    notes               TEXT
);
CREATE INDEX IF NOT EXISTS idx_mr_type_prod ON model_registry(model_type, is_production);

-- ── Training log (every training run, successful or not) ───────────────────
CREATE TABLE IF NOT EXISTS training_log (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts              TEXT    NOT NULL,
    run_type            TEXT    NOT NULL,    -- 'initial' | 'fine_tune' | 'full_retrain' | 'rollback'
    model_type          TEXT    NOT NULL,
    model_version       TEXT,                -- NULL if run failed before saving
    samples_used        INTEGER,
    epochs_run          INTEGER,
    final_loss          REAL,
    val_loss            REAL,
    val_accuracy        REAL,
    duration_seconds    REAL,
    status              TEXT    NOT NULL,    -- 'PROMOTED' | 'REJECTED' | 'MARGINAL' | 'FAILED' | 'ROLLED_BACK'
    notes               TEXT
);
CREATE INDEX IF NOT EXISTS idx_tl_ts ON training_log(run_ts);

-- ── Drift alerts (auto-generated by monitoring layer) ──────────────────────
CREATE TABLE IF NOT EXISTS drift_alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_ts        TEXT    NOT NULL,
    alert_type      TEXT    NOT NULL,        -- 'feature_drift' | 'concept_drift' | 'accuracy_drop' | 'data_stale'
    severity        TEXT    NOT NULL,        -- 'INFO' | 'WARN' | 'CRITICAL'
    model_type      TEXT,
    details         TEXT,                    -- JSON with specifics
    action_taken    TEXT,                    -- 'none' | 'retrain_triggered' | 'rollback' | 'alert_sent'
    resolved_at     TEXT
);
CREATE INDEX IF NOT EXISTS idx_da_ts ON drift_alerts(alert_ts);

-- ── Live accuracy tracking (rolling window) ────────────────────────────────
CREATE TABLE IF NOT EXISTS accuracy_daily (
    date            TEXT    NOT NULL,
    model_type      TEXT    NOT NULL,
    predictions_made    INTEGER NOT NULL,
    outcomes_resolved   INTEGER NOT NULL,
    hit_target_rate     REAL,                -- fraction that hit +3%
    direction_accuracy  REAL,                -- fraction where predicted direction matched actual
    return_mae          REAL,                -- mean abs error of return prediction
    drawdown_mae        REAL,
    PRIMARY KEY (date, model_type)
);

-- ── Schema version tracking (for future migrations) ────────────────────────
CREATE TABLE IF NOT EXISTS schema_meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_version', '1.0');
INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('created_at', datetime('now'));
