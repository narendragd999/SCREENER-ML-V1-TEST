-- ============================================================================
-- Historical Performance Schema v2 — adds 3 tables to existing ml_intel.db
-- ============================================================================
-- Run AFTER schema.sql. These tables track HISTORICAL backtest results
-- (what the model WOULD have predicted for past data, and what actually happened).
--
-- Apply with:
--   sqlite3 data/ml_intel.db < db_schema_v2.sql
-- ============================================================================

-- ── Historical Predictions (one row per (ticker, date, model_version)) ─────
CREATE TABLE IF NOT EXISTS historical_predictions (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                      TEXT    NOT NULL,
    prediction_date             TEXT    NOT NULL,           -- 'YYYY-MM-DD'
    model_version               TEXT    NOT NULL,           -- e.g. 'signal_quality_v012_2026-01-15'

    -- What the model predicted (as of prediction_date)
    predicted_signal_quality    REAL,
    predicted_return            REAL,                       -- %
    predicted_max_drawdown      REAL,                       -- %
    predicted_actionable        TEXT,                       -- BUY / HOLD / SELL / etc.
    predicted_ml_score          REAL,                       -- 0-100
    predicted_trend_regime      TEXT,                       -- UPTREND / SIDEWAYS / DOWNTREND

    -- What actually happened in the next 5 days
    actual_return               REAL,                       -- % (close-to-close)
    actual_max_gain             REAL,                       -- % peak gain
    actual_max_loss             REAL,                       -- % worst loss (negative)
    actual_hit_target           INTEGER,                    -- 1/0/NULL (NULL = still running)
    actual_days_to_target       INTEGER,                    -- NULL if never hit

    -- Status: HIT (target achieved), MISSED (5 days passed, no target),
    --         RUNNING (within 5-day window), NO_DATA (insufficient future data)
    status                      TEXT    NOT NULL,

    -- Performance metrics
    return_error                REAL,                       -- predicted_return - actual_return
    direction_correct           INTEGER,                    -- 1 if signs match, 0 otherwise

    -- Context at prediction time
    entry_price                 REAL,
    target_price                REAL,
    stop_loss_price             REAL,
    exit_price                  REAL,                       -- close at day+5 (if resolved)

    -- Trade simulation
    pnl_pct                     REAL,                       -- actual return (for HIT/MISS)
    pnl_r_multiple              REAL,                       -- pnl / risk_per_share

    -- Metadata
    backtest_run_id             INTEGER,                    -- which backtest run produced this
    computed_at                 TEXT    NOT NULL,

    UNIQUE(ticker, prediction_date, model_version)
);

CREATE INDEX IF NOT EXISTS idx_hp_ticker ON historical_predictions(ticker);
CREATE INDEX IF NOT EXISTS idx_hp_date ON historical_predictions(prediction_date);
CREATE INDEX IF NOT EXISTS idx_hp_model ON historical_predictions(model_version);
CREATE INDEX IF NOT EXISTS idx_hp_status ON historical_predictions(status);
CREATE INDEX IF NOT EXISTS idx_hp_ticker_date ON historical_predictions(ticker, prediction_date);

-- ── Historical Accuracy (aggregated stats per scope) ───────────────────────
CREATE TABLE IF NOT EXISTS historical_accuracy (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_type          TEXT    NOT NULL,   -- 'stock', 'model', 'sector', 'overall'
    scope_id            TEXT    NOT NULL,   -- ticker / model_version / sector_name / 'ALL'
    period              TEXT    NOT NULL,   -- '7d', '30d', '90d', '1y', '5y', '10y', 'all'

    -- Counts
    total_predictions   INTEGER NOT NULL,
    hits                INTEGER NOT NULL,   -- target achieved
    misses              INTEGER NOT NULL,   -- target missed
    running             INTEGER NOT NULL,   -- still in 5-day window
    no_data             INTEGER NOT NULL,   -- insufficient future data

    -- Rates
    hit_rate            REAL,               -- hits / (hits + misses), NULL if no resolved
    direction_accuracy  REAL,               -- predicted direction matched actual
    avg_return          REAL,               -- avg actual_return %
    avg_predicted_return REAL,
    avg_max_gain        REAL,
    avg_max_loss        REAL,
    return_mae          REAL,               -- mean |predicted - actual|
    return_bias         REAL,               -- mean(predicted - actual), + = overpredict

    -- Trade metrics
    avg_days_to_target  REAL,
    win_loss_ratio      REAL,               -- avg_win / avg_loss
    profit_factor       REAL,               -- sum(gains) / |sum(losses)|
    sharpe_ratio        REAL,               -- mean(returns) / std(returns) * sqrt(252)
    expectancy          REAL,               -- (hit_rate * avg_win) - (miss_rate * avg_loss)

    -- Best/worst
    best_trade_pct      REAL,
    worst_trade_pct     REAL,
    avg_hold_days       REAL,

    -- Equity curve (sampled, max 100 points) — stored as JSON array
    equity_curve_json   TEXT,

    computed_at         TEXT    NOT NULL,
    UNIQUE(scope_type, scope_id, period)
);

CREATE INDEX IF NOT EXISTS idx_ha_scope ON historical_accuracy(scope_type, scope_id);
CREATE INDEX IF NOT EXISTS idx_ha_period ON historical_accuracy(period);

-- ── Backtest Runs (audit log of backtest executions) ───────────────────────
CREATE TABLE IF NOT EXISTS backtest_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts              TEXT    NOT NULL,
    run_type            TEXT    NOT NULL,   -- 'initial', 'incremental', 'manual', 'scheduled'
    model_version       TEXT    NOT NULL,
    tickers_count       INTEGER NOT NULL,
    start_date          TEXT    NOT NULL,
    end_date            TEXT    NOT NULL,
    date_range_days     INTEGER NOT NULL,
    predictions_generated INTEGER NOT NULL DEFAULT 0,
    predictions_skipped INTEGER NOT NULL DEFAULT 0,
    duration_seconds    REAL,
    status              TEXT    NOT NULL,   -- 'running', 'completed', 'failed', 'partial'
    progress_pct        REAL    NOT NULL DEFAULT 0,
    current_ticker      TEXT,
    current_date        TEXT,
    error_message       TEXT,
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS idx_br_ts ON backtest_runs(run_ts);
CREATE INDEX IF NOT EXISTS idx_br_status ON backtest_runs(status);

-- ── Schema version tracking ────────────────────────────────────────────────
INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('schema_v2_version', '2.0');
INSERT OR IGNORE INTO schema_meta (key, value) VALUES ('v2_installed_at', datetime('now'));
