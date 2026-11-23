# Local equity research screener

Daily Indian equity analysis built around confirmed EMA9 breakouts, long-term trend, fair-value evidence and swing support structures. The active app is `main.py`, `core/` and `static/index.html`; older routers and the separate ML service are not used to generate this dashboard's recommendations.

## Run

Install `requirements.txt`, then run `python main.py` or the existing `run_screener.bat`. Open http://localhost:8000. Interactive API documentation is at http://localhost:8000/docs.

## How to use it

Read [APP_GUIDE.md](APP_GUIDE.md) for the full beginner workflow, panel controls, progress bars, metrics and local troubleshooting. The dashboard also has **? Help** and **How to use**.

- **Opportunities:** a quick scan of the first 60 symbols in the selected registry. Use the full screener for complete coverage. A prime setup needs technical eligibility, at least 5% fair-value headroom and sufficient valuation evidence.
- **EMA screener:** rank by an explainable score, examine reasons to act or wait, filter by score/actionability, and optionally require an advancing Nifty or 60-session outperformance. Score is a checklist score, not a success probability.
- Full scans defer valuation downloads for watch-only setups so rejected candidates do not delay the technical results. Open **Valuation** for a stock's full fundamental analysis; API callers can request `enrich_watchlist: true`.
- **Stock analysis:** inspect trend, volatility, RSI, 60-session return, available 52-week range, cached valuation and support patterns even when no breakout exists. Price research does not wait for a new valuation download.
- **Swing research:** look for independent support retests, unbroken floors, reversal confirmation and adequate volume. Targets use the nearest confirmed resistance; a distant high or fair value cannot bypass nearer resistance. Proximity alone is watch-only.
- **Position sizing:** enter cash, risk budget, entry, stop and target. Quantity respects cash, loss budget and estimated transaction costs. A gap or EMA-close exit can exceed the estimated stop loss.
- **Paper journal:** record an entry and evaluate subsequent completed sessions. This records hypothetical entries; it does not place orders. Removing one trade ID preserves other entries in the same ticker.
- **Backtest:** compare the original 3%/90-session no-stop close-reference study, a next-open duration simulation with one position per stock, and confirmed EMA with its EMA stop. Inspect complete/mature target counts, costs, waiting time, adverse moves, per-stock results and complete exports. The optional current-FV subset is explicitly retrospective.

## Accuracy rules

1. Daily candles from the current session are excluded before 15:30 India time. Bars are validated, sorted and deduplicated; exchange-local dates are preserved. Scans request at least 365 calendar days to initialize the 200-session trend. Data failures and stale quotes are reported.
2. EMA9 must cross from at/below to above. The following completed candle must close higher and make a higher high. A verified uptrend requires price above SMA50 and SMA200, SMA50 above SMA200 and rising, and a positive 60-session regression slope.
3. Historical setup entries use the next session's open, and targets remain fixed from that reference entry. A fresh confirmation without a subsequent bar has an estimated entry. A new paper entry uses its own entry and target; old setup targets are separate. Historical target hits and stops cannot reactivate merely because the latest price recovers.
4. Recommendation gates require liquidity, confirmation volume against the preceding 20 sessions, limited ATR extension, no resistance before the target, freshness and sufficient estimated reward/risk. The net reward/risk estimate reserves 0.5 ATR of risk and 0.30% default round-trip costs. These are screening assumptions, not measured execution guarantees.
5. EMA invalidation is observed at a completed close and executed at the next available open. Final-bar exits remain pending. Intraday structural stops use conservative stop-first handling when both stop and target appear in one daily candle. Backtest defaults are 5 bps slippage and 10 bps commission per side, configurable in the UI.
6. Accounting March-ending periods align to their correct fiscal year. Missing values stay missing, losses stay negative, and TTM needs consecutive quarters. Small-sample, weak, negative-slope, nonfinite or stale models are excluded. Fair-value gaps use the actual screened close, not a cached quote.
7. Valuation regressions are descriptive. Historical fair-value charts use retrospective fits and lack accounting publication dates. Today's fair value never gates the historical primary strategy. The optional cached-current-FV cohort is retrospective and preserves its snapshots; historical valuation efficacy remains unverified.
8. Profitable closed-trade rate and target-hit rate are separate. Duration studies retain unresolved trades and report both the full-entry observed proportion and a cohort with the entire horizon available. They do not compound overlapping trades. The confirmed-EMA curve shows fixed-allocation realized P&L and excludes open drawdowns. Current ticker lists may contain survivorship bias.

## Verify the logic

With pytest installed, run `python -B -m pytest tests -q -p no:cacheprovider`. Tests use deterministic price paths and mocked data sources to check execution order, fixed targets, missing data, accounting alignment, structural support, valuation eligibility and API calculations.

These checks establish calculation correctness. A higher future trading success rate requires forward paper-trade results and sufficiently large historical samples; the dashboard does not display an invented accuracy figure.
