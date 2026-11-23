# Research: a 3% target with fair-value selection

Research date: 10 November 2026. Available completed price data ends 9 November 2026. The initial measurements describe the confirmed-EMA strategy. The subsequent duration comparison and new research mode are described in the scope correction below.

**Scope correction following the user's clarification:** the measurements below concern confirmed entries, EMA stops and a 20-session maximum. They do not refute the user's breakout-close, no-stop, overlapping-signal, 90-session strategy. An exact extraction of that original duration function on the same 19-stock dataset produced **85.46% target hits over 1,327 entries**, or **86.74% among 1,169 entries with the full 90-session opportunity**. The same entries allowed only 20 sessions produced 66.62%. The existing 10% pullback/no-near-resistance subset had 89.71% overall hits and 90.83% among 218 fully mature entries. These are descriptive technical subsets without historical fair-value selection, not a replication of the full 307-stock claim. See `strategy-research-2026-11-10/duration_reconciliation.json` and `duration-strategy-research-2026-11-10.md` for the subsequent comparison. A separate duration research mode has now been added; the original confirmed-EMA trading rules were preserved.

## What the current logic achieves

The basic EMA9/trend backtest did not show a profitable edge after its modeled execution costs in the app's current Top 20 basket. Nineteen stocks had usable data; TATAMOTORS was unavailable. The three-year entry window begins 11 November 2023, with an additional year of data for indicator initialization. These are a fixed current basket and chronological entry cohorts, with selection/survivorship bias; the recent year is not a pristine independent holdout.

| Strategy / cohort | Closed trades | Net profitable trades | Target-hit rate | Profit factor | Mean net return per trade |
| --- | ---: | ---: | ---: | ---: | ---: |
| EMA9 + verified trend, 3 years | 231 | 40.26% | 36.80% | 0.8371 | -0.2122% |
| Same baseline, latest 12 months | 51 | 35.29% | 29.41% | 0.7990 | -0.2333% |
| Baseline + volume >=1.5x and extension <=1 ATR | 18 | 55.56% | 55.56% | 1.3153 | +0.4068% |
| Current technical gates at confirmation proxy | 5 | 40.00% | 40.00% | 1.5306 | +0.3874% |
| Baseline with fixed 1.5% intraday price stop | 223 | 32.74% | 32.74% | 0.7701 | -0.2767% |

Profit factor below one means aggregate net losses exceed aggregate net gains. Average baseline winners returned +2.7096%, while average losers lost 2.1813%; the implied break-even win rate at those observed payoffs is about 44.6%, above the observed 40.26%. The worst EMA-stop loss was 12.1391%, illustrating why a completed-close exit is not a guaranteed stop price.

The favorable filtered results are small samples: the volume/extension comparator had only four trades in the latest year; its whole-period 95% Wilson win-rate interval is approximately 33.7%–75.4%, without accounting for dependent trades. The technical-gates proxy had only three recent trades, with profit factor 1.0095 and average net return +0.0088%. Adding Nifty uptrend and positive relative strength left only two recent trades, both losses. Tighter stops or more filters cannot be assumed to improve results.

All comparisons use causal completed-bar confirmation, next-session open entries, fixed targets from actual entry, one position per ticker, 20-session maximum holding, 5 bps slippage and 10 bps commission per side. EMA-close exits execute at the next available open; intraday price-stop ties resolve conservatively stop-first. Baseline replay was checked against the existing engine for RELIANCE and DIXON. Five accepted technical-proxy confirmations were also checked against the live scanner at their historical decision clock.

The proxy includes volume, turnover, extension, risk and resistance checks at confirmation. It does not reproduce eligibility after a setup has aged or moved, and it does not test fair-value selection. Full execution costs, market impact, publication-timed valuations and an investable concurrent-position portfolio were not validated. Realized allocation curves omit open-position drawdowns.

## What “fair value” currently means

The app fits historical stock prices against annual operating profit/sales and quarterly TTM quantities. This estimates a historical pricing relationship. It is not a cash-flow valuation or a forecast of future earnings. Its annual and TTM models are correlated, and their minimum/maximum estimates are not a statistical confidence interval.

Historical estimates and expanding diagnostics lack accounting publication timestamps. The existing backtest correctly does not apply today's valuation to historical entries. Consequently, the improvement attributable to the fair-value gate is currently unmeasured.

The live gate permits 5% fair-value headroom with medium-confidence evidence. That classification can permit model dispersion up to 25% and diagnostic MAE up to 30%. A 5% gap is therefore too small to treat as a reliable margin of safety. Fundamental CSV files also lack a refresh TTL; date-based model exclusion can reject stale data without ensuring that the latest statements have been downloaded.

An additional live entry inconsistency deserves correction before a new strategy is promoted: eligibility uses remaining reward and resistance relative to the original setup target, while a new paper entry receives its own target. New-entry risk and resistance checks should use the actual proposed new entry and its target consistently.

## Suggested setup to paper-test

Use valuation as a selection filter, then require a fresh recovery in an established trend. These thresholds are research hypotheses, not evidence of a reliable 3% gain.

1. **Valuation evidence:** use the latest available statements; inspect earnings quality, cash flow and debt. For nonfinancial companies, cross-check the app's historical estimate with conservative normalized earnings/peer multiples or cash-flow valuation. For banks, use appropriate bank valuation measures. Do not count correlated regressions as independent confirmation.
2. **Valuation headroom:** propose at least 15% upside to a conservative supported estimate, calculated as `fair_value / proposed_entry - 1`. For example, entry 100 and estimated value 115 give 15% headroom. Fifteen percent is a proposed screening buffer and does not remove valuation uncertainty.
3. **Trend:** require the app's verified SMA50/SMA200 uptrend and inspect Nifty regime and matched 60-session relative strength. Treat market/RS requirements as separately testable conditions; this study did not establish their incremental benefit.
4. **Trigger:** a completed close crosses above EMA9 and the next completed session closes higher and makes a higher high. Prefer confirmation volume >=1.5x its previous 20-session average and price no more than 1 ATR above EMA9. Avoid chasing a later extended price.
5. **Actual entry check:** enter hypothetically at the following session's observed open. Recalculate target, valuation headroom, resistance and reward/risk from that actual fill; skip a gap that invalidates the plan. The nearest confirmed resistance must lie beyond the proposed target.
6. **Exit/risk:** hold a fixed gross 3% target and retain a clearly specified stop/maximum-holding policy for comparison with the baseline. Skip setups whose logical invalidation gives insufficient estimated net reward/risk (proposed minimum 1.5). Do not force a narrow 1.5% stop into every stock: the broad fixed-stop comparator performed worse. Record gaps, costs and unresolved outcomes separately. The current 20-session maximum is the measured baseline; shorter holding limits need their own tests.

## Gross 3% versus net 3%

A normal 3% gross limit-target winner nets 2.7942% under the engine's 10 bps commission per side. Commission alone requires a gross target of approximately 3.2062% for a 3% net return. With roughly 0.30% total price/transaction friction, about 3.3% gross is a starting approximation; actual fees, slippage and fixed charges determine the result. Personal capital-gains tax is not modeled here.

Indian equity delivery STT is 0.1% on purchase and 0.1% on sale; other charges also apply. See the [NSE official charges schedule](https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies).

## Current candidate check

A broad scan of all 307 symbols in this app's selected registry, with a 15-session confirmation window, returned 202 watchlist setups and **zero actionable entries**. One symbol had unavailable data. Nifty's recorded regime was SIDEWAYS as of 9 November 2026. This is the supplied registry, not proof of coverage of every listed NSE stock.

ZYDUSLIFE was the highest-scoring watchlist candidate, but fails the proposed setup:

- Recorded close: 1215.90; confirmation date: 9 November 2026.
- Gross 3% target from that close: 1252.38.
- Nearest confirmed resistance: 1232.10, only about 1.33% above the recorded close.
- Confirmation volume: 1.17x, below the proposed 1.5x filter.
- App composite valuation: 976.00, about 19.73% below the recorded close; model range 942.93–999.95.
- Valuation confidence: medium; latest eligible fundamental observation: 31 March 2026, 224 days old. Quarterly model fits were excluded.

This is a rejected watchlist example, not a suggested current entry. A high technical score does not override resistance or valuation conditions. No current stock in this scan passed the complete entry rules.

## Research support and the next useful validation

[CFA Institute](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/market-based-valuation-price-enterprise-value-multiples) treats momentum indicators and valuation multiples as complementary selection information. [Asness, Moskowitz and Pedersen](https://www.aqr.com/insights/research/journal-article/value-and-momentum-everywhere) report value/momentum evidence across markets and asset classes. That supports investigating their combination; it does not validate this app's EMA rule, threshold choices or a 3% target. [Damodaran's valuation discussion](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/background/valintro.htm) explains why estimates need explicit uncertainty and sensitivity analysis.

The most useful next step is to save every qualifying and rejected signal with its decision timestamp, latest published fundamentals, immutable model estimate, entry/risk rules and later realized costs/outcome. Compare the same technical candidates with and without the valuation filter using a broader dated universe and future paper observations. Use positive net expectancy, loss tails and sufficient independent observations as the evidence standard; do not select rules by the highest win rate on a tiny sample.

Detailed trade-by-trade measurements and reproducible code are in `strategy-research-2026-11-10/`, the valuation audit is `fair-value-research-2026-11-10.md`, and live scan/ZYDUSLIFE model payloads are saved alongside this report.
