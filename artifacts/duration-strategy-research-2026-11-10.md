# Duration strategy comparison

Reviewed on 10 November 2026 using saved prices only. No network calls or production changes were made by this script.

## What the original result establishes

The exact original duration engine produces **85.46% target hits with 90 sessions**, versus **66.62% with 20** on the same 19 available stocks. The user's longer holding period materially changes the result; the earlier short-hold comparison did not represent it.

The original pullback-at-least-10%-below-prior-high plus away-from-resistance subset reaches **89.71% across 272 entries**, or **90.83% across 218 entries with a complete 90-session follow-up**. Its modeled average net return is **+1.5408% per entry**, profit factor **2.756**, worst net trade **−17.81%**, and worst adverse excursion **−30.12%**. Adding SMA50 raises target hits to 93.33%, but leaves just 15 entries. These descriptive subsets make a 90%+ result plausible; they do not replicate all 307 stocks or historically available fair value.

## Executable next-open comparison

Scope: 19 stocks from the saved 20-stock preset; TATAMOTORS has no validated cached prices. Entries run from 2023-11-11 through 2026-11-09, with earlier bars used for indicator warmup. Fair value is excluded because dated historical valuation inputs are unavailable.

Each strict EMA9 cross is known at its close. Buy at the next observed open with 5 bps slippage; hold one position per ticker, with no stop. The target is an unrounded 3% above the executed buy price. Commission is 10 bps per side. Target touches assume a sell-limit fill without exit slippage; market time exits use 5 bps exit slippage. The entry session counts as session one. An unhit horizon close triggers an exit at the following open. Incomplete follow-up remains open/censored, and an unavailable exit open remains pending.

The full-horizon cohort includes an entry only if its entire configured target observation period exists, even when it won early. This avoids comparing recent early winners with older losses while dropping recent unresolved trades. Net expectancy below is the arithmetic mean of net returns for closed trades in that full-horizon cohort; its closed count is shown whenever a pending exit prevents a realized return. It is not a compounded portfolio return.

| Next-open variant | All entries / open | Full-horizon entries / closed | Full-horizon target hits | Net expectancy | Net profit factor | Worst net loss | Worst observed drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 sessions, all trends | 639 / 2 | 621 / 621 | 66.18% | 0.1815% | 1.102 | -20.80% | -26.25% |
| 90 sessions, all trends | 385 / 11 | 349 / 348 | 85.39% | 0.8178% | 1.493 | -32.05% | -42.98% |
| 90 sessions, above SMA50 | 324 / 10 | 295 / 294 | 85.42% | 0.8331% | 1.510 | -32.05% | -42.98% |
| 90 sessions, trend + volume + extension | 57 / 0 | 55 / 55 | 80.00% | 0.0143% | 1.006 | -21.52% | -33.21% |
| 90 sessions, away from resistance | 322 / 9 | 289 / 289 | 87.54% | 1.0247% | 1.668 | -32.05% | -42.98% |
| 90 sessions, pullback at least 10% | 123 / 2 | 100 / 100 | 84.00% | 0.5765% | 1.295 | -33.89% | -37.43% |
| 90 sessions, pullback + away from resistance | 117 / 2 | 95 / 95 | 85.26% | 0.7703% | 1.428 | -33.89% | -37.43% |

Moving to next-open execution changes the buy price and target barrier. One position per ticker suppresses signals while a trade is open. A longer holding period also changes which later trades are admitted. Thus these variants evaluate executable policies with different accepted entries; the original overlapping study isolates target touches on a common signal population more directly.

The additional research gate requires a verified SMA50/SMA200 uptrend (including rising SMA50 and positive 60-session slope), signal volume at least 1.5 times its previous 20-session average, and EMA extension from zero to one ATR. These fixed thresholds came from the predefined hypotheses. No thresholds were searched or retuned from this comparison.

The practical legacy gates are also compared separately and together: avoid a prior 20-session high above the signal close when it is within 3% of that close; require the signal close to be at least 10% below the prior 60-session high. Both use only history available at the signal close. A next-open gap can alter actual entry resistance distance; the signal-close filter does not inspect that future opening price.

## Recent year and earlier periods

Each row uses entry dates within the stated period and the full-horizon membership rule. The latest year's mature 90-session entries stop well before the final price date; recent unresolved trades remain visible in the JSON.

| Variant | Entry cohort | Full-horizon entries / closed | Target hits | Net expectancy | Net profit factor |
| --- | --- | ---: | ---: | ---: | ---: |
| 20 sessions, all trends | 2023-11-11 to 2024-11-09 | 218 / 218 | 73.85% | 0.9559% | 1.794 |
| 20 sessions, all trends | 2024-11-10 to 2025-11-09 | 209 / 209 | 66.51% | 0.5561% | 1.393 |
| 20 sessions, all trends | 2025-11-10 to 2026-11-09 | 194 / 194 | 57.22% | -1.0923% | 0.615 |
| 90 sessions, all trends | 2023-11-11 to 2024-11-09 | 171 / 171 | 91.23% | 1.5707% | 2.501 |
| 90 sessions, all trends | 2024-11-10 to 2025-11-09 | 103 / 103 | 84.47% | 1.0099% | 1.709 |
| 90 sessions, all trends | 2025-11-10 to 2026-11-09 | 75 / 74 | 73.33% | -1.1894% | 0.650 |
| 90 sessions, above SMA50 | 2023-11-11 to 2024-11-09 | 153 / 153 | 90.20% | 1.4220% | 2.216 |
| 90 sessions, above SMA50 | 2024-11-10 to 2025-11-09 | 82 / 82 | 82.93% | 0.8194% | 1.528 |
| 90 sessions, above SMA50 | 2025-11-10 to 2026-11-09 | 60 / 59 | 76.67% | -0.6748% | 0.772 |
| 90 sessions, trend + volume + extension | 2023-11-11 to 2024-11-09 | 35 / 35 | 82.86% | 0.7513% | 1.479 |
| 90 sessions, trend + volume + extension | 2024-11-10 to 2025-11-09 | 9 / 9 | 77.78% | 0.0218% | 1.010 |
| 90 sessions, trend + volume + extension | 2025-11-10 to 2026-11-09 | 11 / 11 | 72.73% | -2.3370% | 0.495 |
| 90 sessions, away from resistance | 2023-11-11 to 2024-11-09 | 134 / 134 | 93.28% | 1.7681% | 2.931 |
| 90 sessions, away from resistance | 2024-11-10 to 2025-11-09 | 90 / 90 | 85.56% | 1.1713% | 1.886 |
| 90 sessions, away from resistance | 2025-11-10 to 2026-11-09 | 65 / 65 | 78.46% | -0.7109% | 0.771 |
| 90 sessions, pullback at least 10% | 2023-11-11 to 2024-11-09 | 27 / 27 | 96.30% | 2.3559% | 7.045 |
| 90 sessions, pullback at least 10% | 2024-11-10 to 2025-11-09 | 38 / 38 | 86.84% | 1.1805% | 1.848 |
| 90 sessions, pullback at least 10% | 2025-11-10 to 2026-11-09 | 35 / 35 | 71.43% | -1.4520% | 0.616 |
| 90 sessions, pullback + away from resistance | 2023-11-11 to 2024-11-09 | 26 / 26 | 96.15% | 2.3391% | 6.779 |
| 90 sessions, pullback + away from resistance | 2024-11-10 to 2025-11-09 | 34 / 34 | 88.24% | 1.5133% | 2.385 |
| 90 sessions, pullback + away from resistance | 2025-11-10 to 2026-11-09 | 35 / 35 | 74.29% | -1.1167% | 0.683 |

The latest mature 90-session baseline cohort has **73.33% target hits** and **-1.1894% net expectancy** (75 entries, 74 closed). The longer horizon's pooled improvement does not establish stable returns across market periods.

## What to adopt

**90 sessions, above SMA50:** Inconclusive filter benefit: target frequency and net expectancy do not both improve consistently across adequately populated time blocks.

**90 sessions, trend + volume + extension:** Inconclusive filter benefit: target frequency and net expectancy do not both improve consistently across adequately populated time blocks.

**90 sessions, away from resistance:** Candidate for further testing: target frequency and net expectancy improve in the pooled cohort and all three time blocks. Still unvalidated; the latest cohort remains net negative.

**90 sessions, pullback at least 10%:** Inconclusive filter benefit: target frequency and net expectancy do not both improve consistently across adequately populated time blocks.

**90 sessions, pullback + away from resistance:** Inconclusive filter benefit: individual time blocks improve, but pooled target frequency and net expectancy do not. Accepted-entry and time-cohort weighting differ.

Preserve the 90-session no-stop target study as a distinct research view. Show executable next-open results beside it, with separate target-hit, profitable-trade, open-position and mature-cohort denominators. A filter should not be presented as an accuracy improvement based on pooled target hits alone.

## Saved workbook evidence

`sample-data-5-year-250-days.xlsx`, Sheet1!A1:U5235, contains 5,234 successful 3% examples across 294 stocks; median target time is three sessions and p90 is 21. It contains only WIN rows and no original run configuration or missing-trade list, so it cannot supply the full win-rate denominator. Of these rows, 4,366 are Prime and 2,975 are Prime plus uptrend. There are 1,350 overlapping same-ticker entries. Fair values are assigned per April–March fiscal year, but publication/availability dates are absent. These examples support investigating the setup without proving historical valuation effectiveness.

## Additional older price cache

The read-only ML database contains 212,312 OHLCV rows across 106 tickers (93 have at least 500 bars), covering July 2016 to July 2026. No structurally invalid OHLC rows were found. Downloader code uses daily auto-adjusted Yahoo history, but the rows do not preserve source, interval, or adjustment version and share their schema with demo/test writers. This distinct July snapshot was not blended into the November comparison. Its ticker coverage is retained in the JSON for a separate, explicitly dated study after provenance checks.

## Limits and reproduction

The sample uses currently selected stocks, daily target-touch fills, and correlated entries. It excludes historical fair value, delisted-stock reconstruction, order liquidity, and a fixed-capital portfolio. MAE includes the target day's full low because intraday ordering cannot be determined from daily bars. Results are descriptive and cannot establish a stable future 90% success probability.

Run `python artifacts/strategy-research-2026-11-10/compare_duration_execution.py` from the repository root. Inputs are the saved `data_manifest.json`, its 19 stock price CSVs, `duration_reconciliation.json`, and `saved-backtest-reconciliation-2026-11-10.json`. The comparison JSON records each cohort, ticker breakdown, execution parameters, filter assessment and older database coverage. All data downloads are disabled by design.
