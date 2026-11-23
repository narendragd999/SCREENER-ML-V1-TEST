# Legacy 90-bar duration backtest: reconciliation audit

Read-only review on 10 November 2026 of `ema9_router.py`, specifically `_duration_backtest_ticker` and `/api/ema9/duration-backtest`. No network requests, imports of the legacy module, or production changes were made. The user's reported 90%+ figure has not yet been reproduced from their exact saved request/data; this audit establishes what that figure can mean in this engine.

## What a high result can legitimately mean

With a fixed 3% target, `max_hold_days=90` and all optional filters disabled, the statistic is the proportion of eligible EMA-cross signals whose later daily high reaches 3% above the breakout close before the engine exits them. This is a valid historical target-touch question under its stated assumptions. It differs substantially from the current core strategy's confirmed next-session entry, 20-session holding limit, EMA-close stop and net-profitable-trade definition. A 90%+ duration result does not conflict automatically with a much lower win rate from that other strategy.

Longer holding periods and absence of a stop let a stock fall substantially, recover, and still count as a 3% winner. The result can therefore be high without demonstrating a small drawdown, fast payoff, net execution profit, or a deployable portfolio return.

## Exact engine conventions

| Item | Legacy duration behavior and source |
| --- | --- |
| Entry trigger | Strict cross: prior close `<` EMA9 and current close `>` EMA9. No next-day confirmation (`2107-2119`). |
| Entry price | Same breakout day's closing price (`2251-2257`). This is an idealized reference fill; the final closing-price cross is only fully known after that close. |
| Optional trend filter | `require_uptrend=True` requires only close above SMA50 (`2111-2113`). SMA200/rising-trend fields are annotations, not this gate. |
| Multiple signals | Every eligible cross becomes a distinct trade, even while earlier positions in that stock remain open (`1882-1886`, `2299-2300`). There is no cash or position-count limit. |
| Holding duration | `days_held = current_bar_index - entry_bar_index` (`1997`). Ninety means 90 trading observations after entry, not 90 calendar days. |
| First outcome check | The next bar after entry. Exits are checked before new signals (`1991-1994`). The breakout day's earlier high is not used to award a target hit. |
| Winner | A later daily high reaches the fixed target; exit assigned to target (`2004-2007`). There is no stop. A large prior adverse move does not invalidate the entry. |
| Timeout/loss | At the holding limit, exit at that day's close. Close below entry is LOSS; close at/above entry but short of target is TIMEOUT (`2008-2011`). A positive timeout is still not a WIN. |
| Costs | `gain_pct` is gross price change (`2014`); no commission, taxes, liquidity model or slippage is deducted. |
| Date range | Start/end filter breakout/entry dates (`1966-1978`, `2115-2117`). Simulation continues using available price data after the requested end date. End date is an entry cutoff, not an observation cutoff. This is legitimate for a historical entry cohort if disclosed. |
| Final-data treatment | All still-open trades are force-exited at the last available bar, even when fewer than 90 sessions have elapsed (`2008-2011`). Entry on the last bar is skipped (`2096-2099`). |
| Denominator | Pooled number of primary trades, including WIN, LOSS and TIMEOUT; numerator is target-hit WIN count (`2983-2996`). It is trade weighted, not the mean of ticker win rates. |

**Important:** unfinished 90-session opportunities are not silently omitted or counted as open. Their final-data closes become LOSS/TIMEOUT. This gives recent entries less opportunity to reach the target and generally reduces their observed hit frequency relative to full follow-up. A corrected study should separately report mature target outcomes and right-censored recent signals, or state that the last-data forced exit is part of the policy.

The engine also computes a future post-target peak until an EMA breakdown (`2020-2073`). This is an explicitly retrospective what-if annotation and does not change the primary WIN outcome. It should not be interpreted as an attainable exit selected in advance.

## Universe and downloaded history

- The route silently takes the **first 300 input symbols**, not all 307 (`2696-2700`). It does not deduplicate the request; duplicate inputs can repeat trade records and change trade weighting.
- `DOWNLOAD_PERIOD` is **`10y`** (`126`). The route comment saying five years is stale.
- `_batch_download_async(tickers, "1d", 500)` (`2722`) does **not** request 500 days. The downloader ignores its `interval` and `lookback_days` parameters and passes `DOWNLOAD_PERIOD` to `_download_single_ticker` (`553-572`). The number of available bars is still limited by Yahoo/listing history.
- Download uses adjusted OHLC from `.NS`, falling back to `.BO` (`464-476`). Price-cache TTL is four hours (`129`, `163-177`); the engine itself only copies/drops NaNs (`1894`) and does not apply the active core's completed-candle validation. Exact as-of dates and the exchange source should be recorded for reproduction.

## Fair value and optional filters

Fair value is calculated/enriched once per stock using the stock's latest available close (`2790-2797`). The same current composite estimate is copied to every historical trade, and `is_undervalued` means historical entry price below that current estimate (`2826-2835`). This is a **retrospective current-valuation cohort**, not evidence that the stock was knowably undervalued on its historical entry date.

`fetch_fv=True` annotates results. With `prime_only=False`, it does not itself change the primary aggregate. `prime_only=True` retains only `is_undervalued=True`, automatically enabling FV enrichment (`2714-2718`, `2873-2884`). Thus an all-signals 90% figure and a current-FV-prime 90% figure have different meanings. Establishing historical fair-value efficacy requires publication-timed valuation snapshots.

Resistance, pullback and structural filters are applied as subsets after all signal simulations (`2893-2968`). Their annotations principally use pre-entry price windows. Because every signal was simulated independently and overlaps are allowed, removing an earlier trade does not block a later signal. Those filters can form a causal technical signal cohort if thresholds were fixed in advance; choosing thresholds after inspecting outcome tables does not constitute out-of-sample validation.

Returned cohorts differ: with only prime mode enabled, the top-level `all_trades` may retain the full set while `aggregate` uses prime trades; later technical filters overwrite `all_trades`. Fair-value subgroup aggregates preserve earlier subsets, whereas technical subgroup tables are staged along the filter sequence. Compare the exact primary count and config, rather than assuming every displayed table has the same denominator.

## Returns and risk

The route compounds each trade's eventual gross return as though the entire capital were sequentially available, sorted by entry date (`3013-3018`). The equity curve recognizes that eventual return at the entry date (`3022-3035`). Since trades overlap both within and across tickers, this is **not a capital-constrained portfolio return or reliable portfolio drawdown curve**. Per-signal average gross gain remains interpretable, and it should be supplemented with actual costs, adverse excursion, holding durations, concurrent exposure and concentration.

A 90% target-hit rate can coexist with negative expectancy if the remaining failures are sufficiently large. For example, hypothetical 90% wins at +3% and 10% losses at -30% average -0.3% before costs. This is an illustration, not a measurement of the user's result. Multiple entries in the same ticker/market episode are correlated, so uncertainty should account for ticker/time clustering.

## Reproduction and useful validation

Preserve the exact input symbol order and count, target mode (fixed vs ATR), holding sessions, requested entry start/end dates, all flags/thresholds, actual downloaded coverage/as-of dates and primary trade count. Re-run the original no-stop signal study first; do not substitute a confirmed-entry/EMA-stop strategy and label it a correction.

Then compare separately: (1) full-follow-up versus final-data-censored cohorts, (2) same-close reference versus executable next-open fills with costs, (3) 90-session target-touch probability and median time-to-target/adverse excursion, (4) limited concurrent exposure and portfolio equity, and (5) publication-timed or forward-recorded fair-value subsets. These comparisons preserve the valid question behind the user's result while measuring whether it supports a practical 3% setup.
