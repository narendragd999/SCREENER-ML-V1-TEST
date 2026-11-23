# 3% strategy effectiveness research

Research date: 2026-11-10. Entry window: 2023-11-11 onward. Recent cohort: 2025-11-10 onward.

The exact app Top 20 Liquid Leaders preset was used. Current fair value was excluded from historical selection.

The current baseline did not show a profitable edge after modeled costs in this basket: 231 closed trades over three years, 40.26% profitable trades, a 0.8371 profit factor, and -0.2122% average net return per trade. The latest twelve months also had negative expectancy (-0.2333% across 51 trades).

The additional technical gates reduced the sample to five trades in three years. Their positive average result cannot establish effectiveness. The volume/extension variants had more favorable overall results but only 18 or 11 trades, and very few recent observations. These are candidates for broader and forward testing, not proven improvements.

Fair value's effect is still unmeasured. Historical publication-timed valuation snapshots would be needed; present-day fair values cannot be applied retrospectively. A 3% gross limit target produces about 2.79% net at the engine's 10 bps commission per side, before any additional unmodeled costs.

| Variant | Cohort | Closed | Net wins | Target hit | PF | Net expectancy | Avg loss | Win interval (95%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| current_baseline | whole_3years | 231 | 40.26% | 36.8% | 0.8371 | -0.2122% | -2.1813% | [34.14, 46.69] |
| current_baseline | latest_12months | 51 | 35.29% | 29.41% | 0.799 | -0.2333% | -1.7937% | [23.63, 49.01] |
| no_trend | whole_3years | 604 | 38.91% | 34.93% | 0.8007 | -0.2588% | -2.1257% | [35.1, 42.85] |
| no_trend | latest_12months | 183 | 40.44% | 36.61% | 0.8993 | -0.124% | -2.0662% | [33.59, 47.67] |
| volume_not_extended | whole_3years | 18 | 55.56% | 55.56% | 1.3153 | 0.4068% | -2.903% | [33.72, 75.44] |
| volume_not_extended | latest_12months | 4 | 50.0% | 50.0% | 2.2676 | 0.781% | -1.2323% | [15.0, 85.0] |
| volume_market_rs | whole_3years | 11 | 54.55% | 54.55% | 1.3519 | 0.4146% | -2.5925% | [28.01, 78.73] |
| volume_market_rs | latest_12months | 2 | 0.0% | 0.0% | 0.0 | -1.2323% | -1.2323% | [0, 65.76] |
| baseline_price_stop_1_5pct | whole_3years | 223 | 32.74% | 32.74% | 0.7701 | -0.2767% | -1.7897% | [26.91, 39.14] |
| baseline_price_stop_1_5pct | latest_12months | 46 | 36.96% | 36.96% | 0.8771 | -0.145% | -1.8719% | [24.52, 51.4] |
| volume_price_stop_1_5pct | whole_3years | 18 | 44.44% | 44.44% | 1.386 | 0.3744% | -1.746% | [24.56, 66.28] |
| volume_price_stop_1_5pct | latest_12months | 4 | 75.0% | 75.0% | 4.801 | 1.6591% | -1.746% | [30.06, 95.44] |
| technical_gates_confirmation_proxy | whole_3years | 5 | 40.0% | 40.0% | 1.5306 | 0.3874% | -1.2171% | [11.76, 76.93] |
| technical_gates_confirmation_proxy | latest_12months | 3 | 33.33% | 33.33% | 1.0095 | 0.0088% | -1.3839% | [6.15, 79.23] |

## Rules

All variants confirm an EMA9 crossing using a following higher close and higher high, enter the next open, and use a fixed 3% entry target with a 20-session maximum holding. The current baseline requires the engine's full verified uptrend and an EMA-close stop executing at the next open.

Volume variants require confirmation volume >= 1.5x the preceding 20-session average and a confirmation close no more than 1 ATR above EMA9. The market/RS variant additionally requires exact-date Nifty uptrend and positive stock-minus-Nifty 60-session returns. PRICE variants use a fixed 1.5% stop from actual entry, with conservative stop-first ambiguous daily candles.

The technical-gates-at-confirmation proxy requires verified uptrend, volume coverage, Rs 1 crore 20-session turnover, volume >= 1x, extension <= 1.5 ATR, estimated net reward/risk >= 1.5 and no causally confirmed 90-session pivot resistance before the estimated 3% confirmation-close target. It is a proxy for the current live technical gates, not a replay of every latest-scan decision or fair-value selection.

## Coverage

- RELIANCE: 1006 bars; 2022-10-13 to 2026-11-09; .
- TCS: 1006 bars; 2022-10-13 to 2026-11-09; .
- HDFCBANK: 1006 bars; 2022-10-13 to 2026-11-09; .
- INFY: 1006 bars; 2022-10-13 to 2026-11-09; .
- ICICIBANK: 1006 bars; 2022-10-13 to 2026-11-09; .
- BHARTIARTL: 1006 bars; 2022-10-13 to 2026-11-09; .
- SBIN: 1006 bars; 2022-10-13 to 2026-11-09; .
- ITC: 1006 bars; 2022-10-13 to 2026-11-09; .
- HINDUNILVR: 1006 bars; 2022-10-13 to 2026-11-09; .
- LT: 1006 bars; 2022-10-13 to 2026-11-09; .
- BAJFINANCE: 1006 bars; 2022-10-13 to 2026-11-09; .
- TATAMOTORS: 0 bars; None to None; Provider returned no validated completed bars.
- SUNPHARMA: 1006 bars; 2022-10-13 to 2026-11-09; .
- TITAN: 1006 bars; 2022-10-13 to 2026-11-09; .
- MARUTI: 1006 bars; 2022-10-13 to 2026-11-09; .
- AXISBANK: 1006 bars; 2022-10-13 to 2026-11-09; .
- ONGC: 1006 bars; 2022-10-13 to 2026-11-09; .
- NTPC: 1006 bars; 2022-10-13 to 2026-11-09; .
- KALYANKJIL: 1006 bars; 2022-10-13 to 2026-11-09; .
- DIXON: 1006 bars; 2022-10-13 to 2026-11-09; .
- ^NSEI: 1005 bars; 2022-10-13 to 2026-11-09; .

## Baseline verification

- RELIANCE: PASS; 10 replayed trades exactly match existing engine entry, target, exit, net P&L and status fields.
- DIXON: PASS; 16 replayed trades exactly match existing engine entry, target, exit, net P&L and status fields.
- All 5 accepted technical-proxy confirmations pass the actual scanner when its clock is set to the historical confirmation close.

## Limits

- This is a small fixed current 20-stock basket, with survivorship and selection bias; unavailable stocks are not replaced.
- The preceding two years and latest twelve months are chronological entry cohorts from continuous replay, not independently reset portfolios.
- The variants were not tuned in this run, but their specification was informed by known market conventions and prior baseline results; the recent cohort is not a pristine untouched holdout.
- Daily OHLC cannot establish intraday barrier order; PRICE stops resolve ambiguous stop/target candles conservatively stop-first.
- The existing engine's execution model is reused: 5 bps buy/slippage on next-open exits,10 bps commission per side, target limit exits, no modeled market impact or order-book liquidity.
- EMA close stops can lose substantially more than a planned fixed percentage during gaps.
- Realized fixed-allocation return and drawdown omit unrealized positions, compounding, concurrent position constraints, and dividends as separate cashflows.
- Adjusted price histories may be retrospectively revised for corporate actions; transaction-reproducible unadjusted execution is not tested here.
- This comparison includes the named technical gates only. The complete live liquidity/resistance/freshness policy and fair-value policy are not replayed.
- The technical-gates-at-confirmation proxy tests the extra volume,turnover,extension,risk/resistance checks at confirmation close; live latest-scan eligibility can differ after confirmation, lifecycle changes, and next-open gaps.
- No current fair value was applied to historical trades. Publication-timed historical fundamental/model snapshots are required to measure the fair-value filter.
- Trade-level Wilson win intervals and approximate mean-loss intervals ignore dependence across stocks and market days; treat them as descriptive.
