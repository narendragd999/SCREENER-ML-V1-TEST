# Saved backtest evidence reconciliation

Reviewed on 10 November 2026. The user confirmed a 3% target, up to 90 days **per trade**, all 307 tickers, fair value, and filters described as “all.” The exact backtest start/end dates and individual filter switches remain unspecified.

## Main finding

The project contains substantial saved examples of successful 3% equity setups. The main workbook contains **5,234 wins across 294 tickers**, with a median target time of **3 trading sessions**. This supports taking the user's result seriously and warrants replaying their longer holding setup.

The available workbook contains **only WIN outcomes**, and does not contain the original request configuration, failed ticker list, losses, timeouts, or open trades. Its 100% exported-row hit rate cannot establish the full strategy's win rate. This does **not** show the user's 90%+ claim is false; the original full denominator is missing.

## Equity duration export

Source: `E:\application\SCREENER-NEW-V1\sample-data-5-year-250-days.xlsx`, `Sheet1!A1:U5235`.

| Exported population | Rows | Distinct tickers | WIN rows | Median sessions to target | 90th percentile sessions |
| --- | ---: | ---: | ---: | ---: | ---: |
| All exported rows | 5,234 | 294 | 5,234 | 3 | 21 |
| Prime / undervalued = YES | 4,366 | 278 | 4,366 | 3 | 21 |
| Uptrend | 3,668 | 280 | 3,668 | 3 | 21 |
| Prime and uptrend | 2,975 | 254 | 2,975 | 3 | 20 |

Every row has `Is Near Resistance = NO`, so adding that flag to any subgroup leaves the counts unchanged. Pullback and structural support fields are absent from this export, so “all filters” cannot be reconstructed.

- Entry dates span **1 January 2020 to 24 June 2026**. Exit dates span **2 January 2020 to 25 June 2026**. The filename is not reliable evidence of the actual date or holding settings.
- Every entry date equals its breakout date. None exits on its entry date. This is consistent with the legacy breakout-close duration model; it differs from confirmation-day or next-open entry rules.
- Exported winners held **1–59 trading sessions**. All hit within 90 sessions. Calendar holding had a median of **5 days**, a 90th percentile of **32 days**, and a maximum of **94 days**.
- Three exported wins took more than 90 calendar days: MAZDOCK (91), TVSHLTD (91), and MINDACORP (94). Thus the selected sample's target-hit fraction within 90 **calendar** days is **99.9427%**, versus 100% within 90 sessions. Neither rate supplies the missing full denominator.
- **1,350 entries (25.79%)** began while another exported trade in the same ticker was still open. These trades require overlapping positions, or a replay that treats each signal independently. They cannot all represent one position per ticker.
- There are no duplicate ticker/entry-date pairs. Rows are sorted by descending post-target peak gain, rather than by entry date.

### Fair value provenance

The workbook does contain varying historical-year fair values: exactly **one fair value per ticker per April–March fiscal year** across **1,231 ticker/fiscal-year groups**. This is stronger evidence than a single current fair value reused for every historical trade.

The workbook has no valuation calculation timestamp, financial publication date, source/model version, or availability date for each fiscal year's value. Therefore it does not establish that the displayed valuation was available on the entry date. The archived frontend's `_getFVForTrade` selects a fiscal-year value from yearly data and can fall back to an adjacent year. Historical-year assignment alone does not resolve publication timing.

### Displayed 3% gains

Most exported gains are very close to 3%. Nine WIN rows display gains below 2.9%; the lowest is **2.54%** for CUPID, with entry ₹1.97 and target/exit ₹2.02 (`Sheet1!E235:N235`, gain in `S235`). Low adjusted historical prices and two-decimal serialized targets materially change percentage calculations. This should be checked against full-precision prices before treating every WIN as an exact 3% gross gain.

No costs, worst interim drawdown, or portfolio capital ledger appears in the workbook. These successful target touches alone do not determine net expectancy or the capital needed to hold overlapping trades.

## Other equity workbook uses different rules

Source: `E:\application\SCREENER-NEW-V1\9EMA-ANALYSIS.xlsx`.

- `TICKERS!A1:H204` contains 203 ticker summaries. Summing its `W / L / T` counts gives **2,482 trades: 1,082 wins, 1,399 losses, and 1 timeout**, a **43.5939% target-hit rate**.
- `ALL TRADES!A1:J250` contains only **249 trades across 23 tickers**: 122 wins and 127 losses (**48.9960%**). It is a partial detail set, not all 2,482 summary trades.
- Its entry dates span 3 September 2024 to 10 June 2026. All entries occur after their breakout date. The shown short losses are consistent with a different exit policy.

These figures do not rebut the 90-day no-stop duration result because the entry/exit model and exported population differ.

## Cached options results

`backtest_cache` contains 137 JSON files (21,880,852 bytes), all with surge/drop and short-CE options configurations. Ten have `max_gate3_days = 90`; this parameter belongs to the options strategy.

| Cache file | Tickers | Trades | Wins / losses | Accuracy | Cached at (IST) |
| --- | ---: | ---: | ---: | ---: | --- |
| `d8d89cd7b8cc61fcd9fd177886628b39.json` | 241 | 1,014 | 948 / 66 | 93.5% | 14 June 2026 13:01:24 |
| `571147adf11bb3065041e9d14e59e473.json` | 241 | 34 | 33 / 1 | 97.1% | 14 June 2026 12:34:13 |

The first uses current expiry and the second next expiry. These confirm saved 90%+ results exist, but for short-call options rather than the equity fair-value/3% setup. `Report-Complete` also describes the short-CE strategy.

## Additional records checked

- `scan_log.json`: 57 scan records, with a maximum of **307 tickers**. The last record is 27 July 2026. This confirms the scanner's 307-ticker universe, not the backtest's completed universe.
- `trade_log.json`: two manual KALYANKJIL trades, both from a Prime signal with a 3% target. One closed at −0.41%; the other closed at +2.86% after four calendar days. They are not a 307-ticker backtest.
- `signals.json` is empty. The support cache, fundamentals cache, chartink signals, and ML decision/paper-trade files contain scans or decisions, not a saved full duration result.
- Both SQLite files were opened read-only. `ml_service/data/ml_intel.db` has 212,312 OHLCV rows and 539 predictions, but **zero rows** in `backtest_runs`, `historical_predictions`, `historical_accuracy`, `outcomes`, and `accuracy_daily`. `ml_service/ml_intel.db` has empty prediction/history tables.
- A precise search for saved `max_hold_days`, `prime_only`, `win_rate_pct`, and duration-backtest payloads found no full run outside the existing research artifacts and source code.

## Required reconciliation

The comparison should reproduce the user's 90-session legacy duration setup first, then report how execution and risk controls change it. Preserve the original target-touch result alongside the stricter result so the difference is understandable.

To certify the original 90%+ figure, the saved run must include its exact ticker request and completed/failed ticker counts, start/end dates, each filter switch, holding unit, every trade outcome, and valuation availability dates. A full report export or persisted request/result JSON would supply that evidence. The available successful-trade workbook cannot supply those missing fields.

No production code, data, workbooks, or databases were changed by this evidence audit.
