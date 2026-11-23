# Fair value and the 3% setup: read-only logic audit

Reviewed on 10 November 2026. No production code or thresholds changed.

## What the app's fair price means

The engine fits historical share-price levels to total operating profit, sales and trailing-twelve-month fundamentals. It evaluates those fitted relationships at the latest observed fundamental, which is already included in the final training sample. This is a descriptive estimate of the historical pricing relationship. It is not a discounted-cash-flow estimate, a per-share earnings valuation or a forecast of future earnings or a 3% trading gain.

Evidence: `sma_router.py:463-471`, `503-507`, `559-587`, `597-626`. Models share correlated profit/revenue inputs, so agreement does not constitute independent evidence. `core/fair_value.py:62-64` states this and caps confidence at medium (`128-129`).

## Three principal limits on effectiveness claims

1. **Publication-timed predictive evidence is missing.** Annual prices are the fiscal year's last close (`sma_router.py:455-456`); quarterly TTM is aligned with quarter-end prices. A period's completed accounts are ordinarily released later. The expanding diagnostic trains on earlier rows, but predicts the held-out period's price using that period's reported fundamental (`314-333`). It is a diagnostic of the relationship, not a tradable forecast known at the price date. Historical valuation curves use the final full-sample fit; `fv_router.py:142-143` explicitly labels them retrospective and not point-in-time safe.
2. **The historical backtest is a different policy.** `core/backtest.py:241` replays EMA confirmation, the selected trend rule and execution. It does not replay live volume, liquidity, ATR-extension, resistance, reward/risk, market/relative-strength or fair-value gates. The fair-value comparison is correctly unavailable (`256-258`). The measured baseline's 231 trades, 40.26% net wins, profit factor 0.837 and -0.212% expectancy therefore describe the tested EMA/trend policy. The confirmation-time technical proxy has only five trades and cannot establish a dependable edge. Neither statistic measures the combined valuation setup.
3. **The 5% fair-value gate is smaller than allowed estimation uncertainty.** A candidate can qualify with 5% composite headroom (`core/screener.py:230`) while eligible models can disagree by 25% and have diagnostic mean absolute errors up to 30% (`core/fair_value.py:120-128`). The displayed model range is the min/max of model estimates, not a calibrated confidence interval (`122-123`). Five percent is a selection convention, not a demonstrated safety margin or downside floor.

Underlying P&L/quarterly CSVs also have no time-to-live or automatic refresh (`sma_router.py:336-354`). The valuation engine excludes very stale observations, but permits annual ages up to 550 days and TTM ages up to 240 days (`core/fair_value.py:101`). A model can therefore use older accounts even when newer reports exist. Review actual source reports and publication dates before using current valuation.

## 3% target, costs and risk

The target is fixed at 3% above the actual entry (`core/backtest.py:51`). With 10 bps commission per side, an exact target fill returns approximately **2.794% net**, excluding taxes and other charges not represented by that commission assumption. Five bps entry slippage changes the actual entry from which the target is anchored. From an actual fill, a 3% net return needs approximately **3.206% gross** under commissions alone; an approximate 30 bps round-trip cost convention implies about 3.3% gross.

The screener uses a 0.30% round-trip cost reserve and an estimated risk floor of 0.5 ATR (`core/screener.py:53-55`). Its EMA close breach exits at the next open; this is not a hard intraday stop and losses can exceed the displayed EMA distance. A 3% gross target with a 1.5% stop and 0.30% costs has estimated net reward/risk `(3-.3)/(1.5+.3)=1.5`; a 2% stop lowers it to about 1.17.

A separate live-entry mismatch needs review before implementation: resistance and net reward/risk are tested against the original setup target (`core/screener.py:53-55`, `96`), but a new paper entry has its own 3% target (`121`). Recheck resistance and costs at the actual new entry rather than carrying over historical setup eligibility.

## A testable setup hypothesis

Use fair value as a research filter alongside price confirmation, rather than a promise of fast price convergence:

- Verify the latest published accounts, sustained positive profit/revenue and at least two eligible models. Study a conservative model estimate with at least 10-15% headroom over the planned entry; these thresholds are hypotheses and the model minimum is not a statistical lower bound.
- Require a rising SMA50 above SMA200, price above both, Nifty in a verified uptrend and positive matched 60-session relative strength.
- Study a pullback near EMA9 followed by a completed cross and confirmation candle, confirmation volume at least 1.5 times its preceding 20-session average and extension no more than one ATR above EMA9.
- Plan the next-session entry, reject a gap that destroys reward/risk, and require the nearest confirmed resistance beyond the chosen target.
- Use an explicitly defined structural PRICE stop below recent support. If the structural stop requires more than about 1.5% risk for a 3% gross target, skip the setup rather than moving the stop inside market noise to make the ratio look better. Recompute after costs. A PRICE stop is a different policy from the default EMA-close exit and needs separate testing.
- Freeze target, stop, holding limit, gates and costs before evaluation. Record prospective paper signals with their actual valuation available-at timestamp. Assess closed net expectancy, profit factor, target-hit rate, drawdown, costs, coverage and uncertainty across sufficient trades and market regimes.

This recipe is an unvalidated research candidate. Establishing whether fair value adds useful predictive information requires publication-timed historical snapshots or prospective evidence; today's fitted fair value must not be retroactively applied to old trades.
