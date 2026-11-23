# Equity Atlas: how to use the app

Equity Atlas is a local research and paper-trading workspace for NSE shares. It combines daily price signals, support structures, valuation evidence, a risk calculator and historical testing. It does not connect to a broker or place orders.

The app has six working panels: **Opportunities**, **Reversals**, **EMA screener**, **Paper journal**, **Valuation** and **Backtests**. **How to use** and the header's **? Help** button open the built-in guide. The **Position Risk Planner** sits below the panels and can be opened from a research card.

## Start the local app

1. Open the project folder, `E:\application\SCREENER-NEW-V1`.
2. Double-click `run_screener.bat`.
3. Keep the server window open while using the dashboard.
4. Open [http://localhost:8000](http://localhost:8000) in your browser.
5. Check that the header says **Local service connected**. It may initially say **Connecting**.

If dependencies have not been installed, open a terminal in the project folder and run:

```powershell
python -m pip install -r requirements.txt
```

Then start `run_screener.bat`. You can also run `python main.py` from the project folder. The batch file is the usual local startup path and runs the server on port 8000. If that port is already occupied by this app, open the existing dashboard rather than starting another copy.

The calculations run locally, but fresh price and fundamental downloads need an internet connection. Cached data can sometimes remain usable when a new download fails. Closing the server window stops new dashboard requests. A server restart also removes unfinished scan jobs; start those scans again after reconnecting.

The API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs). You do not need it for normal dashboard use.

## A beginner workflow

1. **Research a familiar stock.** On Opportunities, enter a bare NSE symbol such as `DIXON`, `HINDCOPPER`, `RELIANCE` or `TCS`, then click **Analyze stock**. Read the date and available history before interpreting its trend.
2. **Open the price chart.** Click **Inspect price chart** or **View chart**. Compare the latest price, moving averages, target and stop reference. A price chart can open without valuation or a qualifying setup.
3. **Build a shortlist.** Open EMA screener. Start with the **F&O registry**, **Last 10 Trading Days**, **Verified uptrend (50 + 200 SMA)** and **Actionable only**, then click **Scan Now**.
4. **Read the reasons.** Open **Evidence & limitations** on a candidate. Check confirmation volume, remaining reward/risk, resistance, data date and entry blockers. The largest score is only a starting point for research.
5. **Study rejected setups.** Turn off **Actionable only** and scan again. Watchlist rows explain why a pattern does not qualify for a new entry. Broader filters do not remove those blockers.
6. **Plan a paper risk budget.** Click **Plan risk**, review the filled entry, stop and target, and enter your capital and chosen risk percentage. Click **Calculate paper size**.
7. **Save a hypothetical entry.** **Paper track** is available when the candidate passes the entry checks. Open Paper journal and review the saved entry and its fixed target.
8. **Review later sessions.** Use **Refresh outcomes** after more daily candles complete. Compare closed net results with open marks. Use Backtests to study the rules over a larger historical sample.

An empty shortlist can be a valid result. First check data coverage; then inspect the watchlist. Avoid assuming that every scan must produce a trade.

## Read dates and progress before results

The app uses completed daily candles. Before the normal NSE market closes at **15:30 India time**, today's daily candle is excluded from confirmed signals. The latest completed close will usually belong to the previous trading session. Prices shown here are not a live intraday quote stream.

**As of** is the date of the latest completed price observation. **History bars** describes how much usable history was received. A short listing can still have a basic chart and research values while its long-term trend stays **UNKNOWN**. EMA breakout confirmation needs at least 31 completed bars; the verified long-term trend needs at least 200. A displayed 52-week observed range uses the available portion of up to 252 trading sessions, so short history is not a full year's range.

Four bulk operations have progress bars: the Opportunities quick scan, the full EMA scan, the Reversals scan and Backtests.

| Progress item | How to read it |
| --- | --- |
| Preparing / queued | The app is loading a basket or waiting for a local scan slot. The initial bar may be indeterminate or show zero. |
| Download | Completed ticker downloads out of the requested ticker count. A download can finish with unavailable data. |
| Analysis | The engine is checking price rules or replaying historical trades. |
| Valuation | The engine is checking available valuation evidence for relevant candidates. |
| Current symbol | The ticker currently being processed. It is announced before its download starts; the completed count updates after that attempt. |
| Completed / total | Work within the current phase. The total can change when moving from a whole universe to a smaller candidate set. |
| Unavailable | Symbols whose history or required evidence could not be used. These are separate from valid stocks with no signal. |
| Percentage | Progress through the job's reported phases, not a predicted time remaining or trading probability. |
| 100% / completed | The requested work finished. Read the result's coverage and warnings; this does not mean every ticker downloaded successfully. |

Downloads, technical checks and valuation checks take different amounts of time. Progress-enabled downloads process one ticker at a time, so a bar can pause while the displayed ticker's provider request is slow. Wait for the result rather than repeatedly starting the same scan. The app joins duplicate starts of the same scan kind. On Opportunities, it temporarily disables the shortlist's Refresh button and universe selector while keeping single-stock research available. Other bulk panels disable their controls until completion or failure. On failure, the app shows the reason and makes those controls available again.

Completed results replace the progress display on Opportunities, Reversals and EMA screener. Backtests keeps its completed progress panel above the results.

## 1. Opportunities

Opportunities is the quickest way to research a stock or preview potential candidates.

### Analyze one stock

Enter the symbol beside **Analyze stock** and press the button or Enter. The research panel shows:

- The stock symbol and completed price date.
- **Last completed close** and **Verified trend**.
- **ATR14 / RSI14**, **Volume vs prior average** and **60-session return**.
- Available fair value and its confidence label.
- The observed price range and number of history bars.
- An EMA setup and a support reversal when those structures exist.

**Inspect price chart** opens the basic chart. **Inspect valuation models** opens Valuation for that stock. **Close research** hides the research panel.

Basic stock research and charts use cached valuation when it is available. They do not wait for a new fundamental download. A blank fair value in price research can therefore mean that valuation has not been requested yet. Use Valuation for a full analysis.

### Read the quick shortlist

Choose **F&O universe** or **All NSE equities**, then click **Refresh**. Changing this universe also starts a new quick scan. The quick scan checks the **first 60 symbols in the selected registry** and displays up to 12 shortlisted results. It does not cover the entire selected registry. Use EMA screener for the full scan.

The shortlist prefers actionable technical setups with qualifying valuation confluence. That confluence needs at least **5% estimated fair-value headroom** and sufficient valuation evidence. If none qualifies, the panel can display a technical shortlist instead. The caption identifies which list you are viewing.

The four summary cards show **Actionable shortlist**, **Nifty regime**, **Best setup score** and **Coverage**. Read the coverage notice for requested, usable and unavailable symbols. Each opportunity offers **Evidence & limitations**, **View chart**, **Plan risk** and, when eligible, **Paper track**.

## 2. Reversals

Reversals studies support retests and recovery structures, including double bottoms and shelves with several independent touches. A stock being near a low is a watch condition; it does not establish a confirmed entry.

Choose a **Universe** and press **Scan reversals** or **Reload**. Changing the universe also starts a scan. The scan finds candidates with at least 10% chart distance to the first resistance; the display initially filters for at least 15%.

Use these display controls after the scan:

| Control | Purpose |
| --- | --- |
| Min resistance distance | Show at least 15%, 20% or 25% to resistance, or the broader 10% group. This is chart distance, not promised upside. |
| Alert Status | Focus on accumulation watches, confirmed EMA reversals or pullbacks approaching support. |
| Pattern Type | Compare **Double Bottom (W-Pattern)** with **Triple/Multi-Touch Shelf**, or show all formations. |
| Search symbol | Find a ticker within the returned candidates. |
| Actionable only | Hide candidates with unresolved confirmation, volume, risk or data blockers. It is initially unchecked here. |

Read **Support floor**, **First resistance**, **Structural stop**, **Net reward / risk**, **Evidence score**, **Support retests**, **Valuation**, **As of** and **State** together. The summary cards describe accumulation watches, confirmed reversals, average resistance distance and median net reward/risk across the returned candidates.

The first resistance is the nearest confirmed overhead barrier. A farther high or fair-value estimate does not replace a nearer barrier to make the target look larger. The structural stop lies below support with an ATR buffer. A paper reversal uses a **PRICE** stop evaluated against daily price ranges, rather than the EMA completed-close rule.

Reversal actionability also checks independent retests, recent confirmation, price behavior, traded-value evidence, confirmation volume and sufficient structural reward/risk. A **BREAKOUT CONFIRMED** label can still have other blockers; read the evidence before saving a paper entry.

## 3. EMA screener

The EMA screener scans the selected registry for confirmed daily EMA9 breakouts. It needs a close crossing from at or below EMA9 to above it. The next completed candle must close higher, make a higher high and remain above EMA9.

A **verified uptrend** requires the close above SMA50 and SMA200, SMA50 above SMA200 and rising over its recent comparison, and a positive 60-session price slope. Incomplete long-term history produces **UNKNOWN**, not a verified trend.

| Control | What it changes |
| --- | --- |
| Stock Universe | **F&O registry** or **All NSE registry**. A full scan requests the selected registry, rather than the first-60 preview. |
| Breakout Window | Search the last 5, 10 or 15 trading sessions for confirmation. The default is 10. |
| Trend Filter | **Verified uptrend (50 + 200 SMA)**, **Uptrend + Sideways** or **No Trend Filter**. Broader choices expose weaker research; they do not make it actionable. |
| Fair Value Gate | Display **All Signals** or only **Reliable valuation confluence** from the returned results. |
| Actionable only | Initially checked. Turn it off and rescan to inspect blocked or resolved setups. |
| Require Nifty uptrend | Require a verified advancing benchmark. Unknown benchmark history cannot satisfy this requirement. |
| Require positive relative strength | Require the stock to outperform Nifty over matched 60-session endpoints. |
| Min score | Restrict results to the selected checklist score threshold. |
| Sort | Order by **Evidence score**, **Net reward / risk**, **Latest confirmation** or **Relative strength**. |
| Search symbol | Search the returned stock names and symbols. |
| Export CSV | Download the research rows matching the current display filters. |

Changes to universe, breakout window, trend, actionability, benchmark requirements or minimum score start a scan. Fair Value Gate, search and sort work on the already returned rows. **Scan Now** explicitly runs the selected settings again.

Read **Setup target** and **New entry target** separately. The setup target is fixed from the historical signal's reference entry. A historical entry uses the following session's open; if that session has not completed, the entry is estimated. The new-entry target uses the latest completed close. A price move can make these targets different. An old target hit or stop does not reactivate that old setup when the stock later recovers.

The actionability checks include freshness, verified trend, liquidity, confirmation volume, ATR extension, remaining reward/risk and resistance before the target. Net screening reward/risk includes default round-trip costs and a risk allowance; it is an estimate of the setup's economics, not an execution guarantee. Watch-only setups can defer new valuation downloads so they do not slow technical results. Use Valuation to inspect a particular stock's models.

## Scores, indicators and statuses

**A score of 80/100 does not mean an 80% chance of profit.** It ranks the evidence in the engine's checklist. Scores can help choose what to inspect first, but eligibility also depends on blockers. The EMA and reversal checklists differ, so equal scores across those two strategies do not establish equal future performance.

| Metric | Plain-language meaning |
| --- | --- |
| EMA9 | A recent closing-price average that gives more weight to newer closes. |
| SMA50 / SMA200 | Medium-term and long-term averages of closing prices. |
| ATR14 | A measure of recent typical daily price range in rupees. It helps judge extension and stop distance. |
| RSI14 | Momentum on a 0–100 scale. A high or low reading is not a complete buy or sell rule. |
| Confirmation volume | Confirmation-day volume divided by its preceding 20-session average. 1.5× means 50% above that average. |
| Relative strength | Stock return minus Nifty return over matched 60-session endpoints. +3% means three percentage points of outperformance. |
| Net reward / risk | Estimated gain after costs divided by estimated loss. Read the entry and stop assumptions behind it. |
| Fair-value headroom | Estimated model value relative to the displayed price. It has no promised completion date or return. |

| Status | Meaning |
| --- | --- |
| FRESH | A recent confirmed EMA setup; its reference entry can still be estimated if a following session is unavailable. |
| ACTIVE | An unresolved historical setup or open paper entry. It is not automatically eligible for a new entry. |
| NEARING TARGET | The original setup has little distance left to its fixed target. New-entry reward may be limited. |
| STOP PENDING | An EMA completed-close invalidation exists, but the next available open has not yet been observed. |
| TIME EXIT PENDING | The holding limit triggered a close decision, but its following-open exit is unobserved. |
| TARGET HIT | The entry's target exit was observed. Read net P&L separately. |
| STOPPED OUT | A stop or invalidation exit was observed. A later recovery does not erase that outcome. |
| TIME EXIT | An exit at the holding limit was observed. It can have a positive, negative or zero net result. |
| BREAKOUT CONFIRMED | A recent EMA reversal confirmation exists near a support structure. Other entry blockers can remain. |
| ACCUMULATION ZONE | A support-base watch close to the floor, generally within 3.5%. It is not a confirmed buy. |
| APPROACHING SUPPORT | A pullback or falling-price watch that needs stabilization. |
| SWING IN PROGRESS | A recovery watch outside the nearer support zone. |
| Watchlist / entry blocked | A setup exists, but some required evidence or entry check failed. |
| UNKNOWN | Available data cannot establish the requested trend or classification. |

The dashboard displays status names with spaces. A colored badge is a label, not a substitute for the evidence and data date.

## Position Risk Planner

Open **Position Risk Planner** below the panels, use **Open risk planner** on Opportunities, or click **Plan risk** on a candidate. Plan risk fills the latest completed close, the candidate's stop reference and its new-entry target. You can review or change these inputs before calculating.

Enter **Research capital (₹)**, **Risk budget (%)**, **Entry price (₹)**, **Stop reference (₹)**, **Target (₹)** and **Round-trip cost (bps)**. For a long position, the calculator requires **stop < entry < target**. One basis point is 0.01%; the default 30 bps represents 0.30% estimated round-trip cost.

The calculator chooses whole shares within both your cash and estimated loss budget. It reserves estimated costs in the cash calculation. It shows **Paper quantity**, **Capital required**, **Estimated maximum loss** and **Net reward / risk**, including potential net gain. A quantity of zero means the current cash or loss budget cannot fund even one share under those inputs.

Example:

| Input or result | Value |
| --- | --- |
| Capital | ₹100,000 |
| Risk budget | 1% = ₹1,000 |
| Entry / stop / target | ₹100 / ₹95 / ₹110 |
| Round-trip cost | 30 bps = ₹0.30 per share |
| Whole-share quantity | 188 |
| Capital required, including estimated costs | ₹18,856.40 |
| Estimated loss at the stop | ₹996.40 |
| Potential net gain at the target | ₹1,823.60 |
| Net reward / risk | About 1.83 : 1 |

The maximum-loss label is an estimate. Overnight gaps, slippage and an EMA close exit can produce a worse fill than the stop reference. Calculating a quantity does not place an order or attach that quantity to the Paper journal. Editing the risk planner's prices also does not change what Paper track saves from a research card. The current journal tracks an entry's price and percentage outcome rather than a cash portfolio of these sized quantities.

## 4. Paper journal

Click **Paper track** on an eligible research card to save a hypothetical entry at its displayed latest completed close. EMA entries use the new-entry target; reversal entries use the structural resistance target. The journal gives that saved entry its own fixed target. This differs from the original historical setup target.

The card shows **Actual paper entry**, **Fixed target**, its stop reference, current completed price or exit fill, and a date. Open cards show a **gross unrealized mark**. Closed cards show the **net realized return**, including the journal's modeled execution costs. Those two returns should not be compared as though they were calculated on the same basis.

Use **Show** to choose **Open & pending entries**, **Closed outcomes** or **All paper entries**. **Refresh outcomes** evaluates subsequent completed candles. Entry-day intraday highs are excluded for these saved-close entries, so a target touched before the saved entry does not count as a later success.

EMA close invalidation produces an exit at the next available open. Structural reversal stops use daily high/low evidence. If a daily candle could contain both a structural stop and a target, the replay handles the stop first because the within-day order is unknown. Pending exits stay pending until the following execution is observable.

The summary shows **Active Signals**, **Target exits**, **Invalidation exits** and **Signal Win Rate**, with closed counts, pending exits, target-hit rate and average closed net result. Win rate counts positive net realized results among closed entries; it does not count every target label as a profitable trade or include open entries.

**View chart** on a journal card opens that specific saved entry's chart. **Remove entry** removes that trade's saved record. **Clear List** asks for confirmation and removes the whole journal and its outcome history. Entries are stored locally in `signals.json` in the project folder. Back up that file if you want to preserve your research history before clearing it.

Tracking the same ticker while it already has an open entry updates that entry rather than adding a separate simultaneous open trade. A later re-entry after closure preserves the earlier closed record. Review the saved entry after any update.

## 5. Valuation

Enter **Lookup Ticker**, choose **FY Start** and click **Analyze Valuation**, or use a **Quick Select** stock. FY Start is the earliest fiscal year requested for the fundamental analysis; its default is 2014. Valuation can take longer than a basic chart because it may download fundamentals and aligned historical prices.

Read the stock's **Current Market Price**, **Profit Metric**, **Composite Fair Value**, estimated gap and confidence notice together. Banking entities can use a different profit metric; the label identifies it. Positive estimated headroom means the model value is above the price, while negative headroom means it is below it.

The current composite classification is **UNDERVALUED** above 15% estimated headroom, **OVERVALUED** below -15%, and **FAIR** within that band. This classification is different from the Opportunities confluence threshold of 5%. A FAIR stock can therefore still have some positive estimated headroom.

The composite uses eligible models. A model may be shown for inspection while being excluded from the composite. Read confidence, model count, fundamental dates and warnings. Too few observations, weak or invalid fit, nonpositive slopes, losses or invalid values, stale fundamentals and disagreement can reduce what the valuation establishes. **Low** confidence means the evidence does not meet the stronger confluence criteria. **Unavailable**, **N/A**, a dash or a no-model notice means there is no supported estimate for that field; it is not a zero valuation. The present descriptive confidence is capped at medium without publication-timed validation.

The four subtabs provide different views:

| Subtab | What to inspect |
| --- | --- |
| Historical model estimates & EMA9 | Price, EMA9, retrospective composite estimates and illustrative ±15% guides. |
| Annual OP Model Trend | Annual profit and fiscal-year-end prices on separate axes; use **Inspect FY Period** for a snapshot. |
| Annual Sales Model Trend | Annual sales and fiscal-year-end prices; use **Inspect FY Period**. |
| Quarterly TTM Model Trend | Trailing-twelve-month profit and sales views when available; use **Inspect Quarter**. |

**All Bands**, **Composite Stepped** and **Price + 9 EMA** change the first chart's overlays. **1Y**, **3Y**, **5Y** and **ALL** change its displayed window without changing FY Start or refitting the requested models. The year buttons show the last 365, 1,095 or 1,825 calendar days measured from the last chart observation; ALL shows the available history. Available data can be shorter than the chosen window. These window and overlay choices stay selected when you research another ticker.

Use **Annual Model Breakdown (Year-by-Year / FY)** and the quarterly table to inspect observations, period prices, model values and gaps. Snapshot **Misprice Gap** uses actual price relative to fair value, so a negative snapshot gap indicates price below the model estimate. The headline estimated-headroom convention has the opposite sign. Read each label before comparing percentages.

R² describes how well the historical fitted relationship matched its observations. It is not a trade-win probability. The stepped historical fair-value lines are retrospective model fits and are not values proven to have been known at those dates. The ±15% guides are illustrative bands, not confidence intervals. A price below a retrospective line does not establish a historical buying opportunity or guaranteed convergence.

If one model has no chart series, its chart shows a no-history message. Other models, numbers or price charts can remain useful. A chart-rendering failure should not hide the valuation tables or change missing values into zeroes.

## 6. Backtests

Choose a **Strategy** first. The default is **3% duration · close reference, every cross**, with a 3% target, 90 trading sessions and five years of entries. A 90-session maximum is roughly four calendar months. Choose a ticker basket, including **Full F&O registry** or **All NSE registry**, to process the entire selected registry with completed/failed counts; the duration engine does not silently cap it at 300.

The duration choices are:

| Strategy | Entry and exit policy |
| --- | --- |
| Close reference, every cross | Strict EMA9 cross; reference entry at that close; every cross may overlap earlier trades. No stop. Observe the next 90 sessions for the fixed target; otherwise use the horizon close. Same-close execution is idealized. |
| Next-open, one position / stock | Observe the cross at the close and enter the following open with modeled costs. Only one position per ticker. No stop. Entry session counts as session one; the horizon close decides an exit at the next available open. |
| Confirmed EMA, EMA stop | Wait for higher-high/higher-close confirmation, enter the next open and exit following an EMA-close invalidation or holding expiry. |

For duration studies, **Mature cohort target hits** uses only entries with the entire selected observation horizon available, including early winners. **Observed hits / all entries** includes recent unresolved entries; it is the observed lower bound, not their final success rate. Net profitable closed trades, target hits and unresolved entries have separate denominators. An incomplete trade is **CENSORED**, or **TIME EXIT PENDING** when its exit decision awaits the next open. It is not forced into a loss at the dataset end.

Inspect median/90th-percentile target time, worst closed loss, observed adverse moves and simultaneous positions. Daily lows conservatively include the target day's full range because the order of its high and low is unavailable. Duration results show a target-time curve with a fixed mature denominator. They do not compound overlapping positions into a portfolio return.

Optional duration filters use the signal's completed history: close above SMA50; a prior 20-session high within 3% overhead; or a 10% pullback from the prior 60-session high. The separate candidate filter requires a verified long-term trend, volume at least 1.5 times its prior average and EMA extension no more than one ATR. These are research comparisons, not proven improvements. Use the same dates/basket and compare net returns and loss tails as well as hits.

**Load resistance-filter candidate** sets next-open duration, a 3% target, 90 sessions and the nearby-prior-high exclusion. It clears the other optional gates and keeps your selected basket and years. The saved 19-stock study found this candidate improved pooled results and each yearly comparison, but its latest-year net expectancy stayed negative. Run a matched comparison before using that result as evidence.

**Also inspect cached current-FV subset** labels entries below today's available valuation separately. It is a retrospective comparison; it does not establish what valuation was known at a historical entry. The saved result records those snapshots. Publication-dated evidence is needed for historical fair-value validation.

The outcome dropdown and ticker search filter the displayed log. **Export full CSV** includes every matching entry, including nonwinners and unresolved outcomes when **All outcomes** is selected. **Save full run JSON** preserves all entries, parameters, counts, coverage and valuation notes regardless of the display filter. Save this complete run when comparing accuracy; a winner-only export cannot supply the full denominator.

For **Confirmed EMA**, choose 1, 2, 3 or 5 historical years and use the following metrics.

Set **Target %** and **Max holding sessions**; selecting Confirmed EMA sets holding to 20 sessions and enables verified uptrend. Review **Slippage / side (bps)** and **Commission / side (bps)**, which default to 5 and 10. Click **Run historical backtest** and watch the ticker progress above the results.

Confirmation is observed at a completed close, and entry fills at the following session open with modeled costs. The target is fixed from that entry. EMA invalidation or the holding limit creates a close decision and exits at a following available open. If that exit is not yet observable, the trade remains unresolved. Target hits, gaps, close decisions and unresolved entries follow the displayed execution assumptions rather than idealized fills at any chosen daily price.

| Result | Meaning |
| --- | --- |
| Net profitable trade rate | Positive net closed results divided by all closed trades. Open trades are excluded. |
| Profit Factor | Sum of positive net trade returns divided by the absolute sum of negative net trade returns. Infinity means no losses in that sample, not proof of safety. |
| Total Trades Taken | The closed-trade count used for the main statistics. Unresolved entries are reported separately. |
| Average holding sessions | Mean trading-session duration of the closed trades. |
| Target-hit rate | Closed entries that reached their target. This differs from the rate of net profitable trades. |
| Net expectancy / trade | Mean percentage return per closed trade after the modeled costs. |
| Unresolved entries | Trades whose terminal execution has not been observed. They are not assumed losses or wins. |
| Win-rate interval (95%) | A statistical interval around the observed profitable-trade proportion. Few trades usually produce a wide interval. It is not a forecast of future returns. |

Open **Execution assumptions & coverage**, **Performance by ticker**, **Chronological monthly results** and **Historical Trade Log**. The trade log explains entry, target, exit, holding duration, net P&L, outcome and exit reason. Its **Entry target** column shows the fixed target calculated from the entry and the Target % setting you ran.

The **Realized return index · fixed allocation** begins at 100, splits initial capital equally across the requested universe and adds realized closed-trade P&L. It does not mark open holdings to market or compound profits as a continuously rebalanced brokerage portfolio. Its drawdown therefore excludes open-position losses. Monthly results group trades by their exit dates.

The technical replay tests EMA confirmation and the selected trend rule. It is a causal historical simulation and does not perform rolling model training or parameter optimization. It does not replay every current liquidity, ATR-extension, resistance or valuation recommendation gate. Historical valuation comparison is unavailable until publication-dated fundamental evidence exists; today's fair value is not applied to old entries. The selected current ticker baskets can have survivorship bias. A good historical sample is useful evidence, but it does not establish a future accuracy percentage.

## Help, refreshing and troubleshooting

The header's **Refresh workspace** acts on the current panel: Opportunities refreshes its quick scan; Reversals and EMA screener rescan; Paper journal refreshes outcomes; Valuation analyzes its entered ticker; Backtests runs its selected settings. On How to use, it scrolls to the top. It does not refresh every panel at once.

Open **? Help**, **How to use**, or [http://localhost:8000/#help](http://localhost:8000/#help) for the built-in guide.

### “Cannot reach the local service” or “Failed to fetch”

The browser could not complete a request to the local server. Check that the server window is still open. Start `run_screener.bat` if necessary, open the dashboard at its localhost URL and refresh. Then use **Retry chart** or rerun the scan. The header badge reports a connection check; it cannot guarantee that the process is still running later.

If the server has just been updated or restarted, refresh the page so browser controls and backend routes use the same version. A message that a scan job does not exist means its in-memory job was lost or expired; start the scan again.

### “Insufficient completed daily history” or UNKNOWN trend

Check the NSE symbol and read the available-bar count and diagnostic message. A valid new listing can have a chart but lack enough bars for long-term indicators. An established stock can also appear short when a provider download failed or returned invalid data. Retry after checking the connection; do not interpret a provider error as proof that the stock has no historical prices or no setup.

### Chart unavailable or a chart-library error

Read the error in the modal and click **Retry chart**. The loading state is replaced by a diagnostic instead of being left indefinitely. **Open troubleshooting** opens the in-app guide. Missing valuation or a missing reversal pattern should still allow a basic price chart when usable candles exist. If chart rendering fails, price values or valuation tables can remain available; refresh the page after ensuring the local server is running.

### Valuation unavailable or no reliable models

Use **Analyze Valuation** for the correct symbol and inspect the warnings and fundamental dates. A price-only view may simply have deferred valuation. A full valuation result can also lack sufficient models. **Retry valuation** repeats that request; **Open price chart** lets you continue price research. Do not treat a missing field as ₹0, 0% headroom or FAIR.

### No actionable candidates

Check coverage, universe, breakout window and optional benchmark requirements. Turn off Actionable only and rescan to inspect blocked rows. On Reversals, broaden Min resistance distance or the pattern/status display filters. A completed scan can validly have no qualifying entries, even if many stocks have charts or watch structures.

### A slow scan or timeout

Read the phase and current symbol. Full registries and full historical baskets take longer than a first-60 preview or one-stock chart. Provider rate limits or unavailable downloads can delay a phase. A timeout or **Job stopped** notice is an error to inspect and retry, not a finished trading result. Avoid multiple repeated starts while a job is still running.

### How to use the charts

Hover over candles or points to inspect values, drag across a region to zoom, and double-click to reset the view. The valuation view also has its own overlay and window controls. Read the labels for price, EMA, moving averages, target, support and stop reference. A horizontal support or stop line is an estimated level, not protection against a gap.

## A useful daily routine

After completed daily prices become available, check the data dates, run a chosen universe, read the strongest eligible evidence, compare nearby resistance and plan the hypothetical risk. Review older paper entries before saving more. Record what you expected and compare it with later closed results. Use a consistent forward paper sample and historical assumptions to judge usefulness; neither a high checklist score nor a few successful trades establishes future accuracy.
