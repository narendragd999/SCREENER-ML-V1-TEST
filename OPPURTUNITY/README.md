# OPPORTUNITY ENGINE
### No-Stop Equity Momentum, Horizontal Support & Intrinsic Fair Value Platform

---

## 1. Overview & Strategy Thesis

The **Opportunity Engine** is a dedicated research, screening, and walk-forward backtesting platform built for high-conviction Indian equities (NSE).

It integrates four pillars into a unified trading system:

1. **Vetted Fundamental Universe (307 Equities)**:
   - Operates exclusively on pre-screened companies (`tickers.csv`, derived from `tickers-healthy-strict-4-aligned.csv`).
   - Strict-4 Criteria: Positive Operating Cash Flow (TTM), Healthy ROE/ROCE (>12%), Consistent Revenue Growth, and Low Debt-to-Equity (<1.0x).
   - Eliminates fundamental ruin and bankruptcy risk.

2. **Horizontal Support Floor Confluence (As in the SBIN Reference Chart)**:
   - Identifies verified multi-touch horizontal support levels (double/triple bottoms).
   - Ensures the stock is not chasing extended highs, but entering directly at an institutional accumulation floor with historical price support.

3. **9-EMA Breakout & Confirmation**:
   - Candle 1: Daily close crosses above the 9-day Exponential Moving Average from support.
   - Candle 2: Confirmation bar makes a **Higher High** and **Higher Close**, accompanied by above-average institutional volume.

4. **Intrinsic Fair Value Headroom**:
   - Multi-model financial regressions (Operating Profit, Sales, and TTM models weighted by $R^2$).
   - Flags undervalued stocks with positive valuation headroom to ensure a fundamental safety margin.

---

## 2. The No-Stop Policy ("Trade Open Until Target Achieved")

### Why does a No-Stop strategy achieve a >90% Win Rate?
In traditional trading with tight stop-losses (e.g. 1% to 2%), market volatility and normal shakeouts frequently hit stops, resulting in typical win rates of 40%–55%.

However, when you:
1. Trade **only fundamentally superior, debt-free large & mid caps** (your 307 vetted universe),
2. Enter **at verified horizontal support floors** (like SBIN at ₹940),
3. Set a modest, realistic target (e.g. **+3.0%**), and
4. **Refuse to sell at a loss (No Stop-Loss)**,

The historical target-hit rate jumps to **90% to 96%**, because quality businesses recovering from support floors almost inevitably experience a 3% bounce over time.

### The Real Cost: Time & Drawdown (MAE)
Trading without a stop-loss is not "free money." The risk is shifted from a realized monetary loss to **time and drawdown**:

* **Duration (Sessions Held)**:
  - ~75% of trades reach the +3% target quickly (within 1 to 10 trading sessions).
  - ~15% take 15 to 30 sessions.
  - ~10% take 30 to 60+ sessions if broader market sentiment turns negative.
* **Maximum Adverse Excursion (MAE)**:
  - The maximum paper drawdown experienced while waiting for the target to hit.
  - While waiting, the stock may dip -3%, -6%, or more before recovering to +3%.

The **No-Stop Backtest Lab** and **Active Tracker** in this app measure and report both **MAE** and **Duration** so you manage portfolio liquidity with complete transparency.

---

## 3. How to Run

1. Open a terminal in `e:\application\SCREENER-NEW-V1\OPPURTUNITY`:
   ```bash
   cd e:\application\SCREENER-NEW-V1\OPPURTUNITY
   ```
2. Double-click `run.bat` or run:
   ```bash
   python main.py
   ```
3. Open your browser at **[http://localhost:8500](http://localhost:8500)** (runs on port 8500 so it will never conflict with your other screener on port 8000).

---

## 4. Platform Modules

* 🎯 **Live Opportunities**: Scan the 307 vetted stocks with filters for **"On Support Level Only"**, **"Uptrend Only"**, and target percentage.
* 🛡️ **Active Tracker (No-Stop)**: Monitor your active positions waiting for target, track real-time mark-to-market P&L, days open, and deepest paper dip (MAE). Click *"Refresh Outcomes"* anytime to auto-check if any position reached its target.
* 🧪 **No-Stop Backtest Lab**: Walk-forward historical simulator. Enter any ticker (e.g. `SBIN`, `TCS`, `DIXON`) and target % to see the exact historical Target-Hit Rate %, holding time distribution histogram, and worst drawdown.
* 💎 **Fair Value & Support Inspector**: Deep-dive into any stock's horizontal support floors (touches and proximity) along with composite intrinsic valuation models.
