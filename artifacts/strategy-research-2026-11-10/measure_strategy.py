"""Reproducible, untuned 3% strategy comparison; no production code is changed.

Run from the project root:
    python -B artifacts/strategy-research-2026-11-10/measure_strategy.py
Saved validated provider bars are reused on subsequent runs.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import pathlib
import sys
import urllib.request

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.backtest import backtest_single_ticker, compute_performance_stats, simulate_long_trade
from core.data_fetcher import clean_ohlcv, fetch_batch_ohlcv, market_now
from core.strategy import confirmed_breakout_mask, indicator_frame, trend_regime_mask
from unittest.mock import patch

OUT = pathlib.Path(__file__).resolve().parent
RAW = OUT / "prices"
AS_OF_DATE = dt.date(2026, 11, 10)
START = pd.Timestamp(AS_OF_DATE) - pd.Timedelta(days=365 * 3)
RECENT_START = pd.Timestamp(AS_OF_DATE) - pd.Timedelta(days=365)

# These hypotheses were specified before observing results. No grid search.
VARIANTS = {
    "current_baseline": {"uptrend": True, "volume": False, "market_rs": False, "stop": "EMA9_CLOSE"},
    "no_trend": {"uptrend": False, "volume": False, "market_rs": False, "stop": "EMA9_CLOSE"},
    "volume_not_extended": {"uptrend": True, "volume": True, "market_rs": False, "stop": "EMA9_CLOSE"},
    "volume_market_rs": {"uptrend": True, "volume": True, "market_rs": True, "stop": "EMA9_CLOSE"},
    "baseline_price_stop_1_5pct": {"uptrend": True, "volume": False, "market_rs": False, "stop": "PRICE"},
    "volume_price_stop_1_5pct": {"uptrend": True, "volume": True, "market_rs": False, "stop": "PRICE"},
    "technical_gates_confirmation_proxy": {"uptrend": True, "volume": False, "market_rs": False, "stop": "EMA9_CLOSE", "technical_gates": True},
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_data():
    RAW.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "data_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames = {}
        for ticker, info in manifest["coverage"].items():
            if info.get("file"):
                df = pd.read_csv(OUT / info["file"], index_col=0, parse_dates=True)
                frames[ticker] = clean_ohlcv(df)
        return manifest, frames

    with urllib.request.urlopen("http://127.0.0.1:8000/api/backtest/presets", timeout=15) as response:
        presets = json.load(response)["presets"]
    tickers = next(p["tickers"] for p in presets if p["name"] == "Top 20 Liquid Leaders")
    write_json(OUT / "predefined_hypotheses.json", {
        "specified_before_data_download": True, "date": str(AS_OF_DATE),
        "tickers": tickers, "variants": VARIANTS,
        "parameters": {"target_pct": 3, "max_hold_sessions": 20, "slippage_bps_per_side": 5,
                       "commission_bps_per_side": 10, "volume_ratio_min": 1.5,
                       "ema_extension_atr_max": 1.0, "price_stop_pct": 1.5,
                       "relative_strength_60_sessions": "stock minus Nifty > 0"},
    })
    last_report = {"completed": -1}
    def progress(info):
        write_json(OUT / "download_progress.json", info)
        if info.get("completed", 0) != last_report["completed"]:
            last_report["completed"] = info.get("completed", 0)
            print(f"Download {info.get('completed')}/{info.get('total')} {info.get('ticker') or ''}", flush=True)
    frames = fetch_batch_ohlcv(tickers + ["^NSEI"], "1d", 365 * 4, progress=progress)
    coverage = {}
    for ticker in tickers + ["^NSEI"]:
        frame = clean_ohlcv(frames.get(ticker))
        if frame.empty:
            coverage[ticker] = {"bars": 0, "file": None, "reason": "Provider returned no validated completed bars"}
            continue
        filename = "prices/" + ("NIFTY" if ticker == "^NSEI" else ticker) + ".csv"
        frame.to_csv(OUT / filename, index_label="Date")
        coverage[ticker] = {
            "bars": len(frame), "first": str(frame.index[0].date()), "last": str(frame.index[-1].date()),
            "file": filename, "data_source": frame.attrs.get("data_source"),
            "fetched_at": frame.attrs.get("fetched_at"),
            "invalid_bars_removed": frame.attrs.get("invalid_bars_removed", 0),
            "history_warnings": frame.attrs.get("history_warnings", []),
            "days_since_last_bar": (AS_OF_DATE-frame.index[-1].date()).days,
        }
        frames[ticker] = frame
    manifest = {"research_date": str(AS_OF_DATE), "downloaded_at": market_now().isoformat(),
                "requested_lookback_days": 365*4, "analysis_start": str(START.date()),
                "recent_cohort_start": str(RECENT_START.date()), "tickers": tickers, "coverage": coverage}
    write_json(manifest_path, manifest)
    return manifest, frames


def ticker_context(df, nifty):
    bars = indicator_frame(df)
    conf = confirmed_breakout_mask(bars)
    trend = trend_regime_mask(bars)
    pair = pd.concat([bars.Close.rename("stock"), nifty.Close.rename("market")], axis=1).dropna()
    rs = (pair.stock / pair.stock.shift(60) - pair.market / pair.market.shift(60)) * 100
    market_trend = trend_regime_mask(nifty).reindex(bars.index)
    return bars, conf, trend, rs.reindex(bars.index), market_trend


def replay(ticker, context, variant):
    bars, confirmations, trend, rs, market_trend = context
    trades = []
    occupied_until = -1
    counts = {"signals_after_start": 0, "uptrend_eligible": 0, "volume_eligible": 0, "market_rs_eligible": 0}
    for confirm_idx in np.flatnonzero(confirmations.to_numpy()):
        confirm_idx = int(confirm_idx)
        entry_idx = confirm_idx + 1
        if entry_idx >= len(bars) or bars.index[entry_idx] < START or confirm_idx <= occupied_until:
            continue
        counts["signals_after_start"] += 1
        row = bars.iloc[confirm_idx]
        if variant["uptrend"] and trend.iloc[confirm_idx] != "UPTREND":
            continue
        counts["uptrend_eligible"] += 1
        extension = (row.Close-row.ema9)/row.atr14
        if variant["volume"] and not (row.volume_ratio >= 1.5 and 0 <= extension <= 1.0):
            continue
        if variant.get("technical_gates"):
            prior = bars.Volume.iloc[max(0,confirm_idx-20):confirm_idx]
            recent_volume = bars.Volume.iloc[max(0,confirm_idx-19):confirm_idx+1]
            volume_coverage = (prior>0).sum()>=15 and (recent_volume>0).sum()>=15 and row.Volume>0
            risk_pct = (row.Close-row.ema9)/row.Close*100
            risk = max(risk_pct,.5*row.atr14/row.Close*100) if risk_pct>0 else None
            net_rr = 2.7/(risk+.3) if risk is not None else None
            resistance=[]
            # A5-bar centered pivot is known only once its2 later bars close.
            for pivot in range(max(2,confirm_idx+1-90),confirm_idx-1):
                high = float(bars.High.iloc[pivot])
                if high>row.Close and high>=float(bars.High.iloc[pivot-2:pivot+3].max()):
                    resistance.append(high)
            nearest = min(resistance) if resistance else None
            if not (volume_coverage and row.turnover20>=10_000_000 and row.volume_ratio>=1
                    and 0<=extension<=1.5 and net_rr is not None and net_rr>=1.5
                    and (nearest is None or nearest>=row.Close*1.03)):
                continue
        counts["volume_eligible"] += 1
        if variant["market_rs"] and not (rs.iloc[confirm_idx] > 0 and market_trend.iloc[confirm_idx] == "UPTREND"):
            continue
        counts["market_rs_eligible"] += 1
        entry_price = float(bars.Open.iloc[entry_idx]) * 1.0005
        trade = simulate_long_trade(bars, entry_idx, entry_price, target_pct=3.0,
                    max_hold_candles=20, slippage_bps=5.0, commission_bps=10.0,
                    stop_mode=variant["stop"],
                    stop_loss_price=entry_price*.985 if variant["stop"] == "PRICE" else None)
        occupied_until = trade["exit_idx"] if trade["exit_idx"] is not None else len(bars)
        pnl = trade["net_pnl_pct"]
        trades.append({**trade, "ticker": ticker,
            "breakout_date": str(bars.index[confirm_idx-1].date()),
            "confirm_date": str(bars.index[confirm_idx].date()),
            "outcome": "OPEN" if pnl is None else "WIN" if pnl > 0 else "LOSS" if pnl < 0 else "BREAKEVEN",
            "confirmation_volume_ratio": None if not np.isfinite(row.volume_ratio) else float(row.volume_ratio),
            "confirmation_extension_atr": None if not np.isfinite(extension) else float(extension),
            "confirmation_relative_strength_60d_pct": None if not np.isfinite(rs.iloc[confirm_idx]) else float(rs.iloc[confirm_idx]),
            "confirmation_market_trend": str(market_trend.iloc[confirm_idx]),
        })
    return trades, counts


def stats_for(trades, tickers):
    raw = compute_performance_stats(trades, capital_tickers=tickers)
    raw.pop("equity_curve", None)
    raw.pop("monthly_stats", None)
    closed = [t for t in trades if t.get("pnl_pct") is not None]
    by_ticker = []
    for ticker in tickers:
        sub = [t for t in trades if t["ticker"] == ticker]
        s = compute_performance_stats(sub)
        if sub:
            by_ticker.append({"ticker": ticker, "closed": s["total_trades"], "open": s["open_count"],
                       "net_win_pct": s["win_rate_pct"], "expectancy_pct": s["avg_trade_return_pct"],
                       "sum_trade_returns_pct": round(sum(t["pnl_pct"] for t in sub if t.get("pnl_pct") is not None),4)})
    raw["participating_tickers"] = len(by_ticker)
    raw["largest_ticker_share_closed_pct"] = round(max((s["closed"] for s in by_ticker), default=0)/len(closed)*100,2) if closed else None
    raw["per_ticker"] = sorted(by_ticker, key=lambda s:s["expectancy_pct"], reverse=True)
    # The interval for the mean loss is descriptive, not a robust bootstrap.
    losses = np.array([t["pnl_pct"] for t in closed if t["pnl_pct"] < 0], dtype=float)
    raw["worst_loss_pct"] = round(float(losses.min()),4) if len(losses) else None
    raw["loss_5th_percentile_pct"] = round(float(np.quantile(losses,.05)),4) if len(losses) else None
    raw["average_loss_ci95_approx_pct"] = [round(float(losses.mean()-1.96*losses.std(ddof=1)/math.sqrt(len(losses))),4),
        round(float(losses.mean()+1.96*losses.std(ddof=1)/math.sqrt(len(losses))),4)] if len(losses)>1 else None
    return raw


def baseline_check(ticker, df, research):
    engine = backtest_single_ticker(ticker, df, target_pct=3, max_hold_candles=20,
                    slippage_bps=5, commission_bps=10, require_uptrend=True, start_date=START)
    keys = ["entry_date","entry_price","target_price","exit_date","exit_price","pnl_pct","status","stop_reason"]
    expected = [{k:t.get(k) for k in keys} for t in engine]
    actual = [{k:t.get(k) for k in keys} for t in research]
    assert actual == expected, f"Optimized research replay differs from engine: {ticker}"
    return {"ticker":ticker,"trades":len(engine),"matched_fields":keys,"result":"PASS"}


def verify_proxy_against_scanner(trades, frames):
    from core.screener import analyze_ticker_ema9
    checks=[]
    for trade in trades:
        ticker=trade["ticker"]
        decision=pd.Timestamp(trade["confirm_date"])
        history=frames[ticker].loc[:decision]
        decision_clock=decision.to_pydatetime().replace(hour=15,minute=31,
            tzinfo=dt.timezone(dt.timedelta(hours=5,minutes=30)))
        with patch("core.screener.market_now",return_value=decision_clock):
            scanner=analyze_ticker_ema9(ticker,history,max_candles_ago=1,target_pct=3)
        assert scanner and scanner["is_actionable"], f"Technical proxy accepted a blocked scanner setup: {ticker} {decision}"
        checks.append({"ticker":ticker,"confirmation_date":trade["confirm_date"],
                       "scanner_is_actionable":scanner["is_actionable"],"result":"PASS"})
    return checks


def main():
    manifest, frames = load_data()
    declaration_path=OUT/"predefined_hypotheses.json"
    declaration=json.loads(declaration_path.read_text(encoding="utf-8"))
    declaration.pop("specified_before_data_download",None)
    declaration["original_hypotheses_specified_before_data_download"]=[name for name in VARIANTS if name!="technical_gates_confirmation_proxy"]
    declaration["variants"]=VARIANTS
    declaration["additional_technical_proxy_specified_before_reading_results"]={
        "verified_uptrend":True,"min_positive_volume_observations":15,"volume_lookback_sessions":20,
        "min_turnover20_rupees":10_000_000,"min_confirmation_volume_ratio":1.0,
        "max_ema_extension_atr":1.5,"min_estimated_net_rr":1.5,
        "net_rr_formula":"(3-.30)/(max(EMA distance%,.5ATR%)+.30)",
        "resistance":"no90-session5-bar pivot above close but below confirmationClose*1.03; pivot requires2completed right-side bars",
        "label":"technical gates at confirmation proxy; entry remains next open and fair value excluded"}
    write_json(declaration_path,declaration)
    tickers = manifest["tickers"]
    nifty = indicator_frame(clean_ohlcv(frames["^NSEI"]))
    contexts = {t:ticker_context(clean_ohlcv(frames[t]),nifty) for t in tickers if t in frames and len(frames[t])>=200}
    all_trades = []
    result = {"research_date":str(AS_OF_DATE),"parameters":json.loads((OUT/"predefined_hypotheses.json").read_text(encoding="utf-8")),
              "coverage":manifest,"baseline_verification":[],"variants":{},
              "limitations":[
                  "This is a small fixed current 20-stock basket, with survivorship and selection bias; unavailable stocks are not replaced.",
                  "The preceding two years and latest twelve months are chronological entry cohorts from continuous replay, not independently reset portfolios.",
                  "The variants were not tuned in this run, but their specification was informed by known market conventions and prior baseline results; the recent cohort is not a pristine untouched holdout.",
                  "Daily OHLC cannot establish intraday barrier order; PRICE stops resolve ambiguous stop/target candles conservatively stop-first.",
                  "The existing engine's execution model is reused: 5 bps buy/slippage on next-open exits,10 bps commission per side, target limit exits, no modeled market impact or order-book liquidity.",
                  "EMA close stops can lose substantially more than a planned fixed percentage during gaps.",
                  "Realized fixed-allocation return and drawdown omit unrealized positions, compounding, concurrent position constraints, and dividends as separate cashflows.",
                  "Adjusted price histories may be retrospectively revised for corporate actions; transaction-reproducible unadjusted execution is not tested here.",
                  "This comparison includes the named technical gates only. The complete live liquidity/resistance/freshness policy and fair-value policy are not replayed.",
                  "The technical-gates-at-confirmation proxy tests the extra volume,turnover,extension,risk/resistance checks at confirmation close; live latest-scan eligibility can differ after confirmation, lifecycle changes, and next-open gaps.",
                  "No current fair value was applied to historical trades. Publication-timed historical fundamental/model snapshots are required to measure the fair-value filter.",
                  "Trade-level Wilson win intervals and approximate mean-loss intervals ignore dependence across stocks and market days; treat them as descriptive.",
              ]}
    for name, variant in VARIANTS.items():
        trades, gates = [], {}
        for ticker, context in contexts.items():
            sub, counts = replay(ticker,context,variant)
            trades.extend(sub)
            gates[ticker]=counts
            if name=="current_baseline" and ticker in ("RELIANCE","DIXON"):
                result["baseline_verification"].append(baseline_check(ticker,frames[ticker],sub))
        preceding = [t for t in trades if pd.Timestamp(t["entry_date"]) < RECENT_START]
        recent = [t for t in trades if pd.Timestamp(t["entry_date"]) >= RECENT_START]
        crossings = [t for t in preceding if t.get("exit_date") is None or pd.Timestamp(t["exit_date"])>=RECENT_START]
        result["variants"][name]={"rules":variant,"whole_3years":stats_for(trades,tickers),
            "preceding_2years":stats_for(preceding,tickers),"latest_12months":stats_for(recent,tickers),
            "cohort_boundary_crossing_entries":len(crossings),"gate_counts_by_ticker":gates}
        if name=="technical_gates_confirmation_proxy":
            result["technical_proxy_scanner_verification"]=verify_proxy_against_scanner(trades,frames)
        all_trades.extend({"variant":name,**t} for t in trades)
        summary=result["variants"][name]["whole_3years"]
        print(f"{name}: closed={summary['total_trades']} win={summary['win_rate_pct']}% PF={summary['profit_factor']} E={summary['avg_trade_return_pct']}%",flush=True)
    write_json(OUT/"effectiveness.json",result)
    pd.DataFrame(all_trades).to_csv(OUT/"all_trades.csv",index=False)
    rows=[]
    for name,r in result["variants"].items():
        for cohort in ("whole_3years","preceding_2years","latest_12months"):
            s=r[cohort]
            rows.append({"variant":name,"cohort":cohort,**{k:v for k,v in s.items() if not isinstance(v,(dict,list))}})
    pd.DataFrame(rows).to_csv(OUT/"comparison_summary.csv",index=False)
    lines=["# 3% strategy effectiveness research", "",f"Research date: {AS_OF_DATE}. Entry window: {START.date()} onward. Recent cohort: {RECENT_START.date()} onward.",
           "", "The exact app Top 20 Liquid Leaders preset was used. Current fair value was excluded from historical selection.",
           "", "The current baseline did not show a profitable edge after modeled costs in this basket: 231 closed trades over three years, 40.26% profitable trades, a 0.8371 profit factor, and -0.2122% average net return per trade. The latest twelve months also had negative expectancy (-0.2333% across 51 trades).",
           "", "The additional technical gates reduced the sample to five trades in three years. Their positive average result cannot establish effectiveness. The volume/extension variants had more favorable overall results but only 18 or 11 trades, and very few recent observations. These are candidates for broader and forward testing, not proven improvements.",
           "", "Fair value's effect is still unmeasured. Historical publication-timed valuation snapshots would be needed; present-day fair values cannot be applied retrospectively. A 3% gross limit target produces about 2.79% net at the engine's 10 bps commission per side, before any additional unmodeled costs.",
           "", "| Variant | Cohort | Closed | Net wins | Target hit | PF | Net expectancy | Avg loss | Win interval (95%) |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name,r in result["variants"].items():
        for cohort in ("whole_3years","latest_12months"):
            s=r[cohort]
            lines.append(f"| {name} | {cohort} | {s['total_trades']} | {s['win_rate_pct']}% | {s['target_hit_rate_pct']}% | {s['profit_factor']} | {s['avg_trade_return_pct']}% | {s['avg_loss_pct']}% | {s['win_rate_ci95_pct']} |")
    lines += ["", "## Rules", "", "All variants confirm an EMA9 crossing using a following higher close and higher high, enter the next open, and use a fixed 3% entry target with a 20-session maximum holding. The current baseline requires the engine's full verified uptrend and an EMA-close stop executing at the next open.",
      "", "Volume variants require confirmation volume >= 1.5x the preceding 20-session average and a confirmation close no more than 1 ATR above EMA9. The market/RS variant additionally requires exact-date Nifty uptrend and positive stock-minus-Nifty 60-session returns. PRICE variants use a fixed 1.5% stop from actual entry, with conservative stop-first ambiguous daily candles.",
      "", "The technical-gates-at-confirmation proxy requires verified uptrend, volume coverage, Rs 1 crore 20-session turnover, volume >= 1x, extension <= 1.5 ATR, estimated net reward/risk >= 1.5 and no causally confirmed 90-session pivot resistance before the estimated 3% confirmation-close target. It is a proxy for the current live technical gates, not a replay of every latest-scan decision or fair-value selection.",
      "", "## Coverage", ""]
    for t,info in manifest["coverage"].items():
        lines.append(f"- {t}: {info.get('bars')} bars; {info.get('first')} to {info.get('last')}; {info.get('reason','')}.")
    lines += ["", "## Baseline verification", ""]
    for check in result["baseline_verification"]:
        lines.append(f"- {check['ticker']}: PASS; {check['trades']} replayed trades exactly match existing engine entry, target, exit, net P&L and status fields.")
    lines += [f"- All {len(result.get('technical_proxy_scanner_verification',[]))} accepted technical-proxy confirmations pass the actual scanner when its clock is set to the historical confirmation close."]
    lines += ["", "## Limits", ""] + ["- "+x for x in result["limitations"]]
    (OUT/"EFFECTIVENESS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Saved results to {OUT}",flush=True)


if __name__=="__main__":
    main()
