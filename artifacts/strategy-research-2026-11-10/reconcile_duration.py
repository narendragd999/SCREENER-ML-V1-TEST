"""Replay the original duration function without importing its side-effectful router.

Uses the already saved, validated 19-stock OHLCV dataset. This is a scope-matched
duration comparison, not replication of the user's full 307-symbol/FV cohort.
"""
from __future__ import annotations
import ast
import json
import logging
import pathlib
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
START = "2023-11-11"


def original_engine():
    source = (ROOT / "ema9_router.py").read_text(encoding="utf-8-sig")
    parsed = ast.parse(source)
    names = {"_duration_backtest_ticker", "_compute_250_bar_structural_low"}
    selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"pd": pd, "np": np, "Dict": Dict, "List": List, "Optional": Optional,
                 "MIN_HISTORY_DAYS": 60, "logger": logging.getLogger("duration_reconciliation")}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "legacy_duration_functions", "exec"), namespace)
    return namespace["_duration_backtest_ticker"]


def stats(trades, horizon):
    wins = [t for t in trades if t["outcome"] == "WIN"]
    mature = [t for t in trades if t["full_horizon_available"]]
    censored = [t for t in trades if not t["full_horizon_available"] and t["outcome"] != "WIN"]
    net = np.array([t["net_return_pct"] for t in trades], dtype=float)
    negative = net[net < 0]
    maes = np.array([t["mae_pct"] for t in trades], dtype=float)
    periods = np.array([t["days_held"] for t in wins], dtype=float)
    events = {}
    for t in trades:
        events[t["entry_date"]] = events.get(t["entry_date"], 0) + 1
        # Close-entry happens after intraday exits, so prior trades don't
        # require capital concurrently with close entries on their exit day.
        events[t["exit_date"]] = events.get(t["exit_date"], 0) - 1
    active = peak = 0
    for d in sorted(events):
        active += events[d]
        peak = max(peak, active)
    return {
        "horizon_sessions": horizon, "entries": len(trades), "wins_target_hit": len(wins),
        "losses": sum(t["outcome"] == "LOSS" for t in trades),
        "positive_timeouts": sum(t["outcome"] == "TIMEOUT" for t in trades),
        "legacy_target_hit_rate_pct": round(len(wins)/len(trades)*100, 4) if trades else None,
        "fully_mature_entries": len(mature),
        "mature_target_hit_rate_pct": round(sum(t["outcome"] == "WIN" for t in mature)/len(mature)*100, 4) if mature else None,
        "short_followup_nonwins": len(censored),
        "net_win_rate_pct_at_30bps_roundtrip": round(float(np.mean(net > 0)*100), 4),
        "mean_net_return_pct": round(float(net.mean()), 4),
        "net_profit_factor": round(float(net[net > 0].sum()/-negative.sum()), 4) if len(negative) else None,
        "worst_net_return_pct": round(float(net.min()), 4),
        "average_net_loss_pct": round(float(negative.mean()), 4) if len(negative) else None,
        "median_target_hold_sessions": round(float(np.median(periods)), 2) if len(periods) else None,
        "p90_target_hold_sessions": round(float(np.quantile(periods, .9)), 2) if len(periods) else None,
        "median_adverse_excursion_pct": round(float(np.median(maes)), 4),
        "worst_adverse_excursion_pct": round(float(maes.min()), 4),
        "maximum_concurrent_independent_signals": peak,
        "participating_tickers": len({t["ticker"] for t in trades}),
    }


def main():
    replay = original_engine()
    manifest = json.loads((OUT / "data_manifest.json").read_text())
    variants = {}
    for horizon in (20, 90):
        for uptrend in (False, True):
            key = f"legacy_{horizon}_sessions_" + ("above_sma50" if uptrend else "all_trends")
            all_trades = []
            for ticker in manifest["tickers"]:
                info = manifest["coverage"].get(ticker, {})
                if not info.get("file"):
                    continue
                bars = pd.read_csv(OUT / info["file"], index_col=0, parse_dates=True)
                trades = replay(ticker, bars, target_pct=3, max_hold_days=horizon,
                                require_uptrend=uptrend, start_date=START)["trades"]
                for t in trades:
                    i = bars.index.get_indexer([pd.Timestamp(t["entry_date"])])[0]
                    j = bars.index.get_indexer([pd.Timestamp(t["exit_date"])])[0]
                    low = float(bars.Low.iloc[i+1:j+1].min())
                    t["full_horizon_available"] = len(bars)-1-i >= horizon
                    t["mae_pct"] = min(0.0, (low/t["entry_price"]-1)*100)
                    # Cost sensitivity, not an executable same-close fill model.
                    t["net_return_pct"] = (t["exit_price"]*.9985/(t["entry_price"]*1.0015)-1)*100
                all_trades.extend(trades)
            variants[key] = stats(all_trades, horizon)
            if horizon == 90:
                variants[key+"_no_near_resistance"] = stats([t for t in all_trades if not t["is_near_resistance"]], horizon)
                variants[key+"_pullback10_and_no_near_resistance"] = stats([t for t in all_trades if t["is_pullback"] and not t["is_near_resistance"]], horizon)
            pd.DataFrame(all_trades).to_csv(OUT / (key+"_trades.csv"), index=False)
            print(key, json.dumps(variants[key]), flush=True)
    result = {"scope": "19 currently available liquid-preset stocks; excludes TATAMOTORS with unavailable data",
              "entry_start": START, "observation_end": "2026-11-09", "fair_value_filter": False,
              "exact_original_function_extracted": True, "same_close_fill": True,
              "no_stop": True, "independent_overlapping_signals": True,
              "legacy_tail_force_close": True, "variants": variants,
              "caveats": ["Not a 307-ticker reproduction", "No historical FV snapshots", "No portfolio compounding",
                          "Filter subsets are descriptive, not independently validated strategies"]}
    (OUT / "duration_reconciliation.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
