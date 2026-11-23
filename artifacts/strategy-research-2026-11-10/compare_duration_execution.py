"""Compare predefined no-stop duration variants using saved prices only.

Run from the repository root with Python. This script never downloads prices,
changes production state, or selects thresholds from the resulting scores.
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import sys
from collections import Counter

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from core.duration_backtest import backtest_duration_ticker, compute_duration_stats


JSON_PATH = HERE / "duration_execution_comparison.json"
REPORT_PATH = ROOT / "artifacts" / "duration-strategy-research-2026-11-10.md"
VARIANTS = {
    "next_open_20_all_trends": {"max_hold_sessions": 20, "require_uptrend": False, "research_filters": False},
    "next_open_90_all_trends": {"max_hold_sessions": 90, "require_uptrend": False, "research_filters": False},
    "next_open_90_above_sma50": {"max_hold_sessions": 90, "require_uptrend": True, "research_filters": False},
    "next_open_90_research_filters": {"max_hold_sessions": 90, "require_uptrend": False, "research_filters": True},
    "next_open_90_away_resistance": {"max_hold_sessions": 90, "avoid_near_resistance": True},
    "next_open_90_pullback10": {"max_hold_sessions": 90, "require_pullback": True},
    "next_open_90_pullback10_away_resistance": {"max_hold_sessions": 90, "avoid_near_resistance": True, "require_pullback": True},
}
LABELS = {
    "next_open_20_all_trends": "20 sessions, all trends",
    "next_open_90_all_trends": "90 sessions, all trends",
    "next_open_90_above_sma50": "90 sessions, above SMA50",
    "next_open_90_research_filters": "90 sessions, trend + volume + extension",
    "next_open_90_away_resistance": "90 sessions, away from resistance",
    "next_open_90_pullback10": "90 sessions, pullback at least 10%",
    "next_open_90_pullback10_away_resistance": "90 sessions, pullback + away from resistance",
}


def cohort_stats(trades):
    """Keep complete-horizon membership independent of whether a trade won."""
    return {
        "all_entries": compute_duration_stats(trades),
        "full_horizon_cohort": compute_duration_stats([t for t in trades if t["full_horizon_available"]]),
    }


def validate_execution(trades):
    for ticker in sorted({t["ticker"] for t in trades}):
        previous = None
        for trade in sorted([t for t in trades if t["ticker"] == ticker], key=lambda t: t["entry_idx"]):
            if previous is not None:
                assert previous["exit_idx"] is not None, "An open position cannot admit another entry"
                assert trade["entry_idx"] > previous["exit_idx"], "Same-ticker positions overlap"
            assert trade["entry_date"] > trade["signal_date"], "Entry must follow its signal"
            if trade["target_hit"]:
                assert trade["exit_price"] >= trade["target_price"] - 1e-8
                assert trade["target_hit_sessions"] <= trade["max_hold_sessions"]
            if trade["exit_idx"] is None:
                assert trade["net_pnl_pct"] is None, "Unexecuted exits cannot contribute realized returns"
            previous = trade


def candidate_assessment(base, candidate):
    """Descriptive consistency check; no significance claims or optimization."""
    rows = []
    for block in base["time_blocks"]:
        b = base["time_blocks"][block]["full_horizon_cohort"]
        c = candidate["time_blocks"][block]["full_horizon_cohort"]
        rows.append({
            "block": block, "base_entries": b["total_entries"], "candidate_entries": c["total_entries"],
            "target_hit_rate_difference_pp": None if c["mature_target_hit_rate_pct"] is None or b["mature_target_hit_rate_pct"] is None
                else round(c["mature_target_hit_rate_pct"] - b["mature_target_hit_rate_pct"], 4),
            "net_expectancy_difference_pct": None if c["avg_net_return_pct"] is None or b["avg_net_return_pct"] is None
                else round(c["avg_net_return_pct"] - b["avg_net_return_pct"], 4),
        })
    adequate = candidate["cohorts"]["full_horizon_cohort"]["total_entries"] >= 30 and all(r["candidate_entries"] >= 10 for r in rows)
    target_consistent = all(r["target_hit_rate_difference_pp"] is not None and r["target_hit_rate_difference_pp"] > 0 for r in rows)
    net_consistent = all(r["net_expectancy_difference_pct"] is not None and r["net_expectancy_difference_pct"] > 0 for r in rows)
    pooled_base = base["cohorts"]["full_horizon_cohort"]
    pooled_candidate = candidate["cohorts"]["full_horizon_cohort"]
    target_delta = None if pooled_candidate["mature_target_hit_rate_pct"] is None or pooled_base["mature_target_hit_rate_pct"] is None \
        else round(pooled_candidate["mature_target_hit_rate_pct"] - pooled_base["mature_target_hit_rate_pct"], 4)
    net_delta = None if pooled_candidate["avg_net_return_pct"] is None or pooled_base["avg_net_return_pct"] is None \
        else round(pooled_candidate["avg_net_return_pct"] - pooled_base["avg_net_return_pct"], 4)
    pooled_improvement = target_delta is not None and net_delta is not None and target_delta > 0 and net_delta > 0
    if adequate and target_consistent and net_consistent and pooled_improvement:
        finding = "Candidate for further testing: target frequency and net expectancy improve in the pooled cohort and all three time blocks. Still unvalidated; the latest cohort remains net negative."
    elif adequate and target_consistent and net_consistent:
        finding = "Inconclusive filter benefit: individual time blocks improve, but pooled target frequency and net expectancy do not. Accepted-entry and time-cohort weighting differ."
    else:
        finding = "Inconclusive filter benefit: target frequency and net expectancy do not both improve consistently across adequately populated time blocks."
    return {"finding": finding, "validated": False, "minimum_descriptive_counts_met": adequate,
            "target_rate_improves_in_every_block": target_consistent, "net_expectancy_improves_in_every_block": net_consistent,
            "pooled_target_hit_difference_pp": target_delta, "pooled_net_expectancy_difference_pct": net_delta,
            "pooled_target_and_net_improve": pooled_improvement,
            "time_block_differences": rows}


def inspect_older_database():
    """Coverage evidence only; do not blend a different historical source."""
    path = ROOT / "ml_service" / "data" / "ml_intel.db"
    if not path.exists():
        return {"available": False}
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        coverage = [dict(r) for r in db.execute(
            "SELECT ticker,COUNT(*) bars,MIN(date) first_date,MAX(date) last_date,SUM(volume=0) zero_volume "
            "FROM ohlcv GROUP BY ticker ORDER BY ticker")]
        invalid = db.execute("SELECT COUNT(*) FROM ohlcv WHERE open<=0 OR high<=0 OR low<=0 OR close<=0 "
                             "OR high<low OR high<open OR high<close OR low>open OR low>close").fetchone()[0]
    return {"available": True, "path": str(path), "read_only": True,
            "tickers": len(coverage), "rows": sum(r["bars"] for r in coverage),
            "date_min": min(r["first_date"] for r in coverage), "date_max": max(r["last_date"] for r in coverage),
            "tickers_at_least_500_bars": sum(r["bars"] >= 500 for r in coverage), "invalid_ohlc_rows": invalid,
            "last_date_counts": dict(Counter(r["last_date"] for r in coverage)), "coverage": coverage,
            "download_code_default": "yfinance history, auto_adjust=True; delta downloader defaults to 1d",
            "row_level_source_and_adjustment_provenance": False,
            "used_for_strategy_comparison": False,
            "reason_not_blended": "Rows lack source, interval and adjustment version; demo/test writers share the schema. July snapshot is distinct from November prices."}


def value(number, digits=2):
    return "n.a." if number is None else f"{number:.{digits}f}"


def write_report(result):
    legacy = result["original_engine_reconciliation"]["variants"]
    rows = [
        "# Duration strategy comparison",
        "",
        "Reviewed on 10 November 2026 using saved prices only. No network calls or production changes were made by this script.",
        "",
        "## What the original result establishes",
        "",
        f"The exact original duration engine produces **{legacy['legacy_90_sessions_all_trends']['legacy_target_hit_rate_pct']:.2f}% target hits with 90 sessions**, "
        f"versus **{legacy['legacy_20_sessions_all_trends']['legacy_target_hit_rate_pct']:.2f}% with 20** on the same 19 available stocks. "
        "The user's longer holding period materially changes the result; the earlier short-hold comparison did not represent it.",
        "",
        "The original pullback-at-least-10%-below-prior-high plus away-from-resistance subset reaches **89.71% across 272 entries**, "
        "or **90.83% across 218 entries with a complete 90-session follow-up**. Its modeled average net return is **+1.5408% per entry**, "
        "profit factor **2.756**, worst net trade **−17.81%**, and worst adverse excursion **−30.12%**. "
        "Adding SMA50 raises target hits to 93.33%, but leaves just 15 entries. These descriptive subsets make a 90%+ result plausible; "
        "they do not replicate all 307 stocks or historically available fair value.",
        "",
        "## Executable next-open comparison",
        "",
        f"Scope: {result['scope']['available_tickers']} stocks from the saved 20-stock preset; TATAMOTORS has no validated cached prices. "
        f"Entries run from {result['scope']['entry_start']} through {result['scope']['observation_end']}, with earlier bars used for indicator warmup. "
        "Fair value is excluded because dated historical valuation inputs are unavailable.",
        "",
        "Each strict EMA9 cross is known at its close. Buy at the next observed open with 5 bps slippage; hold one position per ticker, "
        "with no stop. The target is an unrounded 3% above the executed buy price. Commission is 10 bps per side. "
        "Target touches assume a sell-limit fill without exit slippage; market time exits use 5 bps exit slippage. "
        "The entry session counts as session one. An unhit horizon close triggers an exit at the following open. "
        "Incomplete follow-up remains open/censored, and an unavailable exit open remains pending.",
        "",
        "The full-horizon cohort includes an entry only if its entire configured target observation period exists, even when it won early. "
        "This avoids comparing recent early winners with older losses while dropping recent unresolved trades. "
        "Net expectancy below is the arithmetic mean of net returns for closed trades in that full-horizon cohort; "
        "its closed count is shown whenever a pending exit prevents a realized return. It is not a compounded portfolio return.",
        "",
        "| Next-open variant | All entries / open | Full-horizon entries / closed | Full-horizon target hits | Net expectancy | Net profit factor | Worst net loss | Worst observed drawdown |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, data in result["variants"].items():
        a = data["cohorts"]["all_entries"]
        m = data["cohorts"]["full_horizon_cohort"]
        rows.append(f"| {LABELS[key]} | {a['total_entries']} / {a['open_count']} | {m['total_entries']} / {m['closed_count']} | "
                    f"{value(m['mature_target_hit_rate_pct'])}% | {value(m['avg_net_return_pct'], 4)}% | {value(m['profit_factor'], 3)} | "
                    f"{value(m['worst_net_loss_pct'])}% | {value(a['worst_mae_pct'])}% |")
    rows += [
        "",
        "Moving to next-open execution changes the buy price and target barrier. One position per ticker suppresses signals while a trade is open. "
        "A longer holding period also changes which later trades are admitted. Thus these variants evaluate executable policies with different accepted entries; "
        "the original overlapping study isolates target touches on a common signal population more directly.",
        "",
        "The additional research gate requires a verified SMA50/SMA200 uptrend (including rising SMA50 and positive 60-session slope), "
        "signal volume at least 1.5 times its previous 20-session average, and EMA extension from zero to one ATR. "
        "These fixed thresholds came from the predefined hypotheses. No thresholds were searched or retuned from this comparison.",
        "",
        "The practical legacy gates are also compared separately and together: avoid a prior 20-session high above the signal close "
        "when it is within 3% of that close; require the signal close to be at least 10% below the prior 60-session high. "
        "Both use only history available at the signal close. A next-open gap can alter actual entry resistance distance; "
        "the signal-close filter does not inspect that future opening price.",
        "",
        "## Recent year and earlier periods",
        "",
        "Each row uses entry dates within the stated period and the full-horizon membership rule. "
        "The latest year's mature 90-session entries stop well before the final price date; recent unresolved trades remain visible in the JSON.",
        "",
        "| Variant | Entry cohort | Full-horizon entries / closed | Target hits | Net expectancy | Net profit factor |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key, data in result["variants"].items():
        for block, cohorts in data["time_blocks"].items():
            m = cohorts["full_horizon_cohort"]
            rows.append(f"| {LABELS[key]} | {block} | {m['total_entries']} / {m['closed_count']} | {value(m['mature_target_hit_rate_pct'])}% | "
                        f"{value(m['avg_net_return_pct'], 4)}% | {value(m['profit_factor'], 3)} |")
    recent_base = result["variants"]["next_open_90_all_trends"]["recent_12_months"]["full_horizon_cohort"]
    rows += [
        "",
        f"The latest mature 90-session baseline cohort has **{value(recent_base['mature_target_hit_rate_pct'])}% target hits** "
        f"and **{value(recent_base['avg_net_return_pct'], 4)}% net expectancy** "
        f"({recent_base['total_entries']} entries, {recent_base['closed_count']} closed). "
        "The longer horizon's pooled improvement does not establish stable returns across market periods.",
        "",
        "## What to adopt",
        "",
    ]
    for key, assessment in result["filter_assessments"].items():
        rows.append(f"**{LABELS[key]}:** {assessment['finding']}")
        rows.append("")
    rows += [
        "Preserve the 90-session no-stop target study as a distinct research view. Show executable next-open results beside it, "
        "with separate target-hit, profitable-trade, open-position and mature-cohort denominators. "
        "A filter should not be presented as an accuracy improvement based on pooled target hits alone.",
        "",
        "## Saved workbook evidence",
        "",
        "`sample-data-5-year-250-days.xlsx`, Sheet1!A1:U5235, contains 5,234 successful 3% examples across 294 stocks; "
        "median target time is three sessions and p90 is 21. It contains only WIN rows and no original run configuration or missing-trade list, "
        "so it cannot supply the full win-rate denominator. Of these rows, 4,366 are Prime and 2,975 are Prime plus uptrend. "
        "There are 1,350 overlapping same-ticker entries. Fair values are assigned per April–March fiscal year, "
        "but publication/availability dates are absent. These examples support investigating the setup without proving historical valuation effectiveness.",
        "",
        "## Additional older price cache",
        "",
        "The read-only ML database contains 212,312 OHLCV rows across 106 tickers (93 have at least 500 bars), "
        "covering July 2016 to July 2026. No structurally invalid OHLC rows were found. "
        "Downloader code uses daily auto-adjusted Yahoo history, but the rows do not preserve source, interval, "
        "or adjustment version and share their schema with demo/test writers. This distinct July snapshot was not blended into the November comparison. "
        "Its ticker coverage is retained in the JSON for a separate, explicitly dated study after provenance checks.",
        "",
        "## Limits and reproduction",
        "",
        "The sample uses currently selected stocks, daily target-touch fills, and correlated entries. It excludes historical fair value, "
        "delisted-stock reconstruction, order liquidity, and a fixed-capital portfolio. MAE includes the target day's full low because intraday ordering "
        "cannot be determined from daily bars. Results are descriptive and cannot establish a stable future 90% success probability.",
        "",
        "Run `python artifacts/strategy-research-2026-11-10/compare_duration_execution.py` from the repository root. "
        "Inputs are the saved `data_manifest.json`, its 19 stock price CSVs, `duration_reconciliation.json`, and "
        "`saved-backtest-reconciliation-2026-11-10.json`. The comparison JSON records each cohort, ticker breakdown, "
        "execution parameters, filter assessment and older database coverage. All data downloads are disabled by design.",
    ]
    REPORT_PATH.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    manifest = json.loads((HERE / "data_manifest.json").read_text(encoding="utf-8"))
    start = manifest["analysis_start"]
    recent = manifest["recent_cohort_start"]
    frames = {}
    skipped = []
    for ticker in manifest["tickers"]:
        info = manifest["coverage"].get(ticker, {})
        if not info.get("file"):
            skipped.append({"ticker": ticker, "reason": info.get("reason", "NO_SAVED_PRICE_FILE")})
            continue
        frames[ticker] = pd.read_csv(HERE / info["file"], index_col=0, parse_dates=True)
    end = max(str(b.index[-1].date()) for b in frames.values())
    previous_start = str((pd.Timestamp(recent) - pd.DateOffset(years=1)).date())
    blocks = {
        "2023-11-11 to 2024-11-09": (start, str((pd.Timestamp(previous_start) - pd.Timedelta(days=1)).date())),
        "2024-11-10 to 2025-11-09": (previous_start, str((pd.Timestamp(recent) - pd.Timedelta(days=1)).date())),
        "2025-11-10 to 2026-11-09": (recent, end),
    }
    variants = {}
    for key, parameters in VARIANTS.items():
        parameters = {"require_uptrend": False, "research_filters": False,
                      "avoid_near_resistance": False, "require_pullback": False, **parameters}
        trades = []
        for ticker, bars in frames.items():
            trades.extend(backtest_duration_ticker(ticker, bars, target_pct=3.0, mode="duration_next_open",
                          slippage_bps=5.0, commission_bps=10.0, start_date=start, end_date=end, **parameters))
        trades.sort(key=lambda t: (t["entry_date"], t["ticker"]))
        validate_execution(trades)
        variants[key] = {
            "parameters": {"mode": "duration_next_open", "target_pct": 3.0, "slippage_bps": 5.0, "commission_bps": 10.0, **parameters},
            "cohorts": cohort_stats(trades),
            "prior_combined": cohort_stats([t for t in trades if t["entry_date"] < recent]),
            "recent_12_months": cohort_stats([t for t in trades if t["entry_date"] >= recent]),
            "time_blocks": {label: cohort_stats([t for t in trades if first <= t["entry_date"] <= last]) for label, (first, last) in blocks.items()},
            "ticker_breakdowns": {ticker: cohort_stats([t for t in trades if t["ticker"] == ticker]) for ticker in frames},
            "trades": trades,
        }
        mature = variants[key]["cohorts"]["full_horizon_cohort"]
        print(key, json.dumps({k: mature[k] for k in ("total_entries", "closed_count", "mature_target_hit_rate_pct",
              "avg_net_return_pct", "profit_factor", "worst_net_loss_pct", "worst_mae_pct")}), flush=True)
    base = variants["next_open_90_all_trends"]
    result = {
        "research_date": "2026-11-10", "no_network": True,
        "scope": {"requested_tickers": manifest["tickers"], "available_tickers": len(frames), "skipped": skipped,
                  "entry_start": start, "observation_end": end, "recent_cohort_start": recent,
                  "historical_fair_value_gate": False, "time_blocks": blocks},
        "variants": variants,
        "filter_assessments": {key: candidate_assessment(base, variants[key]) for key in variants
                               if key not in ("next_open_20_all_trends", "next_open_90_all_trends")},
        "original_engine_reconciliation": json.loads((HERE / "duration_reconciliation.json").read_text(encoding="utf-8")),
        "saved_workbook_evidence": json.loads((ROOT / "artifacts" / "saved-backtest-reconciliation-2026-11-10.json").read_text(encoding="utf-8")),
        "older_database_coverage": inspect_older_database(),
        "predefined_thresholds": {"volume_ratio_min": 1.5, "ema_extension_atr_range": [0.0, 1.0], "verified_uptrend": True,
                                 "source": str(HERE / "predefined_hypotheses.json"), "tuned_from_comparison_results": False},
        "predefined_legacy_gates": {"resistance_lookback_sessions": 20, "near_resistance_threshold_pct": 3.0,
                                   "pullback_lookback_sessions": 60, "minimum_pullback_pct": 10.0,
                                   "gate_evaluation": "signal_close", "tuned_from_comparison_results": False},
        "verification": {"next_session_entries": True, "same_ticker_positions_do_not_overlap": True,
                         "unresolved_entries_have_no_realized_pnl": True, "target_fills_reach_unrounded_barrier": True},
    }
    JSON_PATH.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(result)
    print("Wrote", JSON_PATH)
    print("Wrote", REPORT_PATH)


if __name__ == "__main__":
    main()
