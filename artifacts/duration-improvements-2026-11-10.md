# Duration research and app changes

The user's maximum holding duration is 90 sessions per trade. This update preserves that no-stop target question as a separate mode rather than substituting the confirmed-EMA, 20-session, EMA-stop strategy.

## Implemented

- Original strict-cross, breakout-close target study with independent overlapping signals.
- Next-open duration comparison with costs, one position per ticker and explicit pending time exits.
- Full-universe processing without the legacy 300-symbol cap; completed/failed ticker progress and coverage.
- Unrounded targets to avoid serialized two-decimal prices changing a supposed 3% gain.
- Separate all-entry, mature-horizon, closed-target and net-profitable denominators.
- Target-time curve, median/p90 waiting time, adverse excursions, losses and simultaneous exposures.
- Explicit censored tails instead of assigning an incomplete 90-session opportunity a terminal result.
- Optional existing resistance/pullback gates, the separately unvalidated trend/volume gate and a resistance research preset.
- Cached-current-FV subset explicitly retrospective; it does not alter the primary historical strategy.
- Complete CSV/JSON exports, including failures and unresolved entries; the screen displays up to 100 matching rows.
- Updated dashboard and APP_GUIDE.md; charts retain measured tables/exports when rendering fails.

## Verification

128 Python tests and 26 UI regression groups passed. Nineteen duration tests cover horizon boundaries, causal filters, full universe coverage, costs, MAE, overlap, pending exits and denominators. Two API tests preserve the full 307-symbol request and reject invalid policies.

Live browser verification used the six-stock banking preset, five years, 3%, 90 sessions, next-open fills and the resistance filter. All six completed, with 161 entries: 143 target hits, 15 time exits and three censored. The full-horizon cohort contained 153 entries and 138 target hits (90.1961%). The observed all-entry target proportion was 88.8199%; the two denominators remain distinct. Net average across 158 closed entries was +1.5343% with configured costs.

Browser JSON was exported while the display showed nonwinners; its file still retained all 161 entries. Browser CSV with All outcomes also contained all 161, including the 15 time exits and three censored entries. Charts, complete outcome filtering and controls after job completion were verified. The live payload is saved in `duration-live-banking-2026-11-10.json`.

The independently launched local service remains running at http://127.0.0.1:8000, process 16584. Refresh the dashboard to load the new controls.

## Interpretation

The earlier 20-session confirmed strategy does not invalidate the user's duration result. The original 19-stock duration replay produced 85.46% pooled target hits and a legacy pullback/resistance subset produced 90.83% with full follow-up. The next-open resistance candidate improved the matched sample, but its latest year still had negative net expectancy. No strategy has been certified as a stable future 90% probability, and the exact 307-stock historical fair-value run remains unverified because the saved winner-only workbook lacks the full denominator and publication-timed valuations.

See `duration-strategy-research-2026-11-10.md`, `duration-backtest-audit-2026-11-10.md` and `saved-backtest-reconciliation-2026-11-10.md` for the measured comparison and provenance.
