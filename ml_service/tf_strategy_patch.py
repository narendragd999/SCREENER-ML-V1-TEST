"""
PATCH: Make Trade Suggestions Use TF Models
=============================================
This patch modifies ml_stock_analyzer.py so that the strategy engine
USES the trained TF model predictions (instead of just displaying them).

BEFORE (current behavior):
  - TF model predictions are SHOWN in "ML Prediction" panel
  - Trade suggestions (STRONG_BUY/BUY/WATCH/WAIT) use intelligent_scoring.py
  - These two systems DON'T CONNECT

AFTER (with this patch):
  - TF model predictions DRIVE the trade suggestions
  - intelligent_scoring.py is used only for EXPLANATION (the 10 components)
  - When TF model says BUY, the strategy says BUY
  - When TF model says STRONG_BUY, the strategy says STRONG_BUY

This is the COMBINED STUDY you want:
  Historical Learning + Continuous Learning → TF Model → Trade Suggestions

HOW TO APPLY:
  1. Backup your ml_stock_analyzer.py
  2. Find the suggest_strategy() function (around line 562)
  3. Find this block (around line 597):

     if HAS_INTELLIGENT_SCORING:
         intel_score = compute_intelligent_score(df)
         ml_score = intel_score.get("ml_score", 50)
         signal_quality = intel_score.get("signal_quality", 0.5)
         regime = intel_score.get("trend_regime", "SIDEWAYS")
         expected_return = intel_score.get("expected_return", 0)
         max_drawdown = intel_score.get("max_drawdown", 0)
         risk_reward_base = intel_score.get("risk_reward_ratio", 0)
         score_components = intel_score.get("components", {})
         scoring_method = "intelligent_v2"
     else:
         ml_score = ml_pred.get("ml_score", 50)
         signal_quality = ml_pred.get("signal_quality", 0.5)
         regime = ml_pred.get("trend_regime", "SIDEWAYS")
         expected_return = ml_pred.get("expected_return", 0)
         max_drawdown = ml_pred.get("max_drawdown", 0)
         risk_reward_base = ml_pred.get("risk_reward_ratio", 0)
         score_components = {}
         scoring_method = "ml_model"

  4. REPLACE that entire block with the new code below
"""

# ============================================================================
# NEW CODE — Replace the if/else block with this
# ============================================================================

NEW_STRATEGY_BLOCK = '''    # ══════════════════════════════════════════════════════════════
    #  COMBINED STUDY: TF Model + Intelligent Scoring (NEW)
    #  - TF model drives the DECISION (signal_quality, regime, expected_return)
    #  - Intelligent scoring provides EXPLANATION (10 components)
    # ══════════════════════════════════════════════════════════════

    ml_pred = ml_prediction or {}

    # Check if TF model is trained (model_status == "trained")
    tf_model_trained = (
        ml_pred.get("model_status") == "trained"
        and ml_pred.get("signal_quality") is not None
    )

    if tf_model_trained:
        # ✅ USE TF MODEL for the decision (this is what you want!)
        ml_score = ml_pred.get("ml_score", 50)
        signal_quality = ml_pred.get("signal_quality", 0.5)
        regime = ml_pred.get("trend_regime", "SIDEWAYS")
        expected_return = ml_pred.get("expected_return", 0)
        max_drawdown = ml_pred.get("max_drawdown", 0)
        risk_reward_base = ml_pred.get("risk_reward_ratio", 0)
        score_components = {}
        scoring_method = "tf_model_trained"

        # Also compute intelligent scoring for EXPLANATION (not for decision)
        if HAS_INTELLIGENT_SCORING:
            try:
                intel_score = compute_intelligent_score(df)
                score_components = intel_score.get("components", {})
                # Store for later use in reasons
                _intel_context = intel_score
            except Exception:
                _intel_context = {}
        else:
            _intel_context = {}
    elif HAS_INTELLIGENT_SCORING:
        # ⚠️ FALLBACK: TF model not trained, use intelligent scoring
        intel_score = compute_intelligent_score(df)
        ml_score = intel_score.get("ml_score", 50)
        signal_quality = intel_score.get("signal_quality", 0.5)
        regime = intel_score.get("trend_regime", "SIDEWAYS")
        expected_return = intel_score.get("expected_return", 0)
        max_drawdown = intel_score.get("max_drawdown", 0)
        risk_reward_base = intel_score.get("risk_reward_ratio", 0)
        score_components = intel_score.get("components", {})
        scoring_method = "intelligent_v2_fallback"
        _intel_context = intel_score
    else:
        # ❌ NO MODEL AVAILABLE — use whatever ml_pred has
        ml_score = ml_pred.get("ml_score", 50)
        signal_quality = ml_pred.get("signal_quality", 0.5)
        regime = ml_pred.get("trend_regime", "SIDEWAYS")
        expected_return = ml_pred.get("expected_return", 0)
        max_drawdown = ml_pred.get("max_drawdown", 0)
        risk_reward_base = ml_pred.get("risk_reward_ratio", 0)
        score_components = {}
        scoring_method = "no_model"
        _intel_context = {}'''


# ============================================================================
# Additional enhancement: Add a TF-specific reason to the strategy
# ============================================================================

ADDITIONAL_TF_REASON = '''    # ── TF MODEL SIGNAL (NEW — shows that trained model is driving decision) ──
    if tf_model_trained:
        model_version = ml_pred.get("model_version", "unknown")
        if signal_quality > 0.6:
            boost_total += 12
            reasons.append({"text": f"🤖 TF Model ({model_version[:25]}): {signal_quality:.0%} confidence — HIGH quality signal", "sentiment": "positive", "impact": 12})
        elif signal_quality > 0.45:
            boost_total += 6
            reasons.append({"text": f"🤖 TF Model: {signal_quality:.0%} confidence — moderate signal", "sentiment": "positive", "impact": 6})
        elif signal_quality < 0.3:
            penalty_total += 10
            reasons.append({"text": f"🤖 TF Model: {signal_quality:.0%} confidence — LOW quality, avoid", "sentiment": "negative", "impact": -10})
        else:
            reasons.append({"text": f"🤖 TF Model: {signal_quality:.0%} confidence — neutral", "sentiment": "neutral", "impact": 0})

    # ── INTELLIGENT SCORING CONTEXT (explanation only, not for decision) ──
    if _intel_context:
        comp = _intel_context.get("components", {})
        if comp:
            top_components = sorted(comp.items(), key=lambda x: x[1], reverse=True)[:3]
            for name, score in top_components:
                if score >= 8:
                    reasons.append({"text": f"📊 {name.replace('_', ' ').title()}: {score}/10 — strong", "sentiment": "positive", "impact": 0})
                elif score <= 3:
                    reasons.append({"text": f"📊 {name.replace('_', ' ').title()}: {score}/10 — weak", "sentiment": "negative", "impact": 0})
'''


# ============================================================================
# Auto-patch script
# ============================================================================

def apply_patch(filepath: str) -> bool:
    """Apply the patch to ml_stock_analyzer.py."""
    from pathlib import Path
    import shutil

    path = Path(filepath)
    if not path.exists():
        print(f"ERROR: File not found: {filepath}")
        return False

    content = path.read_text(encoding="utf-8")

    # Backup
    backup_path = path.with_suffix(".py.bak_tf")
    shutil.copy2(path, backup_path)
    print(f"  [BACKUP] Saved to: {backup_path}")

    # Find the old block to replace
    old_block_start = '    if HAS_INTELLIGENT_SCORING:\n        intel_score = compute_intelligent_score(df)'
    old_block_end = '        scoring_method = "ml_model"'

    start_idx = content.find(old_block_start)
    if start_idx == -1:
        print("  [SKIP] Old block not found — may already be patched")
        return False

    end_idx = content.find(old_block_end, start_idx)
    if end_idx == -1:
        print("  [SKIP] End of old block not found")
        return False

    # Include the full line
    end_idx = content.find('\n', end_idx) + 1

    # Replace
    old_block = content[start_idx:end_idx]
    content = content[:start_idx] + NEW_STRATEGY_BLOCK.strip() + '\n' + content[end_idx:]
    print("  [OK] Replaced strategy block with TF model integration")

    # Find a good place to add the TF reason (after the pattern conflict resolution)
    # Look for "# ── ML/INTELLIGENT REGIME ──"
    regime_marker = '    # ── ML/INTELLIGENT REGIME ──'
    regime_idx = content.find(regime_marker)
    if regime_idx != -1:
        content = content[:regime_idx] + ADDITIONAL_TF_REASON.strip() + '\n\n' + content[regime_idx:]
        print("  [OK] Added TF model reason block")
    else:
        print("  [SKIP] Could not find location for TF reason (not critical)")

    # Write back
    path.write_text(content, encoding="utf-8")
    print(f"\n  [DONE] Patched: {path}")
    print("\n  Next steps:")
    print("  1. Restart your ml_service")
    print("  2. Visit /stock-analysis/{TICKER}")
    print("  3. Look for '🤖 TF Model' in the reasons — this confirms TF is driving decisions")
    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tf_strategy_patch.py <path_to_ml_stock_analyzer.py>")
        print("\nOr copy the NEW_STRATEGY_BLOCK code manually.")
        sys.exit(1)
    apply_patch(sys.argv[1])
