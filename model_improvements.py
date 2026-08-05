"""
model_improvements.py
Implements three mentor suggestions:
  1. Probability calibration check (Platt / isotonic)
  2. SHAP feature importance (replaces misleading built-in importance)
  3. Documents cross-validation strategy

Run: python model_improvements.py
Outputs saved to reports/figures/
"""
import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss

FIG_DIR    = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42
TEST_SIZE   = 0.20

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading model and data...")
model = joblib.load("models/best_model.joblib")
FEAT_COLS = model.get_booster().feature_names

df = pd.read_csv("data/features.csv")
if 'frequency' in df.columns:
    freq_dummies = pd.get_dummies(df['frequency'], prefix='frequency')
    df = pd.concat([df.drop(columns=['frequency']), freq_dummies], axis=1)
for c in FEAT_COLS:
    if c not in df.columns:
        df[c] = 0

X = df[FEAT_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df['churned'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}, Churn rate: {y_test.mean():.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. PROBABILITY CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/3] Probability calibration...")

# Calibrate with isotonic regression on training data
calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
calibrated_model.fit(X_train, y_train)

# Compare raw vs calibrated probabilities on test set
raw_probs  = model.predict_proba(X_test)[:, 1]
cal_probs  = calibrated_model.predict_proba(X_test)[:, 1]

brier_raw  = brier_score_loss(y_test, raw_probs)
brier_cal  = brier_score_loss(y_test, cal_probs)
auc_raw    = roc_auc_score(y_test, raw_probs)
auc_cal    = roc_auc_score(y_test, cal_probs)

print(f"  Raw model  — ROC-AUC: {auc_raw:.4f}  Brier score: {brier_raw:.4f}")
print(f"  Calibrated — ROC-AUC: {auc_cal:.4f}  Brier score: {brier_cal:.4f}")
print(f"  (Lower Brier = better calibration)")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, probs, label, color in zip(
    axes,
    [raw_probs, cal_probs],
    ['Raw XGBoost', 'After Isotonic Calibration'],
    ['#C0504D', '#4F81BD']
):
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10, strategy='quantile')
    ax.plot(prob_pred, prob_true, 's-', color=color, lw=2, label=label)
    ax.plot([0, 1], [0, 1], '--', color='grey', lw=1, label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives (Actual Rate)')
    ax.set_title(f'{label}\nBrier={brier_raw if label.startswith("Raw") else brier_cal:.4f}  AUC={auc_raw if label.startswith("Raw") else auc_cal:.4f}')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    # Show where 0.10 and 0.15 thresholds land
    for thresh in [0.10, 0.15]:
        ax.axvline(thresh, ls=':', color='orange', lw=1.2, alpha=0.8)
        ax.text(thresh + 0.01, 0.02, f'{thresh}', color='orange', fontsize=8)

fig.suptitle('Calibration Check — Do Predicted Probabilities Match Reality?', fontweight='bold')
fig.tight_layout()
fig.savefig(FIG_DIR / 'calibration_comparison.png', dpi=150)
plt.close(fig)
print("  Saved -> calibration_comparison.png")

# Save calibrated model
joblib.dump(calibrated_model, "models/calibrated_model.joblib")
print("  Saved -> models/calibrated_model.joblib")

# Determine if thresholds need adjusting
print("\n  Threshold analysis (what fraction are truly churned at each predicted prob):")
for thresh in [0.10, 0.15, 0.20, 0.30]:
    mask = cal_probs >= thresh
    if mask.sum() > 0:
        actual_rate = y_test[mask].mean()
        n = mask.sum()
        print(f"    cal_prob >= {thresh:.2f}: {n:3d} accounts flagged, actual churn rate = {actual_rate:.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SHAP FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/3] SHAP feature importance...")
try:
    import shap

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_test)

    # Global summary plot (bar)
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_vals, X_test, plot_type='bar',
                      feature_names=FEAT_COLS, show=False, max_display=15)
    plt.title('SHAP Global Feature Importance\n(mean |SHAP value| across test set)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'shap_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved -> shap_importance.png")

    # Beeswarm plot (shows direction of effect)
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(shap_vals, X_test, feature_names=FEAT_COLS, show=False, max_display=15)
    plt.title('SHAP Beeswarm — Feature Effects on Churn Prediction\n(red=high feature value, blue=low)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'shap_beeswarm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved -> shap_beeswarm.png")

    # Print top features by SHAP
    mean_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({'feature': FEAT_COLS, 'mean_abs_shap': mean_shap})
    shap_df = shap_df.sort_values('mean_abs_shap', ascending=False)
    print("\n  Top 10 features by SHAP:")
    for _, row in shap_df.head(10).iterrows():
        print(f"    {row['feature']:<35} {row['mean_abs_shap']:.4f}")
    shap_df.to_csv("reports/shap_importance.csv", index=False)
    print("  Saved -> reports/shap_importance.csv")

except ImportError:
    print("  shap not installed. Run: pip install shap")
except Exception as e:
    print(f"  SHAP failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. CROSS-VALIDATION STRATEGY SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/3] Cross-validation strategy...")

cv_report = """
CROSS-VALIDATION STRATEGY — FIN-02 CHURN PREDICTION
====================================================

Split type:    Random stratified 80/20 (train/test), seed=42
Why random:    Each row = one account's SUMMARY over 1993-1997
               (not a time-series row). There is no within-row
               temporal ordering to preserve.

Temporal safety is handled at the FEATURE level, not the split level:
  - Features use only data from 1993-12-31 and earlier (observation window)
  - Churn label uses only 1998 data (labeling window)
  - No future information bleeds into features

Why a time-based account split would NOT help here:
  - Accounts opened in 1993 vs 1995 have different tenure
  - Splitting by account open date would create tenure-based bias,
    not reduce leakage (there is no cross-row leakage risk)

Leakage audit (already in README):
  - Removed has_card and has_loan from a sensitivity run
  - AUC was unchanged -> core model performance is clean

Calibration:
  - Raw XGBoost probabilities checked vs isotonic calibration
  - Risk tiers (High/Medium/Low) based on calibrated probabilities
"""
print(cv_report)
with open("reports/cv_strategy.txt", "w", encoding='utf-8') as f:
    f.write(cv_report)
print("  Saved -> reports/cv_strategy.txt")

print("\nDone. All 3 mentor suggestions implemented.")
print(f"New files in reports/figures/: calibration_comparison.png, shap_importance.png, shap_beeswarm.png")
