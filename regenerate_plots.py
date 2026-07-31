"""
regenerate_plots.py
Regenerate all evaluation plots for the current best_model.joblib.
Run once after training to update reports/figures/.
"""
import sys, os
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
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve

# ---- Paths ----------------------------------------------------------------
BASE      = Path(__file__).parent
FIG_DIR   = BASE / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH  = BASE / "models" / "best_model.joblib"
FEAT_PATH   = BASE / "data" / "features.csv"
LABEL_PATH  = BASE / "data" / "churn_labels.csv"

RANDOM_SEED = 42
TEST_SIZE   = 0.20

# ---- Load -----------------------------------------------------------------
print("Loading model and data...")
model = joblib.load(MODEL_PATH)
FEAT_COLS = model.get_booster().feature_names

# features.csv already contains the churned column
df = pd.read_csv(FEAT_PATH)

# One-hot encode 'frequency' to match model training
# (model was trained with pd.get_dummies which drops the most frequent category)
if 'frequency' in df.columns:
    freq_dummies = pd.get_dummies(df['frequency'], prefix='frequency')
    df = pd.concat([df.drop(columns=['frequency']), freq_dummies], axis=1)

# Select model features (model's FEAT_COLS must all be present now)
missing_cols = [c for c in FEAT_COLS if c not in df.columns]
if missing_cols:
    print(f"Adding missing columns as 0: {missing_cols}")
    for c in missing_cols:
        df[c] = 0

# Coerce all feature columns to numeric (handles '?' strings in district data)
X = df[FEAT_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)
y = df['churned'].astype(int)

print(f"Dataset: {len(df)} accounts, churn rate: {y.mean():.1%}")

# ---- Split (reproduce test set) -------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)
print(f"Test set: {len(X_test)} accounts, churn rate: {y_test.mean():.1%}")

# ---- Predictions ----------------------------------------------------------
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)
f1      = f1_score(y_test, y_pred, zero_division=0)

print(f"\nTest ROC-AUC : {roc_auc:.4f}")
print(f"Test PR-AUC  : {pr_auc:.4f}")
print(f"Test F1      : {f1:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Active','Churned'])}")

# ---- Plot 1: ROC Curve ----------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color='#4F81BD', lw=2, label=f'XGBoost (AUC = {roc_auc:.3f})')
ax.plot([0,1],[0,1],'--', color='grey', lw=1, label='Random baseline')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — FIN-02 Churn Prediction')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / 'roc_curve.png', dpi=150)
plt.close(fig)
print("Saved -> roc_curve.png")

# ---- Plot 2: Precision-Recall Curve ---------------------------------------
prec, rec, _ = precision_recall_curve(y_test, y_prob)
baseline = y_test.mean()
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(rec, prec, color='#C0504D', lw=2, label=f'XGBoost (PR-AUC = {pr_auc:.3f})')
ax.axhline(baseline, ls='--', color='grey', lw=1, label=f'Random baseline ({baseline:.2f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve — FIN-02 Churn Prediction')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / 'pr_curve.png', dpi=150)
plt.close(fig)
print("Saved -> pr_curve.png")

# ---- Plot 3: Calibration Curve --------------------------------------------
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='quantile')
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(prob_pred, prob_true, 's-', color='#9BBB59', lw=2, label='XGBoost')
ax.plot([0,1],[0,1],'--', color='grey', lw=1, label='Perfect calibration')
ax.set_xlabel('Mean Predicted Probability'); ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curve — FIN-02 Churn Prediction')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / 'calibration.png', dpi=150)
plt.close(fig)
print("Saved -> calibration.png")

# ---- Plot 4: Confusion Matrix ---------------------------------------------
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Active','Churned']); ax.set_yticklabels(['Active','Churned'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix — Test Set')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=14)
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(FIG_DIR / 'confusion_matrix.png', dpi=150)
plt.close(fig)
print("Saved -> confusion_matrix.png")

# ---- Plot 5: Feature Importance -------------------------------------------
importance = model.feature_importances_
feat_df = pd.DataFrame({'feature': FEAT_COLS, 'importance': importance})
feat_df = feat_df.sort_values('importance', ascending=True).tail(15)
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(feat_df['feature'], feat_df['importance'], color='#4F81BD')
ax.set_xlabel('Feature Importance (gain)')
ax.set_title('Top 15 Feature Importances — XGBoost')
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / 'feature_importance.png', dpi=150)
plt.close(fig)
print("Saved -> feature_importance.png")

print(f"\nAll 5 plots saved to: {FIG_DIR}")
print(f"ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}  F1={f1:.4f}")
