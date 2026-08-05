"""
rebuild_pipeline.py
Fixes all bugs identified in code review, rebuilds features.csv, retrains model.

Fixes applied:
  Bug 1 - has_card join: now correctly routes card.disp_id -> disp.disp_id -> account_id
  Bug 2 - tx_count_per_year: normalized by actual tenure, not fixed 5 years
  Bug 3 - XGBoost training: adds eval_set + early stopping (20 rounds)
  Bug 4 - Model save: one explicit save cell, saves the final leakage-checked model
"""
import os, sys
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import joblib
import mlflow
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

RANDOM_SEED = 42
DATA_DIR    = Path("data")
FIG_DIR     = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ALL TABLES
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading raw tables")
print("=" * 60)

account  = pd.read_csv(DATA_DIR / "account.csv",  sep=';')
disp     = pd.read_csv(DATA_DIR / "disp.csv",     sep=';')
card     = pd.read_csv(DATA_DIR / "card.csv",     sep=';')
loan     = pd.read_csv(DATA_DIR / "loan.csv",     sep=';')
order    = pd.read_csv(DATA_DIR / "order.csv",    sep=';')
trans    = pd.read_csv(DATA_DIR / "trans.csv",    sep=';', low_memory=False)
district = pd.read_csv(DATA_DIR / "district.csv", sep=';', header=None)

# Parse dates
account['date_parsed'] = pd.to_datetime(account['date'], format='%y%m%d')
trans['date_parsed']   = pd.to_datetime(trans['date'],   format='%y%m%d')

# District column names
district.columns = [
    'district_id','district_name','region','n_inhabitants',
    'n_muni_lt499','n_muni_500_1999','n_muni_2000_9999','n_muni_gt10000',
    'n_cities','urban_ratio','avg_salary','unemp_95','unemp_96',
    'n_entrepreneurs_per1000','crimes_95','crimes_96'
]
district = district[pd.to_numeric(district['district_id'], errors='coerce').notna()]
for col in ['unemp_95','unemp_96','crimes_95','crimes_96','avg_salary','urban_ratio']:
    district[col] = pd.to_numeric(district[col], errors='coerce')
district['district_id'] = district['district_id'].astype(int)

print(f"  account:  {account.shape}")
print(f"  trans:    {trans.shape}")
print(f"  disp:     {disp.shape}")
print(f"  card:     {card.shape}")
print(f"  loan:     {loan.shape}")
print(f"  order:    {order.shape}")
print(f"  district: {district.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. CHURN LABEL (balance-ratio bottom 10th percentile)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2: Building churn labels")
print("=" * 60)

OBS_END   = pd.Timestamp('1997-12-31')
LABEL_END = pd.Timestamp('1998-12-31')

trans_obs   = trans[trans['date_parsed'] <= OBS_END]
trans_label = trans[(trans['date_parsed'] > OBS_END) & (trans['date_parsed'] <= LABEL_END)]

# Eligible: account open >= 180 days before obs end
eligible = account[account['date_parsed'] <= OBS_END - pd.Timedelta(days=180)]['account_id']

churn_df = pd.DataFrame({'account_id': eligible}).set_index('account_id')

# 1997 balance stats for ratio
bal_1997  = trans_obs[trans_obs['date_parsed'].dt.year == 1997].groupby('account_id')['balance'].mean()
bal_label = trans_label.groupby('account_id')['balance'].mean()

churn_df['avg_balance_obs']   = bal_1997.reindex(churn_df.index)
churn_df['avg_balance_label'] = bal_label.reindex(churn_df.index)
churn_df['balance_ratio']     = churn_df['avg_balance_label'] / churn_df['avg_balance_obs']

# Only accounts with positive 1997 balance (ratio is meaningful)
valid = churn_df[churn_df['avg_balance_obs'] > 0].copy()
threshold = valid['balance_ratio'].quantile(0.10)
valid['churned'] = (valid['balance_ratio'] <= threshold).astype(int)

print(f"  Eligible accounts:  {len(churn_df)}")
print(f"  Valid (pos 1997 bal): {len(valid)}")
print(f"  Balance-ratio threshold (10th pct): {threshold:.4f}")
print(f"  Churn rate: {valid['churned'].mean():.1%}  ({valid['churned'].sum()} churned)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING (all bugs fixed)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3: Feature engineering (bugs fixed)")
print("=" * 60)

account_ids = valid.index.tolist()

# --- Account metadata ---
acct = account[account['account_id'].isin(account_ids)].copy()
acct['tenure_days'] = (OBS_END - acct['date_parsed']).dt.days

# --- Transaction features (obs window only) ---
tx = trans_obs[trans_obs['account_id'].isin(account_ids)].copy()

tx_feat = tx.groupby('account_id').agg(
    tx_count     = ('trans_id', 'count'),
    tx_amount_mean = ('amount', 'mean'),
    tx_amount_std  = ('amount', 'std'),
    balance_mean   = ('balance', 'mean'),
    balance_min    = ('balance', 'min'),
    balance_last   = ('balance', 'last'),
).reset_index()

# FIX: Normalize tx_count by actual tenure, not fixed 5 years
acct_tenure = acct[['account_id','tenure_days']].set_index('account_id')
tx_feat = tx_feat.set_index('account_id')
tx_feat['tenure_days'] = acct_tenure['tenure_days']
tx_feat['tx_count_per_year'] = tx_feat['tx_count'] / (tx_feat['tenure_days'] / 365.25)
tx_feat = tx_feat.reset_index()
print(f"  tx_count_per_year: normalized by actual tenure (not fixed /5)")

# --- Loan feature ---
loan_accts = loan[loan['account_id'].isin(account_ids)]['account_id'].unique()

# --- Card feature (BUG FIX: correct join via disp_id) ---
# WRONG (old): disp[disp['account_id'].isin(card['disp_id'])]
# CORRECT: card.disp_id -> disp.disp_id -> disp.account_id
card_feat = (
    disp.merge(card[['disp_id']], on='disp_id', how='inner')[['account_id']]
    .drop_duplicates()
)
card_accts = card_feat[card_feat['account_id'].isin(account_ids)]['account_id'].unique()
print(f"  has_card: {len(card_accts)} accounts ({len(card_accts)/len(account_ids):.1%}) — FIXED from {553} (old buggy count)")

# --- Order features ---
ord_feat = order[order['account_id'].isin(account_ids)].groupby('account_id').agg(
    n_orders       = ('order_id', 'count'),
    order_amount_sum = ('amount', 'sum'),
).reset_index()

# --- District features ---
dist_feat = district[[
    'district_id','n_inhabitants','n_muni_lt499','n_muni_500_1999',
    'n_muni_2000_9999','n_muni_gt10000','n_cities','urban_ratio',
    'avg_salary','unemp_95','unemp_96','n_entrepreneurs_per1000',
    'crimes_95','crimes_96'
]]

# --- Assemble master feature table ---
feat = pd.DataFrame({'account_id': account_ids})

feat = feat.merge(acct[['account_id','district_id','frequency','tenure_days']], on='account_id', how='left')
feat = feat.merge(tx_feat[['account_id','tx_count','tx_amount_mean','tx_amount_std',
                             'balance_mean','balance_min','balance_last','tx_count_per_year']], on='account_id', how='left')
feat['has_loan'] = feat['account_id'].isin(loan_accts).astype(int)
feat['has_card'] = feat['account_id'].isin(card_accts).astype(int)
feat = feat.merge(ord_feat, on='account_id', how='left')
feat = feat.merge(dist_feat, on='district_id', how='left')

# Merge churn label
feat = feat.merge(valid[['churned']].reset_index(), on='account_id', how='inner')

# Fill nulls
feat['n_orders']        = feat['n_orders'].fillna(0)
feat['order_amount_sum'] = feat['order_amount_sum'].fillna(0)
feat['tx_amount_std']   = feat['tx_amount_std'].fillna(0)

feat.to_csv(DATA_DIR / "features.csv", index=False)
print(f"  Features saved: {feat.shape[0]} accounts x {feat.shape[1]} columns")
print(f"  Churn rate: {feat['churned'].mean():.1%}")
print(f"  Null check: {feat.isnull().sum().sum()} total nulls")

# ══════════════════════════════════════════════════════════════════════════════
# 4. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4: Train/test split (80/20, stratified, seed=42)")
print("=" * 60)

# One-hot encode frequency
freq_dummies = pd.get_dummies(feat['frequency'], prefix='frequency')
feat_enc = pd.concat([feat.drop(columns=['frequency','district_name','region'], errors='ignore'), freq_dummies], axis=1)

FEAT_COLS = [c for c in feat_enc.columns if c not in ['account_id','churned']]
X = feat_enc[FEAT_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)
y = feat_enc['churned'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)
# Validation set for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=RANDOM_SEED, stratify=y_train
)
print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
print(f"  Churn rate - train: {y_train.mean():.1%}, test: {y_test.mean():.1%}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAIN MODELS with MLflow
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5: Training models")
print("=" * 60)

mlflow.set_experiment("fin02-churn-rebuild")
results = {}

# Dummy
with mlflow.start_run(run_name="dummy_baseline"):
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train, y_train)
    auc = roc_auc_score(y_test, dummy.predict_proba(X_test)[:,1])
    mlflow.log_metric("test_roc_auc", auc)
    results['Dummy'] = auc
    print(f"  Dummy         AUC: {auc:.4f}")

# Logistic Regression
with mlflow.start_run(run_name="logistic_regression"):
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight='balanced')
    lr.fit(X_train, y_train)
    auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:,1])
    mlflow.log_metric("test_roc_auc", auc)
    results['LogReg'] = auc
    print(f"  LogReg        AUC: {auc:.4f}")

# Random Forest
with mlflow.start_run(run_name="random_forest"):
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_SEED,
                                 class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])
    mlflow.log_metric("test_roc_auc", auc)
    results['RF'] = auc
    print(f"  RandomForest  AUC: {auc:.4f}")

# XGBoost with eval_set + early stopping (BUG FIX)
with mlflow.start_run(run_name="xgboost_fixed"):
    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        random_state=RANDOM_SEED,
        eval_metric='auc',
        use_label_encoder=False,
        verbosity=0,
        early_stopping_rounds=20,
    )
    xgb.fit(X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False)
    auc = roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1])
    mlflow.log_metric("test_roc_auc", auc)
    mlflow.log_param("best_iteration", xgb.best_iteration)
    mlflow.log_param("scale_pos_weight", scale_pos)
    results['XGBoost'] = auc
    print(f"  XGBoost       AUC: {auc:.4f}  (best iter: {xgb.best_iteration})")

# XGBoost without has_card (leakage sensitivity check)
with mlflow.start_run(run_name="xgboost_no_card"):
    FEAT_NO_CARD = [c for c in FEAT_COLS if c != 'has_card']
    xgb_nc = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, random_state=RANDOM_SEED,
        eval_metric='auc', use_label_encoder=False, verbosity=0,
        early_stopping_rounds=20,
    )
    xgb_nc.fit(X_tr[FEAT_NO_CARD], y_tr,
               eval_set=[(X_val[FEAT_NO_CARD], y_val)],
               verbose=False)
    auc_nc = roc_auc_score(y_test, xgb_nc.predict_proba(X_test[FEAT_NO_CARD])[:,1])
    mlflow.log_metric("test_roc_auc", auc_nc)
    results['XGBoost_no_card'] = auc_nc
    print(f"  XGBoost_no_card AUC: {auc_nc:.4f}  (leakage check)")
    delta = auc_nc - results['XGBoost']
    print(f"  Delta (no_card vs full): {delta:+.4f}")
    if abs(delta) < 0.01:
        print("  LEAKAGE CHECK PASSED: has_card does not drive model performance")
    else:
        print(f"  NOTE: has_card contributes {abs(delta):.4f} AUC — now with correct join")

# ══════════════════════════════════════════════════════════════════════════════
# 6. SAVE BEST MODEL (explicit single save)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6: Saving best model")
print("=" * 60)

best_model = xgb  # XGBoost with fixed has_card, scale_pos_weight, early stopping
joblib.dump(best_model, "models/best_model.joblib")
print(f"  Saved models/best_model.joblib")
print(f"  Best model: XGBoost, AUC={results['XGBoost']:.4f}")
print(f"  Features: {len(FEAT_COLS)}")

# Save feature importance
fi = pd.DataFrame({'feature': FEAT_COLS, 'importance': xgb.feature_importances_})
fi = fi.sort_values('importance', ascending=False)
fi.to_csv("reports/feature_importance.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# 7. REGENERATE ALL PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7: Regenerating evaluation plots")
print("=" * 60)

from sklearn.metrics import (roc_curve, precision_recall_curve,
                              confusion_matrix, average_precision_score,
                              f1_score, brier_score_loss)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

y_prob = xgb.predict_proba(X_test)[:,1]
y_pred = (y_prob >= 0.5).astype(int)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)
f1      = f1_score(y_test, y_pred, zero_division=0)

print(f"  ROC-AUC: {roc_auc:.4f}")
print(f"  PR-AUC:  {pr_auc:.4f}")
print(f"  F1:      {f1:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Active','Churned'])}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color='#4F81BD', lw=2, label=f'XGBoost (AUC={roc_auc:.3f})')
ax.plot([0,1],[0,1],'--', color='grey', lw=1, label='Random baseline')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve -- FIN-02 Churn Prediction (Rebuilt)')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG_DIR/'roc_curve.png', dpi=150); plt.close(fig)

# PR Curve
prec, rec, _ = precision_recall_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(rec, prec, color='#C0504D', lw=2, label=f'XGBoost (PR-AUC={pr_auc:.3f})')
ax.axhline(y_test.mean(), ls='--', color='grey', lw=1, label=f'Baseline ({y_test.mean():.2f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve -- FIN-02 Churn Prediction (Rebuilt)')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG_DIR/'pr_curve.png', dpi=150); plt.close(fig)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Active','Churned']); ax.set_yticklabels(['Active','Churned'])
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix -- Test Set (Rebuilt)')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=14)
fig.colorbar(im); fig.tight_layout()
fig.savefig(FIG_DIR/'confusion_matrix.png', dpi=150); plt.close(fig)

# Calibration
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='quantile')
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(prob_pred, prob_true, 's-', color='#9BBB59', lw=2, label='XGBoost (rebuilt)')
ax.plot([0,1],[0,1],'--', color='grey', lw=1, label='Perfect calibration')
ax.set_xlabel('Mean Predicted Probability'); ax.set_ylabel('Fraction of Positives')
ax.set_title('Calibration Curve -- FIN-02 Churn Prediction (Rebuilt)')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(FIG_DIR/'calibration.png', dpi=150); plt.close(fig)

# Feature Importance
top_fi = fi.head(15).sort_values('importance', ascending=True)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_fi['feature'], top_fi['importance'], color='#4F81BD')
ax.set_xlabel('Feature Importance (gain)')
ax.set_title('Top 15 Feature Importances -- XGBoost (Rebuilt)')
ax.grid(axis='x', alpha=0.3)
fig.tight_layout(); fig.savefig(FIG_DIR/'feature_importance.png', dpi=150); plt.close(fig)

print("  All 5 plots saved to reports/figures/")
print("\n" + "=" * 60)
print("REBUILD COMPLETE")
print("=" * 60)
print(f"  Model AUC before fix: 0.7439  (buggy has_card, no early stopping)")
print(f"  Model AUC after fix:  {roc_auc:.4f}  (correct has_card, early stopping)")
print(f"  Delta: {roc_auc - 0.7439:+.4f}")
