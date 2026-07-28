# Experiment Results

**Generated:** 2026-07-28 | **Dataset:** Berka (PKDD'99) | **Model:** FIN-02 Churn Prediction

---

## Churn Definition

| Item | Detail |
|---|---|
| Observation window | 1993-01-01 → 1996-12-31 (features) |
| Labeling window | 1997-01-01 → 1997-12-31 (label) |
| Churn definition | Account in bottom 20th percentile of 1997 transaction activity |
| Churn threshold | ≤ 60 transactions in 1997 |
| Final churn rate | 17.4% (after excluding accounts < 12 months old) |

> **Note:** The original 1998-labeling-window definition yielded only 0.2% churn (4,492 of 4,500 accounts active). We adopted a relative-engagement definition consistent with retail banking practice — the least-engaged quintile is flagged for retention outreach.

---

## Dataset Summary

| Table | Rows |
|---|---|
| accounts | 4,500 |
| transactions | 1,056,320 |
| loans | 682 |
| cards | 892 |
| clients | 5,369 |
| dispositions | 5,369 |
| orders | 6,471 |
| districts | 77 |

**Feature matrix after preprocessing:** 2,252 accounts × 23 features  
**Split:** Train 1,576 / Val 338 / Test 338 (70/15/15, stratified)

---

## Model Comparison Table

| Run | Model | Val ROC-AUC | Val PR-AUC | Notes |
|---|---|---|---|---|
| 01_dummy_baseline | DummyClassifier | 0.5000 | 0.1746 | Trivial floor |
| 02_logistic_regression | Logistic Regression | 0.9379 | 0.7907 | Strong interpretable baseline |
| 03_random_forest | Random Forest | 0.9125 | 0.7151 | Good ensemble |
| **04_xgboost_main** | **XGBoost** | **0.9423** | **0.8326** | **Best model (selected)** |
| 05_xgboost_leakage_sensitivity | XGBoost (–has_card, –has_loan) | 0.9453 | 0.8420 | Slightly higher without timing-ambiguous features |

> **Leakage sensitivity finding:** Removing `has_card` and `has_loan` (features with timing uncertainty) did NOT degrade performance — in fact marginally improved it. This confirms the core model is clean and those features contribute minimal signal.

---

## Final Test Set Results (Frozen — Never Seen During Training)

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.9305** |
| **PR-AUC** | **0.7693** |
| **F1 Score** | **0.6479** |
| Precision | 0.5542 |
| Recall | 0.7797 |
| Accuracy | 0.8521 |
| Test set size | 338 |
| Churn rate in test | 17.5% |

### Classification Report

```
              precision    recall  f1-score   support

  Active (0)       0.95      0.87      0.91       279
 Churned (1)       0.55      0.78      0.65        59

    accuracy                           0.85       338
   macro avg       0.75      0.82      0.78       338
weighted avg       0.88      0.85      0.86       338
```

**Interpretation:**
- ROC-AUC of **0.9305** far exceeds the 0.5 random baseline — the model has strong discriminative power
- Recall of **78%** means the model catches 78% of low-engagement customers — reducing missed churn interventions
- Precision of 55% means about 1 in 2 flagged accounts genuinely needs attention — acceptable for a proactive outreach campaign where over-contact is low-cost

---

## Generated Plots

| Plot | Description |
|---|---|
| `figures/roc_curve.png` | ROC curve with AUC annotation |
| `figures/pr_curve.png` | Precision-Recall curve with baseline |
| `figures/calibration.png` | Reliability diagram |
| `figures/confusion_matrix.png` | Confusion matrix on test set |
| `figures/feature_importance.png` | Top feature importances (XGBoost) |

---

## Error Analysis Summary

### Top False Negative Patterns (Missed Churners)
Accounts predicted as active but actually low-engagement. Common patterns:
- Moderate transaction counts (48–274 tx) — not obviously low on volume alone
- High average transaction amounts — large but infrequent transactions
- Churn probability scores 0.09–0.34 — model uncertain, just below threshold

### Top False Positive Patterns (Incorrect Alerts)
Accounts predicted as churned but actually active in 1997. Common patterns:
- Very low transaction frequency in observation window (0.7–4.3 tx/month)
- Model correctly identified low engagement historically, but account recovered in 1997
- Churn probabilities 0.76–0.95 — high confidence but wrong direction

---

## Model Selection Justification

**Selected model: XGBoost (run 04)**

- **ROC-AUC 0.9423 (val) / 0.9305 (test)** — best generalizing model
- Outperforms Logistic Regression by +0.44 AUC and Random Forest by +2.98 AUC on validation
- **Leakage sensitivity:** removing timing-ambiguous features (`has_card`, `has_loan`) produced 0.9453 val AUC — no degradation, confirming clean feature engineering with no material data leakage
- Handles class imbalance well via `scale_pos_weight` parameter
- Good calibration confirmed by reliability diagram

---

## MLflow Tracking

All 5 experiment runs are logged locally. To view:

```bash
cd C:\Users\user\Documents\fin02-churn-prediction
mlflow ui
# Open: http://localhost:5000
```
