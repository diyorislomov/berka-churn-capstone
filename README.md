# FIN-02 -- Digital Banking Customer Churn Prediction

**Course:** AI/ML Fundamentals -- Module 8 Capstone
**Track:** Field-Based Scenario (FIN-02)
**Dataset:** Berka Dataset (PKDD'99 Financial Discovery Challenge)
**Author:** Diyorbek Islomov

## Project Overview

A Czech bank wants to identify customers at elevated risk of churning. This project builds a supervised binary classification model on the Berka dataset (1993-1998) to predict churn risk and prioritize retention outreach.

## Problem Definition

| Item | Definition |
|---|---|
| ML Task | Supervised binary classification |
| Unit of prediction | One bank account |
| Churn label | Self-defined -- see Methodology below |
| Observation window | 1993-01-01 - 1997-12-31 (features built here) |
| Labeling window | 1998-01-01 - 1998-12-31 (label determined here) |
| Primary metric | ROC-AUC (threshold-independent, robust to class imbalance) |
| Supporting metric | Precision/Recall by risk tier |

## Churn Label Methodology -- Why It Took Three Attempts

This is the most important design decision in the project, and it did not work on the first try.

**Attempt 1 -- "any transaction in 1998":** Failed. 99.8% of accounts (4,492/4,500) had at least one transaction in 1998, because Berka accounts carry automated standing orders that post monthly regardless of customer engagement. Churn rate came back 0.2% -- unusable.

**Attempt 2 -- "transaction count dropped vs. 5-year average":** Failed. Transaction volume grew steadily 1993-1997 as the banking system matured, so comparing 1998 against a flat 5-year average was an unfair baseline. Still under 1% churn.

**Attempt 3 -- balance-trend, percentile-based (final):** An account's average balance in 1998 is compared to its average balance in 1997 (`balance_ratio`). Accounts in the **bottom 10th percentile** of this ratio are labeled churned. This is data-driven (threshold set by the distribution itself, not a guessed number) and captures real disengagement even when automated payments keep transaction counts alive.

- **Final churn rate: 10.0%** (406 of 4,057 eligible accounts)
- **Threshold:** balance_ratio <= 0.73 (1998 balance dropped below 73% of 1997 level)

## Dataset

Berka Dataset (PKDD'99) -- real anonymized data from a Czech bank, 1993-1998.

| Table | Description |
|---|---|
| account.csv | 4,500 accounts, open date, branch district |
| trans.csv | ~1,056,320 transactions, amount, balance, type |
| loan.csv | 682 loans |
| card.csv | 892 credit cards |
| client.csv | 5,369 clients, birth number (encodes DOB + gender) |
| disp.csv | Client-account dispositions |
| order.csv | 6,471 standing orders |
| district.csv | 77 districts, demographics |

Download: [kaggle.com/datasets/marceloventura/the-berka-dataset](https://www.kaggle.com/datasets/marceloventura/the-berka-dataset)
License: released for PKDD'99 academic research.

## Repository Structure

```
berka-churn-capstone/
├── README.md
├── requirements.txt
├── app.py                  # Flask REST API
├── rebuild_pipeline.py     # canonical training pipeline (use this to reproduce results)
├── .gitignore
├── data/
│   ├── churn_labels.csv    # account_id, churned + intermediate features
│   └── features.csv        # final merged feature table (4057 x 28)
├── notebooks/              # exploratory analysis (reference only)
│   ├── 01_data_audit_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   └── predict.py          # inference function used by app.py
├── models/
│   └── best_model.joblib   # trained XGBoost model (AUC 0.7793)
└── reports/
    ├── figures/            # ROC, PR, calibration, SHAP, confusion matrix plots
    ├── feature_importance.csv
    ├── shap_importance.csv
    └── cv_strategy.txt     # cross-validation methodology notes
```

## Setup & Installation

```bash
git clone https://github.com/diyorislomov/berka-churn-capstone.git
cd berka-churn-capstone
pip install -r requirements.txt
```

Download the 8 Berka CSVs from Kaggle (link above) and place them in `data/`.

## How to Reproduce Results

The canonical pipeline is `rebuild_pipeline.py` -- run it to regenerate `features.csv`, retrain the model, and reproduce all metrics:

```bash
python rebuild_pipeline.py
```

The notebooks (`01`, `02`, `03`) document the exploratory process and the churn-label iteration story. They are reference documents, not the production pipeline.

## Running the API

```bash
python app.py
```

Then:
```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d @sample_request.json
```

Response:
```json
{"churn_probability": 0.53, "risk_tier": "High"}
```

## Features (28 total)

| Group | Features |
|---|---|
| Transaction | tx_count, tx_amount_mean, tx_amount_std, tx_count_per_year* |
| Balance | balance_mean, balance_min, balance_last |
| Account | tenure_days, frequency (3 one-hot dummies) |
| Product | has_loan, n_orders, order_amount_sum |
| District | n_inhabitants, n_muni_lt499, n_muni_500_1999, n_muni_2000_9999, n_muni_gt10000, n_cities, urban_ratio, avg_salary, unemp_95, unemp_96, n_entrepreneurs_per1000, crimes_95, crimes_96 |

*`tx_count_per_year` normalized by actual account tenure, not a fixed 5-year denominator.

Note: `has_card` was removed from the final model after the leakage sensitivity check (see below).

## Models & Experiments (MLflow-tracked)

### Exploratory runs (notebooks)

| Run | Model | ROC-AUC |
|---|---|---|
| 00_dummy | Majority-class baseline | 0.500 |
| 01_logistic_regression | Logistic Regression | 0.693 |
| 02_random_forest | Random Forest (200 trees) | 0.741 |
| 03_xgboost_full | XGBoost (all features) | 0.745 |
| 04_xgboost_subset_leaksensitivity | XGBoost (has_card removed) | 0.750 |

### Rebuilt pipeline (rebuild_pipeline.py) -- final results

| Run | Model | ROC-AUC | Notes |
|---|---|---|---|
| dummy_baseline | Majority-class baseline | 0.500 | |
| logistic_regression | Logistic Regression | 0.697 | |
| random_forest | Random Forest (300 trees) | 0.749 | class_weight=balanced |
| xgboost_fixed | XGBoost (with has_card) | 0.764 | scale_pos_weight=9, early stopping |
| **xgboost_no_card** | **XGBoost (no has_card)** | **0.779** | **FINAL MODEL** |

**Final model: XGBoost without has_card, AUC 0.779, PR-AUC 0.262.**

## Leakage-Sensitivity Check

`has_card` was flagged as a potential leakage risk. The exploratory notebooks used a broken join (`account_id` matched against `disp_id` -- different ID spaces), which produced ~90% zeros and made the feature look like noise. After fixing the join (`card.disp_id -> disp.disp_id -> account_id`), 807 accounts (19.9%) are correctly identified as cardholders.

With the **correct join**, removing `has_card` **improved AUC by +0.016** (0.764 -> 0.779). This suggests the feature carries timing ambiguity that hurts discrimination -- so it was dropped from the final model. This validates the leakage concern from a different angle.

## Cross-Validation Strategy

**Split type:** Random stratified 80/20 (train/test), seed=42 + 20% validation split from training set for early stopping.

**Why random split is correct here:** Each row is one account's summary over 1993-1997, not a time-series row. Temporal safety is enforced at the **feature level** -- features use only data up to 1997-12-31, the churn label uses only 1998 data. No future information bleeds into features.

See `reports/cv_strategy.txt` for detailed rationale.

## Probability Calibration

Raw XGBoost probabilities are not literal percentages. Isotonic calibration was applied:
- Brier score improved: 0.171 -> 0.116 (lower = better calibrated)
- AUC is reported on raw probabilities (better discrimination); Brier score reported on calibrated probabilities (better accuracy)

## Evaluation

At the default 0.5 threshold, recall on the churned class is low -- expected on a 10% imbalanced problem. Risk tiers are built from threshold analysis validated against actual churn outcomes:

| Threshold | Accounts flagged | Actual churn rate in flagged group |
|---|---|---|
| prob >= 0.20 | ~63 | ~28% (~2.9x base rate) |
| prob >= 0.10 | ~96 | ~22% (~2.2x base rate) |

## Risk Tiers

| Tier | Threshold | Meaning |
|---|---|---|
| High | prob >= 0.20 | Priority retention outreach |
| Medium | 0.10 <= prob < 0.20 | Monitor, soft outreach |
| Low | prob < 0.10 | Routine monitoring |

**Recommended action:** High and Medium tiers cover ~12% of accounts but capture 60%+ of actual churners -- a workable, actionable segment size.

## SHAP Feature Importance (final model)

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | balance_mean | 0.759 |
| 2 | balance_last | 0.703 |
| 3 | tenure_days | 0.519 |
| 4 | tx_amount_mean | 0.388 |
| 5 | tx_amount_std | 0.345 |
| 6 | tx_count | 0.190 |
| 7 | order_amount_sum | 0.177 |
| 8 | balance_min | 0.164 |
| 9 | n_orders | 0.127 |
| 10 | unemp_95 | 0.096 |

Full SHAP outputs: `reports/shap_importance.csv`, `reports/figures/shap_beeswarm.png`

## Limitations & Responsible AI

**Known Limitations**
- **Self-defined churn label:** No ground-truth churn label exists in Berka; "churn" here is an engineered proxy (bottom-decile balance-trend), not a company-verified outcome.
- **Dataset scope:** Berka is 1990s Czech retail banking -- no digital engagement signals (no app usage, no online banking data) exist in this era of banking.
- **Sample size:** ~4,057 eligible accounts is modest; district-level splits should be treated cautiously.
- **Geographic bias:** District-level features could act as a proxy for factors correlated with, but not caused by, individual customer behavior.
- **Temporal scope:** Trained on 1993-1997 to predict 1998; not validated on modern digital-banking behavior.

**Prohibited Uses**
- Not for automatic account closure or service restriction decisions.
- Not to be used as the sole basis for individual customer treatment without human review.
- Not validated for markets or time periods outside 1990s Czech retail banking.

**Fairness Considerations**
District-level aggregate features introduce potential geographic bias; some districts have very few represented clients, so group-level churn-rate claims should not be over-interpreted.

## Academic Integrity Statement

This project was completed individually by Diyorbek Islomov, with AI assistance used for debugging, reviewing code, and explaining library behavior -- not for generating the churn-label logic, feature engineering decisions, or model selection, which were iteratively developed and understood by the author. All reported metrics come from running the pipeline on the Berka dataset as described above.

## License

Dataset: Berka Dataset, released for PKDD'99 academic research (public domain).
