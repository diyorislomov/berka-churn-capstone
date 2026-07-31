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

**Attempt 2 -- "transaction count dropped vs. 5-year average":** Failed. Transaction volume grew steadily 1993→1997 as the banking system matured, so comparing 1998 against a flat 5-year average was an unfair baseline. Still under 1% churn.

**Attempt 3 -- balance-trend, percentile-based (final):** An account's average balance in 1998 is compared to its average balance in 1997 (`balance_ratio`). Accounts in the **bottom 10th percentile** of this ratio are labeled churned. This is data-driven (threshold set by the distribution itself, not a guessed number) and captures real disengagement even when automated payments keep transaction counts alive.

- **Final churn rate: 10.0%** (406 of 4,056 eligible accounts)
- **Threshold:** balance_ratio ≤ 0.73 (1998 balance dropped below 73% of 1997 level)

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
berka-churn-capstone/
├── README.md
├── requirements.txt
├── app.py # Flask REST API
├── .gitignore
├── data/
│ ├── README.md
│ ├── churn_labels.csv # account_id, churned + intermediate features
│ └── features.csv # final merged feature table (4057 × 31)
├── notebooks/
│ ├── 01_data_audit_eda.ipynb # load 8 tables, parse dates, build churn label
│ ├── 02_feature_engineering.ipynb # merge into final feature table
│ └── 03_modeling.ipynb # baseline, RF, XGBoost, leakage check, eval
├── src/
│ ├── README.md
│ └── predict.py # inference function used by app.py
├── models/
│ ├── README.md
│ └── best_model.joblib # trained XGBoost model
└── reports/
├── feature_importance.csv
└── test_predictions_with_risk_tiers.csv
## Setup & Installation

```bash
git clone https://github.com/diyorislomov/berka-churn-capstone.git
cd berka-churn-capstone
pip install -r requirements.txt
```

Download the 8 Berka CSVs from Kaggle (link above) and place them in `data/`.

## How to Run

Run notebooks in order:
1. `01_data_audit_eda.ipynb` -- loads tables, parses dates, builds churn label → `data/churn_labels.csv`
2. `02_feature_engineering.ipynb` -- merges into feature table → `data/features.csv`
3. `03_modeling.ipynb` -- trains models, logs to MLflow, saves `models/best_model.joblib`

## Running the API

```bash
python app.py
```

Then:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d @sample_request.json
```

Response:
```json
{"churn_probability": 0.1549, "risk_tier": "High"}
```

## Features (27 total)

| Group | Features |
|---|---|
| Transaction | tx_count, tx_amount_mean, tx_amount_std, tx_count_per_year |
| Balance | balance_mean, balance_min, balance_last |
| Account | tenure_days, frequency (statement type) |
| Product | has_loan, has_card, n_orders, order_amount_sum |
| District | population, urban_ratio, avg_salary, unemployment, crime rate, entrepreneurs per 1000 |

## Models & Experiments (MLflow-tracked)

| Run | Model | ROC-AUC |
|---|---|---|
| 00_dummy | Majority-class baseline | 0.500 |
| 01_logistic_regression | Logistic Regression | 0.693 |
| 02_random_forest | Random Forest (200 trees) | 0.741 |
| 03_xgboost_full | XGBoost (all features) | **0.745** |
| 04_xgboost_subset_leaksensitivity | XGBoost (has_card removed) | 0.750 |

**Final model: XGBoost (full feature set), AUC 0.745.**

## Leakage-Sensitivity Check

`has_card` (whether the account holder has a credit card) was flagged in the project brief as a possible timing-leakage risk, since card issue dates could theoretically fall inside the labeling window. Removing it and retraining produced **no meaningful change** in AUC (0.745 → 0.750, within noise) -- this indicates `has_card` was not contributing leaked signal, and the full-feature model is trustworthy.

## Evaluation

At the default 0.5 probability threshold, recall on the churned class was poor (0.04) -- expected on a 10% base-rate imbalanced problem, since the model defaults toward the majority class. Rather than use a single binary cutoff, risk tiers were built from the precision/recall tradeoff across thresholds:

| Threshold | Recall (churned) | Precision (churned) |
|---|---|---|
| 0.10 | 0.54 | 0.20 |
| 0.15 | 0.35 | 0.19 |
| 0.20 | lower | higher |

## Risk Tiers

| Tier | Threshold | Test set count |
|---|---|---|
| High | prob ≥ 0.15 | 149 |
| Medium | 0.10 ≤ prob < 0.15 | 69 |
| Low | prob < 0.10 | 594 |

**Recommended action:** High and Medium tiers (218 of 812 test accounts, ~27%) are flagged for retention outreach -- a workable, actionable segment size rather than flagging the entire customer base.

## Top Feature Importances

1. tx_amount_std -- transaction amount volatility
2. balance_last -- most recent balance
3. n_orders -- standing order count
4. tenure_days -- account age
5. balance_mean -- average balance
6. frequency_POPLATEK PO OBRATU -- statement frequency type
7. balance_min
8. tx_amount_mean
9. crimes_96 -- district crime rate
10. n_entrepreneurs_per1000 -- district economic indicator

## Limitations & Responsible AI

**Known Limitations**
- **Self-defined churn label:** No ground-truth churn label exists in Berka; "churn" here is an engineered proxy (bottom-decile balance-trend), not a company-verified outcome.
- **Dataset scope:** Berka is 1990s Czech retail banking -- no digital engagement signals (no app usage, no online banking data) exist in this era of banking.
- **Sample size:** ~4,056 eligible accounts is modest; district-level splits should be treated cautiously.
- **Geographic bias:** District-level features could act as a proxy for factors correlated with, but not caused by, individual customer behavior.
- **Temporal scope:** Trained on 1993-1997 to predict 1998; not validated on modern digital-banking behavior or Central Asian markets.

**Prohibited Uses**
- Not for automatic account closure or service restriction decisions.
- Not to be used as the sole basis for individual customer treatment without human review.
- Not validated for markets or time periods outside 1990s Czech retail banking.

**Fairness Considerations**
District-level aggregate features introduce potential geographic bias; some districts have very few represented clients, so group-level churn-rate claims should not be over-interpreted.

## Academic Integrity Statement

This project was completed individually by Diyorbek Islomov, with AI assistance (Claude) used for debugging, explaining pandas/sklearn behavior, and reviewing code -- not for generating the churn-label logic, feature engineering decisions, or model selection, which were iteratively developed and understood by the author. All reported metrics come from running the pipeline on the Berka dataset as described above.

## License

Dataset: Berka Dataset, released for PKDD'99 academic research (public domain).

