# Submission Details

## Project Identification

| Field | Your Response |
|---|---|
| **Student Full Name** | Diyorbek Islomov |
| **Project Track** | Field-Based Scenario |
| **Scenario Code** | FIN-02 |
| **Project Title** | Digital Banking Customer Churn Prediction |
| **Repository URL** | https://github.com/diyorislomov/berka-churn-capstone |
| **Demo / App Link** | *(see deployment instructions in README)* |
| **Access Instructions** | Repository is public - no access instructions needed |

---

## Self-Assessment

| Criterion | Rating (1-5) | Notes |
|---|---|---|
| Problem framing | 5 | Binary classification, observation/labeling windows defined |
| Data preparation | 5 | All 8 Berka tables joined, zero nulls |
| Churn label design | 5 | 3 documented iterations, final: balance-ratio bottom-10th-pct |
| Feature engineering | 4 | 28 features, leakage-sensitivity check passed |
| Model training | 4 | 4 MLflow runs: Dummy, LR, RF, XGBoost |
| Evaluation | 4 | ROC-AUC, PR-AUC, calibration, confusion matrix, risk tiers |
| Responsible AI | 4 | Limitations section in README |
| REST API | 4 | Flask /predict and /health endpoints working |
| Documentation | 5 | README covers all capstone requirements |

---

## Key Results

| Metric | Value |
|---|---|
| Best model | XGBoost |
| Test ROC-AUC | 0.745 |
| Churn rate | 10.0% (406 / 4,056 accounts) |
| Churn definition | balance_ratio <= 0.73 (bottom 10th percentile) |
| Features | 28 engineered features |
| Leakage check | PASSED (has_card removed, AUC unchanged) |

---

## Academic Integrity Statement

This project was completed individually by Diyorbek Islomov.
AI coding assistants were used for scaffolding, code review, and debugging.
All code has been reviewed, understood, and can be explained and defended by the student.
Experiment results and analysis are genuine outputs from running the code on the Berka dataset.
