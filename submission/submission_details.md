# LMS Submission Details

## Required Submission Information

Fill in all fields below before submitting to the LMS.

---

| Field | Your Response |
|---|---|
| **Student Full Name** | *(Your full name)* |
| **Project Track** | Field-Based Scenario |
| **Scenario Code** | FIN-02 |
| **Project Title** | Digital Banking Customer Churn Prediction |
| **Repository URL** | *(Your GitHub repo URL, e.g. https://github.com/yourname/fin02-churn-prediction)* |
| **Demo / App Link** | *(Link to hosted demo if available, otherwise leave blank)* |
| **Access Instructions** | *(If private repo, describe how to grant mentor access)* |

---

## Short Project Description

*(Write 3–5 sentences describing your project. Example:)*

This project addresses the customer churn prediction problem for a digital banking platform using the Berka Dataset (PKDD'99), a real anonymized dataset from a Czech bank covering 1993–1998. Churn is defined as account inactivity (zero transactions) during the 1998 calendar year, with features engineered exclusively from the preceding observation window to prevent data leakage. Five models were trained and tracked with MLflow: a DummyClassifier baseline, Logistic Regression, Random Forest, XGBoost, and a leakage-sensitivity variant. The final model outputs a churn probability (0–1) and a Low/Medium/High risk tier, enabling the retention team to prioritize outreach. A reproducible Colab demo notebook demonstrates end-to-end inference.

---

## Checklist Before Submitting

- [ ] GitHub repository is accessible (public or mentor granted access)
- [ ] `README.md` is complete with all required sections
- [ ] `requirements.txt` is present and up to date
- [ ] `demo.ipynb` runs clean from top-to-bottom in a fresh Colab runtime
- [ ] `models/best_model.joblib` is accessible (uploaded to repo release, Google Drive, or similar)
- [ ] `data/README.md` includes download instructions for the Berka dataset
- [ ] `reports/results.md` is filled in with actual experiment results
- [ ] All figures are committed to `reports/figures/`
- [ ] Academic integrity: AI assistance acknowledged in README
- [ ] LMS submission PDF/DOCX matches this information exactly
