# Project Status

**Current stage:** Modeling & evaluation complete, README finalized.

**Completed:**
- Data loaded, audited, 8 Berka tables joined
- Churn label built (balance-trend, 10th percentile, 10% churn rate) — 3 iterations documented in README
- Feature engineering (27 features, zero nulls)
- Models trained & MLflow-tracked: Dummy, LogReg, RF, XGBoost (AUC 0.745)
- Leakage-sensitivity check completed (has_card removed, no meaningful AUC change)
- Flask REST API built and tested (/predict endpoint working)
- README rewritten with real results

**Next:**
- Final repo review (check all folders render correctly on GitHub)
- Defense/pitch prep

**Blockers:** None currently.