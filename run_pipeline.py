"""
run_pipeline.py
Full end-to-end pipeline: load -> features -> train -> evaluate.
Run this from the project root.
"""
import sys, os
sys.path.insert(0, '.')
os.environ['PYTHONIOENCODING'] = 'utf-8'

print("=" * 60)
print("FIN-02 Churn Prediction — Full Pipeline")
print("=" * 60)

# ── Step 1: Load data ────────────────────────────────────────
print("\n[1/4] Loading all 8 Berka tables...")
from src.data_loader import load_all_tables, get_table_summary
tables = load_all_tables()
get_table_summary(tables)

# ── Step 2: Build feature matrix ─────────────────────────────
print("\n[2/4] Building feature matrix...")
from src.features import build_feature_matrix
fm = build_feature_matrix(tables)
print(f"Feature matrix: {fm.shape}, churn rate: {fm['churned'].mean():.1%}")

# Save
import pandas as pd
fm.to_parquet('data/feature_matrix.parquet', index=False)
print("Saved: data/feature_matrix.parquet")

# ── Step 3: Train all models with MLflow ─────────────────────
print("\n[3/4] Training models with MLflow tracking...")
from src.train import train_all_models
results = train_all_models(fm)

# ── Step 4: Final evaluation on test set ─────────────────────
print("\n[4/4] Final evaluation on frozen test set...")
from src.evaluate import run_full_evaluation

best_model   = results['best_model']
X_test       = results['X_test']
y_test       = results['y_test']
feature_cols = results['feature_cols']

metrics = run_full_evaluation(
    best_model, X_test, y_test, feature_cols,
    model_label=results['best_model_name'].upper()
)

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"Best model     : {results['best_model_name']}")
print(f"Test ROC-AUC   : {metrics['roc_auc']}")
print(f"Test PR-AUC    : {metrics['pr_auc']}")
print(f"Test F1        : {metrics['f1']}")
print(f"Plots saved to : reports/figures/")
print(f"Model saved to : models/best_model.joblib")
print("\nTo view MLflow UI: mlflow ui  (then open http://localhost:5000)")
