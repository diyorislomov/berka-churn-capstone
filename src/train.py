"""
train.py
MLflow-tracked training pipeline for FIN-02 Churn Prediction.

Runs five experiments:
  1. DummyClassifier          — trivial majority-class baseline
  2. Logistic Regression      — interpretable statistical baseline
  3. Random Forest            — ensemble model
  4. XGBoost                  — main gradient-boosting model
  5. XGBoost (leakage check)  — same as #4 but with timing-sensitive features removed

Best model by validation ROC-AUC is saved to models/best_model.joblib.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score
)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: xgboost not installed. XGBoost runs will be skipped.")

from src.features import FEATURE_COLS

MODELS_DIR  = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
TEST_SIZE   = 0.15   # 15% frozen test set
VAL_SIZE    = 0.15   # 15% validation

# Features that could be timing-ambiguous (re-evaluated during leakage sensitivity run)
# Based on EDA findings — update this list if needed.
LEAKAGE_SENSITIVE_COLS = ['has_card', 'has_loan']   # Card/loan dates may post-date cutoff


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _get_feature_cols(feature_matrix):
    """Return available feature columns (intersection with FEATURE_COLS)."""
    return [c for c in FEATURE_COLS if c in feature_matrix.columns]


def prepare_splits(feature_matrix):
    """
    Create stratified train / validation / test splits.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols
    """
    feature_cols = _get_feature_cols(feature_matrix)
    X = feature_matrix[feature_cols].fillna(0).astype(float)
    y = feature_matrix['churned'].astype(int)

    # Split off test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    # Split val from train
    val_ratio_adjusted = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio_adjusted, random_state=RANDOM_SEED, stratify=y_temp
    )

    print(f"\nSplit sizes  — Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")
    print(f"Churn rates  — Train: {y_train.mean():.1%}  Val: {y_val.mean():.1%}  Test: {y_test.mean():.1%}")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


def _compute_metrics(model, X, y, prefix):
    """Return a dict of metric_name → value for a given split."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)
    return {
        f"{prefix}_roc_auc":   round(roc_auc_score(y, y_prob), 4),
        f"{prefix}_pr_auc":    round(average_precision_score(y, y_prob), 4),
        f"{prefix}_f1":        round(f1_score(y, y_pred, zero_division=0), 4),
        f"{prefix}_precision": round(precision_score(y, y_pred, zero_division=0), 4),
        f"{prefix}_recall":    round(recall_score(y, y_pred, zero_division=0), 4),
    }


def _run_experiment(run_name, pipeline, X_train, y_train, X_val, y_val, params=None, tags=None):
    """
    Train pipeline, log everything to MLflow, return (fitted_pipeline, val_metrics).
    """
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        if tags:
            mlflow.set_tags(tags)

        pipeline.fit(X_train, y_train)

        train_metrics = _compute_metrics(pipeline, X_train, y_train, "train")
        val_metrics   = _compute_metrics(pipeline, X_val,   y_val,   "val")
        mlflow.log_metrics({**train_metrics, **val_metrics})

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        v_auc = val_metrics['val_roc_auc']
        v_pr  = val_metrics['val_pr_auc']
        print(f"  [{run_name:<35}] Val ROC-AUC: {v_auc:.4f}  PR-AUC: {v_pr:.4f}")

    return pipeline, val_metrics


# ─────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────

def train_all_models(feature_matrix, experiment_name="fin02-churn-prediction"):
    """
    Run all 5 MLflow experiments and save the best model.

    Parameters
    ----------
    feature_matrix : pd.DataFrame  (output of features.build_feature_matrix)
    experiment_name : str

    Returns
    -------
    dict with keys: splits, models, results, best_model_name, feature_cols
    """
    mlflow.set_experiment(experiment_name)

    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_splits(feature_matrix)
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    named_models   = {}  # run_label → fitted pipeline
    results        = {}  # run_label → val metrics dict

    print("\n" + "="*65)
    print("MLflow Experiment Runs")
    print("="*65)

    # ── Run 1: DummyClassifier ───────────────────────────────
    dummy_pipe = Pipeline([('clf', DummyClassifier(strategy='most_frequent', random_state=RANDOM_SEED))])
    named_models['dummy'], results['dummy'] = _run_experiment(
        "01_dummy_baseline", dummy_pipe,
        X_train, y_train, X_val, y_val,
        params={"model": "DummyClassifier", "strategy": "most_frequent"},
        tags={"run_type": "trivial_baseline"}
    )

    # ── Run 2: Logistic Regression ───────────────────────────
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            class_weight='balanced', max_iter=1000,
            C=1.0, solver='lbfgs', random_state=RANDOM_SEED
        ))
    ])
    named_models['lr'], results['lr'] = _run_experiment(
        "02_logistic_regression", lr_pipe,
        X_train, y_train, X_val, y_val,
        params={"model": "LogisticRegression", "class_weight": "balanced",
                "C": 1.0, "max_iter": 1000},
        tags={"run_type": "interpretable_baseline"}
    )

    # ── Run 3: Random Forest ─────────────────────────────────
    rf_pipe = Pipeline([('clf', RandomForestClassifier(
        n_estimators=200, max_depth=6, min_samples_leaf=5,
        class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
    ))])
    named_models['rf'], results['rf'] = _run_experiment(
        "03_random_forest", rf_pipe,
        X_train, y_train, X_val, y_val,
        params={"model": "RandomForest", "n_estimators": 200,
                "max_depth": 6, "class_weight": "balanced"},
        tags={"run_type": "main_model"}
    )

    # ── Run 4: XGBoost ───────────────────────────────────────
    if XGB_AVAILABLE:
        xgb_pipe = Pipeline([('clf', xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=round(scale_pos_weight, 2),
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', use_label_encoder=False,
            random_state=RANDOM_SEED
        ))])
        named_models['xgb'], results['xgb'] = _run_experiment(
            "04_xgboost_main", xgb_pipe,
            X_train, y_train, X_val, y_val,
            params={"model": "XGBoost", "n_estimators": 300, "max_depth": 4,
                    "learning_rate": 0.05, "scale_pos_weight": round(scale_pos_weight, 2)},
            tags={"run_type": "main_model"}
        )

        # ── Run 5: XGBoost leakage-sensitivity ───────────────
        safe_cols = [c for c in feature_cols if c not in LEAKAGE_SENSITIVE_COLS]
        xgb_safe_pipe = Pipeline([('clf', xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=round(scale_pos_weight, 2),
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', use_label_encoder=False,
            random_state=RANDOM_SEED
        ))])
        named_models['xgb_safe'], results['xgb_safe'] = _run_experiment(
            "05_xgboost_leakage_sensitivity", xgb_safe_pipe,
            X_train[safe_cols], y_train, X_val[safe_cols], y_val,
            params={"model": "XGBoost_safe", "removed_features": str(LEAKAGE_SENSITIVE_COLS)},
            tags={"run_type": "leakage_sensitivity"}
        )
    else:
        print("  [XGBoost runs skipped — package not installed]")

    # ── Select and save best model ────────────────────────────
    candidate_keys = ['lr', 'rf'] + (['xgb'] if 'xgb' in results else [])
    best_key = max(candidate_keys, key=lambda k: results[k]['val_roc_auc'])
    best_model = named_models[best_key]
    best_auc   = results[best_key]['val_roc_auc']

    save_path = MODELS_DIR / "best_model.joblib"
    artifact  = {
        'model':        best_model,
        'feature_cols': feature_cols,
        'model_label':  best_key,
        'val_roc_auc':  best_auc,
    }
    joblib.dump(artifact, save_path)

    print("\n" + "="*65)
    print(f"Best model  : {best_key}  (Val ROC-AUC = {best_auc:.4f})")
    print(f"Saved to    : {save_path}")
    print("="*65)

    return {
        'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
        'models':  named_models,
        'results': results,
        'best_model_name': best_key,
        'best_model': best_model,
        'feature_cols': feature_cols,
    }
