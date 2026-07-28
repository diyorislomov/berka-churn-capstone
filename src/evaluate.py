"""
evaluate.py
Final model evaluation for FIN-02 Churn Prediction.

Includes:
  - Final test-set metrics (ROC-AUC, PR-AUC, F1, precision, recall)
  - ROC curve plot
  - Precision-Recall curve plot
  - Calibration (reliability diagram) plot
  - Confusion matrix plot
  - Per-slice evaluation (by region, tenure band, product type)
  - Error analysis (top false negatives and false positives)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # headless-safe backend
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)
from sklearn.calibration import calibration_curve

FIGURES_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_on_test(model, X_test, y_test):
    """
    Compute all final test-set metrics.

    Returns
    -------
    metrics : dict
    y_prob  : ndarray of predicted probabilities
    y_pred  : ndarray of predicted class labels
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        'roc_auc':    round(float(roc_auc_score(y_test, y_prob)), 4),
        'pr_auc':     round(float(average_precision_score(y_test, y_prob)), 4),
        'f1':         round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'precision':  round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'recall':     round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'n_test':     int(len(y_test)),
        'churn_rate': round(float(y_test.mean()), 4),
    }

    print("\n" + "="*55)
    print("FINAL EVALUATION — FROZEN TEST SET")
    print("="*55)
    for k, v in metrics.items():
        print(f"  {k:<20} : {v}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Active (0)', 'Churned (1)']))

    return metrics, y_prob, y_pred


# ─────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────

def _save(fig, filename):
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved -> {path}")
    plt.close(fig)


def plot_roc_curve(y_test, y_prob, model_label="Model"):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2.5, color='steelblue', label=f'{model_label} (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.6, label='Random classifier')
    ax.fill_between(fpr, tpr, alpha=0.08, color='steelblue')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve — Final Model (Test Set)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    _save(fig, 'roc_curve.png')


def plot_pr_curve(y_test, y_prob, model_label="Model"):
    """Plot and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc   = average_precision_score(y_test, y_prob)
    baseline = float(y_test.mean())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2.5, color='darkorange', label=f'{model_label} (AUC = {pr_auc:.4f})')
    ax.axhline(baseline, color='gray', ls='--', lw=1.2, label=f'Baseline (churn = {baseline:.1%})')
    ax.fill_between(recall, precision, baseline, alpha=0.08, color='darkorange')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision–Recall Curve — Final Model (Test Set)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    _save(fig, 'pr_curve.png')


def plot_calibration(y_test, y_prob, n_bins=10):
    """Plot and save calibration (reliability diagram)."""
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(prob_pred, prob_true, 's-', color='steelblue', lw=2, markersize=7, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Observed Churn Fraction', fontsize=12)
    ax.set_title('Calibration Plot (Reliability Diagram) — Test Set', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    _save(fig, 'calibration.png')


def plot_confusion_matrix(y_test, y_pred):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    labels = ['Active', 'Churned']

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual',    fontsize=12)
    ax.set_title('Confusion Matrix — Test Set', fontsize=13, fontweight='bold')
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i, j]:,}', ha='center', va='center',
                    color=color, fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    _save(fig, 'confusion_matrix.png')


def plot_feature_importance(model, feature_cols, top_n=15):
    """
    Plot feature importances if the model supports it
    (Random Forest or XGBoost inside a Pipeline).
    """
    # Unwrap pipeline
    clf = model.named_steps.get('clf', model)
    if not hasattr(clf, 'feature_importances_'):
        print("  Model does not expose feature_importances_ — skipping.")
        return

    importances = clf.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_cols    = [feature_cols[i] for i in indices]
    top_imp     = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(top_n), top_imp[::-1], color='steelblue', alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_cols[::-1], fontsize=10)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    _save(fig, 'feature_importance.png')


# ─────────────────────────────────────────────────────────────
# Slice evaluation
# ─────────────────────────────────────────────────────────────

def evaluate_by_slice(model, X_test, y_test, feature_matrix_test, slice_col):
    """
    Compute per-group ROC-AUC for subgroups defined by slice_col.
    Groups with fewer than 20 samples are skipped.
    """
    if slice_col not in feature_matrix_test.columns:
        print(f"  Column '{slice_col}' not in feature_matrix_test — skipping.")
        return pd.DataFrame()

    y_prob   = model.predict_proba(X_test)[:, 1]
    test_idx = X_test.index
    slice_vals = feature_matrix_test.loc[test_idx, slice_col] if test_idx is not None else feature_matrix_test[slice_col]

    rows = []
    for val in sorted(slice_vals.dropna().unique()):
        mask = (slice_vals == val).values
        n    = mask.sum()
        if n < 20:
            continue
        y_sub    = y_test.values[mask]
        prob_sub = y_prob[mask]
        if len(np.unique(y_sub)) < 2:
            continue
        auc = roc_auc_score(y_sub, prob_sub)
        rows.append({
            'slice_value':  val,
            'n':            n,
            'churn_rate':   round(float(y_sub.mean()), 3),
            'roc_auc':      round(float(auc), 4),
        })

    df = pd.DataFrame(rows)
    print(f"\nSlice evaluation — '{slice_col}':")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────
# Error analysis
# ─────────────────────────────────────────────────────────────

def error_analysis(model, X_test, y_test, feature_cols, n=10):
    """
    Identify the worst false negatives (missed churners) and
    false positives (incorrectly flagged active customers).

    Returns
    -------
    fn_df, fp_df : DataFrames of worst false negatives / positives
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    res = X_test.copy()
    res['y_true']      = y_test.values
    res['y_pred']      = y_pred
    res['churn_prob']  = y_prob

    # False negatives: churned but predicted active (sorted by lowest prob = most surprising miss)
    fn = res[(res['y_true'] == 1) & (res['y_pred'] == 0)].sort_values('churn_prob').head(n)
    # False positives: active but predicted churned
    fp = res[(res['y_true'] == 0) & (res['y_pred'] == 1)].sort_values('churn_prob', ascending=False).head(n)

    display_cols = ['churn_prob'] + [c for c in feature_cols[:6] if c in res.columns]

    print(f"\n{'='*55}")
    print(f"ERROR ANALYSIS — Top {n} False Negatives (missed churners)")
    print(f"{'='*55}")
    print(fn[display_cols].to_string())

    print(f"\n{'='*55}")
    print(f"ERROR ANALYSIS — Top {n} False Positives (incorrect alerts)")
    print(f"{'='*55}")
    print(fp[display_cols].to_string())

    return fn, fp


# ─────────────────────────────────────────────────────────────
# Master evaluation runner
# ─────────────────────────────────────────────────────────────

def run_full_evaluation(model, X_test, y_test, feature_cols,
                        model_label="Final Model", feature_matrix_test=None):
    """
    Run the complete evaluation pipeline:
    metrics → ROC → PR → calibration → confusion matrix
    → feature importance → slice eval → error analysis.

    Returns
    -------
    metrics : dict
    """
    print("\nRunning full evaluation pipeline...")
    metrics, y_prob, y_pred = evaluate_on_test(model, X_test, y_test)

    print("\nGenerating plots...")
    plot_roc_curve(y_test, y_prob, model_label=model_label)
    plot_pr_curve(y_test, y_prob, model_label=model_label)
    plot_calibration(y_test, y_prob)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(model, feature_cols)

    if feature_matrix_test is not None:
        for col in ['region_code', 'tenure_band']:
            evaluate_by_slice(model, X_test, y_test, feature_matrix_test, col)

    error_analysis(model, X_test, y_test, feature_cols)

    return metrics
