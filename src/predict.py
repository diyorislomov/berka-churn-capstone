"""
predict.py
Reusable churn prediction inference for FIN-02.

Usage:
    from src.predict import predict_churn

    result = predict_churn({
        'trans_count': 48,
        'trans_freq_monthly': 4.0,
        'avg_balance': 15000.0,
        # ... other features ...
    })
    print(result['churn_probability'])   # e.g. 0.73
    print(result['risk_tier'])           # 'High'
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "best_model.joblib"

# Risk tier thresholds (calibrated on validation set)
RISK_THRESHOLDS = {
    'High':   0.70,
    'Medium': 0.40,
}


def get_risk_tier(probability: float) -> str:
    """Map a churn probability to a business risk tier."""
    if probability >= RISK_THRESHOLDS['High']:
        return 'High'
    elif probability >= RISK_THRESHOLDS['Medium']:
        return 'Medium'
    else:
        return 'Low'


def get_recommended_action(risk_tier: str) -> str:
    """Return the recommended retention action for a risk tier."""
    actions = {
        'High':   'Proactive personal outreach or targeted retention offer recommended.',
        'Medium': 'Automated re-engagement message recommended.',
        'Low':    'No immediate retention action required.',
    }
    return actions[risk_tier]


def load_model(model_path=None):
    """
    Load the saved model artifact.

    Returns
    -------
    model        : fitted sklearn Pipeline
    feature_cols : list of feature column names expected by the model
    """
    if model_path is None:
        model_path = MODEL_PATH
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at: {model_path}\n"
            "Run notebooks/03_experiments.ipynb to train and save the model first."
        )
    artifact = joblib.load(model_path)
    return artifact['model'], artifact['feature_cols']


def validate_input(account_features: dict, feature_cols: list) -> list:
    """
    Validate input features and return a list of warning messages.
    Does NOT raise errors — missing features default to 0.
    """
    warnings = []
    provided = set(account_features.keys())
    missing  = [f for f in feature_cols if f not in provided]
    if missing:
        warnings.append(f"Missing features (defaulting to 0): {missing}")

    # Range sanity checks
    range_checks = {
        'trans_freq_monthly':  (0, 200),
        'credit_ratio':        (0, 1),
        'tenure_months':       (0, 600),
        'avg_balance':         (-1e7, 1e8),
    }
    for col, (lo, hi) in range_checks.items():
        if col in provided:
            val = account_features[col]
            if val is not None and not (lo <= float(val) <= hi):
                warnings.append(
                    f"Feature '{col}' = {val} is outside expected range [{lo}, {hi}]. "
                    f"Check your feature engineering."
                )
    return warnings


def predict_churn(account_features, model=None, feature_cols=None, model_path=None):
    """
    Predict churn probability for one or more accounts.

    Parameters
    ----------
    account_features : dict or pd.DataFrame or pd.Series
        Feature vector(s). Keys/columns must match the model's feature_cols.
        Missing features default to 0.
    model : sklearn Pipeline, optional
        Pre-loaded model. If None, loads from model_path.
    feature_cols : list, optional
        Feature column order. If None, loads from model_path.
    model_path : str or Path, optional
        Path to saved .joblib artifact.

    Returns
    -------
    dict (single account) or list of dicts (multiple accounts):
        {
          'churn_probability': float,
          'risk_tier':         str ('Low' | 'Medium' | 'High'),
          'recommended_action': str,
          'input_warnings':    list of str,
        }
    """
    if model is None or feature_cols is None:
        model, feature_cols = load_model(model_path)

    # Normalise input to DataFrame
    if isinstance(account_features, dict):
        single = True
        df = pd.DataFrame([account_features])
    elif isinstance(account_features, pd.Series):
        single = True
        df = account_features.to_frame().T
    elif isinstance(account_features, pd.DataFrame):
        single = False
        df = account_features.copy()
    else:
        raise TypeError(f"account_features must be dict, pd.Series, or pd.DataFrame. Got {type(account_features)}.")

    # Validate
    input_dict   = df.iloc[0].to_dict() if single else {}
    warnings_out = validate_input(input_dict, feature_cols) if single else []

    # Align to expected feature columns, fill missing with 0
    X = df.reindex(columns=feature_cols, fill_value=0).astype(float)

    # Predict
    probabilities = model.predict_proba(X)[:, 1]

    results = []
    for prob in probabilities:
        tier = get_risk_tier(prob)
        results.append({
            'churn_probability':  round(float(prob), 4),
            'risk_tier':          tier,
            'recommended_action': get_recommended_action(tier),
            'input_warnings':     warnings_out,
        })

    return results[0] if single else results


# ─────────────────────────────────────────────────────────────
# CLI entry point for quick testing
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Example accounts
    examples = [
        {
            "label": "High-risk account (low activity, declining balance)",
            "features": {
                'trans_count': 6, 'trans_freq_monthly': 0.5,
                'avg_trans_amount': 800.0, 'std_trans_amount': 400.0,
                'credit_count': 2, 'debit_count': 4, 'credit_ratio': 0.33,
                'avg_balance': 1200.0, 'std_balance': 600.0,
                'min_balance': 200.0, 'max_balance': 3000.0,
                'final_balance': 400.0, 'balance_trend': -120.0,
                'months_since_last_tx': 8.0,
                'tenure_months': 24.0, 'statement_freq_code': 0,
                'has_loan': 0, 'has_card': 0, 'order_count': 0,
                'region_code': 2, 'avg_salary': 7500.0,
                'unemployment_rate_96': 8.0, 'urban_population_ratio': 0.4,
            }
        },
        {
            "label": "Medium-risk account",
            "features": {
                'trans_count': 28, 'trans_freq_monthly': 2.3,
                'avg_trans_amount': 2500.0, 'std_trans_amount': 1200.0,
                'credit_count': 12, 'debit_count': 16, 'credit_ratio': 0.43,
                'avg_balance': 8000.0, 'std_balance': 2000.0,
                'min_balance': 3000.0, 'max_balance': 15000.0,
                'final_balance': 6000.0, 'balance_trend': -30.0,
                'months_since_last_tx': 2.5,
                'tenure_months': 36.0, 'statement_freq_code': 1,
                'has_loan': 0, 'has_card': 1, 'order_count': 1,
                'region_code': 0, 'avg_salary': 9000.0,
                'unemployment_rate_96': 4.0, 'urban_population_ratio': 0.65,
            }
        },
        {
            "label": "Low-risk account (engaged, growing balance)",
            "features": {
                'trans_count': 72, 'trans_freq_monthly': 6.0,
                'avg_trans_amount': 4200.0, 'std_trans_amount': 1800.0,
                'credit_count': 36, 'debit_count': 36, 'credit_ratio': 0.5,
                'avg_balance': 22000.0, 'std_balance': 3000.0,
                'min_balance': 15000.0, 'max_balance': 35000.0,
                'final_balance': 28000.0, 'balance_trend': 180.0,
                'months_since_last_tx': 0.3,
                'tenure_months': 60.0, 'statement_freq_code': 4,
                'has_loan': 1, 'has_card': 1, 'order_count': 3,
                'region_code': 0, 'avg_salary': 12000.0,
                'unemployment_rate_96': 2.5, 'urban_population_ratio': 0.85,
            }
        },
    ]

    print("\nFIN-02 Churn Prediction — Inference Examples")
    print("="*55)
    for ex in examples:
        result = predict_churn(ex['features'])
        print(f"\n{ex['label']}")
        print(f"  Churn probability : {result['churn_probability']:.1%}")
        print(f"  Risk tier         : {result['risk_tier']}")
        print(f"  Action            : {result['recommended_action']}")
        if result['input_warnings']:
            print(f"  Warnings          : {result['input_warnings']}")
