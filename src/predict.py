import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.joblib')

_model = None
_feature_cols = None

def load_model():
    global _model, _feature_cols
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _feature_cols = _model.get_booster().feature_names
    return _model, _feature_cols

def risk_tier(prob):
    """
    Thresholds based on calibration analysis against actual churn rates:
      prob >= 0.20  -> High   (actual churn rate ~28% in test set, ~2.9x base rate)
      prob >= 0.10  -> Medium (actual churn rate ~22% in test set, ~2.2x base rate)
      prob <  0.10  -> Low    (routine monitoring only)
    Note: raw XGBoost probabilities are not literal percentages -- these
    thresholds were validated against actual outcomes, not just model output.
    """
    if prob >= 0.20:
        return 'High'
    elif prob >= 0.10:
        return 'Medium'
    return 'Low'

def predict_churn(features: dict):
    model, feature_cols = load_model()
    missing = [c for c in feature_cols if c not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    row = pd.DataFrame([{c: features[c] for c in feature_cols}])
    prob = float(model.predict_proba(row)[:, 1][0])
    return {
        'churn_probability': round(prob, 4),
        'risk_tier': risk_tier(prob)
    }