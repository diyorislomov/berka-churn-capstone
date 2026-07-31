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
    if prob >= 0.15:
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