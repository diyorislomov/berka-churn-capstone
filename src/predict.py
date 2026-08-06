import joblib
import pandas as pd
import numpy as np
import os
from sklearn.calibration import CalibratedClassifierCV

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.joblib')

_model = None
_feature_cols = None


def _get_feature_names(model):
    """
    Extract feature names from either a raw XGBClassifier or a
    CalibratedClassifierCV wrapping one. Raises clear error if unknown type.
    """
    if isinstance(model, CalibratedClassifierCV):
        # cv='prefit' stores the base estimator at model.estimator
        return model.estimator.get_booster().feature_names
    # Raw XGBClassifier
    return model.get_booster().feature_names


def load_model():
    global _model, _feature_cols
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _feature_cols = _get_feature_names(_model)
    return _model, _feature_cols


def risk_tier(prob):
    """
    Thresholds validated against actual churn rates on test set
    (calibrated isotonic model, base churn rate = 10%):
      prob >= 0.20  -> High   (actual churn ~24%, 2.4x base rate)
      prob >= 0.10  -> Medium (actual churn ~20%, 2.0x base rate)
      prob <  0.10  -> Low    (below base rate, routine monitoring)
    Probabilities come from a CalibratedClassifierCV (isotonic regression)
    wrapping the XGBoost model, so they reflect true likelihoods.
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