from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import os
import joblib

# 🔴 IMPORTANT: update this to match the feature order used in training
FEATURES: List[str] = [
    "transaction_amount",
    "user_txn_count_24h",
    "avg_amount_7d",
    "merchant_risk",
    "device_trust",
    "geo_distance_km",
    "is_international",       # 0/1
    "card_age_days",
    "failed_auth_24h",
    "hour_of_day"
]

_SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")

def _try_load_scaler():
    try:
        if os.path.exists(_SCALER_PATH):
            return joblib.load(_SCALER_PATH)
    except Exception:
        pass
    return None

_SCALER = _try_load_scaler()

def ensure_feature_vector(features: Dict[str, float]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Return (X[1,n], missing, extra) in the exact order required."""
    missing, extra = [], []
    vec = []
    for f in FEATURES:
        if f in features:
            v = features[f]
            # booleans as ints
            if isinstance(v, bool): v = int(v)
            vec.append(float(v))
        else:
            missing.append(f)
            vec.append(0.0)
    for k in features.keys():
        if k not in FEATURES:
            extra.append(k)
    X = np.array(vec, dtype=np.float32).reshape(1, -1)
    if _SCALER is not None:
        try:
            X = _SCALER.transform(X)
        except Exception:
            pass
    return X, missing, extra
