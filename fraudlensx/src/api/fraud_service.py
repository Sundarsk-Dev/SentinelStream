# src/api/fraud_service.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np


# --------------------------------------------------------------------------------------
# Paths: resolve model/scaler saved by your training script
# Looks in (in order):
#   1) ENV: FRAUDLENS_MODEL_PATH / FRAUDLENS_SCALER_PATH
#   2) <project_root>/src/models/
#   3) <project_root>/models/
#   4) <project_root>/
#   5) current working directory
# --------------------------------------------------------------------------------------

DEFAULT_MODEL_FILENAME = "fraud_detection_model.pkl"
DEFAULT_SCALER_FILENAME = "fraud_detection_scaler.pkl"

THIS_FILE = Path(__file__).resolve()
SRC_DIR = THIS_FILE.parents[1]      # .../src
PROJECT_ROOT = THIS_FILE.parents[2] # project root

SEARCH_DIRS: List[Path] = [
    # Highest priority: project tree
    PROJECT_ROOT / "src" / "models",
    PROJECT_ROOT / "models",
    PROJECT_ROOT,
    # Fallback: CWD if someone runs from a different place
    Path.cwd(),
]

ENV_MODEL = os.getenv("FRAUDLENS_MODEL_PATH", "").strip()
ENV_SCALER = os.getenv("FRAUDLENS_SCALER_PATH", "").strip()


def _resolve_path(preferred_env: str, filename: str) -> Path:
    if preferred_env:
        p = Path(preferred_env).expanduser().resolve()
        if p.exists():
            return p
    for d in SEARCH_DIRS:
        p = (d / filename).resolve()
        if p.exists():
            return p
    # Last resort: return the most reasonable default (won't exist -> clear error later)
    return (PROJECT_ROOT / "models" / filename).resolve()


MODEL_PATH: Path = _resolve_path(ENV_MODEL, DEFAULT_MODEL_FILENAME)
SCALER_PATH: Path = _resolve_path(ENV_SCALER, DEFAULT_SCALER_FILENAME)


# --------------------------------------------------------------------------------------
# Feature schema (30 inputs): Time, V1..V28, Amount
# --------------------------------------------------------------------------------------
EXPECTED_FEATURES: List[str] = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

# For quick index lookup when mapping dict -> array
FEATURE_INDEX = {name: i for i, name in enumerate(EXPECTED_FEATURES)}


class FraudService:
    """
    Minimal, production-friendly service around your trained RandomForest + StandardScaler.
    - Loads artifacts saved via `joblib.dump`.
    - Predicts probability and label using a configurable threshold.
    - Enforces exact 30-feature order.
    """

    def __init__(self, threshold: float = 0.5, with_explain: bool = False) -> None:
        self.threshold = float(threshold)
        self.with_explain = bool(with_explain)

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at: {MODEL_PATH}\n"
                f"Set FRAUDLENS_MODEL_PATH or place '{DEFAULT_MODEL_FILENAME}' in one of:\n"
                + "\n".join([f" - {d}" for d in SEARCH_DIRS])
            )
        if not SCALER_PATH.exists():
            raise FileNotFoundError(
                f"Scaler file not found at: {SCALER_PATH}\n"
                f"Set FRAUDLENS_SCALER_PATH or place '{DEFAULT_SCALER_FILENAME}' in one of:\n"
                + "\n".join([f" - {d}" for d in SEARCH_DIRS])
            )

        self.model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        print("✅ Fraud detection model and scaler loaded successfully!")
        print(f"   MODEL_PATH : {MODEL_PATH}")
        print(f"   SCALER_PATH: {SCALER_PATH}")

        # Optional: cache feature importances if present (RandomForest)
        self._feature_importances = None
        if hasattr(self.model, "feature_importances_"):
            self._feature_importances = np.asarray(self.model.feature_importances_, dtype=float)

    # ------------------------------------------------------------------
    # Public helpers (used by FastAPI layer)
    # ------------------------------------------------------------------

    @staticmethod
    def expected_feature_order() -> List[str]:
        """Return the strict feature order required by the model."""
        return list(EXPECTED_FEATURES)

    def prepare_features(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """
        Convert a dict of features into a (1, 30) numpy array in the required order.
        Raises a clear error if any required feature is missing.
        Ignores any extra keys.
        """
        missing = [f for f in EXPECTED_FEATURES if f not in feature_dict]
        if missing:
            raise ValueError(
                f"Missing required features: {missing}. "
                f"Expected exactly these keys: {EXPECTED_FEATURES}"
            )

        x = np.zeros((1, len(EXPECTED_FEATURES)), dtype=float)
        for name, idx in FEATURE_INDEX.items():
            val = feature_dict[name]
            try:
                x[0, idx] = float(val)
            except Exception as e:
                raise ValueError(f"Feature '{name}' must be numeric. Got: {val!r}") from e
        return x

    def predict_from_dict(self, feature_dict: Dict[str, float]) -> Dict:
        """
        End-to-end: map dict -> array -> scale -> proba -> label (+optional explanation).
        """
        x = self.prepare_features(feature_dict)
        return self.predict_from_array(x)

    def predict_from_array(self, x: np.ndarray) -> Dict:
        """
        Predict from a (1, 30) or (N, 30) array.
        Returns dict for single row; list of dicts for batch.
        """
        if x.ndim != 2 or x.shape[1] != len(EXPECTED_FEATURES):
            raise ValueError(
                f"Feature shape mismatch, expected: (*, {len(EXPECTED_FEATURES)}), got {x.shape}"
            )

        x_scaled = self.scaler.transform(x)
        probas = self.model.predict_proba(x_scaled)[:, 1]  # probability of fraud
        preds = (probas >= self.threshold).astype(int)

        if x.shape[0] == 1:
            result = {
                "prediction": int(preds[0]),
                "probability": float(probas[0]),
                "threshold": self.threshold,
            }
            if self.with_explain and self._feature_importances is not None:
                result["feature_importances"] = self._top_feature_importances()
            return result

        # Batch
        results = []
        for i in range(x.shape[0]):
            item = {
                "prediction": int(preds[i]),
                "probability": float(probas[i]),
                "threshold": self.threshold,
            }
            if self.with_explain and self._feature_importances is not None:
                item["feature_importances"] = self._top_feature_importances()
            results.append(item)
        return {"results": results}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _top_feature_importances(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Return top-k (feature_name, importance) pairs from the trained model.
        (Global model importances — not per-row explanations.)
        """
        if self._feature_importances is None:
            return []
        idx = np.argsort(self._feature_importances)[::-1][:top_k]
        return [(EXPECTED_FEATURES[i], float(self._feature_importances[i])) for i in idx]
