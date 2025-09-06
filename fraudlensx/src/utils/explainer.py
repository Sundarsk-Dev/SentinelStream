from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import shap

class ShapExplainer:
    def __init__(self, model):
        # TreeExplainer works well for XGBoost
        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X: np.ndarray, feature_names: List[str], top_k: int = 5) -> Dict[str, Any]:
        # Support shap==0.44 API variants
        try:
            shap_values = self.explainer.shap_values(X)  # old API
        except TypeError:
            exp = self.explainer(X)                      # new API returns Explanation
            shap_values = exp.values

        # Binary classifier: ensure 1D
        if isinstance(shap_values, list):
            # choose positive class if list returned
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        vals = shap_values[0]
        # Top-k absolute contributors
        order = np.argsort(np.abs(vals))[::-1][:top_k]
        contributions = [
            {"feature": feature_names[i], "value": float(vals[i])} for i in order
        ]
        return {
            "base_value": float(getattr(self.explainer, "expected_value", 0.0)
                                if not isinstance(getattr(self.explainer, "expected_value", 0.0), (list, np.ndarray))
                                else getattr(self.explainer, "expected_value")[0]),
            "contributions": contributions,
        }
