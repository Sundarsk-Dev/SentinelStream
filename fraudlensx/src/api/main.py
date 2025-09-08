from fastapi import FastAPI
from src.api.fraud_service import FraudService
from src.api.schemas import TransactionIn, PredictionOut
from datetime import datetime

THRESHOLD = 0.5
service = FraudService(threshold=THRESHOLD, with_explain=True)

app = FastAPI(
    title="FraudLensX API",
    version="1.0",
    description="Fraud detection inference API"
)

@app.post("/api/predict", response_model=PredictionOut)
def predict(request: TransactionIn):
    result = service.predict_from_dict(request.features)

    missing = [f for f in service.expected_feature_order() if f not in request.features]
    extra = [f for f in request.features if f not in service.expected_feature_order()]

    explanation = result.get("feature_importances")
    if isinstance(explanation, list):
        explanation = dict(explanation)  # ✅ fix

    return PredictionOut(
        transaction_id=request.transaction_id,
        timestamp=request.timestamp or datetime.utcnow(),
        score=result["probability"],
        is_fraud=bool(result["prediction"]),
        threshold=result["threshold"],
        missing_features=missing,
        extra_features=extra,
        explanation=explanation
    )
