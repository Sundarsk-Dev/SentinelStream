from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from src.api.fraud_service import FraudService

# Configurable threshold
THRESHOLD = 0.5

# Initialize FraudService
service = FraudService(threshold=THRESHOLD, with_explain=True)

app = FastAPI(title="FraudLensX API", version="1.0")


class TransactionRequest(BaseModel):
    transaction_id: str
    features: Dict[str, Any]


@app.post("/predict")
def predict(request: TransactionRequest):
    result = service.predict(request.features)
    service.log_transaction(request.transaction_id, request.features, result)
    return {
        "transaction_id": request.transaction_id,
        "result": result
    }
