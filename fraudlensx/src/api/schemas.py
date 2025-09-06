from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, Optional
from datetime import datetime

class TransactionIn(BaseModel):
    transaction_id: Optional[str] = Field(default=None, description="Client-side id")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)
    features: Dict[str, float]     # keys must match training features

class PredictionOut(BaseModel):
    transaction_id: Optional[str]
    timestamp: datetime
    score: float                   # model probability for fraud
    is_fraud: bool
    threshold: float
    missing_features: list[str]
    extra_features: list[str]
    explanation: dict | None = None
