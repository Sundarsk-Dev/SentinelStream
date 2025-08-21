# src/model_serving/api.py

from fastapi import FastAPI, HTTPException
import joblib
import redis
import json
import numpy as np

# Connect to Redis
# The 'redis' hostname will resolve to the Redis container due to Docker networking
r = redis.Redis(host='redis', port=6379, db=0)

# Load the trained model and scaler at startup
try:
    # This assumes your trained model is saved as a joblib file
    # and a scaler is also available, which is crucial for preprocessing.
    model = joblib.load('models/fraud_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    print("✅ Fraud detection model and scaler loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    # Raise an error to prevent the service from starting if the model can't be loaded
    raise RuntimeError("Failed to load ML model or scaler.")

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="API for fraud scoring with real-time features."
)

@app.get("/health")
def read_health():
    """
    Health check endpoint to verify the service is running and the model is loaded.
    """
    status = "ok"
    if model is None:
        status = "unhealthy"
    return {"status": status}

@app.post("/score")
def score_transaction(transaction: dict):
    """
    Receives a transaction and returns a fraud score based on real-time features.
    
    Args:
        transaction (dict): A dictionary containing transaction details.
    """
    # Check if the model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    user_id = transaction.get('user_id')
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required.")

    # Retrieve real-time features from Redis
    # These keys should match what your Flink job is writing to Redis.
    velocity_key = f"user:{user_id}:amount_1m"
    real_time_feature_data = r.get(velocity_key)

    # Use 0.0 if the key is not found (first transaction for this user in the window)
    real_time_feature = float(real_time_feature_data) if real_time_feature_data else 0.0

    # Prepare data for the model
    # The order of features must match the order used during model training.
    # This example assumes 'amount' and a single real-time feature.
    features = np.array([[
        transaction.get('amount'),
        real_time_feature,
    ]])

    # Preprocess the features using the loaded scaler
    preprocessed_features = scaler.transform(features)

    # Make a prediction using the model
    # The model should be trained to expect the scaled features.
    prediction_proba = model.predict_proba(preprocessed_features)
    
    # The fraud score is the probability of the positive class (e.g., class 1)
    fraud_score = float(prediction_proba[0][1])
    
    # Determine if the transaction is fraud based on a predefined threshold
    is_fraud = bool(fraud_score > 0.5)
    
    return {
        "score": fraud_score, 
        "is_fraud": is_fraud,
        "features_used": {
            "amount": transaction.get('amount'),
            "real_time_velocity": real_time_feature
        }
    }

