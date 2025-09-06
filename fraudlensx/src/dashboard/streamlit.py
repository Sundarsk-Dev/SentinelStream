import streamlit as st
import requests
import numpy as np

API_URL = "http://127.0.0.1:8000/api/predict"

st.title("🛡️ FraudLens — Fraud Alerts")

if st.button("Run Sample Prediction"):
    np.random.seed(42)
    sample = {"Time": 84692, "Amount": 150.0}
    for i in range(1, 29):
        sample[f"V{i}"] = float(np.random.randn())

    resp = requests.post(API_URL, json={"features": sample})
    result = resp.json()

    st.json(result)
