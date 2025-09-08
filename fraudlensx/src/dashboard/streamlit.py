import streamlit as st
import requests
import numpy as np
import pandas as pd

API_URL = "http://127.0.0.1:8000/api/predict"

st.set_page_config(
    page_title="FraudLens — Fraud Alerts",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ FraudLens — Fraud Alerts")
st.markdown("### Real-time fraud detection results with model explanations")

# --- Run Sample Prediction Button ---
if st.button("🔍 Run Sample Prediction"):
    np.random.seed(42)
    sample = {"Time": 84692, "Amount": 150.0}
    for i in range(1, 29):
        sample[f"V{i}"] = float(np.random.randn())

    resp = requests.post(API_URL, json={"features": sample})
    result = resp.json()

    # --- Layout ---
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🚦 Transaction Status")
        if result["is_fraud"]:
            st.error("⚠️ Fraudulent Transaction Detected!")
        else:
            st.success("✅ Legitimate Transaction")

        st.metric("Fraud Probability", f"{result['score']:.2%}")
        st.metric("Threshold", result["threshold"])

    with col2:
        st.subheader("📊 Feature Importance (Top Factors)")
        explanation = result.get("explanation", {})
        if explanation:
            # Convert dict → DataFrame for chart
            exp_df = pd.DataFrame(list(explanation.items()), columns=["Feature", "Importance"])
            exp_df = exp_df.sort_values("Importance", ascending=False).head(10)
            st.bar_chart(exp_df.set_index("Feature"))

    # --- Extra Info ---
    with st.expander("📝 Raw JSON Output"):
        st.json(result)
