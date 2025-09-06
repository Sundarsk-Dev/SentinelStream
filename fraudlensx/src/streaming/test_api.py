import requests
import numpy as np
import pandas as pd

url = "http://127.0.0.1:8000/api/predict"

# Build sample transaction with all 30 features
np.random.seed(42)
sample = {"Time": 84692, "Amount": 150.0}
for i in range(1, 29):
    sample[f"V{i}"] = float(np.random.randn())

payload = {"features": sample}

resp = requests.post(url, json=payload)
print(resp.json())
