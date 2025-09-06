"""
MVP Producer:
- DIRECT_API_MODE=True -> will POST synthetic transactions directly to the API (no Kafka needed)
- DIRECT_API_MODE=False -> publish to Kafka topic 'transactions' (requires local Kafka)
"""
import os, time, random, uuid, json
import requests
DIRECT_API_MODE = True
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Kafka (optional)
BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "transactions")

def gen_txn():
    return {
        "transaction_id": str(uuid.uuid4()),
        "features": {
            "transaction_amount": round(random.uniform(1, 1000), 2),
            "user_txn_count_24h": random.randint(0, 20),
            "avg_amount_7d": round(random.uniform(1, 500), 2),
            "merchant_risk": round(random.uniform(0, 1), 3),
            "device_trust": round(random.uniform(0, 1), 3),
            "geo_distance_km": round(random.uniform(0, 5000), 1),
            "is_international": random.choice([0, 1]),
            "card_age_days": random.randint(1, 3000),
            "failed_auth_24h": random.randint(0, 5),
            "hour_of_day": random.randint(0, 23),
        }
    }

def run_direct_api():
    while True:
        payload = gen_txn()
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            print("POST", r.status_code, r.json().get("score"))
        except Exception as e:
            print("API error:", e)
        time.sleep(0.5)

def run_kafka():
    from kafka import KafkaProducer
    producer = KafkaProducer(bootstrap_servers=BROKER, value_serializer=lambda v: json.dumps(v).encode())
    while True:
        producer.send(TOPIC, gen_txn())
        producer.flush()
        print("Sent to Kafka")
        time.sleep(0.5)

if __name__ == "__main__":
    if DIRECT_API_MODE:
        run_direct_api()
    else:
        run_kafka()

