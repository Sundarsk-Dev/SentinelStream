"""
MVP Consumer stub:
- If you later run Kafka, this script can consume from the topic and call the API.
- For MVP, you don't need to run this; use kafka_producer.py in DIRECT_API_MODE.
"""
import os, json, requests, time
from kafka import KafkaConsumer
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "transactions")
GROUP = os.getenv("KAFKA_GROUP", "fraudlens")

def run():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BROKER,
        group_id=GROUP,
        auto_offset_reset="latest",
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    for msg in consumer:
        payload = msg.value
        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            print("Scored", r.status_code, r.json().get("score"))
        except Exception as e:
            print("API error:", e)
        time.sleep(0.1)

if __name__ == "__main__":
    run()

