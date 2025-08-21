
import json
import requests
import sys
import time
from typing import Dict
from confluent_kafka import Consumer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer
from confluent_kafka.serialization import StringDeserializer

# --- Configuration ---
# Kafka and Schema Registry configuration
KAFKA_BOOTSTRAP_SERVERS = "broker:29092"
SCHEMA_REGISTRY_URL = "http://schema-registry:8081"
TRANSACTIONS_TOPIC = "transactions"
CONSUMER_GROUP_ID = "fraud-detection-group"
API_URL = "http://model-serving:8000/score"

# Schema Registry client setup
schema_registry_client = SchemaRegistryClient({"url": SCHEMA_REGISTRY_URL})

# Deserializers
key_deserializer = StringDeserializer("utf_8")
avro_deserializer = AvroDeserializer(schema_registry_client)

# Kafka Consumer configuration
consumer_conf = {
    'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
    'group.id': CONSUMER_GROUP_ID,
    'auto.offset.reset': 'earliest'
}

# --- Main Consumer Logic ---
def consume_and_score():
    """
    Consumes messages from the Kafka topic, sends data to the model-serving API,
    and prints the fraud score.
    """
    # Create the consumer instance
    consumer = Consumer(consumer_conf)
    
    # Subscribe to the topic
    consumer.subscribe([TRANSACTIONS_TOPIC])

    print(f"Consumer started. Listening for messages on topic: {TRANSACTIONS_TOPIC}")
    
    try:
        while True:
            # Poll for new messages with a 1-second timeout
            msg = consumer.poll(1.0)
            
            # If no message is received, continue to the next loop iteration
            if msg is None:
                continue
            
            # If an error occurs, print it and handle accordingly
            if msg.error():
                if msg.error().code() == 3:  # _PARTITION_EOF
                    # End of partition event - not a critical error
                    continue
                else:
                    # Other error
                    print(f"Consumer error: {msg.error()}")
                    continue

            # Deserialize the message key and value
            key = key_deserializer(msg.key())
            value = avro_deserializer(msg.value(), None)

            if value:
                # Prepare data for the API request
                # Exclude the 'is_fraud' label which is not a feature
                features_dict = {k: v for k, v in value.items() if k != 'is_fraud' and k != 'timestamp'}
                
                # Make the POST request to the API
                # The service name "model-serving" is used as the hostname in Docker Compose
                try:
                    response = requests.post(API_URL, json=features_dict, timeout=5)
                    response.raise_for_status()  # Raise an exception for bad status codes
                    
                    # Parse the JSON response
                    score_data = response.json()
                    fraud_score = score_data.get("fraud_score")
                    
                    if fraud_score is not None:
                        # Print the transaction ID and the model's fraud score
                        print(f"Transaction ID: {key} | Model Fraud Score: {fraud_score:.4f}")
                    else:
                        print(f"Transaction ID: {key} | Failed to get fraud score from API response.")
                
                except requests.exceptions.RequestException as e:
                    print(f"Error calling API for transaction {key}: {e}")
            
    except KeyboardInterrupt:
        pass
    finally:
        # Close the consumer connection
        consumer.close()

if __name__ == "__main__":
    consume_and_score()

