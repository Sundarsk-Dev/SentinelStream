# rule_engine.py

import json
from confluent_kafka import Consumer, KafkaException

# Configuration for the Kafka Consumer
# 'bootstrap.servers' points to the 'broker' service
# 'group.id' is a consumer group name
# 'auto.offset.reset': 'earliest' ensures we start reading from the beginning of the topic if no offset is saved
conf = {
    'bootstrap.servers': 'broker:9092',
    'group.id': 'rule-engine-group',
    'auto.offset.reset': 'earliest'
}

# Create a Consumer instance
consumer = Consumer(conf)

# Subscribe to the 'transactions' topic
consumer.subscribe(['transactions'])

# Define a simple fraud rule: any transaction over a specific amount is considered suspicious
SUSPICIOUS_AMOUNT_THRESHOLD = 1000.0

print("Starting rule engine, waiting for messages...")

try:
    while True:
        # Poll for new messages. The timeout is 1.0 second.
        msg = consumer.poll(1.0)
        
        if msg is None:
            # No message available within the timeout
            continue
        if msg.error():
            if msg.error().code() == KafkaException._PARTITION_EOF:
                # End of partition event, indicates no more messages for a while
                continue
            else:
                # Other error
                print(f"Consumer error: {msg.error()}")
                break
        
        # Message received successfully
        try:
            # Decode the message value (which is a JSON string)
            transaction_data = json.loads(msg.value().decode('utf-8'))
            
            # Extract relevant information
            transaction_id = transaction_data.get('transaction_id')
            amount = transaction_data.get('amount')
            user_id = transaction_data.get('user_id')
            
            print(f"Received transaction: ID={transaction_id}, User={user_id}, Amount=${amount}")
            
            # Apply the fraud detection rule
            if amount > SUSPICIOUS_AMOUNT_THRESHOLD:
                print(f"🚨 FRAUD ALERT! Suspicious transaction detected: ID={transaction_id}, User={user_id}, Amount=${amount}")
            
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON message: {e}")
            
except KeyboardInterrupt:
    pass
finally:
    # Close down the consumer to commit final offsets
    print("\nClosing consumer...")
    consumer.close()
    print("Rule engine finished.")

