# producer.py

import json
import random
import time
from confluent_kafka import Producer

# Configuration for the Kafka Producer
conf = {
    'bootstrap.servers': 'broker:9092',
}

producer = Producer(conf)
topic = 'transactions'

def generate_transaction():
    """Generates a random transaction dictionary."""
    user_ids = ['user_A', 'user_B', 'user_C', 'user_D', 'user_E']
    locations = ['New York', 'Los Angeles', 'Chicago', 'Miami', 'Houston']
    
    if random.random() < 0.1:
        amount = round(random.uniform(1001.0, 5000.0), 2)
    else:
        amount = round(random.uniform(1.0, 999.99), 2)
    
    return {
        'transaction_id': str(random.randint(10000, 99999)),
        'user_id': random.choice(user_ids),
        'amount': amount,
        'timestamp': int(time.time()),
        'location': random.choice(locations)
    }

def delivery_report(err, msg):
    """Callback function for successful/failed delivery."""
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}')

try:
    print("Starting producer...")
    while True:
        transaction = generate_transaction()
        transaction_json = json.dumps(transaction)
        
        producer.produce(
            topic,
            key=transaction['user_id'],
            value=transaction_json,
            callback=delivery_report
        )
        
        producer.poll(0)
        time.sleep(1)

except KeyboardInterrupt:
    pass
finally:
    print("\nFlushing messages...")
    producer.flush()
    print("Producer finished.")
