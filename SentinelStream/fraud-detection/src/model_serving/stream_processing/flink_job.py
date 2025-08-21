# src/stream_processing/flink_job.py

import json
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaSource
from pyflink.common.serialization import SimpleStringSchema
from pyflink.datastream import TimeCharacteristic
from pyflink.datastream.functions import FlatMapFunction
from pyflink.common import WatermarkStrategy, Duration
import redis

# Define a simple flatmap function to parse JSON
class ParseJsonFlatMap(FlatMapFunction):
    def flat_map(self, value):
        try:
            yield json.loads(value)
        except json.JSONDecodeError:
            pass

# Connect to Redis
r = redis.Redis(host='redis', port=6379, db=0)

# Define a function to process and store features in Redis
def process_and_store_features(transaction):
    user_id = transaction.get('user_id')
    amount = transaction.get('amount')
    
    if user_id and amount:
        # Calculate velocity features (simple example: total amount in a 1-minute window)
        key = f"user:{user_id}:amount_1m"
        r.incrbyfloat(key, amount)
        r.expire(key, 60) # Set a 1-minute expiration

        print(f"Computed and stored velocity feature for user {user_id}")
    
    return transaction # Return the original transaction for further processing

def main():
    # Set up Flink environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_stream_time_characteristic(TimeCharacteristic.ProcessingTime)

    # Kafka source configuration
    kafka_source = KafkaSource.builder() \
        .set_bootstrap_servers("broker:29092") \
        .set_topics("transactions") \
        .set_group_id("flink-feature-engine") \
        .set_starting_offsets(KafkaOffsetsInitializer.earliest()) \
        .set_value_only_deserializer(SimpleStringSchema()) \
        .build()

    # Create a DataStream from the Kafka source
    data_stream = env.from_source(
        kafka_source,
        WatermarkStrategy.for_bounded_out_of_orderness(Duration.of_seconds(20)),
        "Kafka Source"
    )

    # Process the stream
    data_stream.flat_map(ParseJsonFlatMap()) \
        .key_by(lambda x: x['user_id']) \
        .map(process_and_store_features)
    
    # Execute the Flink job
    env.execute("Flink Real-Time Feature Computation")

if __name__ == '__main__':
    main()
