"""
Configuration module for storing constant variables and settings
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define Azure credentials
EVENT_HUB_NAMESPACE = 'iesstsabbadbaa-grp-01-05'

RIDES_EVENTHUB_NAME = 'grp04-ride-events'
RIDES_CONSUMER_EVENTHUB_CONNECTION_STR = 'Endpoint=sb://iesstsabbadbaa-grp-01-05.servicebus.windows.net/;SharedAccessKeyName=Consumer;SharedAccessKey=iNowxPjC+fG9CrLnklmDTAy/J1n0e9Wpe+AEhC107ys=;EntityPath=grp04-ride-events'

SPECIALS_EVENTHUB_NAME = 'grp04-special-events'
SPECIALS_CONSUMER_EVENTHUB_CONNECTION_STR = 'Endpoint=sb://iesstsabbadbaa-grp-01-05.servicebus.windows.net/;SharedAccessKeyName=Consumer;SharedAccessKey=tJYUHSWabtnBVNOhc5TgJMHz1vtPw1NqC+AEhH6h8V4=;EntityPath=grp04-special-events'

# Azure Storage credentials
AZURE_STORAGE_ACCOUNT_NAME = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
AZURE_STORAGE_ACCOUNT_KEY = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
AZURE_STORAGE_CONNECTION_STRING = f"DefaultEndpointsProtocol=https;AccountName={AZURE_STORAGE_ACCOUNT_NAME};AccountKey={AZURE_STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"
AZURE_BLOB_CONTAINER_NAME = os.getenv('AZURE_BLOB_CONTAINER_NAME')

# Schema file paths
RIDE_SCHEMA_PATH = "schemas/ride_datafeed_schema.json"
SPECIAL_SCHEMA_PATH = "schemas/special_events_schema.json"

# Spark configuration
SPARK_CONFIG = {
    "app_name": "StreamingAVROFromKafka",
    "streaming_stop_gracefully": True,
    "jars_packages": "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,org.apache.spark:spark-avro_2.12:3.5.0",
    "sql_shuffle_partitions": 4,
    "driver_host": "localhost",
    "bind_address": "localhost",
    "executor_instances": 1,
    "executor_memory": "4g",
    "driver_memory": "4g",
    "executor_cores": "2",
    "task_cpus": "1",
    "kafka_consumer_cache_timeout": "60s",
    "kafka_consumer_max_poll_records": "500",
    "master": "local[*]"
}

# Kafka configurations
def get_kafka_config(event_hub_namespace, connection_string, eventhub_name):
    """Generate Kafka configuration for a given Event Hub"""
    return {
        "kafka.bootstrap.servers": f"{event_hub_namespace}.servicebus.windows.net:9093",
        "kafka.sasl.mechanism": "PLAIN",
        "kafka.security.protocol": "SASL_SSL",
        "kafka.sasl.jaas.config": f'org.apache.kafka.common.security.plain.PlainLoginModule required username="$ConnectionString" password="{connection_string}";',
        "subscribe": eventhub_name,
        "startingOffsets": "earliest",
        "kafka.request.timeout.ms": "60000",
        "kafka.session.timeout.ms": "60000",
        "failOnDataLoss": "false",
        "enable.auto.commit": "true",
        "groupIdPrefix": "Stream_Analytics_",
        "auto.commit.interval.ms": "5000"
    }

# Output directories
OUTPUT_DIRS = {
    "rides": "output/rides",
    "specials": "output/specials", 
    "user_vectors": "output/user_vectors",
    "aggregated": "output/user_vectors"
}

# Checkpoint directories
CHECKPOINT_DIRS = {
    "rides": "checkpoint/rides",
    "parquet_rides": "checkpoint/parquet/rides",
    "parquet_specials": "checkpoint/parquet/specials",
    "parquet_user_vectors": "checkpoint/parquet/user_vectors",
    "parquet_aggregated": "checkpoint/parquet/aggregated"
}

# Stream query configurations
QUERY_CONFIGS = {
    "rides_memory": {
        "name": "all_rides",
        "output_mode": "update",
        "format": "memory",
        "checkpoint_location": "checkpoint/rides"
    },
    "rides_parquet": {
        "name": "rides_parquet",
        "output_mode": "append",
        "format": "parquet",
        "checkpoint_location": "checkpoint/parquet/rides",
        "path": "output/rides",
        "trigger_interval": "20 seconds"
    },
    "specials_parquet": {
        "name": "specials_parquet",
        "output_mode": "append", 
        "format": "parquet",
        "checkpoint_location": "checkpoint/parquet/specials",
        "path": "output/specials",
        "trigger_interval": "20 seconds"
    },
    "specials_memory": {
        "name": "specials_table",
        "output_mode": "append",
        "format": "memory"
    },
    "enriched_memory": {
        "name": "enriched_table",
        "output_mode": "append",
        "format": "memory"
    },
    "aggregated_parquet": {
        "name": "aggregated_query",
        "output_mode": "append",
        "format": "parquet",
        "checkpoint_location": "checkpoint/parquet/aggregated",
        "path": "output/user_vectors",
        "trigger_interval": "10 seconds"
    },
    "test_console": {
        "name": "test_console_query", 
        "output_mode": "append",
        "format": "console",
        "trigger_interval": "10 seconds"
    }
} 