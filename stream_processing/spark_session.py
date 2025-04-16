"""
Spark session module for creating and configuring Spark sessions
"""
import findspark
import pyspark
from pyspark.sql import SparkSession
from stream_processing.config import SPARK_CONFIG

def create_spark_session():
    """
    Create and configure a Spark session
    
    Returns:
        SparkSession: Configured Spark session
    """
    try:
        findspark.init()
        
        # Create a Spark session
        print("Creating Spark session...")
        spark = SparkSession \
            .builder \
            .appName(SPARK_CONFIG["app_name"]) \
            .config("spark.streaming.stopGracefullyOnShutdown", SPARK_CONFIG["streaming_stop_gracefully"]) \
            .config('spark.jars.packages', SPARK_CONFIG["jars_packages"]) \
            .config("spark.sql.shuffle.partitions", SPARK_CONFIG["sql_shuffle_partitions"]) \
            .config("spark.driver.host", SPARK_CONFIG["driver_host"]) \
            .config("spark.driver.bindAddress", SPARK_CONFIG["bind_address"]) \
            .config("spark.executor.instances", SPARK_CONFIG["executor_instances"]) \
            .config("spark.executor.memory", SPARK_CONFIG["executor_memory"]) \
            .config("spark.driver.memory", SPARK_CONFIG["driver_memory"]) \
            .config("spark.executor.cores", SPARK_CONFIG["executor_cores"]) \
            .config("spark.task.cpus", SPARK_CONFIG["task_cpus"]) \
            .config("spark.kafka.consumer.cache.timeout", SPARK_CONFIG["kafka_consumer_cache_timeout"]) \
            .config("spark.kafka.consumer.max.poll.records", SPARK_CONFIG["kafka_consumer_max_poll_records"]) \
            .master(SPARK_CONFIG["master"]) \
            .getOrCreate()
        
        # Configure checkpointLocation
        spark.conf.set("spark.sql.streaming.checkpointLocation", "checkpoint")
        
        print("Spark session created successfully")
        return spark
    except Exception as e:
        print(f"Error creating Spark session: {e}")
        raise 