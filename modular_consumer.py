"""
Modular version of the consumer.py file
This script maintains the same functionality as consumer.py but with a modular structure.
"""
import time
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from stream_processing.setup import setup_environment, initialize_directories
from stream_processing.config import (
    EVENT_HUB_NAMESPACE, 
    RIDES_EVENTHUB_NAME, RIDES_CONSUMER_EVENTHUB_CONNECTION_STR,
    SPECIALS_EVENTHUB_NAME, SPECIALS_CONSUMER_EVENTHUB_CONNECTION_STR,
    RIDE_SCHEMA_PATH, SPECIAL_SCHEMA_PATH,
    QUERY_CONFIGS, OUTPUT_DIRS, CHECKPOINT_DIRS,
    AZURE_STORAGE_CONNECTION_STRING, AZURE_BLOB_CONTAINER_NAME,
    get_kafka_config
)
from stream_processing.schema_utils import load_schema
from stream_processing.spark_session import create_spark_session
from stream_processing.stream_readers import (
    read_eventhub_stream, 
    deserialize_rides_stream, 
    deserialize_specials_stream
)
from stream_processing.enrichment import (
    add_time_components,
    enrich_with_time_features,
    enrich_with_event_information
)
from stream_processing.aggregation import (
    create_user_aggregations,
    display_user_vectors
)
from stream_processing.stream_writers import (
    write_stream_to_memory,
    write_stream_to_parquet,
    write_stream_to_parquet_and_blob,
    write_stream_to_console,
    list_active_queries,
    stop_all_queries
)
# from stream_processing.storage_sender import (
#     storage_connection_str,
#     blob_container_name
# )

def main():
    """Main function to run the stream analytics application"""
    print("Starting Stream Analytics Application")
    
    # Setup environment
    setup_environment()
    
    # Initialize directories
    initialize_directories()
    
    # Print connection information
    print(f"\nEvent Hub Namespace: {EVENT_HUB_NAMESPACE}")
    print(f"Rides Event Hub Name: {RIDES_EVENTHUB_NAME}")
    print(f"Rides Consumer Event Hub Connection String: {RIDES_CONSUMER_EVENTHUB_CONNECTION_STR}")
    print(f"Specials Event Hub Name: {SPECIALS_EVENTHUB_NAME}")
    print(f"Specials Consumer Event Hub Connection String: {SPECIALS_CONSUMER_EVENTHUB_CONNECTION_STR}\n")
    
    # Load schemas
    print("Loading schema files...")
    ride_schema = load_schema(RIDE_SCHEMA_PATH)
    special_schema = load_schema(SPECIAL_SCHEMA_PATH)
    
    # Create Spark session
    spark = create_spark_session()
    
    # Configure Kafka
    print("Setting up Kafka configurations...")
    kafka_conf_rides = get_kafka_config(
        EVENT_HUB_NAMESPACE, 
        RIDES_CONSUMER_EVENTHUB_CONNECTION_STR, 
        RIDES_EVENTHUB_NAME
    )
    
    kafka_conf_specials = get_kafka_config(
        EVENT_HUB_NAMESPACE, 
        SPECIALS_CONSUMER_EVENTHUB_CONNECTION_STR, 
        SPECIALS_EVENTHUB_NAME
    )
    print("Kafka configurations set up successfully")
    
    # Read from Event Hub
    print("Starting to read from Event Hubs...")
    try:
        # Read and deserialize rides
        print("Reading from Rides Event Hub...")
        df_rides_raw = read_eventhub_stream(spark, kafka_conf_rides)
        print("Successfully connected to Rides Event Hub")
        
        df_rides = deserialize_rides_stream(df_rides_raw, ride_schema)
        print("Successfully deserialized Rides data")
        
        # Read and deserialize special events
        print("Reading from Special Events Event Hub...")
        df_specials_raw = read_eventhub_stream(spark, kafka_conf_specials)
        print("Successfully connected to Special Events Event Hub")
        
        df_specials = deserialize_specials_stream(df_specials_raw, special_schema)
        print("Successfully deserialized Special Events data")
    except Exception as e:
        print(f"Error connecting to Event Hubs: {e}")
        return
    
    # Write streams to memory
    print("Starting memory stream query...")
    rides_memory_query = write_stream_to_memory(
        df_rides, 
        QUERY_CONFIGS["rides_memory"]["name"],
        QUERY_CONFIGS["rides_memory"]["checkpoint_location"]
    )
    
    # Write rides to parquet and Azure Blob Storage
    rides_parquet_query = write_stream_to_parquet_and_blob(
        df_rides,
        QUERY_CONFIGS["rides_parquet"]["name"],
        OUTPUT_DIRS["rides"],
        CHECKPOINT_DIRS["parquet_rides"],
        AZURE_STORAGE_CONNECTION_STRING,
        AZURE_BLOB_CONTAINER_NAME,
        "rides",
        QUERY_CONFIGS["rides_parquet"]["trigger_interval"]
    )
    
    # Write special events to parquet and Azure Blob Storage
    specials_parquet_query = write_stream_to_parquet_and_blob(
        df_specials,
        QUERY_CONFIGS["specials_parquet"]["name"],
        OUTPUT_DIRS["specials"],
        CHECKPOINT_DIRS["parquet_specials"],
        AZURE_STORAGE_CONNECTION_STRING,
        AZURE_BLOB_CONTAINER_NAME,
        "specials",
        QUERY_CONFIGS["specials_parquet"]["trigger_interval"]
    )
    
    # Write special events to memory
    specials_memory_query = write_stream_to_memory(
        df_specials, 
        QUERY_CONFIGS["specials_memory"]["name"]
    )
    
    # Add time components to rides data
    df_rides_with_time = add_time_components(df_rides)
    
    # Wait a few seconds for data to arrive
    print("Waiting for data to arrive (10 seconds)...")
    time.sleep(10)
    
    # Status check
    print("\nQuery status:", rides_memory_query.status)
    
    # Check if any data is available
    print("\nChecking for available data...")
    try:
        count_df = spark.sql(f'SELECT count(*) as record_count FROM {QUERY_CONFIGS["rides_memory"]["name"]}')
        count_rides = count_df.collect()[0]['record_count']
        columns = spark.sql(f'SELECT * FROM {QUERY_CONFIGS["rides_memory"]["name"]}').columns
        print(f"Columns: {columns}")
        print(f"Number of records received: {count_rides}")
        
        if count_rides > 0:
            print("\nShowing sample data:")
            spark.sql(f'SELECT * FROM {QUERY_CONFIGS["rides_memory"]["name"]}').show(5, truncate=True)
        else:
            print("No data received yet. This could be because:")
            print("1. No events are being published to the Event Hub")
            print("2. There might be authentication issues with the Event Hub connection")
            print("3. The consumer group might not have access to the events")
            print("4. The Event Hub name or connection strings might be incorrect")
    except Exception as e:
        print(f"Error querying data: {e}")
    
    # Get special events data
    try:
        specials_df = spark.sql("SELECT * FROM specials_table")
    except Exception as e:
        print(f"Error getting special events: {e}")
        specials_df = None
    
    # Enrich ride data with time features
    enriched_df = enrich_with_time_features(df_rides_with_time)
    
    # Enrich with event information
    enriched_df = enrich_with_event_information(enriched_df, specials_df)
    
    # Write enriched data to memory
    enriched_memory_query = write_stream_to_memory(
        enriched_df, 
        QUERY_CONFIGS["enriched_memory"]["name"]
    )
    
    # Create view for SQL queries
    enriched_df.createOrReplaceTempView("enriched_table")
    
    # Define the path to the users static data file
    users_static_path = "data/users_static.json"
    
    # Create user aggregations
    aggregated_df = create_user_aggregations(spark, "enriched_table", users_static_path)
    
    # Write aggregated data to parquet and Azure Blob Storage
    aggregated_query = write_stream_to_parquet_and_blob(
        aggregated_df,
        QUERY_CONFIGS["aggregated_parquet"]["name"],
        OUTPUT_DIRS["user_vectors"],
        CHECKPOINT_DIRS["parquet_aggregated"],
        AZURE_STORAGE_CONNECTION_STRING,
        AZURE_BLOB_CONTAINER_NAME,
        "user_vectors",
        QUERY_CONFIGS["aggregated_parquet"]["trigger_interval"]
    )
    
    # Wait for parquet files to build up
    print("Waiting for Parquet files to be created (210 seconds)...")
    time.sleep(210)
    
    print("Aggregated Query Status:", aggregated_query.status)
    
    # List output files
    try:
        rides_output_files = os.listdir("output/rides")
        specials_output_files = os.listdir("output/specials")
        print(f"\nFiles in rides output directory: {len(rides_output_files)}")
        if rides_output_files:
            print("Sample rides files:", rides_output_files[:5])
        else:
            print("No rides output files created yet")

        print(f"\nFiles in specials output directory: {len(specials_output_files)}")
        if specials_output_files:
            print("Sample special events files:", specials_output_files[:5])
        else:
            print("No special events output files created yet")

        # Check for user vectors files too
        try:
            user_vectors_output_files = os.listdir("output/user_vectors")
            print(f"\nFiles in user vectors output directory: {len(user_vectors_output_files)}")
            if user_vectors_output_files:
                print("Sample user vectors files:", user_vectors_output_files[:5])
            else:
                print("No user vectors output files created yet")
        except Exception as e:
            print(f"Error checking user vectors directory: {e}")
    except Exception as e:
        print(f"Error checking output directories: {e}")
    
    # Display user vectors
    try:
        display_user_vectors(spark)
    except Exception as e:
        print(f"Error displaying user vectors: {e}")
    
    # List active queries
    list_active_queries(spark)
    
    # Continue running or stop
    stop_queries = False
    print("\nTo stop the queries, set stop_queries = True and run this script again.")
    
    if stop_queries:
        stop_all_queries(spark)
    else:
        # Keep the application running until manually stopped
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            print("Stopping application...")
            stop_all_queries(spark)

if __name__ == "__main__":
    main() 