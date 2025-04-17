"""
Stream writers module for writing streaming data to different outputs
"""
from pyspark.sql.functions import expr
from azure.storage.blob import BlobServiceClient
import os


def write_stream_to_memory(df, query_name, checkpoint_location=None):
    """
    Write a streaming dataframe to memory
    
    Args:
        df (DataFrame): Streaming DataFrame to write
        query_name (str): Query name
        checkpoint_location (str, optional): Checkpoint location
        
    Returns:
        StreamingQuery: The streaming query
    """
    query_builder = df.writeStream \
        .outputMode("update") \
        .format("memory") \
        .queryName(query_name)
    
    if checkpoint_location:
        query_builder = query_builder.option("checkpointLocation", checkpoint_location)
    
    query_builder = query_builder.option("failOnDataLoss", "false")
    
    query = query_builder.start()
    print(f"Memory stream query '{query_name}' started successfully")
    
    return query


def write_stream_to_parquet(df, query_name, path, checkpoint_location, trigger_interval=None):
    """
    Write a streaming dataframe to parquet files
    
    Args:
        df (DataFrame): Streaming DataFrame to write
        query_name (str): Query name
        path (str): Output path for parquet files
        checkpoint_location (str): Checkpoint location
        trigger_interval (str, optional): Trigger interval
        
    Returns:
        StreamingQuery: The streaming query
    """
    query_builder = df.writeStream \
        .format("parquet") \
        .option("checkpointLocation", checkpoint_location) \
        .option("path", path) \
        .option("failOnDataLoss", "false") \
        .queryName(query_name) \
        .outputMode("append")
    
    if trigger_interval:
        query_builder = query_builder.trigger(processingTime=trigger_interval)
    
    query = query_builder.start()
    print(f"Parquet stream query '{query_name}' started successfully")
    
    return query


# This function uses foreachBatch to write locally and then upload to blob storage
def process_batch(batch_df, batch_id, local_path, blob_connection_str, blob_container_name, blob_folder_prefix):
    # Save the batch locally
    local_output_path = f"{local_path}/batch_{batch_id}"
    batch_df.write.mode("append").parquet(local_path)
    
    # Get a list of files that were just written
    try:
        # Connect to blob storage
        blob_service_client = BlobServiceClient.from_connection_string(blob_connection_str)
        container_client = blob_service_client.get_container_client(blob_container_name)
        
        # Upload each file to blob storage with the appropriate prefix
        for root, dirs, files in os.walk(local_path):
            for file in files:
                if file.endswith(".parquet") and not file.startswith("._"):
                    local_file_path = os.path.join(root, file)
                    # Create a blob path that includes the stream identifier (prefix)
                    relative_path = os.path.relpath(local_file_path, local_path)
                    blob_path = f"{blob_folder_prefix}/{relative_path}"
                    
                    # Upload the file to blob storage
                    with open(local_file_path, "rb") as data:
                        container_client.upload_blob(name=blob_path, data=data, overwrite=True)
                    print(f"Uploaded {local_file_path} to {blob_path}")
    except Exception as e:
        print(f"Error uploading to blob storage: {e}")


def write_stream_to_parquet_and_blob(df, query_name, local_path, checkpoint_location, blob_connection_str, blob_container_name, blob_folder_prefix, trigger_interval=None):
    """
    Write a streaming dataframe to local parquet files and then upload to Azure Blob Storage
    
    Args:
        df (DataFrame): Streaming DataFrame to write
        query_name (str): Query name
        local_path (str): Local output path for parquet files
        checkpoint_location (str): Checkpoint location
        blob_connection_str (str): Azure Blob Storage connection string
        blob_container_name (str): Azure Blob Storage container name
        blob_folder_prefix (str): Prefix for blob storage folders to differentiate streams
        trigger_interval (str, optional): Trigger interval
        
    Returns:
        StreamingQuery: The streaming query
    """
    
    # Configure the streaming query
    query_builder = df.writeStream \
        .foreachBatch(lambda batch_df, batch_id: process_batch(
            batch_df, batch_id, local_path, blob_connection_str, blob_container_name, blob_folder_prefix
        )) \
        .option("checkpointLocation", checkpoint_location) \
        .queryName(query_name)
    
    if trigger_interval:
        query_builder = query_builder.trigger(processingTime=trigger_interval)
    
    query = query_builder.start()
    print(f"Parquet with Blob Storage stream query '{query_name}' started successfully")
    
    return query


def write_stream_to_console(df, query_name, trigger_interval=None):
    """
    Write a streaming dataframe to the console
    
    Args:
        df (DataFrame): Streaming DataFrame to write
        query_name (str): Query name
        trigger_interval (str, optional): Trigger interval
        
    Returns:
        StreamingQuery: The streaming query
    """
    query_builder = df.writeStream \
        .format("console") \
        .outputMode("append") \
        .queryName(query_name)
    
    if trigger_interval:
        query_builder = query_builder.trigger(processingTime=trigger_interval)
    
    query = query_builder.start()
    print(f"Console stream query '{query_name}' started successfully")
    
    return query


def list_active_queries(spark):
    """
    List all active streaming queries
    
    Args:
        spark (SparkSession): Spark session
    """
    print("\n" + "="*50)
    print("STREAMING QUERIES RUNNING")
    print("="*50)
    print("Active queries:")
    for q in spark.streams.active:
        print(f"  - {q.name} (ID: {q.id})")
    print("\nThe script will continue running, leave it open to collect data.")


def stop_all_queries(spark):
    """
    Stop all active streaming queries
    
    Args:
        spark (SparkSession): Spark session
    """
    print("Stopping all streaming queries...")
    for q in spark.streams.active:
        print(f"Stopping {q.name}...")
        q.stop()
    
    print("Stopping Spark session...")
    spark.stop()
    print("All queries and Spark session stopped.")
    
    print("\nData saved to:")
    print(f"  - Rides data: output/rides")
    print(f"  - Special events data: output/specials")
    print(f"  - User profiles data: output/user_vectors") 