"""
Stream writers module for writing streaming data to different outputs
"""
from pyspark.sql.functions import expr

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