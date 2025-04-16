"""
Data aggregation module for aggregating and processing streaming data
"""
from pyspark.sql.functions import window, count, avg, max, sum, variance, collect_list, approx_count_distinct, expr
from pyspark.sql.types import DoubleType

def create_user_aggregations(spark, enriched_table_name="enriched_table"):
    """
    Create user aggregations from the enriched table
    
    Args:
        spark (SparkSession): Spark session
        enriched_table_name (str): Name of the enriched table
        
    Returns:
        DataFrame: Aggregated DataFrame
    """
    # Run a Spark SQL query to perform complex transformations
    aggregated_df = spark.sql(f"""
        SELECT
            user_id,
            window(timestamp, "10 seconds") AS time_window,
            COUNT(*) AS total_rides,
            AVG(distance_km) AS avg_distance_km,
            MAX(distance_km) AS max_distance_km,
            AVG(pickup_latitude) AS avg_pickup_latitude,
            AVG(pickup_longitude) AS avg_pickup_longitude,
            AVG(dropoff_latitude) AS avg_dropoff_latitude,
            AVG(dropoff_longitude) AS avg_dropoff_longitude,
            CAST(SUM(CASE WHEN day_of_week IN (5, 6) THEN 1 ELSE 0 END) AS DOUBLE)/COUNT(*) AS weekend_ride_ratio,
            CAST(SUM(CASE WHEN event_relation = 'to_event' THEN 1 ELSE 0 END) AS DOUBLE)/COUNT(*) AS to_event_ratio,
            CAST(SUM(CASE WHEN event_relation = 'from_event' THEN 1 ELSE 0 END) AS DOUBLE)/COUNT(*) AS from_event_ratio,
            approx_count_distinct(event_name) AS unique_events_count,
            VARIANCE(distance_km) AS distance_variance,
            VARIANCE(hour) AS hour_variance
        FROM {enriched_table_name}
        GROUP BY user_id, window(timestamp, "10 seconds")
    """)
    
    return aggregated_df

def display_user_vectors(spark, user_vectors_table="user_vectors_latest"):
    """
    Display user vectors information
    
    Args:
        spark (SparkSession): Spark session
        user_vectors_table (str): Name of the user vectors table
    """
    try:
        consolidated_count = spark.sql(f"SELECT COUNT(*) as user_count FROM {user_vectors_table}").collect()[0]['user_count']
        print(f"Number of users with profiles: {consolidated_count}")
        
        if consolidated_count > 0:
            print("\nSample user profiles:")
            spark.sql(f"""
                SELECT user_id, total_rides, avg_distance_km, max_distance_km, 
                      most_common_hour, most_common_day_of_week, 
                      weekend_ride_ratio, to_event_ratio, from_event_ratio,
                      most_common_event
                FROM {user_vectors_table}
                LIMIT 5
            """).show(truncate=False)
            
            # Save a snapshot of the consolidated vectors to a file
            spark.sql(f"SELECT * FROM {user_vectors_table}").write.mode("overwrite").parquet("output/consolidated_user_vectors")
            print("Saved consolidated user vectors to output/consolidated_user_vectors")
    except Exception as e:
        print(f"Error displaying consolidated user vectors: {e}")
        print("No user vectors table exists yet. Waiting for data to be processed.") 