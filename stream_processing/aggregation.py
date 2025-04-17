"""
Data aggregation module for aggregating and processing streaming data
"""
import os
from pyspark.sql.functions import window, count, avg, max, sum, variance, collect_list, approx_count_distinct, expr
from pyspark.sql.types import DoubleType, StructType, StructField, StringType, IntegerType, DateType

def create_user_aggregations(spark, enriched_table_name="enriched_table", users_static_path="data/users_static.json"):
    """
    Create user aggregations from the enriched table
    
    Args:
        spark (SparkSession): Spark session
        enriched_table_name (str): Name of the enriched table
        users_static_path (str): Path to the users static JSON file
        
    Returns:
        DataFrame: Aggregated DataFrame
    """
    # Create users_static view
    try:
        # Using a simple approach with multiLine option
        print(f"Attempting to load users static data from {users_static_path}")
        
        # Just use Spark's built-in JSON reader with multiLine option
        users_static_df = spark.read.option("multiLine", "true").json(users_static_path)
        users_static_df.createOrReplaceTempView("users_static")
        
        print(f"Successfully loaded users static data")
    except Exception as e:
        print(f"Error loading users static data: {e}")
        print("Creating empty users_static table as fallback")
        # Create an empty users_static table as fallback
        spark.sql("""
            CREATE OR REPLACE TEMPORARY VIEW users_static (
                user_id STRING,
                first_name STRING,
                last_name STRING,
                age INT,
                gender STRING,
                email STRING,
                phone_number STRING,
                signup_date STRING,
                city STRING
            ) USING CSV OPTIONS (path '')
        """)
    
    # Run a Spark SQL query to perform complex transformations
    print("Running aggregation query with RIGHT JOIN on user data")
    aggregated_df = spark.sql(f"""
        SELECT
            e.user_id,
            window(e.timestamp, "10 seconds") AS time_window,
            COUNT(*) AS total_rides,
            AVG(e.distance_km) AS avg_distance_km,
            MAX(e.distance_km) AS max_distance_km,
            AVG(e.pickup_latitude) AS avg_pickup_latitude,
            AVG(e.pickup_longitude) AS avg_pickup_longitude,
            AVG(e.dropoff_latitude) AS avg_dropoff_latitude,
            AVG(e.dropoff_longitude) AS avg_dropoff_longitude,
            VARIANCE(e.distance_km) AS distance_variance,
            VARIANCE(e.hour) AS hour_variance,
            u.age,
            u.gender,
            u.signup_date
        FROM users_static u
        RIGHT JOIN {enriched_table_name} e ON u.user_id = e.user_id
        GROUP BY e.user_id, window(e.timestamp, "10 seconds"), u.age, u.gender, u.signup_date
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
                      most_common_event, age, gender, signup_date
                FROM {user_vectors_table}
                LIMIT 5
            """).show(truncate=False)
            
            # Debug: Count and show records with NULL age values
            null_age_count = spark.sql(f"SELECT COUNT(*) as null_count FROM {user_vectors_table} WHERE age IS NULL").collect()[0]['null_count']
            print(f"\nNumber of records with NULL age: {null_age_count}")
            if null_age_count > 0:
                print("Sample records with NULL age:")
                spark.sql(f"SELECT user_id, age, gender, signup_date FROM {user_vectors_table} WHERE age IS NULL LIMIT 5").show(truncate=False)
            
            # Save a snapshot of the consolidated vectors to a file
            spark.sql(f"SELECT * FROM {user_vectors_table}").write.mode("overwrite").parquet("output/user_vectors/consolidated")
            print("Saved consolidated user vectors to output/user_vectors/consolidated")
    except Exception as e:
        print(f"Error displaying consolidated user vectors: {e}")
        print("No user vectors table exists yet. Waiting for data to be processed.") 