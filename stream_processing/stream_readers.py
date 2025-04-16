"""
Stream readers module for reading data from Event Hubs
"""
from pyspark.sql.avro.functions import from_avro
from pyspark.sql.functions import col

def read_eventhub_stream(spark, kafka_config):
    """
    Read streaming data from Event Hub using Kafka connector
    
    Args:
        spark (SparkSession): The Spark session
        kafka_config (dict): Kafka configuration for the Event Hub
        
    Returns:
        DataFrame: Streaming DataFrame from Event Hub
    """
    try:
        df = spark \
            .readStream \
            .format("kafka") \
            .options(**kafka_config) \
            .load()
        
        return df
    except Exception as e:
        print(f"Error reading from Event Hub: {e}")
        raise

def deserialize_rides_stream(df, schema):
    """
    Deserialize the AVRO messages in the rides stream
    
    Args:
        df (DataFrame): Streaming DataFrame with AVRO messages
        schema (str): AVRO schema as string
        
    Returns:
        DataFrame: Deserialized DataFrame with flattened schema
    """
    try:
        # Deserialize AVRO messages
        df = df.select(from_avro(df.value, schema).alias("ride_events"))
        
        # Flatten the schema
        flattened_df = df.select(
            col("ride_events.event_id"),
            col("ride_events.ride_id"),
            col("ride_events.event_type"),
            col("ride_events.timestamp"),
            col("ride_events.user_id"),
            col("ride_events.driver_id"),

            col("ride_events.pickup_location.latitude").alias("pickup_latitude"),
            col("ride_events.pickup_location.longitude").alias("pickup_longitude"),
            col("ride_events.pickup_location.address").alias("pickup_address"),
            col("ride_events.pickup_location.city").alias("pickup_city"),

            col("ride_events.dropoff_location.latitude").alias("dropoff_latitude"),
            col("ride_events.dropoff_location.longitude").alias("dropoff_longitude"),
            col("ride_events.dropoff_location.address").alias("dropoff_address"),
            col("ride_events.dropoff_location.city").alias("dropoff_city"),

            col("ride_events.ride_details.distance_km"),
            col("ride_events.ride_details.estimated_duration_minutes"),
            col("ride_events.ride_details.actual_duration_minutes"),
            col("ride_events.ride_details.vehicle_type"),
            col("ride_events.ride_details.base_fare"),
            col("ride_events.ride_details.surge_multiplier"),
            col("ride_events.ride_details.total_fare"),

            col("ride_events.payment_info.payment_method"),
            col("ride_events.payment_info.payment_status"),
            col("ride_events.payment_info.payment_id"),

            col("ride_events.ratings.user_to_driver_rating"),
            col("ride_events.ratings.driver_to_user_rating"),
            col("ride_events.ratings.user_comment"),
            col("ride_events.ratings.driver_comment"),

            col("ride_events.cancellation_info.canceled_by"),
            col("ride_events.cancellation_info.cancellation_reason"),
            col("ride_events.cancellation_info.cancellation_fee"),

            col("ride_events.traffic_conditions.traffic_level"),
            col("ride_events.traffic_conditions.estimated_delay_minutes"),

            col("ride_events.driver_location.latitude").alias("driver_latitude"),
            col("ride_events.driver_location.longitude").alias("driver_longitude"),
            col("ride_events.driver_location.heading").alias("driver_heading"),
            col("ride_events.driver_location.speed_kmh").alias("driver_speed_kmh"),

            col("ride_events.app_version"),
            col("ride_events.platform"),
            col("ride_events.session_id")
        )
        
        return flattened_df
    except Exception as e:
        print(f"Error deserializing rides stream: {e}")
        raise

def deserialize_specials_stream(df, schema):
    """
    Deserialize the AVRO messages in the special events stream
    
    Args:
        df (DataFrame): Streaming DataFrame with AVRO messages
        schema (str): AVRO schema as string
        
    Returns:
        DataFrame: Deserialized DataFrame with flattened schema
    """
    try:
        # Deserialize AVRO messages
        df = df.select(from_avro(df.value, schema).alias("special_event"))
        
        # Flatten the schema
        flattened_df = df.select(
            col("special_event.type").alias("event_type"),
            col("special_event.name").alias("event_name"),
            col("special_event.venue_zone").alias("venue_zone"),

            col("special_event.venue_location.latitude").alias("venue_latitude"),
            col("special_event.venue_location.longitude").alias("venue_longitude"),
            col("special_event.venue_location.address").alias("venue_address"),
            col("special_event.venue_location.city").alias("venue_city"),

            col("special_event.event_start").alias("event_start"),
            col("special_event.event_end").alias("event_end"),
            col("special_event.arrivals_start").alias("arrivals_start"),
            col("special_event.arrivals_end").alias("arrivals_end"),
            col("special_event.departures_start").alias("departures_start"),
            col("special_event.departures_end").alias("departures_end"),

            col("special_event.arrival_rides").alias("arrival_rides"),
            col("special_event.departure_rides").alias("departure_rides"),
            col("special_event.estimated_attendees").alias("estimated_attendees")
        )
        
        return flattened_df
    except Exception as e:
        print(f"Error deserializing special events stream: {e}")
        raise 