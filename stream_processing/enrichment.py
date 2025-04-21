"""
Data enrichment module for adding additional information to the data streams
"""
from pyspark.sql.functions import udf, lit, col, when, month, dayofmonth, hour, dayofweek, dayofyear
from pyspark.sql.types import StringType

def add_time_components(df):
    """
    Add time-based components to the dataframe
    
    Args:
        df (DataFrame): Input DataFrame
        
    Returns:
        DataFrame: DataFrame with time components added
    """
    return df.select(
        "*",
        month("timestamp").alias("month"),
        dayofmonth("timestamp").alias("day"),
        hour("timestamp").alias("hour"),
        dayofweek("timestamp").alias("day_of_week"),
        dayofyear("timestamp").alias("day_of_year")
    ).withWatermark("timestamp", "20 seconds")

def is_within_area(lat, lon, lat_min, lat_max, lon_min, lon_max):
    """Check if a point is within a rectangular area"""
    return (lat_min <= lat <= lat_max and 
            lon_min <= lon <= lon_max)

@udf(returnType=StringType())
def determine_event_relation(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, venue_lat, venue_lon):
    """
    Determine relationship between a ride and an event location
    
    Args:
        pickup_lat: Pickup latitude
        pickup_lon: Pickup longitude
        dropoff_lat: Dropoff latitude
        dropoff_lon: Dropoff longitude
        venue_lat: Venue latitude
        venue_lon: Venue longitude
        
    Returns:
        str: Relationship type ('from_event', 'to_event', or 'none')
    """
    if (pickup_lat is None or pickup_lon is None or 
        dropoff_lat is None or dropoff_lon is None or 
        venue_lat is None or venue_lon is None):
        return "none"
    
    # Constants for latitude/longitude adjustments (0.5 km distance threshold)
    distance_km = 0.5
    latitude_adjustment = 0.009898 * distance_km
    longitude_adjustment = 0.00118 * distance_km

    # Define intervals around the venue
    lat_min = venue_lat - latitude_adjustment
    lat_max = venue_lat + latitude_adjustment
    lon_min = venue_lon - longitude_adjustment
    lon_max = venue_lon + longitude_adjustment

    # Check if pickup/dropoff locations are within the event area
    pickup_in_area = is_within_area(pickup_lat, pickup_lon, lat_min, lat_max, lon_min, lon_max)
    dropoff_in_area = is_within_area(dropoff_lat, dropoff_lon, lat_min, lat_max, lon_min, lon_max)

    if pickup_in_area and not dropoff_in_area:
        return "from_event"
    elif not pickup_in_area and dropoff_in_area:
        return "to_event"
    else:
        return "none"

def enrich_with_time_features(df):
    """
    Enrich dataframe with time-related features
    
    Args:
        df (DataFrame): Input DataFrame
        
    Returns:
        DataFrame: Enriched DataFrame with time features
    """
    return df.withColumn(
        "is_weekend", 
        when((col("day_of_week") == 1) | (col("day_of_week") == 7), 1).otherwise(0)
    )

def enrich_with_event_information(df, specials_df):
    """
    Enrich rides dataframe with event information from special events
    
    Args:
        df (DataFrame): Rides DataFrame
        specials_df (DataFrame): Special events DataFrame
        
    Returns:
        DataFrame: Enriched DataFrame with event information
    """
    # Initialize default values for event fields
    enriched_df = df.withColumn("event_relation", lit("none").cast(StringType()))
    enriched_df = enriched_df.withColumn("event_name", lit("none").cast(StringType()))
    
    try:
        # Process special events if available and not empty
        if specials_df is not None and not specials_df.rdd.isEmpty():
            # For each special event, apply the UDF to determine the ride relation to that event
            for event in specials_df.collect():
                # Extract event info
                event_name = event.event_name
                venue_lat = event.venue_latitude
                venue_lon = event.venue_longitude
                
                # Create a temporary column using the UDF.
                enriched_df = enriched_df.withColumn(
                    "temp_relation",
                    determine_event_relation(
                        col("pickup_latitude"), 
                        col("pickup_longitude"), 
                        col("dropoff_latitude"), 
                        col("dropoff_longitude"), 
                        lit(venue_lat), 
                        lit(venue_lon)
                    )
                )
                
                # Update event_relation and event_name only for rows where the temp_relation is not "none"
                enriched_df = enriched_df.withColumn(
                    "event_relation",
                    when(col("temp_relation") != "none", col("temp_relation")).otherwise(col("event_relation"))
                )
                enriched_df = enriched_df.withColumn(
                    "event_name",
                    when(col("temp_relation") != "none", lit(event_name)).otherwise(col("event_name"))
                )
                
                # Remove the temporary column
                enriched_df = enriched_df.drop("temp_relation")
    except Exception as e:
        print(f"Error processing special events: {e}")
        # Continue with default event relation values
    
    return enriched_df 