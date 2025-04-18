"""
Data Processing Utilities

This module provides reusable data processing functions for the dashboard.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import streamlit as st

def preprocess_ride_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess ride events data for analysis.
    
    Args:
        df: DataFrame containing ride events data
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Convert timestamp to datetime if it's not already
    if pd.api.types.is_string_dtype(processed_df['timestamp']):
        processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])
    
    # Extract hour and date information
    processed_df['hour'] = processed_df['timestamp'].dt.hour
    processed_df['date'] = processed_df['timestamp'].dt.date
    processed_df['day_of_week'] = processed_df['timestamp'].dt.day_name()
    
    # Sort by timestamp
    processed_df.sort_values('timestamp', inplace=True)
    
    return processed_df

def aggregate_hourly_rides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate ride data by hour.
    
    Args:
        df: DataFrame containing ride events data
        
    Returns:
        DataFrame with hourly aggregations
    """
    # Group by hour and count events
    hourly_df = df.groupby('hour').size().reset_index(name='count')
    
    return hourly_df


def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in a DataFrame column.
    
    Args:
        df: DataFrame containing the data
        column: Column name to check for outliers
        method: Method to use ('iqr' for Interquartile Range or 'zscore' for Z-Score)
        threshold: Threshold for outlier detection (1.5 for IQR, 3 for Z-Score)
        
    Returns:
        DataFrame with added 'is_outlier' column
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    if method.lower() == 'iqr':
        # Calculate IQR
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        
        # Define bounds
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        # Flag outliers
        result_df['is_outlier'] = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method.lower() == 'zscore':
        # Calculate Z-Score
        mean = df[column].mean()
        std = df[column].std()
        
        # Flag outliers
        result_df['is_outlier'] = abs((df[column] - mean) / std) > threshold
    
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'iqr' or 'zscore'.")
    
    return result_df

def segment_customers(users_df: pd.DataFrame, rides_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Segment customers based on their ride patterns and available demographic information.
    
    Args:
        users_df: DataFrame containing user data with demographic info and possibly ride metrics
        rides_df: Optional DataFrame containing ride events data (not needed if metrics already in users_df)
        
    Returns:
        DataFrame with added segment column
    """
    # Make a copy to avoid modifying the original
    users = users_df.copy()
    
    # Check if ride metrics are already present in the users DataFrame
    has_ride_metrics = all(col in users.columns for col in ['avg_distance', 'avg_duration', 'frequency', 'total_rides'])
    
    # Calculate user metrics only if needed and rides_df is provided
    if not has_ride_metrics and rides_df is not None:
        user_metrics = {}
        
        for user_id in users['user_id'].unique():
            # Filter rides for this user
            user_rides = rides_df[rides_df['user_id'] == user_id]
            
            # Count rides
            ride_count = user_rides['ride_id'].nunique()
            
            # Calculate average ride distance and duration
            completed_rides = user_rides[user_rides['event_type'] == 'RIDE_COMPLETED']
            avg_distance = completed_rides['distance_km'].mean() if not completed_rides.empty else 0
            avg_duration = completed_rides['actual_duration_minutes'].mean() if not completed_rides.empty else 0
            
            # Calculate frequency (rides per week)
            if not user_rides.empty and len(user_rides) > 1:
                min_date = user_rides['timestamp'].min()
                max_date = user_rides['timestamp'].max()
                days_active = (max_date - min_date).days + 1
                weeks_active = max(1, days_active / 7)
                frequency = ride_count / weeks_active
            else:
                frequency = 0
                
            # Store user metrics
            user_metrics[user_id] = {
                'total_rides': ride_count,
                'avg_distance': avg_distance,
                'avg_duration': avg_duration,
                'frequency': frequency
            }
        
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame.from_dict(user_metrics, orient='index')
        metrics_df.reset_index(inplace=True)
        metrics_df.rename(columns={'index': 'user_id'}, inplace=True)
        
        # Merge with users DataFrame
        users = users.merge(metrics_df, on='user_id', how='left')
    
    # Get available columns for segmentation
    user_columns = users.columns.tolist()
    
    # Determine which demographic features are available
    has_gender = 'gender' in user_columns
    has_age = 'age' in user_columns
    has_income = 'income_level' in user_columns
    has_occupation = 'occupation' in user_columns
    has_marital_status = 'marital_status' in user_columns
    
    # Define segments based on available data
    def assign_segment(row):
        # Initialize ride metrics if they're in the dataframe
        freq = row.get('frequency', 0)
        avg_dist = row.get('avg_distance', 0)
        avg_dur = row.get('avg_duration', 0)
        total_rides = row.get('total_rides', 0)
        
        # Initialize demographic info if available
        gender = row['gender'] if has_gender and 'gender' in row and not pd.isna(row['gender']) else None
        age = row['age'] if has_age and 'age' in row and not pd.isna(row['age']) else None
        income = row['income_level'] if has_income and 'income_level' in row and not pd.isna(row['income_level']) else None
        occupation = row['occupation'] if has_occupation and 'occupation' in row and not pd.isna(row['occupation']) else None
        marital_status = row['marital_status'] if has_marital_status and 'marital_status' in row and not pd.isna(row['marital_status']) else None
        
        # Frequent Commuters (high frequency, short rides)
        if freq >= 2.5 and avg_dist < 10 and avg_dur < 20:
            if gender == 'Male' and age is not None and age < 40:
                return 'Young Male Commuters'
            elif gender == 'Female' and age is not None and age < 40:
                return 'Young Female Commuters'
            else:
                return 'Regular Commuters'
        
        # Long-distance travelers
        elif avg_dist > 15:
            if age is not None and age > 50:
                return 'Senior Long-distance Travelers'
            elif income is not None and income in ['High', 'Very High']:
                return 'Affluent Long-distance Travelers'
            else:
                return 'Long-distance Travelers'
        
        # Business users (weekday, business hours, consistent patterns)
        elif occupation is not None and occupation in ['Business', 'Professional']:
            return 'Business Travelers'
            
        # Infrequent users
        elif total_rides < 5 or freq < 0.5:
            return 'Occasional Riders'
            
        # Nightlife users (to be implemented with time analysis)
        # elif nighttime_rides_ratio > 0.5:
        #    return 'Nightlife Users'
            
        # Default segment
        else:
            return 'Average Users'
    
    # Apply the function to create a new segment column
    users['segment'] = users.apply(assign_segment, axis=1)
    
    return users

def extract_driver_info(ride_events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract driver metrics from ride events data.
    
    Args:
        ride_events_df: DataFrame containing ride events data
        
    Returns:
        DataFrame with driver metrics
    """
    # Group by driver_id
    driver_metrics = []
    
    for driver_id, driver_rides in ride_events_df.groupby('driver_id'):
        # Skip if driver_id is NaN
        if pd.isna(driver_id):
            continue
            
        # Count rides
        total_rides = len(driver_rides)
        
        # Calculate average user rating
        ratings = driver_rides['user_to_driver_rating'].dropna()
        avg_rating = ratings.mean() if len(ratings) > 0 else 0
        
        # Calculate total fare earned
        completed_rides = driver_rides[driver_rides['event_type'] == 'RIDE_COMPLETED']
        total_fare = completed_rides['total_fare'].sum()
        
        # Calculate cancellation rate
        canceled_rides = driver_rides[driver_rides['event_type'].str.contains('CANCELED', regex=False, na=False)]
        cancellation_rate = len(canceled_rides) / total_rides if total_rides > 0 else 0
        
        # Store driver metrics
        driver_metrics.append({
            'driver_id': driver_id,
            'no_of_rides': total_rides,
            'rating': avg_rating,
            'total_fare': total_fare,
            'cancellation_rate': cancellation_rate
        })
    
    # Convert to DataFrame
    return pd.DataFrame(driver_metrics) 

def determine_ride_direction(row, venue_lat, venue_lon, distance_km=0.5):
    """
    Determines if a ride is coming to, leaving from, or unrelated to a venue.
    
    Parameters:
    - row: A row from the rides dataframe
    - venue_lat: Latitude of the venue
    - venue_lon: Longitude of the venue
    - distance_km: Distance threshold in kilometers
    
    Returns:
    - 'coming', 'leaving', or None
    """
    # Constants for latitude/longitude adjustments
    latitude_adjustment = 0.009898 * distance_km
    longitude_adjustment = 0.00118 * distance_km
    
    # Define intervals around the venue
    lat_interval = [venue_lat - latitude_adjustment, venue_lat + latitude_adjustment]
    lon_interval = [venue_lon - longitude_adjustment, venue_lon + longitude_adjustment]
    
    # Check if pickup/dropoff locations are within the area
    def is_within_area(lat, lon):
        if pd.isna(lat) or pd.isna(lon):
            return False
        return (lat_interval[0] <= lat <= lat_interval[1] and 
                lon_interval[0] <= lon <= lon_interval[1])
    
    pickup_in_area = is_within_area(row["pickup_latitude"], row["pickup_longitude"])
    dropoff_in_area = is_within_area(row["dropoff_latitude"], row["dropoff_longitude"])
    
    if pickup_in_area and not dropoff_in_area:
        return 'leaving'
    elif not pickup_in_area and dropoff_in_area:
        return 'coming'
    else:
        return None
            
            # Function to analyze rides for all events
def analyze_rides_for_all_events(rides_df, special_df, distance_km=0.5):
    """
    Creates two new columns in the rides dataframe:
    1. 'event_direction': indicates if a ride was coming to, leaving from, or unrelated to any event
    2. 'event_name': indicates which event the ride was related to, if any
    
    Parameters:
    - rides_df: DataFrame containing ride data
    - special_df: DataFrame containing special event data
    - distance_km: Distance threshold in kilometers
    
    Returns:
    - rides_df with two new columns
    """
    # Create a dictionary of unique events with their coordinates
    events_dict = {}
    
    # If special events data is not available, return the rides_df as is
    if special_df is None:
        st.warning("Special events data not available. Skipping event analysis.")
        rides_df['event_direction'] = None
        rides_df['event_name'] = None
        return rides_df
    
    # Drop duplicates to get unique events (based on name, latitude, and longitude)
    try:
        unique_events = special_df.drop_duplicates(
            subset=['event_name', 'venue_latitude', 'venue_longitude', 'event_type']
        )
        
        for _, event in unique_events.iterrows():
            event_type = event['event_type']
            lat = event['venue_latitude']
            lon = event['venue_longitude']
            events_dict[event_type] = (lat, lon)
    except Exception as e:
        st.warning(f"Error processing special events: {str(e)}. Skipping event analysis.")
        rides_df['event_direction'] = None
        rides_df['event_name'] = None
        return rides_df
    
    # Initialize new columns
    rides_df['event_direction'] = None
    rides_df['event_name'] = None
    
    # Process each ride
    for idx, ride in rides_df.iterrows():
        # Check each event for this ride
        for event_name, (lat, lon) in events_dict.items():
            direction = determine_ride_direction(ride, lat, lon, distance_km)
            
            # If we found a relationship with this event, update the columns and break
            if direction is not None:
                rides_df.at[idx, 'event_direction'] = direction
                rides_df.at[idx, 'event_name'] = event_name
                break
    
    return rides_df
