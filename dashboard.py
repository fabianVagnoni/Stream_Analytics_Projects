"""
Ride Analytics Dashboard

Main entry point for the Streamlit dashboard application.
"""

import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from utils.data_loader import load_data, load_local_data
from utils.data_processing import preprocess_ride_events, calculate_ride_metrics, categorize_drivers
from utils.visualizations import create_metric_card, plot_time_series

# Load environment variables from the .env file
load_dotenv()

# Retrieve the required variables
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
AZURE_STORAGE_ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
AZURE_BLOB_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME")

# Set page configuration
st.set_page_config(
    page_title="Ride Analytics Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create sidebar and app title
st.sidebar.title("🚗 Ride Analytics")
st.sidebar.info(
    """
    Analyze ride-hailing operations data in real-time.
    
    Data Source: Azure Blob Storage
    """
)

# Page title
st.title("🚗 Ride Analytics Dashboard")

# Load data options
data_source = st.sidebar.radio(
    "Select Data Source",
    ["Azure Blob Storage", "Local Files"]
)

if data_source == "Azure Blob Storage":
    # Connect to Azure and load data
    st.sidebar.subheader("Azure Connection")
    st.sidebar.success(f"Connected to {AZURE_STORAGE_ACCOUNT_NAME}/{AZURE_BLOB_CONTAINER_NAME}")
    
    # Load data from Azure Blob Storage
    with st.spinner("Loading data from Azure Blob Storage..."):
        try:
            # Define blob paths based on the Azure blob container structure
            ride_events_path = "rides/rides.snappy.parquet"
            user_vectors_path = "user_vectors/user_vectors.snappy.parquet"
            specials_path = "specials/specials.snappy.parquet"
            
            # Try to load data, fall back to local if not available
            try:
                ride_events_df = load_data(ride_events_path, file_format='parquet')
                user_vectors_df = load_data(user_vectors_path, file_format='parquet')
                specials_df = load_data(specials_path, file_format='parquet')
                
                # Extract driver information from ride events for our dashboard
                drivers_df = extract_driver_info(ride_events_df)
                
                st.success("Data loaded successfully from Azure Blob Storage!")
            except Exception as e:
                st.warning(f"Error loading data from Azure: {str(e)}. Falling back to local files.")
                ride_events_df = load_local_data("data/ride_events.json")
                drivers_dynamic_df = load_local_data("data/drivers_dynamic.json")
                drivers_static_df = load_local_data("data/drivers_static.json")
                special_events_df = load_local_data("data/special_events.json")
                
                st.success("Data loaded successfully from local files!")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()
else:
    # Load data from local files
    with st.spinner("Loading data from local files..."):
        try:
            ride_events_df = load_local_data("data/ride_events.json")
            drivers_dynamic_df = load_local_data("data/drivers_dynamic.json")
            drivers_static_df = load_local_data("data/drivers_static.json")
            special_events_df = load_local_data("data/special_events.json")
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()

# Helper function to extract driver information from ride events
def extract_driver_info(ride_df):
    """
    Extract driver information from ride events data
    
    Args:
        ride_df: DataFrame containing ride events
        
    Returns:
        DataFrame with driver metrics
    """
    # Group by driver_id and calculate metrics
    driver_metrics = ride_df.groupby('driver_id').agg({
        'ride_id': 'count',
        'user_to_driver_rating': 'mean',
        'total_fare': 'sum',
        'actual_duration_minutes': 'sum'
    }).reset_index()
    
    # Rename columns
    driver_metrics = driver_metrics.rename(columns={
        'ride_id': 'no_of_rides',
        'user_to_driver_rating': 'rating',
        'total_fare': 'money_earned',
        'actual_duration_minutes': 'time_driven'
    })
    
    # Calculate cancellation rate (we need a more complex logic in real implementation)
    # This is a placeholder calculation
    cancel_counts = ride_df[ride_df['event_type'].str.contains('CANCEL', na=False)].groupby('driver_id').size()
    total_assigned = ride_df[ride_df['driver_id'].notna()].groupby('driver_id').size()
    
    # Create a DataFrame for the cancellation rate
    cancel_rate_df = pd.DataFrame({
        'driver_id': total_assigned.index,
        'cancellation_rate': cancel_counts.reindex(total_assigned.index).fillna(0) / total_assigned
    })
    
    # Merge with the driver metrics
    driver_metrics = pd.merge(driver_metrics, cancel_rate_df, on='driver_id', how='left')
    driver_metrics['cancellation_rate'] = driver_metrics['cancellation_rate'].fillna(0)
    
    return driver_metrics

# Process the data
try:
    processed_ride_events = preprocess_ride_events(ride_events_df)
    metrics = calculate_ride_metrics(processed_ride_events)
    
    if data_source == "Azure Blob Storage":
        # Use the driver info extracted from ride events
        categorized_drivers = categorize_drivers(drivers_df)
    else:
        # Use the local drivers_dynamic data
        categorized_drivers = categorize_drivers(drivers_dynamic_df)
        
except Exception as e:
    st.error(f"Error processing data: {str(e)}")
    st.stop()

# Main dashboard content
st.info("""
    This is the main dashboard page. Use the sidebar to navigate to specific analytics pages:
    
    1. Ride Operations and Customer Analytics
    2. Carbon Footprint Tracker
    3. Outlier Rides & Customer Segmentation
""")

# Display some basic metrics
st.header("Overview Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    create_metric_card(
        "Active Rides",
        metrics['active_rides'],
        help_text="Number of rides currently in progress"
    )

with col2:
    create_metric_card(
        "Rides Requested",
        metrics['requested_rides'],
        help_text="Total number of ride requests"
    )
    
with col3:
    create_metric_card(
        "Rides Completed",
        metrics['completed_rides'],
        help_text="Total number of completed rides"
    )
    
with col4:
    create_metric_card(
        "Cancellation Rate",
        f"{metrics['cancellation_rate']:.1f}%",
        help_text="Percentage of ride requests that were canceled"
    )

# Display a sample chart
st.header("Hourly Ride Activity")

# Create a DataFrame with hourly ride counts
hourly_rides = processed_ride_events.groupby(processed_ride_events['timestamp'].dt.hour).size().reset_index()
hourly_rides.columns = ['hour', 'count']

# Plot the hourly ride counts
plot_time_series(
    hourly_rides,
    'hour',
    'count',
    'Hourly Ride Requests',
    show_events=False
)
