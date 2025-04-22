"""
Ride Operations Analytics

This page provides analytics and insights about ride operations, including KPIs,
ride patterns, driver performance, and service metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data, load_data_from_azure, load_local_data
from utils.visualizations import create_metric_card, plot_time_series, plot_treemap
import plotly.graph_objects as go
import json
from streamlit_autorefresh import st_autorefresh

# Set page configuration
st.set_page_config(
    page_title="Ride Operations Analytics",
    page_icon="🚗",
    layout="wide"
)

# Page title
st.title("🚗 Ride Operations Analytics")
st.markdown("### Use Case 1: Real-time Ride Operations Monitoring")

# Auto-refresh the page every 30 seconds
st_autorefresh(interval=30000, key="ops_refresher")

# Load data
# @st.cache_data(ttl=3600) # Removed caching to allow refresh
def load_ride_data():
    """Load ride data from Azure Blob Storage"""
    try:
        # Try loading from Azure first
        ride_events_df = load_data_from_azure("rides/*.snappy.parquet")
        
        if ride_events_df is None:
            # Fall back to local data if Azure fails
            ride_events_df = load_local_data("data/ride_events.json")
            
        if ride_events_df is None:
            st.error("Failed to load ride events data from both Azure and local storage.")
            return None
            
        return ride_events_df
    except Exception as e:
        st.error(f"Error loading ride events data: {str(e)}")
        return None

# @st.cache_data(ttl=3600) # Removed caching to allow refresh
def load_drivers_data():
    """Load drivers data from Azure Blob Storage or local file"""
    try:
        # Try loading from Azure first
        drivers_df = load_data_from_azure("drivers/*.snappy.parquet")
        
        if drivers_df is None:
            # Fall back to local data if Azure fails
            drivers_df = load_local_data("data/drivers_dynamic.json")
            
        if drivers_df is None:
            st.error("Failed to load drivers data from both Azure and local storage.")
            return None
            
        return drivers_df
    except Exception as e:
        st.error(f"Error loading drivers data: {str(e)}")
        return None

# Load and process data
with st.spinner("Loading data..."):
    ride_events_df = load_ride_data()
    drivers_dynamic_df = load_drivers_data()
    
    if ride_events_df is None or drivers_dynamic_df is None:
        st.stop()

# Calculate KPIs
def calculate_kpis(df):
    """Calculate key performance indicators from ride data"""
    # Active rides (rides in progress)
    active_rides = df[df['event_type'].isin(['RIDE_STARTED', 'DRIVER_ARRIVED'])].shape[0]
    
    # Total requested rides
    requested_rides = df[df['event_type'] == 'RIDE_REQUESTED'].shape[0]
    
    # Completed rides
    completed_rides = df[df['event_type'] == 'RIDE_COMPLETED'].shape[0]
    
    # Cancellation rate
    cancelled_rides = df[df['event_type'].isin(['RIDE_CANCELED_BY_USER', 'RIDE_CANCELED_BY_DRIVER'])].shape[0]
    total_rides = df[df['event_type'] == 'RIDE_REQUESTED'].shape[0]
    cancellation_rate = (cancelled_rides / total_rides) * 100 if total_rides > 0 else 0
    
    # Acceptance ratio
    accepted_rides = df[df['event_type'] == 'DRIVER_ASSIGNED'].shape[0]
    acceptance_ratio = (accepted_rides / requested_rides) * 100 if requested_rides > 0 else 0
    
    # Filter only USER_RATED_DRIVER events
    rating_events = df[df['event_type'] == 'USER_RATED_DRIVER']

    # Extract 'user_to_driver_rating' safely from the nested 'ratings' field
    if not rating_events.empty and 'ratings' in rating_events.columns:
        rating_events = rating_events[rating_events['ratings'].notnull()]
        rating_events['user_rating'] = rating_events['ratings'].apply(
            lambda x: x.get('user_to_driver_rating') if isinstance(x, dict) else None
        )
        satisfaction_score = rating_events['user_rating'].mean() * 20
    else:
        satisfaction_score = 0
    
    return {
        'active_rides': active_rides,
        'requested_rides': requested_rides,
        'completed_rides': completed_rides,
        'cancellation_rate': cancellation_rate,
        'acceptance_ratio': acceptance_ratio,
        'satisfaction_score': satisfaction_score
    }

# Calculate metrics
kpis = calculate_kpis(ride_events_df)

# Display KPIs
st.header("Key Performance Indicators")
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    create_metric_card(
        "Active Rides",
        kpis['active_rides'],
        help_text="Number of rides currently in progress"
    )

with col2:
    create_metric_card(
        "Requested Rides",
        kpis['requested_rides'],
        help_text="Total number of ride requests"
    )

with col3:
    create_metric_card(
        "Completed Rides",
        kpis['completed_rides'],
        help_text="Total number of completed rides"
    )

with col4:
    create_metric_card(
        "Cancellation Rate",
        f"{kpis['cancellation_rate']:.1f}%",
        help_text="Percentage of rides that were cancelled"
    )

with col5:
    create_metric_card(
        "Acceptance Ratio",
        f"{kpis['acceptance_ratio']:.1f}%",
        help_text="Percentage of ride requests that were accepted"
    )

with col6:
    create_metric_card(
        "Satisfaction Score",
        f"{kpis['satisfaction_score']:.1f}%",
        help_text="Average customer satisfaction rating"
    )

# Time Series Analysis
st.header("Ride Patterns Over Time")

# Prepare data for time series plots
ride_events_df['timestamp'] = pd.to_datetime(ride_events_df['timestamp'])

# Calculate hourly ride requests
hourly_rides = ride_events_df[ride_events_df['event_type'] == 'RIDE_REQUESTED'].groupby(
    ride_events_df['timestamp'].dt.hour
).size().reset_index()
hourly_rides.columns = ['hour', 'count']

# Calculate hourly cancellations
cancelled_rides = ride_events_df[
    ride_events_df['event_type'].isin(['RIDE_CANCELED_BY_USER', 'RIDE_CANCELED_BY_DRIVER'])
].groupby(ride_events_df['timestamp'].dt.hour).size().reset_index()
cancelled_rides.columns = ['hour', 'count']

# Create two columns for the time series plots
col1, col2 = st.columns(2)

with col1:
    plot_time_series(
        hourly_rides,
        'hour',
        'count',
        'Hourly Ride Requests',
        color='#1f77b4'
    )

with col2:
    plot_time_series(
        cancelled_rides,
        'hour',
        'count',
        'Hourly Ride Cancellations',
        color='#e74c3c'
    )

# Driver Category Distribution
st.header("Driver Performance Categories")

# Calculate driver categories based on ratings
if not drivers_dynamic_df.empty:
    # Add category to drivers data
    def categorize_driver(rating):
        if rating >= 4.5:
            return 'Gold'
        elif rating >= 4.0:
            return 'Silver'
        else:
            return 'Bronze'
    
    # Create a copy to avoid modifying the original dataframe
    drivers_data = drivers_dynamic_df.copy()
    drivers_data['category'] = drivers_data['rating'].apply(categorize_driver)
    
    # Count drivers in each category
    category_counts = drivers_data['category'].value_counts()
    
    # Create color mapping for the bar chart
    colors = {'Gold': '#FFD700', 'Silver': '#C0C0C0', 'Bronze': '#CD7F32'}
    
    # Create a bar chart using Streamlit
    st.bar_chart(category_counts)
    
    # Show some statistics
    total_drivers = len(drivers_data)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gold_count = category_counts.get('Gold', 0)
        gold_percent = (gold_count / total_drivers) * 100
        st.metric("Gold Drivers", f"{gold_count} ({gold_percent:.1f}%)")
        
    with col2:
        silver_count = category_counts.get('Silver', 0)
        silver_percent = (silver_count / total_drivers) * 100
        st.metric("Silver Drivers", f"{silver_count} ({silver_percent:.1f}%)")
        
    with col3:
        bronze_count = category_counts.get('Bronze', 0)
        bronze_percent = (bronze_count / total_drivers) * 100
        st.metric("Bronze Drivers", f"{bronze_count} ({bronze_percent:.1f}%)")
else:
    st.info("No driver data available for categorization.")

# Ride Operations and Customer Analytics
st.header("Ride Operations and Customer Analytics")

# Calculate event type distribution
event_counts = ride_events_df['event_type'].value_counts()

# Create a color map for different event types
color_map = {
    'RIDE_REQUESTED': '#2ecc71',      # Green for requests
    'RIDE_COMPLETED': '#3498db',      # Blue for completions
    'DRIVER_ASSIGNED': '#f1c40f',     # Yellow for assignments
    'RIDE_CANCELED_BY_USER': '#e74c3c',    # Red for cancellations
    'RIDE_CANCELED_BY_DRIVER': '#c0392b',  # Dark red for driver cancellations
    'RIDE_STARTED': '#27ae60',        # Dark green for started rides
    'DRIVER_ARRIVED': '#2980b9'       # Dark blue for arrivals
}

# Create bar chart
fig = go.Figure(data=[
    go.Bar(
        x=event_counts.index,
        y=event_counts.values,
        marker_color=[color_map.get(event, '#95a5a6') for event in event_counts.index]
    )
])

fig.update_layout(
    title="Distribution of Ride Events",
    xaxis_title="Event Type",
    yaxis_title="Number of Events",
    xaxis_tickangle=45,
    showlegend=False
)

st.plotly_chart(fig)
