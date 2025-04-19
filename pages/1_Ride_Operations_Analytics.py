"""
Ride Operations Analytics

This page provides analytics and insights about ride operations, including KPIs,
ride patterns, driver performance, and service metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data, load_data_from_azure
from utils.visualizations import create_metric_card, plot_time_series, plot_treemap

# Set page configuration
st.set_page_config(
    page_title="Ride Operations Analytics",
    page_icon="🚗",
    layout="wide"
)

# Page title
st.title("🚗 Ride Operations Analytics")
st.markdown("### Use Case 1: Real-time Ride Operations Monitoring")

# Load data
@st.cache_data(ttl=3600)
def load_ride_data():
    """Load ride data from Azure Blob Storage"""
    try:
        # Load ride events data
        ride_events_df = load_data_from_azure("rides/*.snappy.parquet")
        
        if ride_events_df is None:
            st.error("Failed to load data from Azure. Check your connection and container configuration.")
            return None
            
        # Debug: Print column names and sample data
        st.write("Available columns:", ride_events_df.columns.tolist())
        st.write("Sample data:", ride_events_df.head())
            
        return ride_events_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load and process data
with st.spinner("Loading ride data..."):
    ride_events_df = load_ride_data()
    if ride_events_df is None:
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
hourly_rides = ride_events_df[ride_events_df['event_type'] == 'RIDE_REQUESTED'].groupby(ride_events_df['timestamp'].dt.hour).size().reset_index()
hourly_rides.columns = ['hour', 'count']

# Calculate ride efficiency
completed_rides = ride_events_df[ride_events_df['event_type'] == 'RIDE_COMPLETED']
if not completed_rides.empty and 'ride_details' in completed_rides.columns:
    completed_rides['efficiency_index'] = completed_rides['ride_details'].apply(
        lambda x: x['actual_duration_minutes'] / x['estimated_duration_minutes'] if x and x['actual_duration_minutes'] and x['estimated_duration_minutes'] else None
    )
    hourly_efficiency = completed_rides.groupby(completed_rides['timestamp'].dt.hour)['efficiency_index'].mean().reset_index()
    hourly_efficiency.columns = ['hour', 'efficiency_index']

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
    if 'efficiency_index' in completed_rides.columns:
        plot_time_series(
            hourly_efficiency,
            'hour',
            'efficiency_index',
            'Ride Efficiency Index',
            color='#2ca02c'
        )

# Driver Category Distribution
st.header("Driver Performance Categories")

# Calculate driver categories based on ratings
rating_events = ride_events_df[ride_events_df['event_type'] == 'USER_RATED_DRIVER']
if not rating_events.empty and 'ratings' in rating_events.columns:
    driver_metrics = rating_events.groupby('driver_id').agg({
        'ratings': lambda x: x.apply(lambda y: y['user_to_driver_rating'] if y else None).mean(),
        'ride_id': 'count'
    }).reset_index()
    
    # Categorize drivers based on rating
    driver_metrics['category'] = pd.cut(
        driver_metrics['ratings'],
        bins=[0, 3, 4, 5],
        labels=['Bronze', 'Silver', 'Gold']
    )
    
    category_counts = driver_metrics['category'].value_counts().reset_index()
    category_counts.columns = ['category', 'count']
    
    # Calculate growth percentage (placeholder - in real implementation, compare with previous period)
    category_counts['growth'] = [5, 2, -1]  # Example values
    
    # Display treemap
    plot_treemap(
        category_counts,
        ['category'],
        'count',
        'Driver Category Distribution'
    )

# Service Speed Metrics
st.header("Service Speed Metrics")

# Calculate average metrics
completed_rides = ride_events_df[ride_events_df['event_type'] == 'RIDE_COMPLETED']
if not completed_rides.empty and 'ride_details' in completed_rides.columns:
    avg_duration = completed_rides['ride_details'].apply(
        lambda x: x['actual_duration_minutes'] if x else None
    ).mean()
    
    # Calculate response time (time between RIDE_REQUESTED and DRIVER_ASSIGNED)
    ride_requests = ride_events_df[ride_events_df['event_type'] == 'RIDE_REQUESTED']
    driver_assignments = ride_events_df[ride_events_df['event_type'] == 'DRIVER_ASSIGNED']
    
    response_times = []
    for _, request in ride_requests.iterrows():
        assignment = driver_assignments[driver_assignments['ride_id'] == request['ride_id']]
        if not assignment.empty:
            response_time = (assignment['timestamp'].iloc[0] - request['timestamp']).total_seconds()
            response_times.append(response_time)
    
    avg_response_time = np.mean(response_times) if response_times else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        create_metric_card(
            "Average Ride Duration",
            f"{avg_duration:.1f} min",
            help_text="Average time taken for completed rides"
        )
    
    with col2:
        create_metric_card(
            "Average Response Time",
            f"{avg_response_time:.1f} sec",
            help_text="Average time taken by drivers to accept rides"
        )
