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
    cancellation_rate = (cancelled_rides / total_rides) if total_rides > 0 else 0
    
    # Acceptance ratio
    accepted_rides = df[df['event_type'] == 'DRIVER_ASSIGNED'].shape[0]
    acceptance_ratio = (accepted_rides / requested_rides) * 100 if requested_rides > 0 else 0
    
    # Filter only USER_RATED_DRIVER events and safely extract ratings
    rating_events = df[df['event_type'] == 'USER_RATED_DRIVER']
    ratings = []
    
    if not rating_events.empty and 'ratings' in rating_events.columns:
        for _, event in rating_events.iterrows():
            if isinstance(event['ratings'], dict):
                rating = event['ratings'].get('user_to_driver_rating')
                if rating is not None:
                    ratings.append(rating)
        
        rating_rides = pd.DataFrame({'user_to_driver_rating': ratings})
        avg_rating = rating_rides['user_to_driver_rating'].mean() if not rating_rides.empty else 0
        rating_score = (avg_rating - 1) / 4 if avg_rating > 0 else 0
    else:
        rating_score = 0
    
    # 2. Duration Efficiency Score (30% weight)
    completed_rides_df = df[(df['event_type'] == 'RIDE_COMPLETED') & 
                          (df['estimated_duration_minutes'] > 0)].copy()
    
    if not completed_rides_df.empty:
        completed_rides_df['efficiency_ratio'] = (completed_rides_df['actual_duration_minutes'] /
                                                completed_rides_df['estimated_duration_minutes'])
        avg_efficiency_ratio = completed_rides_df['efficiency_ratio'].mean()
        duration_score = np.clip(2 - avg_efficiency_ratio, 0, 1)
    else:
        duration_score = 0
    
    # 3. Cancellation Score (30% weight)
    requested_ids = set(df[df['event_type'] == 'RIDE_REQUESTED']['ride_id'])
    cancelled_ids = set(df[df['event_type'].isin(['RIDE_CANCELED_BY_USER', 'RIDE_CANCELED_BY_DRIVER'])]['ride_id'])
    effective_cancelled = len(requested_ids.intersection(cancelled_ids))
    cancellation_rate = effective_cancelled / len(requested_ids) if len(requested_ids) > 0 else 0
    cancellation_score = 1 - min(cancellation_rate / 0.5, 1)
    
    # Calculate weighted composite score
    weights = {
        'rating': 0.4,
        'duration': 0.3,
        'cancellation': 0.3
    }
    
    composite_score = (
        weights['rating'] * rating_score +
        weights['duration'] * duration_score +
        weights['cancellation'] * cancellation_score
    ) * 100  # Convert to percentage
    
    return {
        'active_rides': active_rides,
        'requested_rides': requested_rides,
        'completed_rides': completed_rides,
        'cancellation_rate': cancellation_rate * 100,  # Convert to percentage
        'acceptance_ratio': acceptance_ratio,
        'composite_satisfaction_score': composite_score
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
        "Composite Customer Satisfaction Score",
        f"{kpis['composite_satisfaction_score']:.1f}%",
        help_text="A weighted score combining customer ratings, ride duration efficiency, and cancellation rate"
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
if not drivers_dynamic_df.empty and 'rating' in drivers_dynamic_df.columns:
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
    
    # Ensure rating column is numeric
    drivers_data['rating'] = pd.to_numeric(drivers_data['rating'], errors='coerce')
    
    # Only categorize drivers with valid ratings
    drivers_data = drivers_data.dropna(subset=['rating'])
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
    st.info("No rating data available for driver categorization.")

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

# Ride Status Count Analysis
st.header("Operational Overview: Ride Status Count")

# Define terminal event types (completed or canceled)
terminal_events = {"RIDE_COMPLETED", "RIDE_CANCELED_BY_USER", "RIDE_CANCELED_BY_DRIVER"}

# Compute unique ride_ids by event_type
requested_set = set(ride_events_df.loc[ride_events_df['event_type'] == 'RIDE_REQUESTED', 'ride_id'].unique())
started_set = set(ride_events_df.loc[ride_events_df['event_type'] == 'RIDE_STARTED', 'ride_id'].unique())
completed_set = set(ride_events_df.loc[ride_events_df['event_type'] == 'RIDE_COMPLETED', 'ride_id'].unique())
cancelled_set = set(ride_events_df.loc[ride_events_df['event_type'].isin(["RIDE_CANCELED_BY_USER", "RIDE_CANCELED_BY_DRIVER"]), 'ride_id'].unique())

# Terminal rides are those with any terminal event (completed or canceled)
terminal_set = completed_set.union(cancelled_set)

# Calculate ride categories
requested_rides = requested_set - terminal_set  # Pending requests
active_rides = started_set - terminal_set      # Currently active
completed_rides = completed_set                # Successfully completed

# Prepare data for plotting
ride_status_counts = {
    "Pending Requests": len(requested_rides),
    "Active Rides": len(active_rides),
    "Completed Rides": len(completed_rides),
    "Cancelled Rides": len(cancelled_set)
}

# Create bar chart with custom colors
status_colors = {
    "Pending Requests": "#f1c40f",    # Yellow
    "Active Rides": "#2ecc71",        # Green
    "Completed Rides": "#3498db",     # Blue
    "Cancelled Rides": "#e74c3c"      # Red
}

fig = go.Figure(data=[
    go.Bar(
        x=list(ride_status_counts.keys()),
        y=list(ride_status_counts.values()),
        marker_color=[status_colors[status] for status in ride_status_counts.keys()],
        text=list(ride_status_counts.values()),
        textposition='auto'
    )
])

fig.update_layout(
    title="Current Ride Status Distribution",
    xaxis_title="Ride Status",
    yaxis_title="Number of Rides",
    showlegend=False,
    height=500
)

st.plotly_chart(fig, use_container_width=True)

# Speed of Service Analysis
st.header("Operational Overview: Speed of Service")

# Ensure timestamp is in datetime format
ride_events_df['timestamp'] = pd.to_datetime(ride_events_df['timestamp'])

# 1. Driver Response Time Analysis
# Filter events for RIDE_REQUESTED and DRIVER_ASSIGNED
requested_df = ride_events_df[ride_events_df['event_type'] == 'RIDE_REQUESTED'][['ride_id', 'timestamp']]
assigned_df = ride_events_df[ride_events_df['event_type'] == 'DRIVER_ASSIGNED'][['ride_id', 'timestamp']]

# Merge and calculate response time
response_df = pd.merge(requested_df, assigned_df, on='ride_id', suffixes=('_requested', '_assigned'))
response_df['response_time_sec'] = (response_df['timestamp_assigned'] - response_df['timestamp_requested']).dt.total_seconds()
avg_response_time_sec = response_df['response_time_sec'].mean()

# 2. Ride Duration Analysis
# Filter events for RIDE_STARTED and RIDE_COMPLETED
started_df = ride_events_df[ride_events_df['event_type'] == 'RIDE_STARTED'][['ride_id', 'timestamp']]
completed_df = ride_events_df[ride_events_df['event_type'] == 'RIDE_COMPLETED'][['ride_id', 'timestamp']]

# Merge and calculate duration
duration_df = pd.merge(started_df, completed_df, on='ride_id', suffixes=('_started', '_completed'))
duration_df['ride_duration_sec'] = (duration_df['timestamp_completed'] - duration_df['timestamp_started']).dt.total_seconds()
duration_df['ride_duration_min'] = duration_df['ride_duration_sec'] / 60.0
avg_ride_duration_min = duration_df['ride_duration_min'].mean()

# Display metrics in columns
col1, col2 = st.columns(2)

with col1:
    create_metric_card(
        "Average Response Time",
        f"{avg_response_time_sec:.1f} sec",
        help_text="Average time between ride request and driver assignment"
    )

with col2:
    create_metric_card(
        "Average Ride Duration",
        f"{avg_ride_duration_min:.1f} min",
        help_text="Average time between ride start and completion"
    )

# Create time series plots for response times and durations
st.subheader("Response Time Distribution")
fig_response = go.Figure(data=[
    go.Histogram(
        x=response_df['response_time_sec'],
        nbinsx=30,
        name="Response Time",
        marker_color='#2ecc71'
    )
])
fig_response.update_layout(
    xaxis_title="Response Time (seconds)",
    yaxis_title="Number of Rides",
    showlegend=False
)
st.plotly_chart(fig_response, use_container_width=True)

st.subheader("Ride Duration Distribution")
fig_duration = go.Figure(data=[
    go.Histogram(
        x=duration_df['ride_duration_min'],
        nbinsx=30,
        name="Ride Duration",
        marker_color='#3498db'
    )
])
fig_duration.update_layout(
    xaxis_title="Ride Duration (minutes)",
    yaxis_title="Number of Rides",
    showlegend=False
)
st.plotly_chart(fig_duration, use_container_width=True)

# Display summary metrics in columns
col1, col2 = st.columns(2)

with col1:
    response_percentiles = np.percentile(response_df['response_time_sec'], [25, 50, 75])
    st.markdown("**Response Time Statistics**")
    st.write(f"- 25th percentile: {response_percentiles[0]:.1f} sec")
    st.write(f"- Median: {response_percentiles[1]:.1f} sec")
    st.write(f"- 75th percentile: {response_percentiles[2]:.1f} sec")

with col2:
    duration_percentiles = np.percentile(duration_df['ride_duration_min'], [25, 50, 75])
    st.markdown("**Ride Duration Statistics**")
    st.write(f"- 25th percentile: {duration_percentiles[0]:.1f} min")
    st.write(f"- Median: {duration_percentiles[1]:.1f} min")
    st.write(f"- 75th percentile: {duration_percentiles[2]:.1f} min")

# Real-Time Trends Analysis
st.header("Real-Time Trends")
st.markdown("### Time-Based Ride Request Analysis")

# Ensure timestamp is in datetime format (if not already done)
ride_events_df['timestamp'] = pd.to_datetime(ride_events_df['timestamp'])

# 1. Prepare data: Filter for RIDE_REQUESTED events and sort by timestamp
requested_rides = ride_events_df[ride_events_df['event_type'] == 'RIDE_REQUESTED'].copy()
requested_rides = requested_rides.sort_values('timestamp')

# 2. Create hourly windows for ride requests
hourly_request_counts = requested_rides.groupby(
    requested_rides['timestamp'].dt.strftime('%Y-%m-%d %H:00')
)['ride_id'].count().reset_index()
hourly_request_counts.columns = ['hour', 'request_count']

# Create line plot for hourly requests
fig_hourly = go.Figure()

fig_hourly.add_trace(
    go.Scatter(
        x=hourly_request_counts['hour'],
        y=hourly_request_counts['request_count'],
        mode='lines+markers',
        name='Hourly Requests',
        line=dict(color='#2ecc71', width=2),
        marker=dict(size=6)
    )
)

fig_hourly.update_layout(
    title="Ride Requests per Hour",
    xaxis_title="Time",
    yaxis_title="Number of Requests",
    showlegend=True,
    height=400
)

st.plotly_chart(fig_hourly, use_container_width=True)

# Display summary statistics
col1, col2, col3 = st.columns(3)

with col1:
    create_metric_card(
        "Average Requests/Hour",
        f"{hourly_request_counts['request_count'].mean():.1f}",
        help_text="Average number of ride requests per hour"
    )

with col2:
    create_metric_card(
        "Peak Requests/Hour",
        hourly_request_counts['request_count'].max(),
        help_text="Maximum number of ride requests in a single hour"
    )

with col3:
    create_metric_card(
        "Total Hours Analyzed",
        len(hourly_request_counts),
        help_text="Total number of hours in the analysis period"
    )

# Additional temporal analysis
st.subheader("Request Patterns")

# Create two columns for additional visualizations
col1, col2 = st.columns(2)

with col1:
    # Hourly pattern throughout the day
    hourly_pattern = requested_rides.groupby(
        requested_rides['timestamp'].dt.hour
    )['ride_id'].count().reset_index()
    hourly_pattern.columns = ['hour', 'request_count']
    
    fig_daily = go.Figure()
    fig_daily.add_trace(
        go.Bar(
            x=hourly_pattern['hour'],
            y=hourly_pattern['request_count'],
            marker_color='#3498db'
        )
    )
    fig_daily.update_layout(
        title="24-Hour Request Pattern",
        xaxis_title="Hour of Day",
        yaxis_title="Total Requests",
        showlegend=False
    )
    st.plotly_chart(fig_daily, use_container_width=True)

with col2:
    # Day of week pattern
    daily_pattern = requested_rides.groupby(
        requested_rides['timestamp'].dt.day_name()
    )['ride_id'].count().reset_index()
    daily_pattern.columns = ['day', 'request_count']
    
    # Define correct day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_pattern['day'] = pd.Categorical(daily_pattern['day'], categories=day_order, ordered=True)
    daily_pattern = daily_pattern.sort_values('day')
    
    fig_weekly = go.Figure()
    fig_weekly.add_trace(
        go.Bar(
            x=daily_pattern['day'],
            y=daily_pattern['request_count'],
            marker_color='#e74c3c'
        )
    )
    fig_weekly.update_layout(
        title="Weekly Request Pattern",
        xaxis_title="Day of Week",
        yaxis_title="Total Requests",
        showlegend=False
    )
    st.plotly_chart(fig_weekly, use_container_width=True)

# Display peak hours and days
st.subheader("Peak Times Analysis")

# Find peak hours
peak_hours = hourly_pattern.nlargest(3, 'request_count')
st.write("**Peak Hours for Ride Requests:**")
for _, row in peak_hours.iterrows():
    st.write(f"- {row['hour']:02d}:00 - {(row['hour']+1):02d}:00: {row['request_count']} requests")

# Find busiest days
busiest_days = daily_pattern.nlargest(3, 'request_count')
st.write("**Busiest Days for Ride Requests:**")
for _, row in busiest_days.iterrows():
    st.write(f"- {row['day']}: {row['request_count']} requests")

# Ride Efficiency Index (REI) Analysis
st.subheader("Ride Efficiency Index (REI) Analysis")
st.markdown("""
The Ride Efficiency Index (REI) is a composite metric that measures overall ride service efficiency.
It combines completion rate, response time, and driver utilization into a single score.
""")

# Calculate components for REI
# 1. Completion Rate
total_requests = len(ride_events_df[ride_events_df['event_type'] == 'RIDE_REQUESTED'])
completed_rides = len(ride_events_df[ride_events_df['event_type'] == 'RIDE_COMPLETED'])
completion_rate = (completed_rides / total_requests) if total_requests > 0 else 0

# 2. Average Response Time Score (inverse, because lower is better)
avg_response_time = response_df['response_time_sec'].mean()
max_acceptable_response_time = 300  # 5 minutes
response_time_score = 1 - min(avg_response_time / max_acceptable_response_time, 1)

# 3. Driver Utilization Rate
active_time = duration_df['ride_duration_sec'].sum()
total_time_window = (ride_events_df['timestamp'].max() - ride_events_df['timestamp'].min()).total_seconds()
total_drivers = len(ride_events_df['driver_id'].unique())
if total_time_window > 0 and total_drivers > 0:
    utilization_rate = min(active_time / (total_time_window * total_drivers), 1)
else:
    utilization_rate = 0

# Calculate REI (weighted average of components)
weights = {
    'completion_rate': 0.4,
    'response_time': 0.3,
    'utilization': 0.3
}

rei_score = (
    weights['completion_rate'] * completion_rate +
    weights['response_time'] * response_time_score +
    weights['utilization'] * utilization_rate
) * 100  # Convert to percentage

# Display REI Score with gauge chart
fig_rei = go.Figure(go.Indicator(
    mode="gauge+number",
    value=rei_score,
    domain={'x': [0, 1], 'y': [0, 1]},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "#2ecc71"},
        'steps': [
            {'range': [0, 40], 'color': "#e74c3c"},
            {'range': [40, 70], 'color': "#f1c40f"},
            {'range': [70, 100], 'color': "#2ecc71"}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 80
        }
    }
))

fig_rei.update_layout(
    title="Ride Efficiency Index (REI)",
    height=300
)

st.plotly_chart(fig_rei, use_container_width=True)

# Display component metrics
col1, col2, col3 = st.columns(3)

with col1:
    create_metric_card(
        "Completion Rate",
        f"{completion_rate*100:.1f}%",
        help_text="Percentage of requested rides that were completed"
    )

with col2:
    create_metric_card(
        "Response Time Score",
        f"{response_time_score*100:.1f}%",
        help_text="Score based on average driver response time (higher is better)"
    )

with col3:
    create_metric_card(
        "Driver Utilization",
        f"{utilization_rate*100:.1f}%",
        help_text="Percentage of total available driver time spent on rides"
    )

# Component Trends
st.subheader("REI Component Trends")

# Ensure timestamp is in datetime format
ride_events_df['timestamp'] = pd.to_datetime(ride_events_df['timestamp'])

# Create hourly time buckets
hourly_timestamps = pd.date_range(
    start=ride_events_df['timestamp'].min(),
    end=ride_events_df['timestamp'].max(),
    freq='H'
)

# Calculate metrics for each hour
hourly_metrics = []
for start_time in hourly_timestamps[:-1]:  # Exclude the last timestamp as it's the end of the last interval
    end_time = start_time + pd.Timedelta(hours=1)
    
    # Filter events for this hour
    hour_events = ride_events_df[(ride_events_df['timestamp'] >= start_time) & 
                                (ride_events_df['timestamp'] < end_time)]
    
    # Calculate completion rate
    hour_requests = len(hour_events[hour_events['event_type'] == 'RIDE_REQUESTED'])
    hour_completions = len(hour_events[hour_events['event_type'] == 'RIDE_COMPLETED'])
    completion_rate = hour_completions / hour_requests if hour_requests > 0 else 0
    
    # Calculate response time score
    hour_response = response_df[
        (response_df['timestamp_requested'] >= start_time) & 
        (response_df['timestamp_requested'] < end_time)
    ]
    avg_response_time = hour_response['response_time_sec'].mean() if not hour_response.empty else max_acceptable_response_time
    response_score = 1 - min(avg_response_time / max_acceptable_response_time, 1)
    
    # Calculate utilization rate
    hour_duration = duration_df[
        (duration_df['timestamp_started'] >= start_time) & 
        (duration_df['timestamp_started'] < end_time)
    ]
    active_time = hour_duration['ride_duration_sec'].sum() if not hour_duration.empty else 0
    hour_drivers = len(hour_events['driver_id'].unique())
    utilization = min(active_time / (3600 * max(hour_drivers, 1)), 1)
    
    hourly_metrics.append({
        'hour': start_time,
        'completion_rate': completion_rate,
        'response_time_score': response_score,
        'utilization_rate': utilization
    })

# Convert to DataFrame
hourly_metrics_df = pd.DataFrame(hourly_metrics)

# Create the plot
fig_components = go.Figure()

fig_components.add_trace(go.Scatter(
    x=hourly_metrics_df['hour'],
    y=hourly_metrics_df['completion_rate'] * 100,
    name='Completion Rate',
    line=dict(color='#2ecc71', width=2)
))

fig_components.add_trace(go.Scatter(
    x=hourly_metrics_df['hour'],
    y=hourly_metrics_df['response_time_score'] * 100,
    name='Response Time Score',
    line=dict(color='#3498db', width=2)
))

fig_components.add_trace(go.Scatter(
    x=hourly_metrics_df['hour'],
    y=hourly_metrics_df['utilization_rate'] * 100,
    name='Utilization Rate',
    line=dict(color='#e74c3c', width=2)
))

fig_components.update_layout(
    title="Hourly REI Component Trends",
    xaxis_title="Time",
    yaxis_title="Score (%)",
    height=400,
    showlegend=True,
    hovermode='x unified'
)

st.plotly_chart(fig_components, use_container_width=True)

# Insights
st.markdown("### Key Insights")
st.markdown("""
- The REI provides a holistic view of service efficiency by combining multiple performance indicators
- Components are weighted based on their relative importance to overall service quality
- Trends help identify periods of high and low efficiency for targeted improvements
""")

# Recommendations based on REI
st.markdown("### Recommendations")
if rei_score < 60:
    st.error("⚠️ Low REI Score - Priority Areas for Improvement:")
    if completion_rate < 0.7:
        st.write("- Focus on increasing ride completion rate through better matching algorithms")
    if response_time_score < 0.7:
        st.write("- Improve driver response time through better dispatch system")
    if utilization_rate < 0.5:
        st.write("- Optimize driver allocation to increase utilization")
elif rei_score < 80:
    st.warning("🔄 Moderate REI Score - Areas for Optimization:")
    st.write("- Fine-tune dispatch algorithms")
    st.write("- Consider dynamic pricing during peak hours")
    st.write("- Implement driver incentive programs")
else:
    st.success("✅ High REI Score - Maintenance Recommendations:")
    st.write("- Monitor for sustained performance")
    st.write("- Consider expanding service area")
    st.write("- Share best practices across regions")

# Final Values Display
st.header("Final Values Summary")

# Create three columns for the component scores
final_col1, final_col2, final_col3, final_col4 = st.columns(4)

with final_col1:
    st.metric(
        "Final REI Score",
        f"{rei_score:.1f}%",
        help="Overall Ride Efficiency Index"
    )

with final_col2:
    st.metric(
        "Completion Component",
        f"{(weights['completion_rate'] * completion_rate * 100):.1f}%",
        help="Weighted completion rate contribution"
    )

with final_col3:
    st.metric(
        "Response Time Component",
        f"{(weights['response_time'] * response_time_score * 100):.1f}%",
        help="Weighted response time contribution"
    )

with final_col4:
    st.metric(
        "Utilization Component",
        f"{(weights['utilization'] * utilization_rate * 100):.1f}%",
        help="Weighted utilization rate contribution"
    )

# Display the calculation breakdown
st.markdown("### REI Score Calculation Breakdown")
st.markdown(f"""
- **Completion Rate**: {completion_rate*100:.1f}% × {weights['completion_rate']*100}% weight = {(weights['completion_rate'] * completion_rate * 100):.1f}%
- **Response Time Score**: {response_time_score*100:.1f}% × {weights['response_time']*100}% weight = {(weights['response_time'] * response_time_score * 100):.1f}%
- **Utilization Rate**: {utilization_rate*100:.1f}% × {weights['utilization']*100}% weight = {(weights['utilization'] * utilization_rate * 100):.1f}%
- **Final REI Score**: {rei_score:.1f}%
""")

# -----------------------------------------------------------------------------
# Driver Performance Analysis
# -----------------------------------------------------------------------------
st.header("Driver Performance Analysis")

# 1. Driver Categorization Based on User Rating
# Compute average rating per driver using the 'user_to_driver_rating' column
# Only consider rows where a rating is available
rating_rides = ride_events_df[ride_events_df['user_to_driver_rating'].notna()]
avg_driver_rating = rating_rides.groupby('driver_id')['user_to_driver_rating'].mean()

def categorize_rating(rating):
    if rating >= 4.5:
        return 'Gold'
    elif rating >= 4.0:
        return 'Silver'
    else:
        return 'Bronze'

# Apply categorization and create a DataFrame with driver categories
driver_category = avg_driver_rating.apply(categorize_rating).rename('driver_category')
driver_category_df = driver_category.reset_index()

# 2. Average Driver Response Time (in seconds)
# Convert timestamp to datetime
ride_events_df['event_time'] = pd.to_datetime(ride_events_df['timestamp'], unit='ms')

# Filter events for RIDE_REQUESTED and DRIVER_ASSIGNED
requested_df = ride_events_df[ride_events_df['event_type'] == 'RIDE_REQUESTED'][['ride_id', 'event_time']]
assigned_df = ride_events_df[ride_events_df['event_type'] == 'DRIVER_ASSIGNED'][['ride_id', 'event_time', 'driver_id']]

# Merge and compute response time
df_response = pd.merge(requested_df, assigned_df, on='ride_id', suffixes=('_requested', '_assigned'))
df_response['response_time_sec'] = (df_response['event_time_assigned'] - df_response['event_time_requested']).dt.total_seconds()
avg_response_per_driver = df_response.groupby('driver_id')['response_time_sec'].mean().reset_index()
avg_response_per_driver.rename(columns={'response_time_sec': 'avg_response_time_sec'}, inplace=True)

# 3. Total Number of Completed Rides per Driver
completed_by_driver = ride_events_df[ride_events_df['event_type'] == 'RIDE_COMPLETED'].groupby('driver_id')['ride_id'].nunique().reset_index()
completed_by_driver.rename(columns={'ride_id': 'completed_rides'}, inplace=True)

# 4. Average Ride Duration per Driver (in minutes)
# Filter rides for start and completion events
started_df = ride_events_df[ride_events_df['event_type'] == 'RIDE_STARTED'][['ride_id', 'event_time', 'driver_id']]
completed_df = ride_events_df[ride_events_df['event_type'] == 'RIDE_COMPLETED'][['ride_id', 'event_time']]

# Merge and compute duration
df_duration = pd.merge(started_df, completed_df, on='ride_id', suffixes=('_started', '_completed'))
df_duration['ride_duration_sec'] = (df_duration['event_time_completed'] - df_duration['event_time_started']).dt.total_seconds()
df_duration['ride_duration_min'] = df_duration['ride_duration_sec'] / 60.0
avg_duration_per_driver = df_duration.groupby('driver_id')['ride_duration_min'].mean().reset_index()
avg_duration_per_driver.rename(columns={'ride_duration_min': 'avg_ride_duration_min'}, inplace=True)

# 5. Merge Driver Metrics with Driver Category
driver_metrics = driver_category_df.merge(avg_response_per_driver, on='driver_id', how='left') \
                                 .merge(completed_by_driver, on='driver_id', how='left') \
                                 .merge(avg_duration_per_driver, on='driver_id', how='left')

# 6. Aggregate Metrics by Driver Category
category_summary = driver_metrics.groupby('driver_category').agg(
    avg_response_time_sec=('avg_response_time_sec', 'mean'),
    avg_completed_rides=('completed_rides', 'mean'),
    avg_ride_duration_min=('avg_ride_duration_min', 'mean'),
    driver_count=('driver_id', 'nunique')
).reset_index()

# Display summary metrics
st.markdown("### Summary by Driver Category")
for _, row in category_summary.iterrows():
    st.write(f"**Category: {row['driver_category']}**")
    st.write(f"- Average Response Time (sec): {row['avg_response_time_sec']:.2f}")
    st.write(f"- Average Completed Rides: {row['avg_completed_rides']:.2f}")
    st.write(f"- Average Ride Duration (min): {row['avg_ride_duration_min']:.2f}")
    st.write(f"- Total Drivers: {row['driver_count']}")
    st.write("------------------------------------------")

# 7. Visualize Driver Categories Distribution
# Count drivers in each category
driver_category_counts = driver_category_df['driver_category'].value_counts()

# Create pie chart for category distribution
fig = go.Figure(data=[
    go.Pie(
        labels=driver_category_counts.index,
        values=driver_category_counts.values,
        hole=0.4,
        marker_colors=['gold', 'silver', 'saddlebrown']
    )
])

fig.update_layout(
    title="Driver Categories Distribution",
    showlegend=True,
    height=400
)

st.plotly_chart(fig, use_container_width=True)
