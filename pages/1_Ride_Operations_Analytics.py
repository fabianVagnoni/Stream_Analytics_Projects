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

# @st.cache_data(ttl=3600) # Removed caching to allow refresh
def load_special_events_data():
    """Load special events data from JSON file"""
    try:
        # Try loading from local file
        special_events_df = load_local_data("data/special_events.json")
        
        if special_events_df is None:
            st.warning("Failed to load special events data from local storage.")
            return None
            
        return special_events_df
    except Exception as e:
        st.warning(f"Error loading special events data: {str(e)}")
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
col1, col2, col3 = st.columns(3)

with col1:
    create_metric_card(
        "Cancellation Rate",
        f"{kpis['cancellation_rate']:.1f}%",
        help_text="Percentage of rides that were cancelled"
    )

with col2:
    create_metric_card(
        "Acceptance Ratio",
        f"{kpis['acceptance_ratio']:.1f}%",
        help_text="Percentage of ride requests that were accepted"
    )

with col3:
    create_metric_card(
        "Composite Customer Satisfaction Score",
        f"{kpis['composite_satisfaction_score']:.1f}%",
        help_text="A weighted score combining customer ratings, ride duration efficiency, and cancellation rate"
    )

# Ride Status Count Analysis
st.header("Operational Overview")

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

# Real-Time Trends Analysis
st.header("Real-Time Trends")
st.markdown("### Time-Based Ride Request Analysis")

# -----------------------------------------------------------------------------
# Matplotlib-based Time Series Plot with Events
# -----------------------------------------------------------------------------
# Ensure timestamp is in datetime format (if not already done)
ride_events_df['timestamp'] = pd.to_datetime(ride_events_df['timestamp'])

# Load special events data
special_events_df = load_special_events_data()

# Prepare data for Matplotlib visualization
requested_rides_matplotlib = ride_events_df[ride_events_df['event_type'] == 'RIDE_REQUESTED'].copy()
requested_rides_matplotlib['event_time'] = pd.to_datetime(requested_rides_matplotlib['timestamp'])
requested_rides_matplotlib.set_index('event_time', inplace=True)
rides_per_hour_matplotlib = requested_rides_matplotlib['ride_id'].resample('H').nunique()

# Convert special event times to datetime
if special_events_df is not None:
    # Use ISO 8601 format for parsing timestamps and handle errors
    try:
        special_events_df['event_start_dt'] = pd.to_datetime(special_events_df['event_start'], format='ISO8601', errors='coerce')
        if 'event_end' in special_events_df.columns:
            special_events_df['event_end_dt'] = pd.to_datetime(special_events_df['event_end'], format='ISO8601', errors='coerce')
        
        # Filter out rows with invalid dates (NaT values)
        special_events_df = special_events_df.dropna(subset=['event_start_dt'])
        
        # Check if we have any valid events left
        if len(special_events_df) == 0:
            st.warning("No valid event timestamps found in special events data.")
            special_events_df = None
    except Exception as e:
        st.warning(f"Error processing special events timestamps: {str(e)}")
        special_events_df = None

# Create Matplotlib figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure

# Create a figure using Matplotlib
fig = Figure(figsize=(12, 6))
ax = fig.add_subplot(111)

# Plot ride requests per hour
ax.plot(rides_per_hour_matplotlib.index, rides_per_hour_matplotlib.values, 
        marker='o', linestyle='-', color='#3498db', label='Rides per Hour')

# Add vertical lines for special events
if special_events_df is not None:
    max_y = rides_per_hour_matplotlib.max() * 0.95
    
    for _, event in special_events_df.iterrows():
        # Add start time line
        ax.axvline(event['event_start_dt'], color='red', linestyle='--', linewidth=1)
        
        # Add end time line if available
        if 'event_end_dt' in event:
            ax.axvline(event['event_end_dt'], color='red', linestyle='--', linewidth=1)
        
        # Add text label for event
        event_name = event.get('name', event['type'])
        ax.text(event['event_start_dt'], max_y, event_name,
                rotation=90, verticalalignment='top', horizontalalignment='right', 
                color='red', fontsize=8)

# Format x-axis to show dates and times clearly
ax.set_xlabel('Time')
ax.set_ylabel('Number of Ride Requests')
ax.set_title('Hourly Ride Requests with Event Boundaries')
ax.legend()

# Set up date formatting for x-axis
ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))  # Show tick every 4 hours
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
fig.autofmt_xdate()  # Auto-format dates to prevent overlap

# Set the x-axis limits to focus on the first few days of January 2025
ax.set_xlim(pd.Timestamp('2025-01-01'), pd.Timestamp('2025-01-02'))

# Add grid for better readability
ax.grid(True, linestyle='--', alpha=0.7)

# Display the Matplotlib figure in Streamlit
st.pyplot(fig)

# Original Plotly visualization (hourly ride requests)
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
        line=dict(color='#3498db', width=2),
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



# -----------------------------------------------------------------------------
# Speed of Service Calculations (for REI Analysis)
# -----------------------------------------------------------------------------
# Calculate response time metrics
requested_df_sos = ride_events_df[ride_events_df['event_type'] == 'RIDE_REQUESTED'][['ride_id', 'timestamp']]
assigned_df_sos = ride_events_df[ride_events_df['event_type'] == 'DRIVER_ASSIGNED'][['ride_id', 'timestamp']]

# Merge and calculate response time
response_df_sos = pd.merge(requested_df_sos, assigned_df_sos, on='ride_id', suffixes=('_requested', '_assigned'))
response_df_sos['response_time_sec'] = (pd.to_datetime(response_df_sos['timestamp_assigned']) - 
                                      pd.to_datetime(response_df_sos['timestamp_requested'])).dt.total_seconds()
avg_response_time_sec = response_df_sos['response_time_sec'].mean()

# Calculate ride duration metrics
started_df_sos = ride_events_df[ride_events_df['event_type'] == 'RIDE_STARTED'][['ride_id', 'timestamp']]
completed_df_sos = ride_events_df[ride_events_df['event_type'] == 'RIDE_COMPLETED'][['ride_id', 'timestamp']]

# Merge and calculate duration
duration_df_sos = pd.merge(started_df_sos, completed_df_sos, on='ride_id', suffixes=('_started', '_completed'))
duration_df_sos['ride_duration_sec'] = (pd.to_datetime(duration_df_sos['timestamp_completed']) - 
                                      pd.to_datetime(duration_df_sos['timestamp_started'])).dt.total_seconds()
duration_df_sos['ride_duration_min'] = duration_df_sos['ride_duration_sec'] / 60.0
avg_ride_duration_min = duration_df_sos['ride_duration_min'].mean()




# -----------------------------------------------------------------------------
# Driver Performance Analysis
# -----------------------------------------------------------------------------
st.header("Our Drivers")

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

# 7. Visualize Driver Categories Distribution First
# Count drivers in each category
driver_category_counts = driver_category_df['driver_category'].value_counts()

# Create color mapping that ensures each category gets its corresponding color
category_colors = {
    'Gold': 'gold',
    'Silver': 'silver',
    'Bronze': 'saddlebrown'
}

# Order the categories and create corresponding color list
ordered_categories = ['Gold', 'Silver', 'Bronze']
ordered_counts = [driver_category_counts.get(cat, 0) for cat in ordered_categories]
ordered_colors = [category_colors[cat] for cat in ordered_categories]

# Create pie chart for category distribution
fig = go.Figure(data=[
    go.Pie(
        labels=ordered_categories,
        values=ordered_counts,
        hole=0.4,
        marker_colors=ordered_colors
    )
])

fig.update_layout(
    title="Driver Categories Distribution",
    showlegend=True,
    height=400
)

st.plotly_chart(fig, use_container_width=True)

# Display summary metrics after the pie chart
st.markdown("### Summary by Driver Category")
for _, row in category_summary.iterrows():
    st.write(f"**Category: {row['driver_category']}**")
    st.write(f"- Average Response Time (sec): {row['avg_response_time_sec']:.2f}")
    st.write(f"- Average Completed Rides: {row['avg_completed_rides']:.2f}")
    st.write(f"- Average Ride Duration (min): {row['avg_ride_duration_min']:.2f}")
    st.write(f"- Total Drivers: {row['driver_count']}")
    st.write("------------------------------------------")

# -----------------------------------------------------------------------------
# Speed of Service Display
# -----------------------------------------------------------------------------
st.header("Speed of Service")

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

# -----------------------------------------------------------------------------
# REI Analysis from Basic Notebook
# -----------------------------------------------------------------------------
st.header("Ride Efficiency Index (REI)")
st.markdown("""
This analysis evaluates the efficiency of completed rides by comparing actual versus estimated durations.
- Ratio > 1.0: Rides taking longer than estimated
- Ratio < 1.0: Rides completing faster than estimated
- Ratio = 1.0: Perfect estimation accuracy
""")

# Preprocessing: filter completed rides
completed_rides = ride_events_df[ride_events_df['event_type'] == 'RIDE_COMPLETED'].copy()

# Make sure we remove cases where estimated_duration_minutes might be zero (avoid division by zero) and NaN
completed_rides = completed_rides[completed_rides['estimated_duration_minutes'].notna()]
completed_rides = completed_rides[completed_rides['estimated_duration_minutes'] > 0]

# Compute the Ride Efficiency Ratio (REI) for each completed ride
# REI is defined as the ratio of actual to estimated duration
completed_rides['efficiency_ratio'] = completed_rides['actual_duration_minutes'] / completed_rides['estimated_duration_minutes']

# Set event_time for time-based operations
completed_rides['event_time'] = pd.to_datetime(completed_rides['timestamp'])

# Set event_time as index and resample into 1-minute buckets
completed_rides_indexed = completed_rides.set_index('event_time')

# Resample to 1-minute intervals. For each minute, compute the mean efficiency ratio
efficiency_per_minute = completed_rides_indexed['efficiency_ratio'].resample('1T').mean()

# Create a sliding window: Rolling average efficiency ratio over a 15-minute window
rolling_efficiency = efficiency_per_minute.rolling(window=15, min_periods=1).mean()

# Plot the Ride Efficiency Index over time
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=rolling_efficiency.index,
    y=rolling_efficiency.values,
    mode='lines+markers',
    name='15-min Rolling Avg REI',
    line=dict(color='blue', width=2),
    marker=dict(size=5)
))

fig.add_shape(
    type="line",
    x0=rolling_efficiency.index.min(),
    y0=1.0,
    x1=rolling_efficiency.index.max(),
    y1=1.0,
    line=dict(color="red", width=2, dash="dash"),
)

fig.add_annotation(
    x=rolling_efficiency.index.min(),
    y=1.0,
    text="Ideal Efficiency (Ratio=1)",
    showarrow=False,
    yshift=10,
    font=dict(color="red")
)

fig.update_layout(
    title="Real-Time Ride Efficiency Index (REI)",
    xaxis_title="Time",
    yaxis_title="Ride Efficiency Ratio (Actual / Estimated Duration)",
    height=500,
    showlegend=True,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Display statistics
col1, col2 = st.columns(2)

with col1:
    avg_efficiency = completed_rides['efficiency_ratio'].mean()
    create_metric_card(
        "Average Efficiency Ratio",
        f"{avg_efficiency:.2f}",
        help_text="Average of actual/estimated duration (lower is better)"
    )

with col2:
    ideal_rides = len(completed_rides[completed_rides['efficiency_ratio'] <= 1.0])
    total_rides = len(completed_rides)
    ideal_percentage = (ideal_rides / total_rides) * 100 if total_rides > 0 else 0
    create_metric_card(
        "Rides Faster Than Expected",
        f"{ideal_percentage:.1f}%",
        help_text="Percentage of rides with efficiency ratio ≤ 1.0"
    )
