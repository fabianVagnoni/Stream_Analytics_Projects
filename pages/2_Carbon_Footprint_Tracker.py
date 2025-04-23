"""
Carbon Footprint Tracker

This page provides analytics and insights about carbon emissions from rides,
including emissions metrics, trees needed for offset, and emission patterns.
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.data_loader import load_data, load_data_from_azure, load_local_data
from utils.visualizations import create_metric_card, plot_time_series, plot_heatmap, plot_bar_chart, plot_gauge
from utils.data_processing import calculate_emissions
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import datetime # Import datetime for date comparisons
from streamlit_autorefresh import st_autorefresh

# Set page configuration
st.set_page_config(
    page_title="Carbon Footprint Tracker",
    page_icon="🌱",
    layout="wide"
)

# --- Auto-Refresh --- 
st_autorefresh(interval=30000, key="carbon_refresher")

# Page title
st.title("🌱 Carbon Footprint Tracker")
st.markdown("### Use Case 2: Carbon Footprint Analysis and Offset Tracking")

# --- Function Definitions ---

@st.cache_data(ttl=3600)
def load_ride_data():
    """Load ride data from Azure Blob Storage or local file"""
    try:
        ride_events_df = load_data_from_azure("rides/*.snappy.parquet")
        if ride_events_df is None:
            ride_events_df = load_local_data("data/ride_events.json")
        if ride_events_df is None:
             st.error("Failed to load ride events data.")
             return None
        return ride_events_df
    except Exception as e:
        st.error(f"Error loading ride events data: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def load_drivers_data():
    """Load drivers data from Azure Blob Storage or local file"""
    try:
        drivers_df = load_data_from_azure("drivers/*.snappy.parquet")
        if drivers_df is None:
            drivers_df = load_local_data("data/drivers_static.json")
        if drivers_df is None:
             st.warning("No drivers data found. Emissions based on distance only.")
             # Return empty df with expected columns if none found
             return pd.DataFrame(columns=['driver_id', 'vehicle'])
        return drivers_df
    except Exception as e:
        st.warning(f"Error loading drivers data: {str(e)}")
        return pd.DataFrame(columns=['driver_id', 'vehicle'])

# @st.cache_data(ttl=3600) # Removed cache for auto-refresh
def get_preprocessed_data():
    """Loads and preprocesses the base ride and driver data."""
    ride_events_df = load_ride_data()
    drivers_static_df = load_drivers_data()

    if ride_events_df is None:
        return None # Return None if loading fails

    processed_df = ride_events_df.copy()

    # --- Emissions Calculation Logic ---
    # Ensure 'total_emissions' column exists after this block
    default_emission_val = 180 # g CO2/km (fallback)
    processed_df['total_emissions'] = 0 # Initialize

    # Check if necessary columns exist for calculation
    driver_data_available = (drivers_static_df is not None and
                             not drivers_static_df.empty and
                             'driver_id' in drivers_static_df.columns and
                             'vehicle' in drivers_static_df.columns)
    rides_data_available = ('driver_id' in processed_df.columns and
                            'distance_km' in processed_df.columns)

    if driver_data_available and rides_data_available:
        try:
            # Extract vehicle type (e.g., 'Electric', 'Hybrid', 'Gas')
            drivers_static_df['vehicle_emissions_type'] = drivers_static_df['vehicle'].str.extract(r'\((.*?)\)$', expand=False).fillna('Unknown')

            # Map type to emission value (g CO2/km)
            emissions_mapping = {'Electric': 124, 'Hybrid': 162, 'Gas': 218}
            drivers_static_df['emissions_value'] = drivers_static_df['vehicle_emissions_type'].map(emissions_mapping).fillna(default_emission_val)

            # Merge to get emissions value per ride
            processed_df = processed_df.merge(
                drivers_static_df[['driver_id', 'emissions_value', 'vehicle_emissions_type']],
                on='driver_id',
                how='left'
            )

            # Calculate total emissions (g CO2)
            # Ensure distance_km is numeric, fill missing emissions_value with default
            processed_df['distance_km'] = pd.to_numeric(processed_df['distance_km'], errors='coerce')
            processed_df['emissions_value'] = processed_df['emissions_value'].fillna(default_emission_val)
            processed_df['total_emissions'] = processed_df['distance_km'] * processed_df['emissions_value']

        except Exception as e:
            st.warning(f"Error during detailed emissions calculation: {e}. Using default.")
            # Fallback calculation if merge/extraction fails
            processed_df['distance_km'] = pd.to_numeric(processed_df['distance_km'], errors='coerce')
            processed_df['total_emissions'] = processed_df['distance_km'].fillna(0) * default_emission_val
    elif 'distance_km' in processed_df.columns:
         st.warning("Driver/Vehicle data missing or incomplete. Calculating emissions based on distance * default rate.")
         processed_df['distance_km'] = pd.to_numeric(processed_df['distance_km'], errors='coerce')
         processed_df['total_emissions'] = processed_df['distance_km'].fillna(0) * default_emission_val
    else:
         st.warning("Cannot calculate emissions (missing distance/driver data). Setting to 0.")
         processed_df['total_emissions'] = 0
         # Ensure vehicle_emissions_type exists even if calculation failed
         if 'vehicle_emissions_type' not in processed_df.columns:
              processed_df['vehicle_emissions_type'] = 'Unknown'

    # Filter for completed rides and ensure timestamp is datetime
    if 'event_type' in processed_df.columns:
        completed_data = processed_df[processed_df['event_type'] == 'RIDE_COMPLETED'].copy()
    else:
        completed_data = processed_df.copy() # Assume all relevant if no event_type

    if 'timestamp' in completed_data.columns:
        try:
            completed_data['timestamp'] = pd.to_datetime(completed_data['timestamp'], errors='coerce')
            completed_data.dropna(subset=['timestamp'], inplace=True) # Drop rows where conversion failed
        except Exception as e:
             st.warning(f"Could not process timestamp column: {e}")
             # If timestamp is critical and fails, maybe return None or empty df?
             return pd.DataFrame() # Return empty if timestamp fails

    # Ensure essential columns for filtering/display exist, add defaults if necessary
    for col in ['traffic_level', 'estimated_delay_minutes', 'pickup_latitude', 'pickup_longitude']:
         if col not in completed_data.columns:
              completed_data[col] = None # Add column with None if missing

    return completed_data

def calculate_emission_metrics(df):
    """Calculate key emissions metrics from ride data"""
    metrics = {
        'total_emissions': 0, 'avg_emissions_per_ride': 0, 'trees_needed': 0,
        'peak_emissions': 0, 'offpeak_emissions': 0, 'peak_vs_offpeak': 0,
        'spike_sessions': 0
    }
    if df.empty or 'total_emissions' not in df.columns:
        return metrics

    df_copy = df.copy()
    df_copy['total_emissions'] = pd.to_numeric(df_copy['total_emissions'], errors='coerce').fillna(0)

    metrics['total_emissions'] = df_copy['total_emissions'].sum()
    metrics['avg_emissions_per_ride'] = df_copy['total_emissions'].mean()

    kg_per_tree = 25
    g_per_tree = kg_per_tree * 1000
    metrics['trees_needed'] = (metrics['total_emissions'] / g_per_tree) if g_per_tree > 0 else 0

    if 'timestamp' in df_copy.columns:
        try:
            df_copy['hour'] = pd.to_datetime(df_copy['timestamp'], errors='coerce').dt.hour
            df_copy.dropna(subset=['hour'], inplace=True) # Drop rows with invalid timestamps
            df_copy['hour'] = df_copy['hour'].astype(int)

            peak_hours_mask = df_copy['hour'].isin([7, 8, 9, 17, 18, 19])
            peak_rides = df_copy[peak_hours_mask]
            offpeak_rides = df_copy[~peak_hours_mask]

            metrics['peak_emissions'] = peak_rides['total_emissions'].mean() if not peak_rides.empty else 0
            metrics['offpeak_emissions'] = offpeak_rides['total_emissions'].mean() if not offpeak_rides.empty else 0

            if metrics['offpeak_emissions'] > 0:
                 metrics['peak_vs_offpeak'] = ((metrics['peak_emissions'] / metrics['offpeak_emissions']) - 1) * 100
            else:
                 metrics['peak_vs_offpeak'] = 0
        except Exception as e:
            st.warning(f"Could not calculate peak/offpeak emissions: {e}")

    percentile_95 = df_copy['total_emissions'].quantile(0.95)
    metrics['spike_sessions'] = df_copy[df_copy['total_emissions'] > percentile_95].shape[0]

    # Fill NaN results with 0
    for key, value in metrics.items():
        if pd.isna(value):
            metrics[key] = 0

    return metrics

def generate_emission_heatmap(ride_data):
    """Generate a folium heatmap of emissions based on pickup locations"""
    if ride_data is None or ride_data.empty:
        st.info("No data available for heatmap with current filters.")
        return create_sample_heatmap() # Show sample if no data

    required_cols = ['pickup_latitude', 'pickup_longitude', 'total_emissions']
    if not all(col in ride_data.columns for col in required_cols):
        st.warning("Required columns (lat, lon, emissions) missing for heatmap.")
        return create_sample_heatmap()

    map_data = ride_data[required_cols].copy().dropna()
    map_data['pickup_latitude'] = pd.to_numeric(map_data['pickup_latitude'], errors='coerce')
    map_data['pickup_longitude'] = pd.to_numeric(map_data['pickup_longitude'], errors='coerce')
    map_data['total_emissions'] = pd.to_numeric(map_data['total_emissions'], errors='coerce')
    map_data.dropna(inplace=True) # Drop rows where conversion failed

    if map_data.empty:
        st.info("No valid location/emission data points for heatmap.")
        return create_sample_heatmap()

    heat_data = map_data[['pickup_latitude', 'pickup_longitude', 'total_emissions']].values.tolist()

    # Create map centered on Madrid
    m = folium.Map(location=[40.4168, -3.7038], zoom_start=12, tiles="CartoDB positron")
    HeatMap(
        data=heat_data, radius=15, blur=10, min_opacity=0.5,
        gradient={'0.4': 'blue', '0.65': 'lime', '0.8': 'yellow', '1': 'red'}
    ).add_to(m)
    return m

def create_sample_heatmap():
    """Create a sample heatmap for demonstration purposes"""
    import random
    madrid_lat, madrid_lng = 40.4168, -3.7038
    sample_points = []
    for _ in range(300): sample_points.append([madrid_lat + random.uniform(-0.03, 0.03), madrid_lng + random.uniform(-0.03, 0.03), random.uniform(10, 100)])
    # ... (add other sample point generation loops if desired) ...
    m = folium.Map(location=[madrid_lat, madrid_lng], zoom_start=12, tiles="CartoDB positron")
    HeatMap(data=sample_points, radius=15, blur=10, min_opacity=0.5, gradient={'0.4': 'blue', '0.65': 'lime', '0.8': 'yellow', '1': 'red'}).add_to(m)
    return m

def safe_cut(df, col_name, bins, labels):
    """Safely apply pd.cut creating a 'delay_category' column"""
    if df.empty or col_name not in df.columns:
        return df # Return unchanged if no data or column

    df_copy = df.copy()
    # Ensure col_name is numeric before cutting
    df_copy[col_name] = pd.to_numeric(df_copy[col_name], errors='coerce')
    # Perform cut, handle potential NaNs resulting from coercion or being outside bins
    df_copy['delay_category'] = pd.cut(
        df_copy[col_name],
        bins=bins,
        labels=labels,
        right=False # Define how bins are closed, adjust as needed
    )
    # Optionally fill NaNs in delay_category if needed, e.g., with an 'Unknown' category
    # df_copy['delay_category'] = df_copy['delay_category'].cat.add_categories('Unknown').fillna('Unknown')
    return df_copy

def filter_data(df, date_range_val, vehicle_type_val, traffic_level_val, area_val):
    """Apply filters from the dashboard controls to the dataset"""
    if df is None: return pd.DataFrame() # Handle None input
    filtered_df = df.copy()
    original_rows = len(filtered_df)

    # 1. Date filter
    if 'timestamp' in filtered_df.columns and len(date_range_val) == 2:
        try:
            # Ensure timestamp is datetime type
            if not pd.api.types.is_datetime64_any_dtype(filtered_df['timestamp']):
                filtered_df['timestamp'] = pd.to_datetime(filtered_df['timestamp'], errors='coerce')
            filtered_df.dropna(subset=['timestamp'], inplace=True) # Drop if conversion failed

            start_date, end_date = date_range_val
            # Compare date objects directly
            filtered_df = filtered_df[(filtered_df['timestamp'].dt.date >= start_date) & (filtered_df['timestamp'].dt.date <= end_date)]
        except Exception as e:
            st.warning(f"Date filter failed: {e}")

    # 2. Vehicle type filter
    if vehicle_type_val != 'All' and 'vehicle_emissions_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['vehicle_emissions_type'].fillna('Unknown') == vehicle_type_val]

    # 3. Traffic level filter
    if traffic_level_val != 'All' and 'traffic_level' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['traffic_level'].fillna('Unknown') == traffic_level_val]

    # 4. Area filter
    if area_val != 'All Areas':
        lat_col, lon_col = 'pickup_latitude', 'pickup_longitude'
        if lat_col in filtered_df.columns and lon_col in filtered_df.columns:
            try:
                filtered_df[lat_col] = pd.to_numeric(filtered_df[lat_col], errors='coerce')
                filtered_df[lon_col] = pd.to_numeric(filtered_df[lon_col], errors='coerce')
                filtered_df.dropna(subset=[lat_col, lon_col], inplace=True)

                areas = { # Define boundaries (inclusive)
                    'Downtown': {'lat': (40.40, 40.43), 'lon': (-3.71, -3.68)},
                    'Airport': {'lat': (40.46, 40.50), 'lon': (-3.57, -3.54)},
                    'Business District': {'lat': (40.42, 40.45), 'lon': (-3.69, -3.67)}
                    # Add more precise definitions if needed
                }
                if area_val in areas:
                     bounds = areas[area_val]
                     filtered_df = filtered_df[
                         (filtered_df[lat_col] >= bounds['lat'][0]) & (filtered_df[lat_col] <= bounds['lat'][1]) &
                         (filtered_df[lon_col] >= bounds['lon'][0]) & (filtered_df[lon_col] <= bounds['lon'][1])
                     ]
                elif area_val == 'Suburbs':
                     # Example: Define suburbs by excluding other defined areas
                     # This requires careful definition of all non-suburb areas
                     downtown_mask = (
                         (filtered_df[lat_col] >= areas['Downtown']['lat'][0]) & (filtered_df[lat_col] <= areas['Downtown']['lat'][1]) &
                         (filtered_df[lon_col] >= areas['Downtown']['lon'][0]) & (filtered_df[lon_col] <= areas['Downtown']['lon'][1])
                     )
                     airport_mask = (
                         (filtered_df[lat_col] >= areas['Airport']['lat'][0]) & (filtered_df[lat_col] <= areas['Airport']['lat'][1]) &
                         (filtered_df[lon_col] >= areas['Airport']['lon'][0]) & (filtered_df[lon_col] <= areas['Airport']['lon'][1])
                     )
                     biz_mask = (
                         (filtered_df[lat_col] >= areas['Business District']['lat'][0]) & (filtered_df[lat_col] <= areas['Business District']['lat'][1]) &
                         (filtered_df[lon_col] >= areas['Business District']['lon'][0]) & (filtered_df[lon_col] <= areas['Business District']['lon'][1])
                     )
                     filtered_df = filtered_df[~(downtown_mask | airport_mask | biz_mask)]

            except Exception as e:
                st.warning(f"Area filter failed: {e}")

    # Debugging/Info
    if filtered_df.empty and original_rows > 0 :
         st.sidebar.info("No data matches the selected filters.") # Use info

    return filtered_df

# --- Streamlit App Layout ---

# 1. Get Base Data
base_completed_rides = get_preprocessed_data()

if base_completed_rides is None or base_completed_rides.empty:
    st.error("No valid ride data available to display.")
    st.stop()

# 2. Sidebar Controls
st.sidebar.title("Dashboard Controls")
st.sidebar.header("Date Range")

# Get min/max from actual data if timestamp column exists and data is not empty
min_data_date = base_completed_rides['timestamp'].min().date() if 'timestamp' in base_completed_rides.columns and not base_completed_rides.empty else datetime.date(2023, 1, 1)
max_data_date = base_completed_rides['timestamp'].max().date() if 'timestamp' in base_completed_rides.columns and not base_completed_rides.empty else datetime.date(2023, 12, 31)

# Define hardcoded overall min/max possible dates for the widget
# These act as absolute bounds regardless of data
overall_min_date = datetime.date(2023, 1, 1)
overall_max_date = datetime.date(2025, 12, 31)

# Determine the effective min/max range for the widget input
# It must be within overall bounds AND encompass the data range
widget_min_date = min(min_data_date, overall_min_date)
widget_max_date = max(max_data_date, overall_max_date)

# Handle potential edge case where min > max after combining (shouldn't happen if overall bounds are wide)
if widget_min_date > widget_max_date:
    st.sidebar.warning("Inconsistent date ranges detected. Using default widget range.")
    widget_min_date = overall_min_date
    widget_max_date = overall_max_date

# Define your PREFERRED default start and end dates
default_start_pref = datetime.date(2023, 1, 1)
default_end_pref = datetime.date(2023, 1, 31)

# Clamp the preferred defaults to the valid widget range
final_default_start = max(widget_min_date, default_start_pref)
final_default_start = min(final_default_start, widget_max_date) # Ensure start doesn't exceed max

final_default_end = min(widget_max_date, default_end_pref)
final_default_end = max(final_default_end, widget_min_date) # Ensure end doesn't precede min

# Ensure start is not after end AFTER clamping
if final_default_start > final_default_end:
    # If the preferred range is completely outside the valid range, default to the end date
    # Or alternatively, default to the start date: final_default_end = final_default_start
    final_default_start = final_default_end

# Now use the calculated final defaults and widget range
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(final_default_start, final_default_end),
    min_value=widget_min_date,
    max_value=widget_max_date
)

st.sidebar.header("Vehicle Type")
vehicle_types = ['All'] + sorted(base_completed_rides['vehicle_emissions_type'].dropna().unique().tolist()) if 'vehicle_emissions_type' in base_completed_rides.columns else ['All']
selected_vehicle_type = st.sidebar.selectbox("Select Vehicle Type", options=vehicle_types, index=0)

st.sidebar.header("Traffic Level")
traffic_levels = ['All'] + sorted(base_completed_rides['traffic_level'].dropna().unique().tolist()) if 'traffic_level' in base_completed_rides.columns else ['All']
selected_traffic_level = st.sidebar.selectbox("Select Traffic Level", options=traffic_levels, index=0)

st.sidebar.header("Geographical Area")
geographical_areas = ['All Areas', 'Downtown', 'Suburbs', 'Airport', 'Business District']
selected_area = st.sidebar.selectbox("Select Area", options=geographical_areas, index=0)

# 3. Apply Filtering
filtered_data = filter_data(
    base_completed_rides,
    date_range,
    selected_vehicle_type,
    selected_traffic_level,
    selected_area
)

# 4. Display Active Filters
st.sidebar.markdown("---")
active_filters_list = []

# Check if date_range is a valid tuple/list of length 2 before accessing elements
is_valid_range = isinstance(date_range, (tuple, list)) and len(date_range) == 2

# Compare dates correctly ONLY if we have a valid range
if is_valid_range and date_range != (final_default_start, final_default_end):
     active_filters_list.append(f"Date: {date_range[0]} to {date_range[1]}")
elif not is_valid_range:
     # Log or warn if the date_range is not as expected (optional)
     # st.sidebar.warning("Date range input value is not a valid range.")
     pass # Silently ignore if not a valid range for filter display

if selected_vehicle_type != 'All':
     active_filters_list.append(f"Vehicle: {selected_vehicle_type}")
if selected_traffic_level != 'All':
     active_filters_list.append(f"Traffic: {selected_traffic_level}")
if selected_area != 'All Areas':
     active_filters_list.append(f"Area: {selected_area}")

if active_filters_list:
     st.sidebar.markdown("**Active Filters:**")
     for f in active_filters_list:
         st.sidebar.markdown(f"- {f}")
# else: st.sidebar.info("No filters active") # Optional

# 5. Calculate KPIs from Filtered Data
emission_metrics = calculate_emission_metrics(filtered_data)

# 6. Display KPIs
st.markdown("### Key Emission Metrics") # Add overall header
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total CO₂ Emitted (tons)", f"{emission_metrics['total_emissions'] / 1_000_000:.2f}")
with col2:
    st.metric("Trees Needed (Annual Offset)", f"{int(round(emission_metrics['trees_needed']))}")
with col3:
    st.metric("Peak Hour Emission Diff.", f"{emission_metrics['peak_vs_offpeak']:.1f}%")

# 7. Display Charts (using Filtered Data)
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Emissions by Traffic & Delay") # Use smaller header
    if 'traffic_level' in filtered_data.columns and 'estimated_delay_minutes' in filtered_data.columns:
        if not filtered_data.empty:
            try:
                categorized_data = safe_cut(
                    filtered_data, 'estimated_delay_minutes',
                    bins=[0, 10, 20, 30, 40, 50, 60, np.inf], # Add inf bin
                    labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60+']
                )
                # Ensure delay_category exists after safe_cut
                if 'delay_category' in categorized_data.columns:
                    heatmap_pivot = categorized_data.pivot_table(
                        index='delay_category', columns='traffic_level',
                        values='total_emissions', aggfunc='mean'#, observed=True # Add observed=True if pandas >= 1.5
                    ).fillna(0)

                    if not heatmap_pivot.empty:
                         fig = px.imshow(
                             heatmap_pivot, text_auto=True, aspect="auto",
                             labels=dict(x="Traffic Level", y="Delay (min)", color="Avg Emissions (gCO₂)"),
                             color_continuous_scale="Greens"
                         )
                         fig.update_layout(
                             title_text="Avg Emissions by Traffic & Delay", title_x=0.5,
                             margin=dict(t=50, l=25, r=25, b=25), height=400 # Adjust layout
                         )
                         st.plotly_chart(fig, use_container_width=True)
                    else:
                         st.info("No data to display in heatmap for current filters.")
                else:
                     st.warning("Delay category could not be created for heatmap.")
            except Exception as e:
                st.warning(f"Error creating heatmap: {str(e)}")
        else:
            st.info("No data available for heatmap with current filters.")
    else:
        st.info("Required columns missing for heatmap.")

with col2:
    st.markdown("#### Hourly Average Emissions")
    if 'timestamp' in filtered_data.columns:
        if not filtered_data.empty:
            try:
                hourly_calc_data = filtered_data.copy()
                hourly_calc_data['hour'] = hourly_calc_data['timestamp'].dt.hour
                hourly_emissions = hourly_calc_data.groupby('hour')['total_emissions'].mean().reset_index() # removed observed=True for compatibility if needed

                if not hourly_emissions.empty:
                     fig = px.line(
                         hourly_emissions, x='hour', y='total_emissions',
                         labels={'hour': 'Hour of Day', 'total_emissions': 'Avg Emissions (gCO₂)'},
                         color_discrete_sequence=['#2ca02c']
                     )
                     fig.update_layout(
                         title_text="Average Emissions per Ride by Hour", title_x=0.5,
                         margin=dict(t=50, l=25, r=25, b=25), height=400 # Adjust layout
                     )
                     st.plotly_chart(fig, use_container_width=True)
                else:
                     st.info("No data to display in hourly chart for current filters.")
            except Exception as e:
                 st.warning(f"Error creating hourly chart: {str(e)}")
        else:
            st.info("No data available for hourly chart with current filters.")
    else:
        st.info("Timestamp column missing for hourly chart.")

# Bottom charts
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Daily Emissions by Weekday")
    if 'timestamp' in filtered_data.columns:
        if not filtered_data.empty:
            try:
                daily_calc_data = filtered_data.copy()
                daily_calc_data['day_of_week'] = daily_calc_data['timestamp'].dt.day_name()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                daily_calc_data['day_of_week'] = pd.Categorical(daily_calc_data['day_of_week'], categories=day_order, ordered=True)
                daily_emissions = daily_calc_data.groupby('day_of_week', observed=False)['total_emissions'].sum().reset_index() # Set observed=False explicitly

                if not daily_emissions.empty:
                     fig = px.bar(
                         daily_emissions, x='day_of_week', y='total_emissions',
                         labels={'day_of_week': 'Day of Week', 'total_emissions': 'Total Emissions (gCO₂)'},
                         color_discrete_sequence=['#2ca02c']
                     )
                     fig.update_layout(
                         title_text="Total Emissions by Day of Week", title_x=0.5,
                         margin=dict(t=50, l=25, r=25, b=25), height=350 # Adjust layout
                     )
                     st.plotly_chart(fig, use_container_width=True)
                else:
                     st.info("No data to display in daily chart for current filters.")
            except Exception as e:
                 st.warning(f"Error creating daily chart: {str(e)}")
        else:
            st.info("No data available for daily chart with current filters.")
    else:
        st.info("Timestamp column missing for daily chart.")

with col2:
    st.markdown("#### Spike Sessions (>95th %)")
    spike_sessions = emission_metrics['spike_sessions']
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=spike_sessions,
        title={'text': "Spike Sessions Count"},
        gauge={'axis': {'range': [None, max(20, spike_sessions * 1.2)]}, # Dynamic range
               'bar': {'color': "darkgreen"},
               'steps': [{'range': [0, 10], 'color': 'lightgreen'}, {'range': [10, 20], 'color': 'lightgray'}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 10}}
    ))
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), height=350) # Adjust layout
    st.plotly_chart(fig, use_container_width=True)

# 8. Geospatial Map (using Filtered Data)
st.markdown("---")
st.markdown("### Geospatial Emission Hotspots")
emission_map = generate_emission_heatmap(filtered_data)
if emission_map:
    st_folium(emission_map, width=None, height=500, use_container_width=True) # Use container width
else:
    # Sample map shown by generate_emission_heatmap if needed
    pass # Function handles displaying sample or warnings

# 9. Download Button (using Filtered Data)
st.sidebar.header("Export Data")
download_cols = ['ride_id', 'timestamp', 'total_emissions', 'vehicle_emissions_type', 'traffic_level'] # Add more relevant columns
available_cols = [col for col in download_cols if col in filtered_data.columns]
if not filtered_data.empty and available_cols:
    try:
        csv_data = filtered_data[available_cols].to_csv(index=False).encode('utf-8')
    except Exception as e:
        st.sidebar.warning(f"Could not prepare download data: {e}")
        csv_data = "".encode('utf-8')
else:
    csv_data = "".encode('utf-8') # Empty data if no rows or columns

st.sidebar.download_button(
    label="Download Filtered Data (CSV)",
    data=csv_data,
    file_name='filtered_emissions_data.csv',
    mime='text/csv',
    disabled=(len(csv_data)==0) # Disable button if no data
)
