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

# Set page configuration
st.set_page_config(
    page_title="Carbon Footprint Tracker",
    page_icon="🌱",
    layout="wide"
)

# Page title
st.title("🌱 Carbon Footprint Tracker")
st.markdown("### Use Case 2: Carbon Footprint Analysis and Offset Tracking")

# Load data
@st.cache_data(ttl=3600)
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

@st.cache_data(ttl=3600)
def load_drivers_data():
    """Load drivers data from Azure Blob Storage or local file"""
    try:
        # Try loading from Azure first - update path to match actual data in Azure
        # Since drivers_static doesn't exist in Azure, try alternative paths
        drivers_df = load_data_from_azure("drivers/*.snappy.parquet")  # Trying "drivers" instead of "drivers_static"
        
        if drivers_df is None:
            # Fall back to local data if Azure fails
            drivers_df = load_local_data("data/drivers_static.json")
            
        if drivers_df is None:
            st.warning("Failed to load drivers data from both Azure and local storage. Using default emissions values.")
            # Create minimal empty DataFrame with required columns to avoid errors
            drivers_df = pd.DataFrame(columns=['driver_id', 'vehicle', 'vehicle_emissions_type', 'emissions_value'])
            
        return drivers_df
    except Exception as e:
        st.warning(f"Error loading drivers data: {str(e)}. Using default emissions values.")
        # Create minimal empty DataFrame with required columns to avoid errors
        return pd.DataFrame(columns=['driver_id', 'vehicle', 'vehicle_emissions_type', 'emissions_value'])

# Load and process data
with st.spinner("Loading data..."):
    ride_events_df = load_ride_data()
    drivers_static_df = load_drivers_data()
    
    if ride_events_df is None:
        st.stop()

    # Process data for emissions calculations
    if 'emissions_kg_co2' not in ride_events_df.columns:
        # Extract vehicle emissions type from the vehicle description
        if 'vehicle' in drivers_static_df.columns and not drivers_static_df.empty:
            drivers_static_df['vehicle_emissions_type'] = drivers_static_df['vehicle'].str.extract(r'\((.*?)\)$')
            
            # Create a mapping of vehicle emissions types to their respective CO2 emissions
            emissions_mapping = {
                'Electric': 124,  # g CO2/km
                'Hybrid': 162,    # g CO2/km
                'Gas': 218        # g CO2/km
            }
            
            # Default for unknown types
            default_emission = 180  # g CO2/km
            
            # Map the emissions type to the corresponding value (g CO2/km)
            drivers_static_df['emissions_value'] = drivers_static_df['vehicle_emissions_type'].map(emissions_mapping).fillna(default_emission)
            
            # Merge drivers with rides to get emissions value per ride
            if 'driver_id' in ride_events_df.columns:
                try:
                    rides_with_emissions = ride_events_df.merge(
                        drivers_static_df[['driver_id', 'emissions_value']], 
                        on='driver_id', 
                        how='left'
                    )
                    
                    # Calculate the total emissions for each ride (g CO2)
                    if 'distance_km' in rides_with_emissions.columns:
                        rides_with_emissions['total_emissions'] = rides_with_emissions['distance_km'] * rides_with_emissions['emissions_value'].fillna(default_emission)
                        ride_events_df = rides_with_emissions
                    else:
                        st.warning("Distance data missing. Using default values for emissions calculation.")
                        ride_events_df['total_emissions'] = 200  # default value
                except Exception as e:
                    st.warning(f"Error during data merging: {str(e)}. Using calculated emissions instead.")
                    ride_events_df = calculate_emissions(ride_events_df, drivers_static_df)
            else:
                st.warning("Driver ID missing in ride data. Using calculated emissions instead.")
                ride_events_df = calculate_emissions(ride_events_df, drivers_static_df)
        else:
            st.warning("Vehicle information missing. Using calculated emissions instead.")
            ride_events_df = calculate_emissions(ride_events_df, drivers_static_df)
            
    # Ensure total_emissions column exists
    if 'total_emissions' not in ride_events_df.columns:
        ride_events_df['total_emissions'] = 200  # default value

# Create a DataFrame with only completed rides for better analysis
completed_rides = ride_events_df[ride_events_df['event_type'] == 'RIDE_COMPLETED'] if 'event_type' in ride_events_df.columns else ride_events_df

# Calculate key metrics
def calculate_emission_metrics(df):
    """Calculate key emissions metrics from ride data"""
    if 'total_emissions' not in df.columns:
        return {
            'total_emissions': 0,
            'avg_emissions_per_ride': 0,
            'trees_needed': 0,
            'peak_emissions': 0,
            'offpeak_emissions': 0,
            'peak_vs_offpeak': 0,
            'spike_sessions': 0
        }
    
    # Create a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    # Total emissions in grams CO2
    total_emissions = df_copy['total_emissions'].sum()
    
    # Average emissions per ride
    avg_emissions = df_copy['total_emissions'].mean()
    
    # Calculate trees needed (assuming 25kg CO2 absorbed per mature tree per year)
    kg_per_tree = 25  # average CO₂ uptake per tree per year
    g_per_tree = kg_per_tree * 1000  # convert to grams
    trees_needed = total_emissions / g_per_tree
    
    # Identify peak hours (7-9 AM and 5-7 PM)
    if 'timestamp' in df_copy.columns:
        df_copy.loc[:, 'hour'] = pd.to_datetime(df_copy['timestamp']).dt.hour
        peak_hours = (df_copy['hour'].isin([7, 8, 9, 17, 18, 19]))
        peak_emissions = df_copy.loc[peak_hours, 'total_emissions'].mean()
        offpeak_emissions = df_copy.loc[~peak_hours, 'total_emissions'].mean()
        peak_vs_offpeak = ((peak_emissions / offpeak_emissions) - 1) * 100 if offpeak_emissions > 0 else 0
    else:
        peak_emissions = offpeak_emissions = peak_vs_offpeak = 0
    
    # Count spike sessions (rides with emissions > 95th percentile)
    percentile_95 = df_copy['total_emissions'].quantile(0.95)
    spike_sessions = df_copy[df_copy['total_emissions'] > percentile_95].shape[0]
    
    return {
        'total_emissions': total_emissions,
        'avg_emissions_per_ride': avg_emissions,
        'trees_needed': trees_needed,
        'peak_emissions': peak_emissions,
        'offpeak_emissions': offpeak_emissions,
        'peak_vs_offpeak': peak_vs_offpeak,
        'spike_sessions': spike_sessions
    }

# Calculate metrics
emission_metrics = calculate_emission_metrics(completed_rides)

# Display KPIs in a 3-column layout for the top cards
col1, col2, col3 = st.columns(3)

with col1:
    # Display total CO2 emissions
    st.markdown("### # tons of CO2 emitted")
    tons_co2 = emission_metrics['total_emissions'] / 1000000  # Convert grams to tons
    st.markdown(f"<h1 style='text-align: center; color: #0f7e9b;'>{tons_co2:.2f}</h1>", unsafe_allow_html=True)
    
with col2:
    # Display trees needed
    st.markdown("### Mature trees planted to absorb emissions")
    trees_needed = emission_metrics['trees_needed']
    st.markdown(f"<h1 style='text-align: center; color: #0f7e9b;'>{int(round(trees_needed))}</h1>", unsafe_allow_html=True)
    
with col3:
    # Display emissions difference in peak hours
    st.markdown("### % more emissions in peak hours")
    peak_vs_offpeak = emission_metrics['peak_vs_offpeak']
    st.markdown(f"<h1 style='text-align: center; color: #0f7e9b;'>{peak_vs_offpeak:.1f}%</h1>", unsafe_allow_html=True)

# Create a layout for the charts
st.markdown("---")

# Create a 2-column layout for the middle charts
col1, col2 = st.columns(2)

with col1:
    # Create heatmap of emissions by traffic level and delay
    st.markdown("### Emissions by Traffic Level and Delay")
    
    if 'traffic_level' in completed_rides.columns and 'estimated_delay_minutes' in completed_rides.columns:
        try:
            # Use safe function to apply categorization without SettingWithCopyWarning
            completed_rides = safe_cut(
                completed_rides, 
                'estimated_delay_minutes',
                bins=[0, 10, 20, 30, 40, 50, 60],
                labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60']
            )
            
            # Create a pivot table for the heatmap
            heatmap_data = completed_rides.pivot_table(
                index='delay_category',
                columns='traffic_level',
                values='total_emissions',
                aggfunc='mean'
            ).fillna(0)
            
            # Check if the pivot table has data
            if not heatmap_data.empty:
                # Convert the pivot table to a format suitable for plotly heatmap
                heatmap_df = heatmap_data.reset_index()
                melted_df = pd.melt(heatmap_df, id_vars='delay_category')
                
                # Use plotly directly instead of the custom function which might have issues
                fig = px.imshow(
                    heatmap_data.values,
                    labels=dict(x="Traffic Level", y="Delay (minutes)", color="Emissions (gCO₂)"),
                    x=heatmap_data.columns.tolist(),
                    y=heatmap_data.index.tolist(),
                    color_continuous_scale="Greens",
                    title="Average Emissions by Traffic Level and Delay (gCO₂)"
                )
                fig.update_layout(
                    title={
                        'text': "Average Emissions by Traffic Level and Delay (gCO₂)",
                        'x': 0.5,
                        'xanchor': 'center',
                        'y': 0.95,
                        'yanchor': 'top'
                    },
                    margin=dict(t=50)  # Increase top margin for title
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                raise ValueError("Pivot table has no data")
        except Exception as e:
            st.warning(f"Error creating heatmap: {str(e)}. Using sample data instead.")
            # Fall back to sample data
            # Create sample data
            traffic_levels = ['LOW', 'MEDIUM', 'HIGH', 'SEVERE']
            delay_categories = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60']
            
            # Create a random matrix for the heatmap
            data = np.random.randint(500, 900, size=(len(delay_categories), len(traffic_levels)))
            
            # Create a DataFrame with the data
            heatmap_df = pd.DataFrame(data, index=delay_categories, columns=traffic_levels)
            heatmap_df.index.name = 'Estimated Delay (min)'
            
            # Plot the heatmap
            fig, ax = plt.subplots(figsize=(10, 6))
            sns_heatmap = sns.heatmap(heatmap_df, annot=True, cmap='Greens', fmt='g', ax=ax)
            plt.title('Heatmap of Avg Emissions by Traffic Level and Delay (gCO₂)')
            st.pyplot(fig)
    else:
        # Create sample data if real data is not available
        # Create sample data if real data is not available
        traffic_levels = ['LOW', 'MEDIUM', 'HIGH', 'SEVERE']
        delay_categories = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60']
        
        # Create a random matrix for the heatmap
        data = np.random.randint(500, 900, size=(len(delay_categories), len(traffic_levels)))
        
        # Create a DataFrame with the data
        heatmap_df = pd.DataFrame(data, index=delay_categories, columns=traffic_levels)
        heatmap_df.index.name = 'Estimated Delay (min)'
        
        # Plot the heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns_heatmap = sns.heatmap(heatmap_df, annot=True, cmap='Greens', fmt='g', ax=ax)
        plt.title('Heatmap of Avg Emissions by Traffic Level and Delay (gCO₂)')
        plt.tight_layout()  # Ensure title fits
        st.pyplot(fig)

with col2:
    st.markdown("### Hourly Average Emissions per Ride")
    
    if 'timestamp' in completed_rides.columns:
        # Make a copy of the dataframe to avoid SettingWithCopyWarning
        hourly_data = completed_rides.copy()
        
        # Convert timestamp to datetime and extract hour
        hourly_data.loc[:, 'hour'] = pd.to_datetime(hourly_data['timestamp']).dt.hour
        
        # Aggregate emissions by hour with observed=True to fix FutureWarning
        hourly_emissions = hourly_data.groupby('hour', observed=True)['total_emissions'].mean().reset_index()
        
        # Plot time series
        try:
            # Use plotly directly for better control of the title
            fig = px.line(
                hourly_emissions, 
                x='hour', 
                y='total_emissions',
                title="Average Emissions per Ride by Hour (gCO₂)",
                color_discrete_sequence=['#2ca02c']
            )
            fig.update_layout(
                title={
                    'text': "Average Emissions per Ride by Hour (gCO₂)",
                    'x': 0.5,
                    'xanchor': 'center',
                    'y': 0.95,
                    'yanchor': 'top'
                },
                xaxis_title="Hour",
                yaxis_title="Emissions (gCO₂)",
                margin=dict(t=50)  # Increase top margin for title
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Error creating time series: {str(e)}. Using the visualization utility instead.")
            # Fallback to the utility function
            plot_time_series(
                hourly_emissions,
                x_col='hour',
                y_col='total_emissions',
                title="Average Emissions per Ride by Hour (gCO₂)",
                color='#2ca02c'
            )
    else:
        # Create sample data if timestamp is not available
        hours = list(range(0, 24))
        emissions = [500, 450, 400, 350, 300, 600, 900, 1200, 1500, 1400, 1300, 1200, 
                    1300, 1200, 1100, 1200, 1300, 1500, 1400, 1300, 1000, 800, 700, 600]
        
        # Create DataFrame with sample data
        hourly_emissions = pd.DataFrame({'hour': hours, 'total_emissions': emissions})
        
        # Plot time series with directly controlled title
        fig = px.line(
            hourly_emissions, 
            x='hour', 
            y='total_emissions',
            title="Average Emissions per Ride by Hour (gCO₂)",
            color_discrete_sequence=['#2ca02c']
        )
        fig.update_layout(
            title={
                'text': "Average Emissions per Ride by Hour (gCO₂)",
                'x': 0.5,
                'xanchor': 'center',
                'y': 0.95,
                'yanchor': 'top'
            },
            xaxis_title="Hour",
            yaxis_title="Emissions (gCO₂)",
            margin=dict(t=50)  # Increase top margin for title
        )
        st.plotly_chart(fig, use_container_width=True)

# Create a 2-column layout for the bottom charts
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Average Daily Emissions by Weekday")
    
    if 'timestamp' in completed_rides.columns:
        # Make a copy of the dataframe to avoid SettingWithCopyWarning
        daily_data = completed_rides.copy()
        
        # Convert timestamp to datetime and extract day of week
        daily_data.loc[:, 'day_of_week'] = pd.to_datetime(daily_data['timestamp']).dt.day_name()
        
        # Ensure day of week is ordered correctly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_data.loc[:, 'day_of_week'] = pd.Categorical(daily_data['day_of_week'], categories=day_order, ordered=True)
        
        # Aggregate emissions by day of week with observed=True to fix FutureWarning
        daily_emissions = daily_data.groupby('day_of_week', observed=True)['total_emissions'].sum().reset_index()
        
        # Plot bar chart with better title control
        try:
            fig = px.bar(
                daily_emissions,
                x='day_of_week',
                y='total_emissions',
                title="Total Emissions by Day of Week (gCO₂)",
                color_discrete_sequence=['#2ca02c']
            )
            fig.update_layout(
                title={
                    'text': "Total Emissions by Day of Week (gCO₂)",
                    'x': 0.5,
                    'xanchor': 'center',
                    'y': 0.95,
                    'yanchor': 'top'
                },
                xaxis_title="Day of Week",
                yaxis_title="Total Emissions (gCO₂)",
                margin=dict(t=50)  # Increase top margin for title
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Error creating bar chart: {str(e)}. Using the visualization utility instead.")
            # Fallback to the utility function
            plot_bar_chart(
                daily_emissions,
                x_col='day_of_week',
                y_col='total_emissions',
                title="Total Emissions by Day of Week (gCO₂)",
                color='#2ca02c'
            )
    else:
        # Create sample data if timestamp is not available
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        emissions = [50000, 60000, 210000, 70000, 55000, 40000, 35000]
        
        # Create DataFrame with sample data
        daily_emissions = pd.DataFrame({'day_of_week': days, 'total_emissions': emissions})
        
        # Plot bar chart with better title control
        fig = px.bar(
            daily_emissions,
            x='day_of_week',
            y='total_emissions',
            title="Total Emissions by Day of Week (gCO₂)",
            color_discrete_sequence=['#2ca02c']
        )
        fig.update_layout(
            title={
                'text': "Total Emissions by Day of Week (gCO₂)",
                'x': 0.5,
                'xanchor': 'center',
                'y': 0.95,
                'yanchor': 'top'
            },
            xaxis_title="Day of Week",
            yaxis_title="Total Emissions (gCO₂)",
            margin=dict(t=50)  # Increase top margin for title
        )
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Spike Sessions (>95th percentile)")
    
    # Display number of spike sessions
    spike_sessions = emission_metrics['spike_sessions']
    
    # Create a gauge chart for spike sessions
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=spike_sessions,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Spike Sessions"},
        gauge={
            'axis': {'range': [None, 20], 'tickwidth': 1, 'tickcolor': "darkgreen"},
            'bar': {'color': "darkgreen"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 10], 'color': 'lightgreen'},
                {'range': [10, 20], 'color': 'lightgray'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 10
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkgreen", 'family': "Arial"}
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Create a separator for the geospatial section
st.markdown("---")
st.header("Geospatial Emission Hotspots")

# Create a function to generate the emission heatmap
def generate_emission_heatmap(ride_data):
    """Generate a folium heatmap of emissions based on pickup/dropoff locations"""
    if ride_data is None or ride_data.empty:
        st.warning("No ride data available for heatmap visualization.")
        return None
    
    # Check if we have the necessary columns
    required_cols = ['pickup_latitude', 'pickup_longitude']
    emission_col = 'total_emissions'
    
    if not all(col in ride_data.columns for col in required_cols):
        st.warning("Required location data missing for heatmap visualization.")
        # Create a sample map centered on Madrid with random data
        return create_sample_heatmap()
    
    # Create a copy of the data to avoid SettingWithCopyWarning
    map_data = ride_data.copy()
    
    if emission_col not in map_data.columns:
        st.warning("Emissions data missing. Using ride count for heatmap intensity.")
        # If emissions data is not available, use count as intensity
        map_data.loc[:, 'heatmap_intensity'] = 1
    else:
        map_data.loc[:, 'heatmap_intensity'] = map_data[emission_col]
    
    # Filter out any rows with missing location data
    map_data = map_data[required_cols + ['heatmap_intensity']].dropna()
    
    if map_data.empty:
        st.warning("No valid data points for heatmap after filtering.")
        return create_sample_heatmap()
    
    # Format data for heatmap: [lat, lng, intensity] and convert to simple list of lists
    # Ensure all values are numeric to avoid issues with the folium HeatMap
    heat_data = []
    for _, row in map_data.iterrows():
        try:
            lat = float(row['pickup_latitude'])
            lng = float(row['pickup_longitude'])
            intensity = float(row['heatmap_intensity'])
            heat_data.append([lat, lng, intensity])
        except (ValueError, TypeError) as e:
            # Skip invalid data points
            continue
    
    if not heat_data:
        st.warning("No valid data points after conversion. Using sample data.")
        return create_sample_heatmap()
    
    # Create a folium map centered on Madrid
    m = folium.Map(location=[40.4168, -3.7038], zoom_start=12, tiles="CartoDB positron")
    
    # Add the HeatMap layer with explicit types for all values
    HeatMap(
        data=heat_data,  # List of [lat, lng, intensity] lists
        radius=15,       # size of each point's influence
        blur=10,         # smoothing factor
        min_opacity=0.5, # how transparent low-value areas are
        gradient={'0.4': 'blue', '0.65': 'lime', '0.8': 'yellow', '1': 'red'}
    ).add_to(m)
    
    return m

def create_sample_heatmap():
    """Create a sample heatmap for demonstration purposes"""
    import random
    
    # Madrid center coordinates
    madrid_lat, madrid_lng = 40.4168, -3.7038
    
    # Create sample points with higher concentration near the center
    sample_points = []
    
    # Downtown area (higher concentration)
    for _ in range(300):
        lat = madrid_lat + random.uniform(-0.03, 0.03)
        lng = madrid_lng + random.uniform(-0.03, 0.03)
        intensity = random.uniform(10, 100)  # Higher values in the center
        sample_points.append([lat, lng, intensity])
    
    # Chamberí area (medium concentration)
    for _ in range(150):
        lat = madrid_lat + 0.02 + random.uniform(-0.02, 0.02)
        lng = madrid_lng - 0.01 + random.uniform(-0.02, 0.02)
        intensity = random.uniform(5, 50)
        sample_points.append([lat, lng, intensity])
    
    # Salamanca area (medium concentration)
    for _ in range(150):
        lat = madrid_lat - 0.01 + random.uniform(-0.02, 0.02)
        lng = madrid_lng + 0.02 + random.uniform(-0.02, 0.02)
        intensity = random.uniform(5, 50)
        sample_points.append([lat, lng, intensity])
    
    # Outer areas (lower concentration)
    for _ in range(200):
        lat = madrid_lat + random.uniform(-0.1, 0.1)
        lng = madrid_lng + random.uniform(-0.1, 0.1)
        intensity = random.uniform(1, 20)  # Lower values in outer areas
        sample_points.append([lat, lng, intensity])
    
    # Create the map
    m = folium.Map(location=[madrid_lat, madrid_lng], zoom_start=12, tiles="CartoDB positron")
    
    # Add the heatmap
    HeatMap(
        data=sample_points,
        radius=15,
        blur=10,
        min_opacity=0.5,
        gradient={'0.4': 'blue', '0.65': 'lime', '0.8': 'yellow', '1': 'red'}
    ).add_to(m)
    
    return m

# Create a function to safely apply cut to a dataframe column
def safe_cut(df, col_name, bins, labels):
    """Safely apply pd.cut to a dataframe column without SettingWithCopyWarning"""
    df_copy = df.copy()
    df_copy.loc[:, 'delay_category'] = pd.cut(
        df_copy[col_name], 
        bins=bins,
        labels=labels
    )
    return df_copy

# Generate and display the heatmap
emission_map = generate_emission_heatmap(completed_rides)

if emission_map:
    # Replace folium_static with st_folium
    st_folium(emission_map, width=800, height=500)
else:
    st.error("Unable to generate the emissions heatmap.")

# Dashboard Controls in Sidebar
st.sidebar.title("Dashboard Controls")

st.sidebar.header("Date Range")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-31')),
    min_value=pd.Timestamp('2023-01-01'),
    max_value=pd.Timestamp('2023-12-31')
)

st.sidebar.header("Vehicle Type")
vehicle_types = ['All', 'Electric', 'Hybrid', 'Gas']
selected_vehicle_type = st.sidebar.selectbox(
    "Select Vehicle Type",
    options=vehicle_types,
    index=0
)

st.sidebar.header("Traffic Level")
traffic_levels = ['All', 'LOW', 'MEDIUM', 'HIGH', 'SEVERE']
selected_traffic_level = st.sidebar.selectbox(
    "Select Traffic Level",
    options=traffic_levels,
    index=0
)

st.sidebar.header("Geographical Area")
geographical_areas = ['All Areas', 'Downtown', 'Suburbs', 'Airport', 'Business District']
selected_area = st.sidebar.selectbox(
    "Select Area",
    options=geographical_areas,
    index=0
)

# Download options
st.sidebar.header("Export Data")
st.sidebar.download_button(
    label="Download Emissions Data (CSV)",
    data=completed_rides[['ride_id', 'timestamp', 'total_emissions']].to_csv().encode('utf-8'),
    file_name='emissions_data.csv',
    mime='text/csv',
)
