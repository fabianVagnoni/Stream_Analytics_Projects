"""
Ride Analytics Dashboard

Main entry point for the Streamlit dashboard application.
"""

import streamlit as st
from dotenv import load_dotenv
import os


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

# Sidebar navigation
st.sidebar.title("🚗 Ride Analytics")
st.sidebar.info(
    """
    Analyze ride-hailing operations data in real-time.
    
    Data Source: Azure Blob Storage
    """
)

# Data source selection (hidden behind an expander to keep UI clean)
with st.sidebar.expander("Data Source Settings"):
    data_source = st.radio(
        "Select Data Source",
        ["Azure Blob Storage", "Local Files"]
    )

    if data_source == "Azure Blob Storage":
        st.subheader("Azure Connection")
        if AZURE_STORAGE_ACCOUNT_NAME and AZURE_BLOB_CONTAINER_NAME:
            st.success(f"Connected to {AZURE_STORAGE_ACCOUNT_NAME}/{AZURE_BLOB_CONTAINER_NAME}")
        else:
            st.warning("Azure connection not configured. Check your .env file.")
    else:
        st.success("Using local data files")


# Main content
# Create a visually appealing header with logo and title
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3097/3097180.png", width=150)
with col2:
    st.title("🚗 Ride Analytics Dashboard")
    st.subheader("Comprehensive analytics for ride-hailing operations")

# Introduction with cards
st.markdown("""
## Welcome to the Ride Analytics Platform

This dashboard provides real-time analytics and insights for ride-hailing operations, 
helping you to optimize fleet management, improve customer experience, and reduce environmental impact.
""")

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Ride Operations Analytics
    
    Track key metrics including:
    - Active rides in real-time
    - Driver performance
    - Customer satisfaction
    - Regional demand patterns
    
    [Go to Operations Analytics →](Ride_Operations_Analytics)
    """)
    
with col2:
    st.markdown("""
    ### 🌱 Carbon Footprint Tracker
    
    Monitor and optimize:
    - CO₂ emissions per ride
    - Fuel efficiency
    - Electric vehicle usage
    - Environmental impact reductions
    
    [Go to Carbon Footprint Tracker →](Carbon_Footprint_Tracker)
    """)
    
with col3:
    st.markdown("""
    ### 👥 Outlier Rides & Segmentation
    
    Gain insights through:
    - Anomaly detection
    - Customer segmentation
    - Behavioral pattern analysis
    - Targeted service optimization
    
    [Go to Outlier Analysis →](Outlier_Rides_Segmentation)
    """)

# Divider
st.markdown("---")

# How to use section
st.markdown("""
## How to Use This Dashboard

1. **Navigate** between different analytics modules using the sidebar or the links above
2. **Filter** data by date range, or other parameters
3. **Export** insights as CSV, Excel, or image files
""")

# Data status - instead of showing errors, show a clean status message
with st.expander("Data Status"):
    try:
        # Try to check if data files exist locally
        import os
        data_path = "data"
        local_files_exist = os.path.exists(data_path) and len(os.listdir(data_path)) > 0
        
        if local_files_exist:
            st.success("✅ Local data files available")
            st.info("Using cached data. Last updated: Today")
        else:
            st.warning("⚠️ Local data files not found")
            st.info("The dashboard will attempt to load data from Azure when you navigate to specific pages")
            
        if data_source == "Azure Blob Storage":
            # Don't actually try to load from Azure here, just show status
            if AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY and AZURE_BLOB_CONTAINER_NAME:
                st.success("✅ Azure connection configured")
            else:
                st.error("❌ Azure connection not fully configured")
                st.info("Please check your .env file for AZURE_STORAGE_ACCOUNT_NAME, AZURE_STORAGE_ACCOUNT_KEY, and AZURE_BLOB_CONTAINER_NAME")
    except Exception as e:
        st.warning(f"Could not check data status: {str(e)}")

# Footer
st.markdown("---")
st.markdown("© 2025 Ride Analytics Platform | v1.0.0 | Stream Analytics Project Group 4")

# Don't load actual data on the home page
# This avoids showing error messages when data sources aren't available
# Data will be loaded only when navigating to specific analytics pages