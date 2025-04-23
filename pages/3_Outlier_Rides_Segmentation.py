"""
Outlier Rides & Customer Segmentation

This page analyzes outlier rides and segments customers based on their behavior.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
from utils.data_loader import load_data, load_local_data, load_data_from_azure
from utils.data_processing import preprocess_ride_events, detect_outliers, segment_customers, analyze_rides_for_all_events
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os
import matplotlib.pyplot as plt
import warnings
import shap
import plotly.express as px

# Suppress warnings to keep the output clean
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Outlier Rides & Segmentation",
    page_icon="👥",
    layout="wide"
)

# Load pre-trained models
@st.cache_resource
def load_advanced_models():
    """Load pre-trained models for outlier detection and user clustering"""
    models = {}
    
    try:
        models_dir = "advanced_analytics_models"
        
        # Try to load each model individually
        try:
            models['outlier_detector'] = joblib.load(os.path.join(models_dir, "outlier_detector.joblib"))
            st.success("Successfully loaded outlier_detector model")
        except Exception as e:
            st.warning(f"Could not load outlier_detector: {str(e)}")
            # Create a fallback isolation forest
            models['outlier_detector'] = IsolationForest(random_state=42)
        
        try:
            models['outlier_scaler'] = joblib.load(os.path.join(models_dir, "outlier_scaler.joblib"))
            st.success("Successfully loaded outlier_scaler model")
        except Exception as e:
            st.warning(f"Could not load outlier_scaler: {str(e)}")
            # Create a fallback scaler
            models['outlier_scaler'] = StandardScaler()
        
        # Initialize outlier_explainer to None
        models['outlier_explainer'] = None
        
        # Try to load or create the SHAP explainer with improved error handling
        try:
            # Try to load the pre-trained explainer if it exists
            outlier_explainer_path = os.path.join(models_dir, "outlier_explainer.joblib")
            if os.path.exists(outlier_explainer_path):
                try:
                    models['outlier_explainer'] = joblib.load(outlier_explainer_path)
                    st.success("Successfully loaded outlier_explainer model")
                except Exception as e:
                    st.warning(f"Could not load saved outlier_explainer: {str(e)}")
                    # Will try creating a new one below
            
            # If loading failed or file doesn't exist, try to create a new explainer
            if models['outlier_explainer'] is None and 'outlier_detector' in models and models['outlier_detector'] is not None:
                try:
                    # Try the simplest form first (most compatible across versions)
                    models['outlier_explainer'] = shap.TreeExplainer(models['outlier_detector'])
                    st.success("Created new outlier_explainer with default parameters")
                except Exception as e1:
                    st.warning(f"Error creating basic TreeExplainer: {str(e1)}")
                    try:
                        # Try with data=None parameter (works in newer SHAP versions)
                        models['outlier_explainer'] = shap.TreeExplainer(
                            model=models['outlier_detector'],
                            data=None,
                            feature_perturbation="interventional"
                        )
                        st.success("Created new outlier_explainer with explicit parameters")
                    except Exception as e2:
                        st.warning(f"Error creating TreeExplainer with explicit parameters: {str(e2)}")
                        # Last resort: create a KernelExplainer instead
                        try:
                            # Generate a small dummy dataset for the background
                            n_features = 10  # Adjust based on your model's expected features
                            background_data = pd.DataFrame(np.zeros((1, n_features)))
                            predict_fn = lambda x: models['outlier_detector'].decision_function(x)
                            models['outlier_explainer'] = shap.KernelExplainer(predict_fn, background_data)
                            st.success("Created KernelExplainer as fallback")
                        except Exception as e3:
                            st.warning(f"Failed to create any SHAP explainer: {str(e3)}")
                            models['outlier_explainer'] = None
        except Exception as e:
            st.warning(f"Could not setup outlier_explainer: {str(e)}")
            models['outlier_explainer'] = None
        
        try:
            models['user_clustering_model'] = joblib.load(os.path.join(models_dir, "user_clustering_model.joblib"))
            st.success("Successfully loaded user_clustering_model")
        except Exception as e:
            st.warning(f"Could not load user_clustering_model: {str(e)}")
            models['user_clustering_model'] = KMeans(n_clusters=4, random_state=42)
        
        try:
            models['user_scaler'] = joblib.load(os.path.join(models_dir, "user_scaler.joblib"))
            st.success("Successfully loaded user_scaler model")
        except Exception as e:
            st.warning(f"Could not load user_scaler: {str(e)}")
            # Create a fallback scaler
            models['user_scaler'] = StandardScaler()
        
        return models
    except Exception as e:
        st.error(f"Error setting up models: {str(e)}")
        return None

# Page title
st.title("👥 Outlier Rides & Customer Segmentation")
st.markdown("### Use Case 3: Identifying Outlier Patterns and Customer Segments")

# Load data (try from Azure first, fall back to local)
@st.cache_data(ttl=3600)
def load_segment_data():
    """Load user and ride data, detect outliers, and segment customers"""
    try:
        # Load advanced models
        models = load_advanced_models()
        if models is None:
            st.warning("Could not load advanced analytics models. Falling back to simple detection methods.")
            use_advanced_models = False
        else:
            use_advanced_models = True
        
        # Try to load from Azure
        from utils.data_loader import load_data_from_azure
        
        # Load user data from user_vectors folder (contains demographic and potentially ride metrics)
        users_static_df = load_data_from_azure("user_vectors/*.snappy.parquet")
        
        # Load ride events data from rides folder
        ride_events_df = load_data_from_azure("rides/*.snappy.parquet")
        
        # Load special events data (if available)
        special_events_df = load_data_from_azure("specials/*.snappy.parquet")
        
        if users_static_df is None or ride_events_df is None:
            st.error("Failed to load data from Azure. Check your connection and container configuration.")
        
        # Process ride events - detect outliers
        with st.spinner("Processing ride data..."):
            # Debug information - show column names
            st.info(f"Ride events columns: {list(ride_events_df.columns)}")
            if users_static_df is not None:
                st.info(f"User static columns: {list(users_static_df.columns)}")
            
            # Step 1: Analyze rides for events (just like in the notebook)
            processed_ride_events = ride_events_df.copy()
            
            # Define function to determine ride direction relative to an event venue
            
            # Apply event analysis if special events data is available
            if special_events_df is not None:
                processed_ride_events = analyze_rides_for_all_events(processed_ride_events, special_events_df)
                st.info("Successfully analyzed rides in relation to special events")
            
            # Step 2: Create subset for user profile creation (exactly as in notebook)
            # Ensure required columns exist
            required_columns = [
                "timestamp", "user_id", "pickup_latitude", "pickup_longitude",
                "dropoff_latitude", "dropoff_longitude", "distance_km"
            ]
            
            # Check for missing columns
            missing_columns = [col for col in required_columns if col not in processed_ride_events.columns]
            if missing_columns:
                st.warning(f"Missing required columns: {missing_columns}. Adding default values.")
                for col in missing_columns:
                    if col == "timestamp":
                        processed_ride_events[col] = pd.Timestamp('2023-01-01')
                    elif col in ["pickup_latitude", "dropoff_latitude"]:
                        processed_ride_events[col] = 40.0
                    elif col in ["pickup_longitude", "dropoff_longitude"]:
                        processed_ride_events[col] = -3.0
                    elif col == "distance_km":
                        processed_ride_events[col] = 0.0
                    else:
                        processed_ride_events[col] = None
            
            # Create the data subset
            columns_for_subset = [
                "timestamp", "user_id", "pickup_latitude", "pickup_longitude",
                "dropoff_latitude", "dropoff_longitude", "distance_km"
            ]
            
            # Add event columns if they exist
            if 'event_direction' in processed_ride_events.columns:
                columns_for_subset.append('event_direction')
            else:
                processed_ride_events['event_direction'] = None
                columns_for_subset.append('event_direction')
                
            if 'event_name' in processed_ride_events.columns:
                columns_for_subset.append('event_name')
            else:
                processed_ride_events['event_name'] = None
                columns_for_subset.append('event_name')
            
            # Check for other columns required for outlier detection
            outlier_columns = ["traffic_level", "estimated_duration_minutes", 
                             "actual_duration_minutes", "estimated_delay_minutes",
                             "driver_speed_kmh"]
            
            for col in outlier_columns:
                if col in processed_ride_events.columns:
                    columns_for_subset.append(col)
            
            # Create the subset
            rides_subset = processed_ride_events[columns_for_subset].copy()
            
            # Extract time features
            rides_subset["month"] = rides_subset["timestamp"].dt.month
            rides_subset["day"] = rides_subset["timestamp"].dt.day
            rides_subset["hour"] = rides_subset["timestamp"].dt.hour
            rides_subset["day_of_week"] = rides_subset["timestamp"].dt.dayofweek
            rides_subset["day_of_year"] = rides_subset["timestamp"].dt.dayofyear
            
            # Drop timestamp after extracting features
            rides_subset_without_timestamp = rides_subset.drop(columns=["timestamp"])
            
            # Step 3: Create user profile features (as in notebook)
            st.info("Creating user profiles from ride data")
            user_features = pd.DataFrame()
            
            # Group by user_id to create aggregated features
            user_groups = rides_subset.groupby('user_id')
            
            # Basic stats features
            user_features['user_id'] = user_groups['user_id'].first()
            user_features['total_rides'] = user_groups.size()
            user_features['avg_distance_km'] = user_groups['distance_km'].mean()
            user_features['max_distance_km'] = user_groups['distance_km'].max()
            
            # Location features - average pickup and dropoff coordinates
            user_features['avg_pickup_latitude'] = user_groups['pickup_latitude'].mean()
            user_features['avg_pickup_longitude'] = user_groups['pickup_longitude'].mean()
            user_features['avg_dropoff_latitude'] = user_groups['dropoff_latitude'].mean()
            user_features['avg_dropoff_longitude'] = user_groups['dropoff_longitude'].mean()
            
            # Time-based features
            user_features['most_common_hour'] = user_groups['hour'].agg(
                lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else -1
            )
            user_features['most_common_day_of_week'] = user_groups['day_of_week'].agg(
                lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else -1
            )
            user_features['weekend_ride_ratio'] = user_groups['day_of_week'].apply(
                lambda x: sum(x.isin([5, 6])) / len(x) if len(x) > 0 else 0
            )
            
            # Event-related features
            user_features['to_event_ratio'] = user_groups['event_direction'].apply(
                lambda x: sum(x == 'coming') / len(x) if len(x) > 0 else 0
            )
            user_features['from_event_ratio'] = user_groups['event_direction'].apply(
                lambda x: sum(x == 'leaving') / len(x) if len(x) > 0 else 0
            )
            
            # Get most common event attended per user
            user_features['most_common_event'] = user_groups['event_name'].agg(
                lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else 'no_event'
            )
            
            # Calculate variance in ride patterns
            user_features['distance_variance'] = user_groups['distance_km'].var()
            user_features['hour_variance'] = user_groups['hour'].var()
            
            # Reset index for the final dataframe
            user_features = user_features.reset_index(drop=True)
            
            # Function to map categorical values to numeric indices
            def map_level(level, mapping):
                return mapping.get(level, len(mapping))
            
            # Encode categorical 'most_common_event'
            unique_events = user_features["most_common_event"].unique()
            events_map = {level: i for i, level in enumerate(unique_events)}
            user_features["most_common_event"] = user_features["most_common_event"].apply(
                map_level, args=(events_map,)
            )
            
            # Step 4: If user static data has gender or other categorical columns, encode them
            if 'gender' in users_static_df.columns:
                unique_genders = users_static_df["gender"].unique()
                genders_map = {level: i for i, level in enumerate(unique_genders)}
                users_static_df["gender"] = users_static_df["gender"].apply(
                    map_level, args=(genders_map,)
                )
            
            # Step 5: Merge user features with static data
            # Make sure user_id is a string type in both dataframes
            users_static_df['user_id'] = users_static_df['user_id'].astype(str)
            user_features['user_id'] = user_features['user_id'].astype(str)
            
            # Display data for debugging
            st.info(f"User features columns: {list(user_features.columns)}")
            st.info(f"Users static columns: {list(users_static_df.columns)}")
            
            # Check if we have user_id in both dataframes
            if 'user_id' in users_static_df.columns and 'user_id' in user_features.columns:
                # Merge the dataframes with custom suffixes
                complete_user_features = pd.merge(user_features, users_static_df, on="user_id", how="outer",
                                                 suffixes=('', '_static'))
                
                # Prioritize the calculated metrics from user_features
                # For any duplicate columns, keep the one without suffix
                columns_to_drop = [col for col in complete_user_features.columns if col.endswith('_static') 
                                  and col.replace('_static', '') in complete_user_features.columns]
                complete_user_features = complete_user_features.drop(columns=columns_to_drop)
                
                st.success("Successfully merged user ride metrics with static user data")
            else:
                complete_user_features = user_features
                st.warning("Could not merge with user static data due to missing user_id column")
            
            # Step 6: Fill NA values with -1 (as in notebook)
            complete_user_features = complete_user_features.fillna(-1)
            
            # Debug after merge - check column names in the merged data
            st.info(f"Complete user features columns: {list(complete_user_features.columns)}")
            
            # Map column names for the dashboard display
            complete_user_features['avg_ride_distance'] = complete_user_features['avg_distance_km']
            if 'actual_duration_minutes' in processed_ride_events.columns:
                complete_user_features['avg_actual_duration_minutes'] = user_groups['actual_duration_minutes'].mean() if 'actual_duration_minutes' in rides_subset.columns else -1
                complete_user_features['avg_ride_duration'] = complete_user_features['avg_actual_duration_minutes']
            
            if 'total_fare' in processed_ride_events.columns:
                complete_user_features['avg_total_fare'] = user_groups['total_fare'].mean() if 'total_fare' in rides_subset.columns else -1
                complete_user_features['avg_fare_amount'] = complete_user_features['avg_total_fare']
            
            # Step 7: Prepare outlier detection data from rides
            if use_advanced_models:
                # Filter to only include completed rides for outlier detection
                completed_rides = processed_ride_events[processed_ride_events['event_type'] == 'RIDE_COMPLETED'].copy() if 'event_type' in processed_ride_events.columns else processed_ride_events.copy()
                
                if len(completed_rides) == 0:
                    st.warning("No completed rides found. Cannot perform outlier detection.")
                    # Fall back to simple detection
                    use_advanced_models = False
                else:
                    # Use the advanced ML models for outlier detection
                    st.info("Using advanced machine learning models for outlier detection")
                    
                    try:
                        # Prepare the features EXACTLY as done in the notebook
                        # Get the subset for outlier detection
                        outlier_columns = ["timestamp", "pickup_latitude", "pickup_longitude", 
                                         "dropoff_latitude", "dropoff_longitude", "traffic_level", 
                                         "distance_km", "estimated_duration_minutes", 
                                         "actual_duration_minutes", "estimated_delay_minutes", 
                                         "driver_speed_kmh"]
                        
                        # Check which columns exist in completed_rides
                        available_columns = [col for col in outlier_columns if col in completed_rides.columns]
                        st.info(f"Available columns for outlier detection: {available_columns}")
                        
                        missing_columns = [col for col in outlier_columns if col not in completed_rides.columns]
                        if missing_columns:
                            st.warning(f"Missing columns for outlier detection: {missing_columns}")
                            for col in missing_columns:
                                if col == "timestamp":
                                    completed_rides[col] = pd.Timestamp('2023-01-01')
                                elif col in ["pickup_latitude", "dropoff_latitude"]:
                                    completed_rides[col] = 40.0
                                elif col in ["pickup_longitude", "dropoff_longitude"]:
                                    completed_rides[col] = -3.0
                                elif col == "traffic_level":
                                    completed_rides[col] = "MEDIUM"
                                else:
                                    completed_rides[col] = 0
                        
                        # Create the outlier subset with all available columns
                        rides_outlier_subset = completed_rides[outlier_columns].copy()
                        
                        # Extract time features from timestamp
                        rides_outlier_subset["month"] = rides_outlier_subset["timestamp"].dt.month
                        rides_outlier_subset["day"] = rides_outlier_subset["timestamp"].dt.day
                        rides_outlier_subset["hour"] = rides_outlier_subset["timestamp"].dt.hour
                        rides_outlier_subset["day_of_week"] = rides_outlier_subset["timestamp"].dt.dayofweek
                        
                        # Drop timestamp after extracting features
                        rides_outlier_subset.drop(columns=["timestamp"], inplace=True)
                        
                        # Encode traffic level if it exists
                        if 'traffic_level' in rides_outlier_subset.columns:
                            # Get unique traffic levels
                            unique_traffic_levels = rides_outlier_subset["traffic_level"].unique()
                            
                            # Create mapping dictionary with index for each level
                            traffic_levels = {}
                            for i, level in enumerate(unique_traffic_levels):
                                traffic_levels[level] = i
                            
                            # Apply mapping
                            rides_outlier_subset["traffic_level"] = rides_outlier_subset["traffic_level"].apply(
                                map_level, args=(traffic_levels,)
                            )
                        
                        # Fill NA values with -100 as done in the notebook
                        rides_outlier_subset = rides_outlier_subset.fillna(-100)
                        
                        # Ensure we have data
                        if len(rides_outlier_subset) == 0:
                            st.warning("After preparing data, no rows remain for outlier detection.")
                            use_advanced_models = False
                        else:
                            # Try using the pre-trained model directly with the properly formatted data
                            try:
                                # Create indices to map back to the original data
                                original_indices = rides_outlier_subset.index
                                
                                # Make predictions with the trained model
                                outlier_predictions = models['outlier_detector'].predict(rides_outlier_subset)
                                outlier_scores = models['outlier_detector'].decision_function(rides_outlier_subset)
                                
                                # Map the results back to the original dataframe
                                completed_rides.loc[original_indices, 'outlier_score'] = outlier_scores
                                completed_rides.loc[original_indices, 'is_outlier'] = outlier_predictions == -1
                                
                                # Find which rides are outliers
                                outlier_indices = original_indices[outlier_predictions == -1]
                                outlier_count = len(outlier_indices)
                                
                                st.success(f"Successfully identified {outlier_count} outlier rides")
                                
                                # Copy outlier information to the original processed_ride_events DataFrame
                                processed_ride_events['outlier_score'] = np.nan
                                processed_ride_events['is_outlier'] = False
                                
                                # Copy values from completed_rides to processed_ride_events
                                processed_ride_events.loc[completed_rides.index, 'outlier_score'] = completed_rides['outlier_score']
                                processed_ride_events.loc[completed_rides.index, 'is_outlier'] = completed_rides['is_outlier']
                                
                                # Add individual metric outlier flags for dashboard
                                processed_ride_events['ride_distance_outlier'] = False
                                processed_ride_events['ride_duration_outlier'] = False
                                processed_ride_events['fare_amount_outlier'] = False
                                
                                # Set all metric outliers to the overall outlier flag
                                processed_ride_events.loc[processed_ride_events['is_outlier'], 'ride_distance_outlier'] = True
                                processed_ride_events.loc[processed_ride_events['is_outlier'], 'ride_duration_outlier'] = True
                                processed_ride_events.loc[processed_ride_events['is_outlier'], 'fare_amount_outlier'] = True
                            
                            except ValueError as e:
                                st.warning(f"Feature mismatch in outlier detection: {str(e)}. Training a new model.")
                                # Create and train a new model with the exact same approach
                                new_detector = IsolationForest(contamination=0.01, random_state=42)
                                new_detector.fit(rides_outlier_subset)
                                
                                # Make predictions with the new model
                                outlier_predictions = new_detector.predict(rides_outlier_subset)
                                outlier_scores = new_detector.decision_function(rides_outlier_subset)
                                
                                # Set up outlier flags
                                processed_ride_events['outlier_score'] = np.nan
                                processed_ride_events['is_outlier'] = False
                                processed_ride_events['ride_distance_outlier'] = False
                                processed_ride_events['ride_duration_outlier'] = False
                                processed_ride_events['fare_amount_outlier'] = False
                                
                                # Map outliers to processed_ride_events
                                outlier_indices = original_indices[outlier_predictions == -1]
                                processed_ride_events.loc[outlier_indices, 'outlier_score'] = outlier_scores[outlier_predictions == -1]
                                processed_ride_events.loc[outlier_indices, 'is_outlier'] = True
                                processed_ride_events.loc[outlier_indices, 'ride_distance_outlier'] = True
                                processed_ride_events.loc[outlier_indices, 'ride_duration_outlier'] = True
                                processed_ride_events.loc[outlier_indices, 'fare_amount_outlier'] = True
                    except Exception as e:
                        st.error(f"Error in outlier detection: {str(e)}")
                        use_advanced_models = False
            
            if not use_advanced_models:
                # Fallback to simpler IQR-based outlier detection
                st.info("Using simple IQR-based outlier detection")
                # Map column names to expected format
                column_mapping = {
                    'distance_km': 'ride_distance',
                    'actual_duration_minutes': 'ride_duration',
                    'total_fare': 'fare_amount'
                }
                
                # Check which columns need to be mapped and create them
                for source_col, target_col in column_mapping.items():
                    if source_col in processed_ride_events.columns:
                        processed_ride_events[target_col] = processed_ride_events[source_col]
                
                # Required columns for basic outlier detection
                required_columns = ['ride_distance', 'ride_duration', 'fare_amount']
                
                # Ensure required columns exist
                missing_columns = [col for col in required_columns if col not in processed_ride_events.columns]
                if missing_columns:
                    st.warning(f"Missing required columns for simple outlier detection: {missing_columns}. Using default values.")
                    for col in missing_columns:
                        processed_ride_events[col] = 0
                
                for metric in required_columns:
                    # Apply outlier detection with IQR method and default threshold (1.5)
                    outlier_result = detect_outliers(processed_ride_events, metric, method='iqr')
                    # Rename the is_outlier column to be metric-specific
                    processed_ride_events[f'{metric}_outlier'] = outlier_result['is_outlier']
                
                # Add an overall outlier flag
                processed_ride_events['is_outlier'] = (
                    processed_ride_events['ride_distance_outlier'] | 
                    processed_ride_events['ride_duration_outlier'] | 
                    processed_ride_events['fare_amount_outlier']
                )
            
            # Step 8: Add outlier metrics to user features
            # This step ensures user_features has outlier percentage
            user_ride_metrics = {}
            
            for user_id, user_rides in processed_ride_events.groupby('user_id'):
                if len(user_rides) == 0:
                    continue
                
                metrics = {
                    'user_id': user_id,
                    'outlier_ride_count': user_rides['is_outlier'].sum() if 'is_outlier' in user_rides.columns else 0,
                    'outlier_percentage': (user_rides['is_outlier'].mean() * 100) if 'is_outlier' in user_rides.columns else 0
                }
                
                user_ride_metrics[user_id] = metrics
            
            # Convert to DataFrame
            outlier_metrics_df = pd.DataFrame(list(user_ride_metrics.values()) if user_ride_metrics else [])
            
            # Merge outlier metrics with complete_user_features if not already there
            if 'outlier_percentage' not in complete_user_features.columns and not outlier_metrics_df.empty:
                # Ensure user_id type consistency
                outlier_metrics_df['user_id'] = outlier_metrics_df['user_id'].astype(str)
                complete_user_features = pd.merge(complete_user_features, outlier_metrics_df, on="user_id", how="left")
                complete_user_features['outlier_percentage'] = complete_user_features['outlier_percentage'].fillna(0)
            
            # Step 9: Cluster users for segmentation
            try:
                # Check that required columns for clustering exist
                required_clustering_columns = [
                    'avg_distance_km',         # Must exist from our user_features
                    'avg_pickup_latitude',     # Must exist from our user_features 
                    'avg_pickup_longitude',    # Must exist from our user_features
                    'avg_dropoff_latitude',    # Must exist from our user_features
                    'avg_dropoff_longitude',   # Must exist from our user_features
                    'outlier_percentage'       # Added in Step 8
                ]
                
                # Check which columns are missing
                missing_cols = [col for col in required_clustering_columns if col not in complete_user_features.columns]
                if missing_cols:
                    st.warning(f"Missing columns for clustering: {missing_cols}. Adding defaults.")
                    for col in missing_cols:
                        complete_user_features[col] = 0
                
                if use_advanced_models and 'user_clustering_model' in models and models['user_clustering_model'] is not None:
                    try:
                        # Check data types before clustering
                        st.info("Checking data types before clustering...")
                        for col in complete_user_features.columns:
                            if col != 'user_id':
                                # Check if any value in the column is a dict or other non-numeric type
                                non_numeric_values = complete_user_features[col].apply(lambda x: not isinstance(x, (int, float, bool, np.number)) and not pd.isna(x))
                                if non_numeric_values.any():
                                    problem_values = complete_user_features.loc[non_numeric_values, col].iloc[0]
                                    st.warning(f"Column {col} contains non-numeric values: {type(problem_values)} - {problem_values}")
                                    # Convert dictionaries or other objects to string and then to a numeric index
                                    if isinstance(problem_values, dict) or not isinstance(problem_values, (int, float)):
                                        st.info(f"Converting column {col} to numeric categories")
                                        # Convert to string first, then to categorical codes
                                        complete_user_features[col] = complete_user_features[col].astype(str).astype('category').cat.codes
                        
                        # Log columns after type conversion
                        st.info(f"Columns after type conversion: {list(complete_user_features.columns)}")
                        
                        # Define the exact 18 features the model was trained on
                        expected_columns = [
                            'total_rides', 'avg_distance_km', 'max_distance_km', 
                            'avg_pickup_latitude', 'avg_pickup_longitude', 
                            'avg_dropoff_latitude', 'avg_dropoff_longitude', 
                            'most_common_hour', 'most_common_day_of_week', 
                            'weekend_ride_ratio', 'to_event_ratio', 'from_event_ratio', 
                            'most_common_event', 'distance_variance', 'hour_variance', 
                            'age', 'gender', 'signup_date'
                        ]
                        
                        # Check which expected columns are missing
                        missing_cols = [col for col in expected_columns if col not in complete_user_features.columns]
                        if missing_cols:
                            st.warning(f"Missing columns required for clustering model: {missing_cols}. Adding with default values.")
                            for col in missing_cols:
                                complete_user_features[col] = -1
                        
                        # Keep only the columns the model was trained on
                        user_id_column = complete_user_features['user_id'].copy()
                        
                        # Create a copy of the complete features for later restoration
                        all_features = complete_user_features.copy()
                        
                        # Select only the columns the model expects
                        clustering_features = complete_user_features[expected_columns].copy()
                        
                        st.info(f"Using these {len(expected_columns)} columns for clustering: {expected_columns}")
                        
                        # Check for NaN or infinite values
                        if clustering_features.isna().any().any() or np.isinf(clustering_features.values).any():
                            st.warning("Found NaN or infinite values in clustering features. Filling with -1.")
                            clustering_features = clustering_features.fillna(-1)
                            clustering_features = clustering_features.replace([np.inf, -np.inf], -1)
                        
                        # Scale features
                        scaler = StandardScaler()
                        scaled_user_features = scaler.fit_transform(clustering_features)
                        
                        # Try to use the pre-trained clustering model
                        try:
                            # Assign clusters
                            cluster_labels = models['user_clustering_model'].predict(scaled_user_features)
                            
                            # Add cluster labels back to the original dataframe
                            complete_user_features = pd.DataFrame(user_id_column).rename(columns={'user_id': 'user_id'})
                            complete_user_features['cluster'] = cluster_labels
                            
                            # Map cluster numbers to meaningful names
                            cluster_names = {
                                0: "Occasional Riders",
                                1: "Regular Commuters",
                                2: "Long-distance Travelers",
                                3: "Business Travelers"
                            }
                            
                            # Map cluster numbers to names
                            complete_user_features['segment'] = complete_user_features['cluster'].map(
                                lambda x: cluster_names.get(x, f"Segment {x}")
                            )
                            
                            st.success("Successfully segmented users using advanced clustering")
                            
                            # Restore all the original columns
                            for col in expected_columns:
                                complete_user_features[col] = clustering_features[col]
                            
                        except Exception as e:
                            st.warning(f"Error using pre-trained clustering model: {str(e)}. Creating a new model.")
                            # Create a new clustering model
                            new_clustering = KMeans(n_clusters=4, random_state=42)
                            cluster_labels = new_clustering.fit_predict(scaled_user_features)
                            
                            # Add cluster labels back to a new dataframe with user_id
                            complete_user_features = pd.DataFrame(user_id_column).rename(columns={'user_id': 'user_id'})
                            complete_user_features['cluster'] = cluster_labels
                            
                            # Restore all the original columns
                            for col in expected_columns:
                                complete_user_features[col] = clustering_features[col]
                            
                            # Map cluster numbers to names
                            cluster_names = {
                                0: "Occasional Riders",
                                1: "Regular Commuters",
                                2: "Long-distance Travelers",
                                3: "Business Travelers"
                            }
                            
                            complete_user_features['segment'] = complete_user_features['cluster'].map(
                                lambda x: cluster_names.get(x, f"Segment {x}")
                            )
                    except Exception as e:
                        st.warning(f"Advanced clustering failed: {str(e)}. Falling back to basic segmentation.")
                        import traceback
                        st.warning(f"Traceback: {traceback.format_exc()}")
                        # Fallback to basic segmentation
                        complete_user_features = segment_customers(complete_user_features)
                else:
                    # Fallback to basic segmentation
                    complete_user_features = segment_customers(complete_user_features)
            except Exception as e:
                st.warning(f"User segmentation error: {str(e)}. Using basic segmentation.")
                import traceback
                st.warning(f"Traceback: {traceback.format_exc()}")
                complete_user_features = segment_customers(complete_user_features)
                
            # Return the data in the expected format
            return {
                'ride_events': ride_events_df,
                'processed_ride_events': processed_ride_events,
                'users_static': users_static_df,
                'segmented_users': complete_user_features,
                'use_advanced_models': use_advanced_models,
                'models': models,
                'rides_outlier_subset': rides_outlier_subset if 'rides_outlier_subset' in locals() else None,
                # Add prepared SHAP data if available
                'outlier_data': {
                    'outlier_indices': [idx for idx in processed_ride_events.index if processed_ride_events.loc[idx, 'is_outlier']] if 'is_outlier' in processed_ride_events.columns else []
                } 
            }
            
    except Exception as e:
        return


# Load and process data
with st.spinner("Loading data..."):
    data = load_segment_data()
    if data is None:
        st.error("Failed to load required data.")
        st.stop()

# Extract data from the loaded dictionary
ride_events = data['ride_events']
users_static = data['users_static']
processed_ride_events = data['processed_ride_events']
segmented_users = data['segmented_users']
use_advanced_models = data.get('use_advanced_models', False)
models = data.get('models', {})
rides_outlier_subset = data.get('rides_outlier_subset', None)
outlier_indices = data.get('outlier_data', {}).get('outlier_indices', [])

st.markdown("---")

# --- Clustering Section ---
st.header("Customer Clustering Analysis")

# Display Variation Clusters Metrics
st.markdown("#### Variation Clusters (vs. Average)")

if 'cluster' in segmented_users.columns and not segmented_users.empty:
    # Calculate overall averages first
    overall_avg_dist = segmented_users['avg_distance_km'].mean() if 'avg_distance_km' in segmented_users.columns else 0
    overall_avg_dur = segmented_users['avg_ride_duration'].mean() if 'avg_ride_duration' in segmented_users.columns else 0
    overall_avg_fare = segmented_users['avg_fare_amount'].mean() if 'avg_fare_amount' in segmented_users.columns else 0

    cluster_cols = st.columns(min(3, len(segmented_users['cluster'].unique())))
    processed_clusters = 0
    for cluster_id, cluster_data in segmented_users.groupby('cluster'):
        if processed_clusters >= len(cluster_cols): break
        if cluster_data.empty: continue # Skip empty clusters

        segment_name = cluster_data['segment'].iloc[0] if 'segment' in cluster_data.columns else f"Cluster {cluster_id}"

        # Calculate cluster averages
        cluster_avg_dist = cluster_data['avg_distance_km'].mean() if 'avg_distance_km' in cluster_data.columns else 0
        cluster_avg_dur = cluster_data['avg_ride_duration'].mean() if 'avg_ride_duration' in cluster_data.columns else 0
        cluster_avg_fare = cluster_data['avg_fare_amount'].mean() if 'avg_fare_amount' in cluster_data.columns else 0

        # Calculate percentage differences
        dist_change = ((cluster_avg_dist / overall_avg_dist) - 1) * 100 if overall_avg_dist != 0 else 0
        dur_change = ((cluster_avg_dur / overall_avg_dur) - 1) * 100 if overall_avg_dur != 0 else 0
        fare_change = ((cluster_avg_fare / overall_avg_fare) - 1) * 100 if overall_avg_fare != 0 else 0

        # Calculate overall change (average of non-zero changes)
        valid_changes = [c for c in [dist_change, dur_change, fare_change] if not pd.isna(c)] # Consider only valid numbers
        overall_change = sum(valid_changes) / len(valid_changes) if valid_changes else 0

        with cluster_cols[processed_clusters]:
            st.metric(f"{segment_name}", f"{overall_change:.1f}%", help="Avg. % deviation in key metrics")
        processed_clusters += 1
else:
     st.info("Cluster information not available for variation metrics.")


st.markdown("&nbsp;") # Add some vertical space

# --- Donut Chart and Descriptions --- 

# Display Donut Chart using Plotly
st.markdown("#### Segment Distribution")
segment_counts_df = segmented_users['segment'].value_counts().reset_index()
segment_counts_df.columns = ['segment', 'count']

if not segment_counts_df.empty:
    fig_donut = px.pie(
        segment_counts_df,
        values='count',
        names='segment',
        title='User Segments Distribution',
        hole=0.4, # Creates the donut effect
        color_discrete_sequence=px.colors.qualitative.Pastel # Use a pleasant color scheme
    )
    fig_donut.update_traces(textposition='inside', textinfo='percent+label')
    fig_donut.update_layout(
        showlegend=True,
        title_x=0.5,
        legend_title_text='Segments',
        margin=dict(t=50, b=0, l=0, r=0), # Adjust margins
        height=400 # Adjust height if needed
    )
    st.plotly_chart(fig_donut, use_container_width=True)
else:
    st.info("No segment data available for distribution chart.")


# Display Quick Segment Descriptions (Moved Here)
st.markdown("#### Segment Descriptions")
segment_unique_quick = segmented_users['segment'].unique().tolist() if 'segment' in segmented_users.columns else []
descriptions = {
    "Occasional Riders": "Infrequent usage, variable patterns.",
    "Regular Commuters": "Frequent, short, consistent rides, often during peak hours.",
    "Long-distance Travelers": "Infrequent, longer rides, higher fares.",
    "Business Travelers": "Weekday/business hour usage.",
    "Average Users": "Moderate frequency, mixed patterns."
    # Add descriptions for any other segments generated by your model
}

if segment_unique_quick:
    # Use columns for better layout if many segments
    num_desc_cols = min(len(segment_unique_quick), 3) # Max 3 columns
    desc_cols = st.columns(num_desc_cols)
    for i, segment_name in enumerate(segment_unique_quick):
        with desc_cols[i % num_desc_cols]:
            st.markdown(f"**{segment_name}:**")
            st.markdown(f"> {descriptions.get(segment_name, 'Unique travel patterns.')}")
            if (i + 1) % num_desc_cols != 0: # Add space between columns
                 st.markdown("&nbsp;")

else:
    st.markdown("_No segment descriptions available._")


st.markdown("---")

# --- Outlier Section ---
st.header("Outlier Ride Analysis")

# Percentage of Outliers (Gauge Plot) - Moved here
st.markdown("#### Percentage of Outliers")
# Calculate outlier percentages (ensure columns exist)
duration_outlier_percent = processed_ride_events['ride_duration_outlier'].mean() * 100 if 'ride_duration_outlier' in processed_ride_events.columns else 0
distance_outlier_percent = processed_ride_events['ride_distance_outlier'].mean() * 100 if 'ride_distance_outlier' in processed_ride_events.columns else 0
fare_outlier_percent = processed_ride_events['fare_amount_outlier'].mean() * 100 if 'fare_amount_outlier' in processed_ride_events.columns else 0
avg_outlier_percent = np.mean([p for p in [duration_outlier_percent, distance_outlier_percent, fare_outlier_percent] if p > 0]) if any(p > 0 for p in [duration_outlier_percent, distance_outlier_percent, fare_outlier_percent]) else 0

# Create gauge chart (adjust range dynamically)
max_gauge = max(10, avg_outlier_percent * 1.5) # Ensure gauge shows value
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number", value=avg_outlier_percent,
    number={'suffix': '%'},
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "% Rides Flagged as Outliers"},
    gauge={
        'axis': {'range': [0, max_gauge], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "royalblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 1.5], 'color': 'lightblue'},
            {'range': [1.5, 5], 'color': 'lightgray'},
            {'range': [5, max_gauge], 'color': 'darkgray'} # Add another step
        ],
        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 1.5}}
))
fig_gauge.update_layout(height=250, margin={'t': 70, 'b': 30, 'l':30, 'r':30})
st.plotly_chart(fig_gauge, use_container_width=True)


st.markdown("&nbsp;") # Add space

# --- Interactive SHAP and Table --- 

# Prepare outlier data for selection and display
selected_ride_id = None # Initialize
outlier_rides_df = pd.DataFrame() # Initialize as empty

if 'is_outlier' in processed_ride_events.columns and processed_ride_events['is_outlier'].any():
    outlier_rides_df = processed_ride_events[processed_ride_events['is_outlier']].copy()
    # Ensure ride_id is present for selection
    if 'ride_id' in outlier_rides_df.columns and not outlier_rides_df.empty:
        outlier_ride_ids = outlier_rides_df['ride_id'].tolist()

        # --- Selection Widget --- 
        selected_ride_id = st.selectbox(
            "Select Outlier Ride ID for SHAP Explanation:",
            options=outlier_ride_ids,
            index=0, # Default to the first one
            key="shap_ride_selector", # Add a key for stability
            help="Choose an outlier ride from the table below to see its specific SHAP analysis."
        )
    else:
         st.warning("Outlier ride IDs are not available for selection.")
else:
    st.info("No outliers detected in the dataset.")

# --- SHAP Plot and Outliers Table Row --- 
col1, col2 = st.columns([2, 3]) # Give SHAP plot a bit more relative space

with col1:
    st.markdown("#### Outlier Explanation (SHAP)")
    # Condition rendering on having necessary components and a selection
    if selected_ride_id and not outlier_rides_df.empty and use_advanced_models and models and 'outlier_explainer' in models and rides_outlier_subset is not None:
        try:
            # Find the index corresponding to the selected ride_id in the *outlier* dataframe
            # This index should map correctly to rides_outlier_subset if it was created via .loc
            selected_index_row = outlier_rides_df[outlier_rides_df['ride_id'] == selected_ride_id]
            
            if not selected_index_row.empty:
                selected_index = selected_index_row.index[0]
                
                # Check if this index exists in the subset used for SHAP
                if selected_index in rides_outlier_subset.index:
                    explainer = models['outlier_explainer']
                    outlier_features_single = rides_outlier_subset.loc[[selected_index]] # Get the specific row

                    # Calculate SHAP values for the single instance
                    shap_values_single = None
                    if hasattr(explainer, 'shap_values'):
                        shap_values_all = explainer.shap_values(outlier_features_single)
                        if isinstance(shap_values_all, list):
                             shap_values_single = shap_values_all[0][0] # Get the first output, first instance
                        else:
                             shap_values_single = shap_values_all[0] # First instance if single output
                    elif callable(explainer):
                         shap_values_single = explainer(outlier_features_single).values[0]
                    
                    if shap_values_single is not None:
                        # Extract expected value (handle potential list/array)
                        expected_value = explainer.expected_value
                        if isinstance(expected_value, (list, np.ndarray)):
                            expected_value = expected_value[0]

                        # Display SHAP waterfall plot
                        fig_shap, ax_shap = plt.subplots(figsize=(6, 4)) # Smaller figure
                        shap.plots.waterfall(shap.Explanation(values=shap_values_single, 
                                                               base_values=expected_value, 
                                                               data=outlier_features_single.iloc[0], 
                                                               feature_names=rides_outlier_subset.columns), 
                                             max_display=10, show=False)
                        plt.tight_layout()
                        st.pyplot(fig_shap)
                        st.caption(f"Showing SHAP explanation for outlier ride: {selected_ride_id}")
                    else:
                         st.warning("Could not compute SHAP values for the selected ride.")
                else:
                    st.warning(f"Selected ride index ({selected_index}) not found in SHAP feature data.")
            else:
                st.warning(f"Could not find selected ride ID ({selected_ride_id}) in outlier details.")

        except Exception as e:
            st.error(f"Error generating SHAP plot: {e}")
            import traceback
            st.error(traceback.format_exc()) # Show full traceback for debugging
            
    elif not selected_ride_id:
        st.info("Select a Ride ID above to see its SHAP explanation.")
    elif not use_advanced_models:
        st.info("SHAP explanations require the advanced ML model to be active.")
    else:
         st.info("SHAP analysis components not available.")


with col2:
    st.markdown("#### Outlier Rides Table")
    if not outlier_rides_df.empty:
        display_cols = ['ride_id', 'user_id', 'distance_km', 'actual_duration_minutes', 'total_fare', 'outlier_score']
        available_display_cols = [col for col in display_cols if col in outlier_rides_df.columns]

        if available_display_cols:
             st.dataframe(outlier_rides_df[available_display_cols].head(10), height=450) # Adjust height to match SHAP plot area better
             if len(outlier_rides_df) > 10:
                 st.caption(f"Showing first 10 of {len(outlier_rides_df)} outliers.")
        else:
             st.info("No relevant columns available to display in the outlier table.")
    else:
        st.info("No outlier details to display (none detected or data missing).")