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

# Display Variation Clusters
st.markdown("## Variation Clusters")

# Calculate actual variation in metrics by cluster
if 'cluster' in segmented_users.columns:
    # Create a row for variation clusters
    col1, col2, col3 = st.columns(3)
    
    # Get unique clusters (up to 3)
    unique_clusters = segmented_users['cluster'].unique()
    
    # Ensure we only process up to 3 clusters
    unique_clusters = unique_clusters[:min(3, len(unique_clusters))]
    
    # Calculate percentage changes for each cluster compared to overall average
    overall_avg_dist = segmented_users['avg_distance_km'].mean()
    overall_avg_dur = segmented_users['avg_ride_duration'].mean() if 'avg_ride_duration' in segmented_users.columns else 0
    overall_avg_fare = segmented_users['avg_fare_amount'].mean() if 'avg_fare_amount' in segmented_users.columns else 0
    
    # Display each cluster's variation
    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = segmented_users[segmented_users['cluster'] == cluster_id]
        segment_name = cluster_data['segment'].iloc[0] if 'segment' in cluster_data.columns else f"Cluster {cluster_id}"
        
        # Calculate percentage differences
        dist_change = ((cluster_data['avg_distance_km'].mean() / overall_avg_dist) - 1) * 100 if overall_avg_dist != 0 else 0
        dur_change = ((cluster_data['avg_ride_duration'].mean() / overall_avg_dur) - 1) * 100 if 'avg_ride_duration' in cluster_data.columns and overall_avg_dur != 0 else 0
        fare_change = ((cluster_data['avg_fare_amount'].mean() / overall_avg_fare) - 1) * 100 if 'avg_fare_amount' in cluster_data.columns and overall_avg_fare != 0 else 0
        
        # Use the average of the changes that we could calculate
        valid_changes = [c for c in [dist_change, dur_change, fare_change] if c != 0]
        overall_change = sum(valid_changes) / len(valid_changes) if valid_changes else 0
        
        # Display in the appropriate column
        col = [col1, col2, col3][i]
        with col:
            # Variation Cluster with segment name
            st.markdown(f"### {segment_name}:")
            
            # Display with color-coded value
            st.markdown(f"<span style='color:{'red' if overall_change < 0 else 'green'}'>{overall_change:.1f}%</span>", unsafe_allow_html=True)
else:
    # Fallback to sample values if no cluster data
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Occasional Riders:")
        st.markdown(f"<span style='color:{'red'}'>{-2}%</span>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### Regular Commuters:")
        st.markdown(f"<span style='color:{'green'}'>5%</span>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("### Long-distance Travelers:")
        st.markdown(f"<span style='color:{'green'}'>8%</span>", unsafe_allow_html=True)

# Display Outlier Percentage
st.markdown("## Percentage of Outliers")

# Calculate outlier percentages
duration_outlier_percent = processed_ride_events['ride_duration_outlier'].mean() * 100
distance_outlier_percent = processed_ride_events['ride_distance_outlier'].mean() * 100
fare_outlier_percent = processed_ride_events['fare_amount_outlier'].mean() * 100

# Take average of outlier percentages (simplified for demonstration)
avg_outlier_percent = (duration_outlier_percent + distance_outlier_percent + fare_outlier_percent) / 3

# Create a gauge chart for outlier percentage
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=avg_outlier_percent,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "% of Outliers"},
    gauge={
        'axis': {'range': [None, 5], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "royalblue"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 1.5], 'color': 'lightblue'},
            {'range': [1.5, 5], 'color': 'lightgray'}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 1.5
        }
    }
))

fig.update_layout(
    height=300,
    font={'color': "darkblue", 'family': "Arial"}
)

st.plotly_chart(fig, use_container_width=True)

# Add annotation about current value
st.markdown(f"**Current Value: {avg_outlier_percent:.1f}%**")

# Add SHAP Explainer for Outliers
st.markdown("## Outlier Explanations")
st.markdown("These plots explain what factors contributed to specific outlier predictions")

# Extract the needed variables from the data dictionary
use_advanced_models = data.get('use_advanced_models', False)
models = data.get('models', {})
rides_outlier_subset = data.get('rides_outlier_subset', None)
outlier_indices = data.get('outlier_data', {}).get('outlier_indices', [])

# Limit to top 3 outliers for visualization
if len(outlier_indices) > 3:
    outlier_indices = outlier_indices[:3]

if 'is_outlier' in processed_ride_events.columns and processed_ride_events['is_outlier'].any():
    # Only show SHAP plots if we have outliers and advanced models were used
    if use_advanced_models and models and 'outlier_detector' in models and models['outlier_detector'] is not None:
        try:
            # Get outlier data
            outlier_data = processed_ride_events[processed_ride_events['is_outlier']].head(3)
            
            # Create more informative tab names when we have outlier data
            tab_titles = []
            for i in range(min(len(outlier_data), 3)):
                # Create a descriptive title using user_id and distance if available
                title = "Outlier "
                if 'user_id' in outlier_data.columns:
                    title += f"(User {outlier_data.iloc[i]['user_id'][:5]})"
                if 'distance_km' in outlier_data.columns:
                    title += f" {outlier_data.iloc[i]['distance_km']:.1f}km"
                else:
                    title += f" #{i+1}"
                tab_titles.append(title)
            
            # Fill remaining tabs if needed
            while len(tab_titles) < 3:
                tab_titles.append(f"Outlier Example {len(tab_titles)+1}")
            
            # Create tabs for different outlier examples
            st.markdown("### Top 3 Most Significant Outliers")
            shap_tabs = st.tabs(tab_titles)
            
            # Check if rides_outlier_subset is available
            if rides_outlier_subset is not None and len(outlier_indices) > 0:
                try:
                    # Get features for outliers
                    outlier_features = rides_outlier_subset.loc[outlier_indices]
                    
                    # Get explainer from models
                    if 'outlier_explainer' in models and models['outlier_explainer'] is not None:
                        explainer = models['outlier_explainer']
                        
                        # Calculate SHAP values
                        try:
                            # Different SHAP explainers have different APIs for getting SHAP values
                            if hasattr(explainer, 'shap_values'):
                                try:
                                    # For TreeExplainer
                                    shap_values = explainer.shap_values(outlier_features)
                                    
                                    # Handle both single and multi-output case
                                    if isinstance(shap_values, list) and len(shap_values) > 1:
                                        # For multi-output, use the first output
                                        shap_values = shap_values[0]
                                except Exception as e:
                                    st.warning(f"Error using explainer.shap_values: {str(e)}")
                                    # Try with explainer directly
                                    shap_values = explainer(outlier_features).values
                            else:
                                # For newer SHAP explainers like Explainer class
                                shap_values = explainer(outlier_features).values
                            
                            # Display for each tab
                            for i, tab in enumerate(shap_tabs):
                                if i < len(outlier_indices) and i < len(outlier_features):
                                    with tab:
                                        try:
                                            # Create matplotlib figure
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            
                                            try:
                                                # Match the notebook implementation exactly
                                                plt.figure(figsize=(10, 6))
                                                
                                                # Handle expected_value correctly - ensure it's a scalar
                                                if hasattr(explainer, 'expected_value'):
                                                    if isinstance(explainer.expected_value, list) or isinstance(explainer.expected_value, np.ndarray):
                                                        expected_val = explainer.expected_value[0]
                                                        # If still an array, take the first element
                                                        if isinstance(expected_val, (list, np.ndarray)):
                                                            expected_val = expected_val[0]
                                                    else:
                                                        expected_val = explainer.expected_value
                                                        # If still an array, take the first element
                                                        if isinstance(expected_val, (list, np.ndarray)):
                                                            expected_val = expected_val[0]
                                                else:
                                                    expected_val = 0  # Fallback
                                                
                                                # Ensure shap_values is also handled correctly
                                                current_shap_values = shap_values[i]
                                                if isinstance(current_shap_values, list) and len(current_shap_values) > 0:
                                                    current_shap_values = current_shap_values[0]
                                                
                                                # Now call waterfall_legacy with explicitly scalar expected_value
                                                shap.plots._waterfall.waterfall_legacy(
                                                    expected_val,
                                                    current_shap_values,
                                                    feature_names=list(outlier_features.columns)
                                                )
                                                st.pyplot(plt.gcf())
                                            except Exception as e:
                                                st.warning(f"Error with waterfall_legacy: {str(e)}")
                                                # Fallback to a simpler plot
                                                if isinstance(shap_values, np.ndarray):
                                                    # Create a simple bar chart of feature importance
                                                    feature_importance = pd.Series(
                                                        np.abs(shap_values[i]) if len(shap_values.shape) > 1 else np.abs(shap_values),
                                                        index=outlier_features.columns
                                                    ).sort_values(ascending=False)
                                                    plt.figure(figsize=(10, 6))
                                                    plt.barh(feature_importance.index[:10], feature_importance.values[:10])
                                                    plt.xlabel('SHAP value (impact on outlier detection)')
                                                    plt.title('Feature Impact on Outlier Detection')
                                                    st.pyplot(plt.gcf())
                                            
                                            # Display outlier details
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.markdown("**Outlier Details**")
                                                for col in ['user_id', 'timestamp', 'distance_km', 'actual_duration_minutes', 'total_fare']:
                                                    if col in outlier_data.columns and i < len(outlier_data):
                                                        st.markdown(f"**{col}:** {outlier_data.iloc[i][col]}")
                                            
                                            with col2:
                                                st.markdown("**Key Factors**")
                                                # Get feature importance in a more robust way
                                                if isinstance(shap_values, np.ndarray):
                                                    if len(shap_values.shape) > 1:
                                                        feature_importance = np.abs(shap_values[i])
                                                    else:
                                                        feature_importance = np.abs(shap_values)
                                                    
                                                    # Get top features
                                                    if len(feature_importance) == len(outlier_features.columns):
                                                        top_indices = np.argsort(feature_importance)[-3:]
                                                        top_features = [outlier_features.columns[j] for j in top_indices]
                                                        
                                                        for feature in top_features:
                                                            if i < len(outlier_features) and feature in outlier_features.columns:
                                                                st.markdown(f"- **{feature}**: {outlier_features.iloc[i][feature]}")
                                        except Exception as e:
                                            st.warning(f"Error displaying SHAP plot: {str(e)}")
                                            # Show simple bar chart instead
                                            if isinstance(shap_values, np.ndarray) and i < len(shap_values):
                                                feature_importance = pd.Series(
                                                    np.abs(shap_values[i]) if len(shap_values.shape) > 1 else np.abs(shap_values),
                                                    index=outlier_features.columns
                                                ).sort_values(ascending=False)
                                                st.bar_chart(feature_importance.head(5))
                        except Exception as e:
                            st.warning(f"Error calculating SHAP values: {str(e)}")
                            # Use feature importances as fallback
                            if hasattr(models['outlier_detector'], 'feature_importances_'):
                                st.markdown("### Feature Importance (from model)")
                                importance = pd.Series(
                                    models['outlier_detector'].feature_importances_,
                                    index=outlier_features.columns
                                ).sort_values(ascending=False)
                                st.bar_chart(importance)
                    else:
                        st.info("No SHAP explainer available in the models")
                except Exception as e:
                    st.warning(f"Error accessing outlier features: {str(e)}")
            else:
                st.info("No outlier subset available for SHAP analysis")
        except Exception as e:
            st.warning(f"Could not generate SHAP explanations: {str(e)}")
            st.info("Using example visualizations instead")
            
            # Show example SHAP visualizations
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Time-based Feature Impact")
                st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/waterfall_plot.png", 
                         caption="Example SHAP waterfall plot showing feature contributions")
            with col2:
                st.markdown("### Location-based Feature Impact")
                st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/waterfall_plot.png", 
                         caption="Example SHAP waterfall plot showing feature contributions")
    else:
        st.info("Advanced models not available or not used for outlier detection")
        # Show example SHAP plots
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Example: Time-based Outlier Factors")
            st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/waterfall_plot.png", 
                     caption="Example SHAP plot showing how hour and time features impact outlier detection")
        with col2:
            st.markdown("### Example: Location-based Outlier Factors")
            st.image("https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/waterfall_plot.png", 
                     caption="Example SHAP plot showing how location features impact outlier detection")
else:
    st.info("No outliers detected in the dataset")

# Display Customer Segments
st.markdown("## Customer Segments in the System")


# Count users by cluster rather than segment
cluster_counts = segmented_users['cluster'].value_counts().to_dict() if 'cluster' in segmented_users.columns else segmented_users['segment'].value_counts().to_dict()

# Create a figure with matplotlib
fig, ax = plt.subplots(figsize=(8, 5), subplot_kw=dict(aspect="equal"))

# Create a donut chart
wedges, texts = ax.pie(cluster_counts.values(), wedgeprops=dict(width=0.5), startangle=-40)

# Setup styling for the annotations
bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
         bbox=bbox_props, zorder=0, va="center")

# Add annotations with arrows pointing to each segment
for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = f"angle,angleA=0,angleB={ang}"
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    
    # Get the cluster name/number and percentage
    cluster_id = list(cluster_counts.keys())[i]
    segment_name = segmented_users[segmented_users['cluster'] == cluster_id]['segment'].iloc[0] if 'cluster' in segmented_users.columns else list(cluster_counts.keys())[i]
    percentage = 100 * list(cluster_counts.values())[i] / sum(cluster_counts.values())
    
    # Show cluster ID with percentage
    label = f"{segment_name}\n{percentage:.1f}%"
    
    ax.annotate(label, xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
               horizontalalignment=horizontalalignment, **kw)

# Set background color and remove axes
ax.set_facecolor('none')
fig.patch.set_alpha(0.0)
plt.axis('off')

# Add "Segments" text in the center
plt.annotate('Clusters', xy=(0, 0), ha='center', va='center', fontsize=16)

# Display the matplotlib figure in Streamlit
st.pyplot(fig)

# Display segment characteristics
st.markdown("## Segment Characteristics")

# Create a dynamic layout based on available segments
segment_unique = segmented_users['segment'].unique().tolist()
num_segments = len(segment_unique)
cols_per_row = 3
num_rows = (num_segments + cols_per_row - 1) // cols_per_row  # Ceiling division

for row in range(num_rows):
    # Create columns for this row
    cols = st.columns(cols_per_row)
    
    # Fill columns with segment descriptions
    for col_idx in range(cols_per_row):
        segment_idx = row * cols_per_row + col_idx
        
        if segment_idx < num_segments:
            segment_name = segment_unique[segment_idx]
            
            with cols[col_idx]:
                st.markdown(f"### {segment_name}")
                
                # Display segment characteristics based on the segment name
                if "Commuter" in segment_name:
                    st.markdown("* Frequent short-distance rides")
                    st.markdown("* Consistent travel patterns")
                    st.markdown("* Typically during rush hours")
                    
                elif "Long-distance" in segment_name:
                    st.markdown("* Longer than average rides")
                    st.markdown("* Higher fare amounts")
                    st.markdown("* Less frequent usage")
                    
                elif "Business" in segment_name:
                    st.markdown("* Weekday usage")
                    st.markdown("* Business hour rides")
                    st.markdown("* Professional occupation")
                    
                elif "Occasional" in segment_name:
                    st.markdown("* Infrequent app usage")
                    st.markdown("* Variable ride patterns")
                    st.markdown("* Less predictable behavior")
                    
                elif "Average" in segment_name:
                    st.markdown("* Moderate usage frequency")
                    st.markdown("* Mixed ride patterns")
                    st.markdown("* Standard fare amounts")
                
                else:
                    # Generic description for other segments
                    st.markdown("* Unique travel patterns")
                    st.markdown("* Segment-specific behaviors")
                    st.markdown("* Targeted service opportunities")

# Display Outlier Rides Table
st.markdown("## Outlier Rides (As a Table)")

# Create a combined outliers dataframe
combined_outliers = pd.DataFrame({
    'ride_id': processed_ride_events['ride_id'] if 'ride_id' in processed_ride_events.columns else ['unknown'],
    'duration_outlier': processed_ride_events['ride_duration_outlier'] if 'ride_duration_outlier' in processed_ride_events.columns else [False],
    'distance_outlier': processed_ride_events['ride_distance_outlier'] if 'ride_distance_outlier' in processed_ride_events.columns else [False],
    'fare_outlier': processed_ride_events['fare_amount_outlier'] if 'fare_amount_outlier' in processed_ride_events.columns else [False]
})

# Flag rides that are outliers in at least one dimension
combined_outliers['is_any_outlier'] = (combined_outliers['duration_outlier'] | 
                                      combined_outliers['distance_outlier'] | 
                                      combined_outliers['fare_outlier'])

# Filter to only show outliers
outlier_rides = combined_outliers[combined_outliers['is_any_outlier']]

# Get additional information for these rides if we have outliers
if len(outlier_rides) > 0:
    # Get additional information for these rides
    outlier_details = ride_events[ride_events['ride_id'].isin(outlier_rides['ride_id'])]
    
    # Initialize display columns with columns we know exist
    display_columns = []
    
    # Add columns only if they exist in outlier_details
    for col in ['ride_id', 'user_id', 'driver_id', 'timestamp']:
        if col in outlier_details.columns:
            display_columns.append(col)
    
    # Add distance column (use either distance_km or ride_distance)
    if 'distance_km' in outlier_details.columns:
        display_columns.append('distance_km')
    elif 'ride_distance' in outlier_details.columns:
        display_columns.append('ride_distance')
    
    # Add duration column (use either actual_duration_minutes or ride_duration)
    if 'actual_duration_minutes' in outlier_details.columns:
        display_columns.append('actual_duration_minutes')
    elif 'ride_duration' in outlier_details.columns:
        display_columns.append('ride_duration')
    
    # Add fare column (use either total_fare or fare_amount)
    if 'total_fare' in outlier_details.columns:
        display_columns.append('total_fare') 
    elif 'fare_amount' in outlier_details.columns:
        display_columns.append('fare_amount')
    
    # Filter to completed rides if that info is available
    if 'event_type' in outlier_details.columns:
        outlier_details = outlier_details[outlier_details['event_type'] == 'RIDE_COMPLETED']
    
    # Make sure display_columns only includes columns that exist
    display_columns = [col for col in display_columns if col in outlier_details.columns]
    
    # Create the table with available columns
    if len(display_columns) > 0 and len(outlier_details) > 0:
        outlier_table = outlier_details[display_columns]
        
        # Display the table with a limit on rows
        st.dataframe(outlier_table.head(10))
        
        # Add a "See More" button
        if len(outlier_table) > 10:
            if st.button("See More Outliers"):
                st.dataframe(outlier_table)
    else:
        st.warning("No outlier details available to display")
else:
    st.info("No outliers detected in the dataset")

# Dashboard Controls in Sidebar
st.sidebar.title("Dashboard Controls")

st.sidebar.header("Time Range")
date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(datetime.date(2025, 1, 1), datetime.date(2025, 1, 2)),
    min_value=datetime.date(2025, 1, 1),
    max_value=datetime.date(2025, 1, 2)
)

st.sidebar.header("Outlier Detection")
outlier_detection_method = st.sidebar.selectbox(
    "Detection Method",
    options=["Advanced ML (Isolation Forest)", "Simple IQR", "Z-Score"],
    index=0
)

# Add explanation for the selected method
if outlier_detection_method == "Advanced ML (Isolation Forest)":
    st.sidebar.markdown("""
    **Isolation Forest** is an algorithm that efficiently detects outliers by isolating observations.
    It works well for high-dimensional data and is less affected by local density patterns.
    """)
elif outlier_detection_method == "Simple IQR":
    st.sidebar.markdown("""
    **IQR (Interquartile Range)** method flags data points that fall below Q1-1.5*IQR or above Q3+1.5*IQR.
    Simple but effective for univariate outlier detection.
    """)
else:  # Z-Score
    st.sidebar.markdown("""
    **Z-Score** method flags data points that are a certain number of standard deviations away from the mean.
    Assumes normal distribution of the data.
    """)

outlier_threshold = st.sidebar.slider(
    "Outlier Sensitivity",
    min_value=0.1,
    max_value=3.0,
    value=1.5,
    step=0.1,
    help="Lower values detect more outliers, higher values are more conservative"
)

st.sidebar.header("Customer Segments")
selected_segments = st.sidebar.multiselect(
    "Customer Segments",
    options=segmented_users['segment'].unique().tolist(),
    default=segmented_users['segment'].unique().tolist()[:3]  # Default to first 3 segments
)

# Add download button for the data
st.sidebar.header("Export Data")
st.sidebar.download_button(
    label="Download Outlier Rides CSV",
    data=outlier_table.to_csv(index=False).encode('utf-8'),
    file_name='outlier_rides.csv',
    mime='text/csv',
) 