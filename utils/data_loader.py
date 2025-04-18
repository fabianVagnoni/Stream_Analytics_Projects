"""
Data Loader Module for Azure Blob Storage

This module handles connections to Azure Blob Storage and provides functions
to load data from the storage in JSON, AVRO, and Parquet formats.
"""

import os
import io
import json
import pandas as pd
import fastavro
import streamlit as st
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
from azure.storage.blob import BlobServiceClient
import glob
import re

# Load environment variables
load_dotenv()

# Get Azure Storage credentials from environment variables
ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
ACCOUNT_KEY = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME")

def get_blob_service_client() -> BlobServiceClient:
    """
    Create and return a BlobServiceClient using the storage account credentials.
    
    Returns:
        BlobServiceClient: The Azure Blob Service client
    """
    connect_str = f"DefaultEndpointsProtocol=https;AccountName={ACCOUNT_NAME};AccountKey={ACCOUNT_KEY};EndpointSuffix=core.windows.net"
    return BlobServiceClient.from_connection_string(connect_str)

@st.cache_data(ttl=3600)  # Cache the data for 1 hour
def list_blobs(folder_path: str = None) -> List[str]:
    """
    List all blobs in the specified folder path.
    
    Args:
        folder_path: Optional folder path within the container
        
    Returns:
        List of blob names
    """
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        prefix = f"{folder_path}/" if folder_path else ""
        blob_list = container_client.list_blobs(name_starts_with=prefix)
        
        return [blob.name for blob in blob_list]
    except Exception as e:
        st.error(f"Error listing blobs: {str(e)}")
        return []

@st.cache_data(ttl=3600)  # Cache the data for 1 hour
def load_json_from_blob(blob_name: str) -> pd.DataFrame:
    """
    Load JSON data from a blob and convert it to a pandas DataFrame.
    
    Args:
        blob_name: The name of the blob to load
        
    Returns:
        DataFrame containing the data from the JSON blob
    """
    try:
        blob_service_client = get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        
        # Download the blob content
        download_stream = blob_client.download_blob()
        data = download_stream.readall()
        
        # Parse JSON data
        json_data = json.loads(data)
        
        # Convert to DataFrame
        return pd.DataFrame(json_data)
    except Exception as e:
        st.error(f"Error loading JSON from blob {blob_name}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache the data for 1 hour
def load_avro_from_blob(blob_name: str) -> pd.DataFrame:
    """
    Load AVRO data from a blob and convert it to a pandas DataFrame.
    
    Args:
        blob_name: The name of the blob to load
        
    Returns:
        DataFrame containing the data from the AVRO blob
    """
    try:
        blob_service_client = get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        
        # Download the blob content
        download_stream = blob_client.download_blob()
        data = download_stream.readall()
        
        # Parse AVRO data
        avro_bytes = io.BytesIO(data)
        avro_reader = fastavro.reader(avro_bytes)
        avro_records = [record for record in avro_reader]
        
        # Convert to DataFrame
        return pd.DataFrame.from_records(avro_records)
    except Exception as e:
        st.error(f"Error loading AVRO from blob {blob_name}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache the data for 1 hour
def load_parquet_from_blob(blob_name: str) -> pd.DataFrame:
    """
    Load Parquet data from a blob and convert it to a pandas DataFrame.
    
    Args:
        blob_name: The name of the blob to load
        
    Returns:
        DataFrame containing the data from the Parquet blob
    """
    try:
        blob_service_client = get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_name)
        
        # Download the blob content
        download_stream = blob_client.download_blob()
        
        # Create a BytesIO object and write the downloaded content to it
        parquet_bytes = io.BytesIO()
        download_stream.readinto(parquet_bytes)
        parquet_bytes.seek(0)  # Reset the stream position to the beginning
        
        # Read Parquet data
        return pd.read_parquet(parquet_bytes)
    except Exception as e:
        st.error(f"Error loading Parquet from blob {blob_name}: {str(e)}")
        return pd.DataFrame()

def load_data(blob_path: str, file_format: str = None) -> pd.DataFrame:
    """
    Load data from a blob using the appropriate loader based on the file extension.
    
    Args:
        blob_path: The path of the blob to load
        file_format: Optional file format override ('json', 'avro', or 'parquet')
        
    Returns:
        DataFrame containing the data from the blob
    """
    # Determine file format from extension if not provided
    
    # Load data using the appropriate loader
    if file_format.lower() == 'json':
        return load_json_from_blob(blob_path)
    elif file_format.lower() == 'avro':
        return load_avro_from_blob(blob_path)
    elif file_format.lower() == 'parquet':
        return load_parquet_from_blob(blob_path)
    else:
        st.error(f"Unsupported file format: {file_format}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)  # Cache the data for 1 hour
def load_data_from_azure(blob_pattern: str) -> Optional[pd.DataFrame]:
    """
    Load data from Azure Blob Storage using a pattern that may include wildcards.
    
    Args:
        blob_pattern: Pattern to match blobs (e.g., "folder/*.snappy.parquet")
        
    Returns:
        DataFrame containing the combined data from all matching blobs,
        or None if no data could be loaded
    """
    try:
        # Extract folder path and file pattern
        folder_path = os.path.dirname(blob_pattern)
        file_pattern = os.path.basename(blob_pattern)
        
        # Convert wildcard pattern to regex pattern
        regex_pattern = file_pattern.replace('.', '\.').replace('*', '.*')
        
        # List all blobs in the folder
        blob_list = list_blobs(folder_path)
        if not blob_list:
            st.warning(f"No blobs found in path: {folder_path}")
            return None
        
        # Filter blobs based on pattern
        matching_blobs = []
        for blob_name in blob_list:
            # Extract just the file name from the blob path
            blob_file = os.path.basename(blob_name)
            if re.match(regex_pattern, blob_file):
                matching_blobs.append(blob_name)
        
        if not matching_blobs:
            st.warning(f"No blobs matched pattern: {blob_pattern}")
            return None
        
        # Load and combine DataFrames
        st.info(f"Loading {len(matching_blobs)} files matching pattern '{blob_pattern}'")
        dfs = []
        for blob_name in matching_blobs:
            # Determine file format based on extension
            if blob_name.endswith('.json'):
                df = load_json_from_blob(blob_name)
            elif blob_name.endswith('.avro'):
                df = load_avro_from_blob(blob_name)
            elif blob_name.endswith('.parquet') or '.snappy.parquet' in blob_name:
                df = load_parquet_from_blob(blob_name)
            else:
                st.warning(f"Skipping file with unknown format: {blob_name}")
                continue
                
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            st.warning(f"No data loaded from any matching blobs")
            return None
        
        # Combine all DataFrames (this will work as long as they have compatible schemas)
        return pd.concat(dfs, ignore_index=True)
        
    except Exception as e:
        st.error(f"Error loading data from Azure using pattern {blob_pattern}: {str(e)}")
        return None

# For local development and testing, also provide functions to load from local files
def load_local_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a local file (JSON, AVRO, or Parquet).
    
    Args:
        file_path: Path to the local file
        
    Returns:
        DataFrame containing the data from the file
    """
    try:
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        elif file_path.endswith('.avro'):
            with open(file_path, 'rb') as f:
                avro_reader = fastavro.reader(f)
                avro_records = [record for record in avro_reader]
            return pd.DataFrame.from_records(avro_records)
        elif file_path.endswith('.parquet') or file_path.endswith('.snappy.parquet'):
            return pd.read_parquet(file_path)
        else:
            st.error(f"Unsupported file format for {file_path}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {str(e)}")
        return pd.DataFrame() 