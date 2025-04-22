"""
Azure Storage connection utilities
"""
from azure.storage.blob import BlobServiceClient
from stream_processing.config import (
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_BLOB_CONTAINER_NAME
)

# Storage connection information
storage_connection_str = AZURE_STORAGE_CONNECTION_STRING
blob_container_name = AZURE_BLOB_CONTAINER_NAME

# Create a Blob service client
blob_service_client = BlobServiceClient.from_connection_string(storage_connection_str)

def get_blob_container_client():
    """
    Get a blob container client
    
    Returns:
        ContainerClient: The blob container client
    """
    return blob_service_client.get_container_client(blob_container_name)
