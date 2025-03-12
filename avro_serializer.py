"""
AVRO Serialization for Ride-Hailing Data

This module provides functionality to serialize and deserialize data
using AVRO format, as required by the project specifications.

Usage:
    from avro_serializer import AvroSerializer
    serializer = AvroSerializer('ride_datafeed_schema.json')
    avro_data = serializer.serialize_event(event_data)
"""

import json
import fastavro
import os
import io
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AvroSerializer:
    """
    Handles serialization and deserialization of data using AVRO format.
    """
    
    def __init__(self, schema_file, parse_schema=True):
        """
        Initialize the serializer with an AVRO schema.
        
        Args:
            schema_file: Path to the schema file (JSON format)
            parse_schema: Whether to parse the schema file (set to False if passing a pre-parsed schema)
        """
        self.schema_file = schema_file
        
        if parse_schema:
            try:
                with open(schema_file, 'r') as f:
                    schema_json = json.load(f)
                self.schema = fastavro.parse_schema(schema_json)
                logger.info(f"Successfully loaded schema from {schema_file}")
            except Exception as e:
                logger.error(f"Error loading schema from {schema_file}: {str(e)}")
                raise
        else:
            # Schema is already parsed
            self.schema = schema_file
    
    def _prepare_data(self, data):
        """
        Prepare data for AVRO serialization by handling special types.
        
        Args:
            data: The data to prepare
            
        Returns:
            dict: Prepared data
        """
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                result[key] = self._prepare_data(value)
            return result
        elif isinstance(data, list):
            return [self._prepare_data(item) for item in data]
        elif isinstance(data, datetime):
            # Convert datetime to timestamp (milliseconds since epoch)
            return int(data.timestamp() * 1000)
        else:
            return data
    
    def serialize_event(self, event_data):
        """
        Serialize a single event to AVRO format.
        
        Args:
            event_data: Event data dictionary
            
        Returns:
            bytes: Serialized AVRO data
        """
        # Prepare data
        prepared_data = self._prepare_data(event_data)
        
        # Serialize to AVRO
        buffer = io.BytesIO()
        fastavro.schemaless_writer(buffer, self.schema, prepared_data)
        buffer.seek(0)
        return buffer.read()
    
    def serialize_events(self, events_data):
        """
        Serialize multiple events to AVRO format.
        
        Args:
            events_data: List of event data dictionaries
            
        Returns:
            bytes: Serialized AVRO data
        """
        # Prepare all events
        prepared_events = [self._prepare_data(event) for event in events_data]
        
        # Serialize to AVRO
        buffer = io.BytesIO()
        fastavro.writer(buffer, self.schema, prepared_events)
        buffer.seek(0)
        return buffer.read()
    
    def serialize_to_file(self, events_data, output_file):
        """
        Serialize events and write to a file.
        
        Args:
            events_data: List of event data dictionaries
            output_file: Path to output file
            
        Returns:
            str: Path to the output file
        """
        # Prepare all events
        prepared_events = [self._prepare_data(event) for event in events_data]
        
        # Serialize to AVRO file
        with open(output_file, 'wb') as out:
            fastavro.writer(out, self.schema, prepared_events)
        
        logger.info(f"Serialized {len(events_data)} events to {output_file}")
        return output_file
    
    def deserialize_file(self, avro_file):
        """
        Deserialize data from an AVRO file.
        
        Args:
            avro_file: Path to AVRO file
            
        Returns:
            list: Deserialized records
        """
        records = []
        try:
            with open(avro_file, 'rb') as f:
                for record in fastavro.reader(f):
                    records.append(record)
            logger.info(f"Deserialized {len(records)} records from {avro_file}")
            return records
        except Exception as e:
            logger.error(f"Error deserializing {avro_file}: {str(e)}")
            raise
    
    def deserialize_bytes(self, avro_bytes):
        """
        Deserialize data from AVRO bytes.
        
        Args:
            avro_bytes: AVRO bytes
            
        Returns:
            list: Deserialized records
        """
        records = []
        try:
            buffer = io.BytesIO(avro_bytes)
            for record in fastavro.reader(buffer):
                records.append(record)
            return records
        except Exception as e:
            logger.error(f"Error deserializing bytes: {str(e)}")
            raise


class UserDriverSerializer:
    """
    Handles serialization for user and driver data using multiple schemas.
    """
    
    def __init__(self, schema_file):
        """
        Initialize with schema file containing multiple schemas.
        
        Args:
            schema_file: Path to the schema file with multiple schemas
        """
        self.schema_file = schema_file
        self.schemas = {}
        
        try:
            # Load the file as a text file
            with open(schema_file, 'r') as f:
                content = f.read()
            
            # Split content by empty lines to get separate schemas
            schema_texts = [s.strip() for s in content.split('\n\n') if s.strip()]
            
            # Parse each schema
            for schema_text in schema_texts:
                try:
                    schema_json = json.loads(schema_text)
                    if 'name' in schema_json:
                        name = schema_json['name']
                        self.schemas[name] = fastavro.parse_schema(schema_json)
                        logger.info(f"Loaded schema for {name}")
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing schema JSON: {str(e)}")
                    logger.error(f"Problematic text: {schema_text[:100]}...")
            
            logger.info(f"Loaded {len(self.schemas)} schemas from {schema_file}")
        except Exception as e:
            logger.error(f"Error loading schemas from {schema_file}: {str(e)}")
            raise
    
    def get_serializer(self, entity_type):
        """
        Get serializer for specific entity type.
        
        Args:
            entity_type: Name of the entity (e.g., 'UserStatic', 'DriverDynamic')
            
        Returns:
            AvroSerializer: Serializer for the specified entity type
        """
        if entity_type in self.schemas:
            return AvroSerializer(self.schemas[entity_type], parse_schema=False)
        else:
            available = ", ".join(self.schemas.keys())
            raise ValueError(f"No schema found for {entity_type}. Available schemas: {available}")
    
    def serialize_data(self, data, entity_type, output_file=None):
        """
        Serialize data for a specific entity type.
        
        Args:
            data: Data to serialize (list of records or single record)
            entity_type: Name of the entity type
            output_file: Path to output file (optional)
            
        Returns:
            bytes or str: Serialized data or path to output file
        """
        serializer = self.get_serializer(entity_type)
        
        # Check if data is a list
        if not isinstance(data, list):
            data = [data]
        
        if output_file:
            return serializer.serialize_to_file(data, output_file)
        else:
            return serializer.serialize_events(data)


def convert_json_to_avro(json_file, schema_file, output_file=None):
    """
    Convert a JSON file containing ride events to AVRO format.
    
    Args:
        json_file: Path to JSON file
        schema_file: Path to schema file
        output_file: Path to output AVRO file (default: json_file with .avro extension)
        
    Returns:
        str: Path to output file
    """
    if output_file is None:
        output_file = os.path.splitext(json_file)[0] + '.avro'
    
    # Initialize serializer
    serializer = AvroSerializer(schema_file)
    
    # Load JSON data
    with open(json_file, 'r') as f:
        events = json.load(f)
    
    # Verify the data structure
    if not isinstance(events, list):
        raise ValueError(f"Expected JSON file to contain a list of events, but got {type(events)}")
    
    # Serialize
    return serializer.serialize_to_file(events, output_file)


# Example usage
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='AVRO serialization for ride-hailing data')
    parser.add_argument('--schema', type=str, required=True, help='Path to schema file')
    parser.add_argument('--input', type=str, help='Path to input JSON file')
    parser.add_argument('--output', type=str, help='Path to output AVRO file')
    parser.add_argument('--entity-type', type=str, help='Entity type for multi-schema files')
    parser.add_argument('--deserialize', type=str, help='Path to AVRO file to deserialize')
    
    args = parser.parse_args()
    
    if args.deserialize:
        # Deserialize mode
        serializer = AvroSerializer(args.schema)
        records = serializer.deserialize_file(args.deserialize)
        print(f"Deserialized {len(records)} records:")
        for i, record in enumerate(records[:5]):
            print(f"Record {i+1}:")
            print(json.dumps(record, indent=2))
        if len(records) > 5:
            print(f"... and {len(records) - 5} more records")
    elif args.input:
        # Serialize mode
        if args.entity_type:
            # Multi-schema mode
            user_driver_serializer = UserDriverSerializer(args.schema)
            
            with open(args.input, 'r') as f:
                data = json.load(f)
            
            output = args.output or f"{args.input.rsplit('.', 1)[0]}_{args.entity_type}.avro"
            user_driver_serializer.serialize_data(data, args.entity_type, output)
            print(f"Serialized data as {args.entity_type} to {output}")
        else:
            # Single schema mode
            output = args.output or f"{args.input.rsplit('.', 1)[0]}.avro"
            convert_json_to_avro(args.input, args.schema, output)
            print(f"Converted {args.input} to {output}")
    else:
        print("Error: Either --input or --deserialize must be specified")
        sys.exit(1)
