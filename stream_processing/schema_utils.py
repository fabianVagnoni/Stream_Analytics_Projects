"""
Schema utilities for loading and processing schema files
"""

def load_schema(schema_path):
    """
    Load schema from JSON file
    
    Args:
        schema_path (str): Path to the schema file
        
    Returns:
        str: Schema content as string
    """
    try:
        with open(schema_path) as f:
            schema = f.read()
            print(f"Successfully loaded schema from {schema_path}")
            return schema
    except Exception as e:
        print(f"Error loading schema from {schema_path}: {e}")
        raise 