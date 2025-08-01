"""
JSON Schema loader for MCP tools in MemoryOS
"""

import json
import os
from typing import Dict, Any
from pathlib import Path


def load_schema(schema_name: str) -> Dict[str, Any]:
    """
    Load and return JSON schema for MCP tools
    
    Args:
        schema_name: Name of the schema file (e.g., "add_conversation_input.json")
        
    Returns:
        Dictionary containing the JSON schema
        
    Raises:
        FileNotFoundError: If schema file doesn't exist
        json.JSONDecodeError: If schema file contains invalid JSON
    """
    # Get the schemas directory relative to this file
    current_dir = Path(__file__).parent.parent
    schema_path = current_dir / "schemas" / schema_name
    
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema = json.load(f)
        return schema
    except FileNotFoundError:
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in schema file {schema_path}: {e}")


def get_all_schemas() -> Dict[str, Dict[str, Any]]:
    """
    Load all available schemas
    
    Returns:
        Dictionary mapping schema names to schema content
    """
    schemas = {}
    schema_files = [
        "add_conversation_input.json",
        "add_conversation_output.json", 
        "add_execution_input.json",
        "add_execution_output.json",
        "retrieve_conversation_input.json",
        "retrieve_conversation_output.json",
        "retrieve_execution_input.json",
        "retrieve_execution_output.json"
    ]
    
    for schema_file in schema_files:
        try:
            schemas[schema_file] = load_schema(schema_file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load schema {schema_file}: {e}")
    
    return schemas


def validate_input(data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate input data against a JSON schema
    
    Args:
        data: Input data to validate
        schema: JSON schema to validate against
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        import jsonschema
        jsonschema.validate(instance=data, schema=schema)
        return True, ""
    except ImportError:
        # If jsonschema is not installed, do basic validation
        return _basic_validation(data, schema)
    except jsonschema.exceptions.ValidationError as e:
        return False, str(e)


def _basic_validation(data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, str]:
    """
    Basic validation when jsonschema library is not available
    
    Args:
        data: Input data to validate
        schema: JSON schema to validate against
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check required fields
    required_fields = schema.get("required", [])
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    
    # Check field types for basic validation
    properties = schema.get("properties", {})
    for field, value in data.items():
        if field in properties:
            expected_type = properties[field].get("type")
            if expected_type == "string" and not isinstance(value, str):
                return False, f"Field {field} must be a string"
            elif expected_type == "integer" and not isinstance(value, int):
                return False, f"Field {field} must be an integer"
            elif expected_type == "boolean" and not isinstance(value, bool):
                return False, f"Field {field} must be a boolean"
            elif expected_type == "array" and not isinstance(value, list):
                return False, f"Field {field} must be an array"
            elif expected_type == "object" and not isinstance(value, dict):
                return False, f"Field {field} must be an object"
    
    return True, ""