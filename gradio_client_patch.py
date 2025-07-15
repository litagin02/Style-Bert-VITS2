"""
Patch for gradio_client to fix TypeError: argument of type 'bool' is not iterable
"""

def patch_gradio_client():
    """Apply patch to gradio_client to fix schema processing bug"""
    try:
        import gradio_client.utils as utils
        
        # Save original functions
        original_get_type = utils.get_type
        original_json_schema_to_python_type = utils._json_schema_to_python_type
        
        def patched_get_type(schema):
            """Patched version that handles bool values in schema"""
            # Handle case where schema is a bool
            if isinstance(schema, bool):
                return "bool"
            
            # Handle case where schema is not a dict
            if not isinstance(schema, dict):
                return str(type(schema).__name__)
            
            # Call original function for normal cases
            return original_get_type(schema)
        
        def patched_json_schema_to_python_type(schema, defs):
            """Patched version that handles non-dict schemas"""
            # Handle bool values directly
            if isinstance(schema, bool):
                return "bool"
            
            # Handle None values
            if schema is None:
                return "None"
            
            # Handle non-dict schemas
            if not isinstance(schema, dict):
                return str(type(schema).__name__)
            
            # Call original function for normal cases
            return original_json_schema_to_python_type(schema, defs)
        
        # Apply patches
        utils.get_type = patched_get_type
        utils._json_schema_to_python_type = patched_json_schema_to_python_type
        print("Applied gradio_client patch successfully")
        
    except Exception as e:
        print(f"Warning: Could not apply gradio_client patch: {e}")