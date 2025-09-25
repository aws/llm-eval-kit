import inspect
from nova_custom_evaluation_sdk.processors.pre_processor import PreProcessor
from nova_custom_evaluation_sdk.processors.post_processor import PostProcessor

# Expected number of parameters for processor functions (data, context)
EXPECTED_PARAM_COUNT = 2

def _validate_function_signature(func):
    """Validate that function has exactly 2 parameters (data, context)"""
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    
    if len(params) != EXPECTED_PARAM_COUNT:
        raise TypeError(f"Function {func.__name__} must have exactly {EXPECTED_PARAM_COUNT} parameters (data, context), got {len(params)}")

def preprocess(func):
    """Decorator to create a PreProcessor from a function"""
    _validate_function_signature(func)
    return PreProcessor(func)

def postprocess(func):
    """Decorator to create a PostProcessor from a function"""
    _validate_function_signature(func)
    return PostProcessor(func)