from llm_eval_kit.processors.decorators import preprocess, postprocess
from llm_eval_kit.lambda_handler import build_lambda_handler
import json

@preprocess
def preprocessor(event: dict, context) -> dict:
    """Preprocesses evaluation data before model inference.
    
    Preprocessing step give you ability to process the input data before sending into the inference server.
    
    Args:
        event (dict): Lambda event containing evaluation data with keys:
            - data (dict): Nested data containing:
                - system (str): System prompt for the model
                - prompt (str): User input prompt
                - gold (str): Expected/gold standard response
        context: Lambda context object
        
    Returns:
        dict: Formatted response with:
            - statusCode (int): HTTP status code (200)
            - body (dict): Processed data containing system, prompt, and gold fields
            
    Example:
        >>> event = {"data": {"system": "You are helpful", "prompt": "Hi", "gold": "Hello!"}}
        >>> result = preprocessor(event, None)
        >>> result["body"]["prompt"]
        "Hi"
    """
    # Example of pass-through pre-processor
    data = event.get('data', {})
    return {
        "statusCode": 200,
        "body": {
            "system": data.get("system"),
            "prompt": data.get("prompt", ""),
            "gold": data.get("gold", "")
        }
    }
    
@postprocess
def postprocessor(event: dict, context) -> dict:
    """Postprocesses model inference results and calculates evaluation metrics.
    
    Post processing step allows you to customize metrics calculation and return with custom metrics.
    
    Args:
        event (dict): Lambda event containing inference results with keys:
            - data (dict): Nested data containing:
                - inference_output (str): Model's generated response
                - gold (str): Expected/gold standard response
                - prompt (str): Original user prompt
        context: Lambda context object
        
    Returns:
        dict: Evaluation results with:
            - statusCode (int): HTTP status code (200)
            - body (list): List of metric dictionaries, each containing:
                - metric (str): Metric name
                - value (float): Metric value
                
    Example:
        >>> event = {"data": {"inference_output": "Hello!", "gold": "Hello!"}}
        >>> result = postprocessor(event, None)
        >>> result["body"][0]["value"]
        1.0
    """
    # Example of post-processor with custom metrics

    data = event.get('data', [])
    inference_output = data.get('inference_output', '')
    gold = data.get('gold', '')
    
    metrics = []
    inverted_accuracy = 0 if inference_output.lower() == gold.lower() else 1.0
    metrics.append({
        "metric": "inverted_accuracy_custom",
        "value": inverted_accuracy
    })
    
    # Add more metrics here
    
    return {
        "statusCode": 200,
        "body": metrics
    }

preprocess_event = {
    "process_type": "preprocess",
    "data": {
        "system": "You are a helpful assistant",
        "prompt": "hello world",
        "gold": "Hello! How can I help you today?"
    }
}

postprocess_event = {
    "process_type": "postprocess",
    "data":
        {
            "prompt": "hello world",
            "inference_output": "Hello! How can I help you today?",
            "gold": "Hello! How can I help you today?"
        }
}

# Build the Lambda handler
lambda_handler = build_lambda_handler(
    preprocessor=preprocessor,
    postprocessor=postprocessor
)
    
print("Testing preprocess:")
result = lambda_handler(preprocess_event, None)
print(json.dumps(result, indent=2))

print("\nTesting postprocess:")
result = lambda_handler(postprocess_event, None)
print(json.dumps(result, indent=2))