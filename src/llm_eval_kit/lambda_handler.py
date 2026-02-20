from typing import Dict, Any, Optional, Union, Callable
from llm_eval_kit.processors.pre_processor import PreProcessor
from llm_eval_kit.processors.post_processor import PostProcessor
from llm_eval_kit.model.models import ProcessType

def build_lambda_handler(
    preprocessor: Optional[PreProcessor] = None,
    postprocessor: Optional[PostProcessor] = None
) -> Callable:
    """
    Build a Lambda handler function with preprocessing and postprocessing capabilities.
    
    Args:
        preprocessor: PreProcessor instance (decorated with @preprocess)
        postprocessor: PostProcessor instance (decorated with @postprocess)
        
    Returns:
        Lambda handler function
    """
    
    def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """
        Lambda handler function that routes requests based on process_type.
        
        Args:
            event: Lambda event containing process_type and data payload
            context: Lambda context object
            
        Returns:
            Processed result
        """
        process_type = event.get('process_type')
        
        if process_type == ProcessType.PREPROCESS and preprocessor:
            return preprocessor.process(event, context)
        elif process_type == ProcessType.POSTPROCESS and postprocessor:
            return postprocessor.process(event, context)
        else:
            return {
                "statusCode": 400,
                "body": {"error": f"Unsupported process_type: {process_type}"}
            }
    
    return lambda_handler