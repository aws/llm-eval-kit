from functools import wraps
from typing import Callable, Dict, Any
from nova_custom_evaluation_sdk.processors.processor import Processor
from nova_custom_evaluation_sdk.processors.exceptions import PostprocessError
from nova_custom_evaluation_sdk.model.models import PostprocessingPayload, PostprocessingResponse
from pydantic import ValidationError

class PostProcessor(Processor):
    """PostProcessor implementation with validation"""
    def __init__(self, func: Callable):
        self.func = func
        self.__name__ = getattr(func, '__name__', 'PostProcessor')
        self.__doc__ = getattr(func, '__doc__', None)

    def process(self, data: Dict, context: Any) -> Dict:
        try:
            # Validate input
            payload = PostprocessingPayload(**data)
            
            # Process data
            result = self.func(data, context)
            
            # Validate output
            response = PostprocessingResponse(**result)
            return response.dict()
            
        except ValidationError as e:
            raise PostprocessError(f"Validation failed: {str(e)}") from e
        except Exception as e:
            raise PostprocessError(f"PostProcessor failed: {str(e)}") from e