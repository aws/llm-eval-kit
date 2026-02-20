from functools import wraps
from typing import Callable, Dict, Any
from llm_eval_kit.processors.processor import Processor
from llm_eval_kit.processors.exceptions import PreprocessError
from llm_eval_kit.model.models import PreprocessingPayload, PreprocessingResponse
from pydantic import ValidationError

class PreProcessor(Processor):
    """Preprocessor implementation with validation"""
    def __init__(self, func: Callable):
        self.func = func
        self.__name__ = getattr(func, '__name__', 'PreProcessor')
        self.__doc__ = getattr(func, '__doc__', None)

    def process(self, data: Dict, context: Any) -> Dict:
        try:
            # Validate input
            payload = PreprocessingPayload(**data)
            
            # Process data
            result = self.func(data, context)
            
            # Validate output
            response = PreprocessingResponse(**result)
            return response.dict()
            
        except ValidationError as e:
            raise PreprocessError(f"Validation failed: {str(e)}") from e
        except Exception as e:
            raise PreprocessError(f"Preprocessing failed: {str(e)}") from e