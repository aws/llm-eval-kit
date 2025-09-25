from abc import ABC, abstractmethod
from typing import Dict, Any

class Processor(ABC):
    """Abstract base class for processors"""
    @abstractmethod
    def process(self, data: Dict, context: Any) -> Dict:
        """
        Process the input data
        
        Args:
            data: Input data to process
            context: AWS Lambda context object
            
        Returns:
            Processed data
        """
        pass