import unittest
from nova_custom_evaluation_sdk.processors.decorators import preprocess
from nova_custom_evaluation_sdk.processors.pre_processor import PreProcessor
from nova_custom_evaluation_sdk.processors.exceptions import PreprocessError


class TestPreprocessorDecorator(unittest.TestCase):
    
    def test_clean_input_success(self):
        """Test the clean_input function from test.py works correctly"""
        @preprocess
        def clean_input(event: dict, context) -> dict:
            data = event.get("data", {})
            return {
                "statusCode": 200,
                "body": {
                    "prompt": data.get("prompt", ""),
                    "gold": data.get("gold", "")
                }
            }
        
        event = {
            "process_type": "preprocess",
            "data": {
                "prompt": "hello world",
                "gold": "Hello! How can I help you today?"
            }
        }
        
        result = clean_input.process(event, None)
        
        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(result["body"]["prompt"], "hello world")
        self.assertEqual(result["body"]["gold"], "Hello! How can I help you today?")
    
    def test_clean_input_missing_fields(self):
        """Test clean_input handles missing input gracefully"""
        @preprocess
        def clean_input(event: dict, context) -> dict:
            data = event.get("data", {})
            return {
                "statusCode": 200,
                "body": {
                    "prompt": data.get("prompt", ""),
                    "gold": data.get("gold", "")
                }
            }
        
        event = {
            "process_type": "preprocess",
            "data": {}
        }
        
        with self.assertRaises(PreprocessError) as cm:
            clean_input.process(event, None)
        
        self.assertIn("Validation failed", str(cm.exception))
    
    def test_preprocess_decorator_creates_preprocessor_instance(self):
        """Test that @preprocess decorator returns PreProcessor instance"""
        @preprocess
        def sample_func(data, context):
            return data
        
        self.assertIsInstance(sample_func, PreProcessor)
    
    def test_preprocess_error_handling(self):
        """Test preprocessor raises PreprocessError on function failure"""
        @preprocess
        def failing_processor(event: dict, context) -> dict:
            raise ValueError("Processing failed")
        
        event = {
            "process_type": "preprocess",
            "data": {
                "prompt": "test",
                "gold": "test"
            }
        }
        
        with self.assertRaises(PreprocessError) as cm:
            failing_processor.process(event, None)
        
        self.assertIn("Preprocessing failed", str(cm.exception))
        self.assertIn("Processing failed", str(cm.exception))
    
    def test_preprocess_with_context(self):
        """Test preprocessor raises validation error with extra fields"""
        @preprocess
        def context_aware_processor(event: dict, context) -> dict:
            data = event.get("data", {})
            return {
                "statusCode": 200,
                "body": {
                    "prompt": data.get("prompt", ""),
                    "gold": data.get("gold", ""),
                    "has_context": context is not None
                }
            }
        
        event = {
            "process_type": "preprocess",
            "data": {
                "prompt": "test",
                "gold": "test"
            }
        }
        
        with self.assertRaises(PreprocessError) as cm:
            context_aware_processor.process(event, "mock_context")
        
        self.assertIn("Validation failed", str(cm.exception))


if __name__ == "__main__":
    unittest.main()