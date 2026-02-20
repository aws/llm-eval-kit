import unittest
from llm_eval_kit.processors.decorators import preprocess
from llm_eval_kit.processors.exceptions import PreprocessError


class TestPreprocessor(unittest.TestCase):
    
    def test_preprocess_decorator_success(self):
        """Test that @preprocess decorator works correctly"""
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
    
    def test_preprocess_decorator_with_missing_fields(self):
        """Test preprocessor handles missing input fields"""
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
                "prompt": "test"
            }
        }
        
        with self.assertRaises(PreprocessError) as cm:
            clean_input.process(event, None)
        
        self.assertIn("Validation failed", str(cm.exception))
    
    def test_preprocess_decorator_error_handling(self):
        """Test that preprocessor raises PreprocessError on function failure"""
        @preprocess
        def failing_processor(event: dict, context) -> dict:
            raise ValueError("Something went wrong")
        
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
        self.assertIn("Something went wrong", str(cm.exception))


if __name__ == "__main__":
    unittest.main()