import unittest
from llm_eval_kit.processors.decorators import preprocess
from llm_eval_kit.processors.pre_processor import PreProcessor
from llm_eval_kit.processors.post_processor import PostProcessor
from llm_eval_kit.processors.exceptions import PreprocessError, PostprocessError


class TestCompleteProcessorSuite(unittest.TestCase):
    """Complete test suite based on test.py example"""
    
    def test_original_clean_input_function(self):
        """Test the exact clean_input function from test.py"""
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
        
        # Use proper event structure
        event = {
            "process_type": "preprocess",
            "data": {
                "prompt": "hello world",
                "gold": "Hello! How can I help you today?"
            }
        }
        
        # Call using .process() method as fixed in test.py
        result = clean_input.process(event, None)
        
        # Verify expected output
        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(result["body"]["prompt"], "hello world")
        self.assertEqual(result["body"]["gold"], "Hello! How can I help you today?")
    
    def test_preprocess_decorator_type(self):
        """Verify @preprocess returns PreProcessor instance"""
        @preprocess
        def sample_func(data, context):
            return data
        
        self.assertIsInstance(sample_func, PreProcessor)
        self.assertTrue(hasattr(sample_func, 'process'))
    
    def test_missing_input_fields(self):
        """Test behavior with missing input fields"""
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
        
        # Test with missing gold field - should raise validation error
        event = {
            "process_type": "preprocess",
            "data": {
                "prompt": "test only"
            }
        }
        
        with self.assertRaises(PreprocessError) as cm:
            clean_input.process(event, None)
        
        self.assertIn("Validation failed", str(cm.exception))
        
        # Test with missing prompt field - should raise validation error
        event = {
            "process_type": "preprocess",
            "data": {
                "gold": "test gold"
            }
        }
        
        with self.assertRaises(PreprocessError) as cm:
            clean_input.process(event, None)
        
        self.assertIn("Validation failed", str(cm.exception))
    
    def test_preprocessing_error_handling(self):
        """Test error handling in preprocessing"""
        @preprocess
        def failing_processor(event: dict, context) -> dict:
            raise ValueError("Intentional failure")
        
        event = {
            "process_type": "preprocess",
            "data": {
                "prompt": "test",
                "gold": "test"
            }
        }
        
        with self.assertRaises(PreprocessError) as cm:
            failing_processor.process(event, None)
        
        error_msg = str(cm.exception)
        self.assertIn("Preprocessing failed", error_msg)
        self.assertIn("Intentional failure", error_msg)
    
    def test_postprocessor_functionality(self):
        """Test PostProcessor works correctly"""
        def format_result(data: dict, context) -> dict:
            return {
                "statusCode": 200,
                "body": [{"metric": "accuracy", "value": 1.0}]
            }
        
        processor = PostProcessor(format_result)
        event = {
            "process_type": "postprocess",
            "data": {
                "prompt": "test",
                "inference_output": "success",
                "gold": "success"
            }
        }
        result = processor.process(event, None)
        
        self.assertEqual(result["statusCode"], 200)
        self.assertEqual(result["body"][0]["metric"], "accuracy")
        self.assertEqual(result["body"][0]["value"], 1.0)
    
    def test_postprocessor_error_handling(self):
        """Test PostProcessor error handling"""
        def failing_postprocessor(data: dict, context) -> dict:
            raise RuntimeError("Post processing error")
        
        processor = PostProcessor(failing_postprocessor)
        event = {
            "process_type": "postprocess",
            "data": {
                "prompt": "test",
                "inference_output": "test",
                "gold": "test"
            }
        }
        
        with self.assertRaises(PostprocessError) as cm:
            processor.process(event, None)
        
        error_msg = str(cm.exception)
        self.assertIn("PostProcessor failed", error_msg)
        self.assertIn("Post processing error", error_msg)


if __name__ == "__main__":
    unittest.main()