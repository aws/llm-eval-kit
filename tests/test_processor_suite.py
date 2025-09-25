import unittest
from nova_custom_evaluation_sdk.processors.decorators import preprocess
from nova_custom_evaluation_sdk.processors.pre_processor import PreProcessor
from nova_custom_evaluation_sdk.processors.post_processor import PostProcessor
from nova_custom_evaluation_sdk.processors.exceptions import PreprocessError, PostprocessError


class TestPreprocessorDecorator(unittest.TestCase):
    
    def test_clean_input_example(self):
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
    
    def test_preprocess_creates_preprocessor(self):
        """Test @preprocess decorator creates PreProcessor instance"""
        @preprocess
        def sample_func(data, context):
            return data
        
        self.assertIsInstance(sample_func, PreProcessor)
    
    def test_preprocess_error_handling(self):
        """Test preprocessor error handling"""
        @preprocess
        def failing_func(data, context):
            raise ValueError("Test error")
        
        with self.assertRaises(PreprocessError):
            failing_func.process({}, None)


class TestPreProcessor(unittest.TestCase):
    
    def test_direct_usage(self):
        """Test PreProcessor direct instantiation"""
        def transform(data, context):
            input_data = data.get("data", {})
            return {
                "statusCode": 200,
                "body": {
                    "prompt": input_data.get("prompt", "default"),
                    "gold": input_data.get("gold", "default")
                }
            }
        
        processor = PreProcessor(transform)
        event = {
            "process_type": "preprocess",
            "data": {
                "prompt": "test",
                "gold": "test"
            }
        }
        result = processor.process(event, None)
        
        self.assertEqual(result["body"]["prompt"], "test")
    
    def test_empty_input(self):
        """Test with empty input"""
        def handle_empty(data, context):
            input_data = data.get("data", {})
            return {
                "statusCode": 200,
                "body": {
                    "prompt": input_data.get("prompt", "fallback"),
                    "gold": input_data.get("gold", "fallback")
                }
            }
        
        processor = PreProcessor(handle_empty)
        event = {
            "process_type": "preprocess",
            "data": {}
        }
        
        with self.assertRaises(PreprocessError) as cm:
            processor.process(event, None)
        
        self.assertIn("Validation failed", str(cm.exception))


class TestPostProcessor(unittest.TestCase):
    
    def test_direct_usage(self):
        """Test PostProcessor direct instantiation"""
        def format_output(data, context):
            return {
                "statusCode": 200,
                "body": [{"metric": "accuracy", "value": 1.0}]
            }
        
        processor = PostProcessor(format_output)
        event = {
            "process_type": "postprocess",
            "data": {
                "prompt": "test",
                "inference_output": "success",
                "gold": "success"
            }
        }
        result = processor.process(event, None)
        
        self.assertEqual(result["body"][0]["metric"], "accuracy")
        self.assertEqual(result["body"][0]["value"], 1.0)
    
    def test_error_handling(self):
        """Test PostProcessor error handling"""
        def failing_func(data, context):
            raise RuntimeError("Post error")
        
        processor = PostProcessor(failing_func)
        event = {
            "process_type": "postprocess",
            "data": {
                "prompt": "test",
                "inference_output": "test",
                "gold": "test"
            }
        }
        
        with self.assertRaises(PostprocessError):
            processor.process(event, None)


if __name__ == "__main__":
    unittest.main()