import unittest
from nova_custom_evaluation_sdk.processors.decorators import preprocess
from nova_custom_evaluation_sdk.processors.pre_processor import PreProcessor
from nova_custom_evaluation_sdk.processors.post_processor import PostProcessor
from nova_custom_evaluation_sdk.processors.exceptions import PreprocessError, PostprocessError


class TestProcessorDecorators(unittest.TestCase):
    
    def test_preprocess_decorator_creates_preprocessor(self):
        """Test that @preprocess decorator creates PreProcessor instance"""
        @preprocess
        def sample_func(data, context):
            return data
        
        self.assertIsInstance(sample_func, PreProcessor)
    
    def test_preprocess_decorator_success(self):
        """Test successful preprocessing with decorator"""
        @preprocess
        def clean_input(event: dict, context) -> dict:
            data = event.get("data", {})
            return {
                "statusCode": 200,
                "body": {
                    "system": data.get("system"),
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
    
    def test_preprocess_error_handling(self):
        """Test preprocessor error handling"""
        @preprocess
        def failing_processor(event: dict, context) -> dict:
            raise ValueError("Test error")
        
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
        self.assertIn("Test error", str(cm.exception))


class TestPreProcessor(unittest.TestCase):
    
    def test_preprocessor_direct_usage(self):
        """Test PreProcessor class direct usage"""
        def transform_data(data, context):
            input_data = data.get("data", {})
            return {
                "statusCode": 200,
                "body": {
                    "prompt": input_data.get("prompt", ""),
                    "gold": input_data.get("gold", "")
                }
            }
        
        processor = PreProcessor(transform_data)
        event = {
            "process_type": "preprocess",
            "data": {
                "prompt": "test data",
                "gold": "test response"
            }
        }
        result = processor.process(event, None)
        
        self.assertEqual(result["body"]["prompt"], "test data")
        self.assertEqual(result["body"]["gold"], "test response")
    
    def test_preprocessor_empty_input(self):
        """Test preprocessor with empty input"""
        @preprocess
        def handle_empty(event: dict, context) -> dict:
            data = event.get("data", {})
            return {
                "statusCode": 200,
                "body": {
                    "prompt": data.get("prompt", "default"),
                    "gold": data.get("gold", "default")
                }
            }
        
        event = {
            "process_type": "preprocess",
            "data": {}
        }
        
        with self.assertRaises(PreprocessError) as cm:
            handle_empty.process(event, None)
        
        self.assertIn("Validation failed", str(cm.exception))


class TestPostProcessor(unittest.TestCase):
    
    def test_postprocessor_direct_usage(self):
        """Test PostProcessor class direct usage"""
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
    
    def test_postprocessor_error_handling(self):
        """Test postprocessor error handling"""
        def failing_postprocessor(data, context):
            raise RuntimeError("Post processing failed")
        
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
        
        self.assertIn("PostProcessor failed", str(cm.exception))
        self.assertIn("Post processing failed", str(cm.exception))


if __name__ == "__main__":
    unittest.main()