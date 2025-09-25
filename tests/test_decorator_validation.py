import unittest
from nova_custom_evaluation_sdk.processors.decorators import preprocess, postprocess


class TestDecoratorValidation(unittest.TestCase):
    
    def test_preprocess_decorator_wrong_parameter_count_zero(self):
        """Test preprocess decorator with zero parameters"""
        with self.assertRaises(TypeError) as cm:
            @preprocess
            def invalid_func():
                pass
        
        self.assertIn("must have exactly 2 parameters", str(cm.exception))
        self.assertIn("got 0", str(cm.exception))
    
    def test_preprocess_decorator_wrong_parameter_count_one(self):
        """Test preprocess decorator with one parameter"""
        with self.assertRaises(TypeError) as cm:
            @preprocess
            def invalid_func(data):
                pass
        
        self.assertIn("must have exactly 2 parameters", str(cm.exception))
        self.assertIn("got 1", str(cm.exception))
    
    def test_preprocess_decorator_wrong_parameter_count_three(self):
        """Test preprocess decorator with three parameters"""
        with self.assertRaises(TypeError) as cm:
            @preprocess
            def invalid_func(data, context, extra):
                pass
        
        self.assertIn("must have exactly 2 parameters", str(cm.exception))
        self.assertIn("got 3", str(cm.exception))
    
    def test_postprocess_decorator_wrong_parameter_count_zero(self):
        """Test postprocess decorator with zero parameters"""
        with self.assertRaises(TypeError) as cm:
            @postprocess
            def invalid_func():
                pass
        
        self.assertIn("must have exactly 2 parameters", str(cm.exception))
        self.assertIn("got 0", str(cm.exception))
    
    def test_postprocess_decorator_wrong_parameter_count_one(self):
        """Test postprocess decorator with one parameter"""
        with self.assertRaises(TypeError) as cm:
            @postprocess
            def invalid_func(data):
                pass
        
        self.assertIn("must have exactly 2 parameters", str(cm.exception))
        self.assertIn("got 1", str(cm.exception))
    
    def test_postprocess_decorator_wrong_parameter_count_three(self):
        """Test postprocess decorator with three parameters"""
        with self.assertRaises(TypeError) as cm:
            @postprocess
            def invalid_func(data, context, extra):
                pass
        
        self.assertIn("must have exactly 2 parameters", str(cm.exception))
        self.assertIn("got 3", str(cm.exception))


if __name__ == "__main__":
    unittest.main()