class ProcessorError(Exception):
    """Base exception for processor errors"""
    pass

class PreprocessError(ProcessorError):
    """Exception raised during preprocessing"""
    pass

class PostprocessError(ProcessorError):
    """Exception raised during postprocessing"""
    pass