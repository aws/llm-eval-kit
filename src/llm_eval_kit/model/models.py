"""Pydantic models for BYOC payload validation."""
 
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum 
 
class ProcessType(str, Enum):
    PREPROCESS = "preprocess"
    POSTPROCESS = "postprocess"
 
 
class PreprocessingData(BaseModel):
    """Data section for preprocessing payload."""
    model_config = {"extra": "forbid"}
    
    system: Optional[str] = Field(None, description="System prompt")
    prompt: str = Field(..., description="User prompt")
    gold: str = Field(..., description="Gold standard answer")
 
 
class PreprocessingPayload(BaseModel):
    """Payload for preprocessing Lambda function."""
    model_config = {"extra": "forbid"}
    
    process_type: ProcessType = Field(..., description="Process type")
    data: PreprocessingData = Field(..., description="Data to process")
    
    @validator('process_type')
    def validate_process_type(cls, v):
        if v != ProcessType.PREPROCESS:
            raise ValueError('process_type must be "preprocess" for preprocessing payload')
        return v
 
 
class PreprocessingResponseBody(BaseModel):
    """Body section for preprocessing response."""
    model_config = {"extra": "forbid"}
    
    system: Optional[str] = Field(None, description="Processed system prompt")
    prompt: str = Field(..., description="Processed user prompt")
    gold: str = Field(..., description="Processed gold standard")
 
 
class PreprocessingResponse(BaseModel):
    """Response from preprocessing Lambda function."""
    model_config = {"extra": "forbid"}
    
    statusCode: int = Field(..., description="HTTP status code")
    body: PreprocessingResponseBody = Field(..., description="Response body")
    
    @validator('statusCode')
    def validate_status_code(cls, v):
        if v != 200:
            raise ValueError('statusCode must be 200')
        return v
 
 
class PostprocessingDataItem(BaseModel):
    """Single item in postprocessing data array."""
    model_config = {"extra": "forbid"}
    
    prompt: str = Field(..., description="User prompt")
    inference_output: str = Field(..., description="Model response")
    gold: str = Field(..., description="Gold standard answer")
 
 
class PostprocessingPayload(BaseModel):
    """Payload for postprocessing Lambda function."""
    model_config = {"extra": "forbid"}
    
    process_type: ProcessType = Field(..., description="Process type")
    data: PostprocessingDataItem = Field(..., description="Data to process")
    
    @validator('process_type')
    def validate_process_type(cls, v):
        if v != ProcessType.POSTPROCESS:
            raise ValueError('process_type must be "postprocess" for postprocessing payload')
        return v
 
 
class MetricResult(BaseModel):
    """Single metric result."""
    model_config = {"extra": "forbid"}
    
    metric: str = Field(..., description="Metric name")
    value: Union[float, int] = Field(..., description="Metric value")


class PostprocessingResponse(BaseModel):
    """Response from postprocessing Lambda function."""
    model_config = {"extra": "forbid"}
    
    statusCode: int = Field(..., description="HTTP status code")
    body: List[MetricResult] = Field(..., description="List of computed metrics")
    
    @validator('statusCode')
    def validate_status_code(cls, v):
        if v != 200:
            raise ValueError('statusCode must be 200')
        return v