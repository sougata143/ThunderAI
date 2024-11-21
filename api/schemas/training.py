from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Optional

class TrainingData(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "texts": ["positive text", "negative text"],
            "labels": [1, 0]
        }
    })

    texts: List[str] = Field(min_length=1, description="List of text samples for training")
    labels: List[int] = Field(min_length=1, description="List of labels corresponding to the texts")

    @property
    def num_samples(self) -> int:
        return len(self.texts)

    @field_validator("labels")
    @classmethod
    def validate_lengths(cls, v: List[int], info):
        if len(info.data.get("texts", [])) != len(v):
            raise ValueError("Number of texts and labels must match")
        return v 

class TrainingConfig(BaseModel):
    modelType: str = Field(..., description="Type of model to train (bert, gpt, transformer, lstm)")
    params: Dict[str, float] = Field(..., description="Training parameters")
    
    @field_validator('params')
    @classmethod
    def validate_params(cls, params):
        # Ensure epochs is a positive integer
        if 'epochs' in params:
            params['epochs'] = int(params['epochs'])
            if params['epochs'] < 1:
                raise ValueError("Epochs must be at least 1")
        
        # Validate other parameters
        if 'learningRate' in params and params['learningRate'] <= 0:
            raise ValueError("Learning rate must be positive")
            
        if 'batchSize' in params:
            params['batchSize'] = int(params['batchSize'])
            if params['batchSize'] < 1:
                raise ValueError("Batch size must be at least 1")
                
        if 'validationSplit' in params:
            if not 0 <= params['validationSplit'] <= 1:
                raise ValueError("Validation split must be between 0 and 1")
                
        return params

class TrainingResponse(BaseModel):
    modelId: str
    status: str
    message: str

class TrainingStatus(BaseModel):
    modelId: str
    status: str
    progress: float
    currentEpoch: Optional[int] = None
    metrics: Optional[Dict[str, list]] = None