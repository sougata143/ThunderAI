from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from ..user import PyObjectId

class TrainingConfig(BaseModel):
    batch_size: int = Field(default=32, description="Training batch size")
    learning_rate: float = Field(default=5e-5, description="Learning rate for optimization")
    epochs: int = Field(default=3, description="Number of training epochs")
    max_length: int = Field(default=512, description="Maximum sequence length")
    model_name: str = Field(default="gpt2", description="Base model architecture to use")
    
    class Config:
        schema_extra = {
            "example": {
                "batch_size": 32,
                "learning_rate": 5e-5,
                "epochs": 3,
                "max_length": 512,
                "model_name": "gpt2"
            }
        }

class ModelMetrics(BaseModel):
    perplexity: float = Field(..., description="Model perplexity score")
    bleu_score: Optional[float] = Field(None, description="BLEU score if applicable")
    accuracy: Optional[float] = Field(None, description="Model accuracy if applicable")
    loss: float = Field(..., description="Training loss")
    
    class Config:
        schema_extra = {
            "example": {
                "perplexity": 15.7,
                "bleu_score": 0.85,
                "accuracy": 0.92,
                "loss": 2.3
            }
        }

class LLMModelBase(BaseModel):
    name: str = Field(..., description="Name of the model")
    description: str = Field(..., description="Description of the model's purpose")
    architecture: str = Field(..., description="Model architecture (e.g., GPT, BERT)")
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)
    metrics: Optional[ModelMetrics] = None
    status: str = Field(default="initialized", description="Current status of the model")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(default="anonymous", description="User ID who created the model")

class LLMModelInDB(LLMModelBase):
    _id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            PyObjectId: str
        }

class LLMModel(LLMModelBase):
    id: str = Field(..., description="Model ID")
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
        schema_extra = {
            "example": {
                "id": "model_id_here",
                "name": "ThunderGPT-1",
                "description": "General-purpose language model for text generation",
                "architecture": "GPT-2",
                "training_config": {
                    "batch_size": 32,
                    "learning_rate": 5e-5,
                    "epochs": 3,
                    "max_length": 512,
                    "model_name": "gpt2"
                },
                "status": "initialized",
                "created_by": "user_id_here"
            }
        }

class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt for generation")
    max_length: int = Field(default=100, description="Maximum length of generated text")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Nucleus sampling parameter")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Once upon a time",
                "max_length": 100,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }

class TextGenerationResponse(BaseModel):
    generated_text: str = Field(..., description="Generated text output")
    prompt: str = Field(..., description="Original input prompt")
    model_id: str = Field(..., description="ID of the model used")
    generation_time: float = Field(..., description="Time taken for generation in seconds")
    
    class Config:
        schema_extra = {
            "example": {
                "generated_text": "Once upon a time in a magical forest...",
                "prompt": "Once upon a time",
                "model_id": "model_id_here",
                "generation_time": 0.45
            }
        }
