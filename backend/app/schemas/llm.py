from typing import Optional, List
from pydantic import BaseModel, Field, validator
from datetime import datetime
from bson import ObjectId

class LLMModelBase(BaseModel):
    name: str = Field(..., description="Name of the model")
    description: Optional[str] = Field(None, description="Description of the model")
    model_type: str = Field(..., description="Type of the model (e.g., GPT, BERT)")
    version: str = Field(..., description="Version of the model")
    parameters: Optional[dict] = Field(default_factory=dict, description="Model parameters")
    metadata: Optional[dict] = Field(default_factory=dict, description="Additional metadata")

class LLMModelCreate(LLMModelBase):
    pass

class LLMModelUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    model_type: Optional[str] = None
    version: Optional[str] = None
    parameters: Optional[dict] = None
    metadata: Optional[dict] = None

class LLMModel(LLMModelBase):
    id: str = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    is_active: bool = True

    class Config:
        allow_population_by_field_name = True
        json_encoders = {
            ObjectId: str,
            datetime: lambda v: v.isoformat(),
        }

class GenerationParameters(BaseModel):
    max_tokens: int = Field(default=100, ge=1, le=2048, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0, description="Presence penalty")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")
    model: Optional[str] = Field(default="gpt-3.5-turbo", description="OpenAI model to use")

    @validator('temperature', 'top_p', 'frequency_penalty', 'presence_penalty')
    def validate_float_range(cls, v, field):
        if v < field.field_info.ge or v > field.field_info.le:
            raise ValueError(f"{field.name} must be between {field.field_info.ge} and {field.field_info.le}")
        return v

    class Config:
        extra = "allow"  # Allow extra fields for flexibility

class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Text prompt for generation")
    parameters: Optional[GenerationParameters] = Field(default_factory=GenerationParameters, description="Generation parameters")

class GenerationResponse(BaseModel):
    model_id: str = Field(..., description="ID of the model used for generation")
    generated_text: str = Field(..., description="Generated text")
    prompt: str = Field(..., description="Original prompt")
    parameters: GenerationParameters = Field(..., description="Parameters used for generation")
    usage: dict = Field(..., description="Token usage statistics")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
