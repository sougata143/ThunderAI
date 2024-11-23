from pydantic import BaseModel, Field
from typing import Optional


class Settings(BaseModel):
    openai_api_key: Optional[str] = Field(None, description="OpenAI API Key")
    default_model: Optional[str] = Field(None, description="Default LLM Model")
    max_tokens: Optional[int] = Field(1000, description="Maximum tokens for generation", ge=1, le=4000)
    temperature: Optional[float] = Field(0.7, description="Temperature for text generation", ge=0.0, le=1.0)
    top_p: Optional[float] = Field(1.0, description="Top P for text generation", ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(0.0, description="Frequency penalty", ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(0.0, description="Presence penalty", ge=-2.0, le=2.0)


class SettingsUpdate(BaseModel):
    openai_api_key: Optional[str] = None
    default_model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
