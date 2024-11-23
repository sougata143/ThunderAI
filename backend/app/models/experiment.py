"""
Temporary mock models for experiments.
Database functionality is currently disabled for simplification.
"""

from datetime import datetime
from typing import Dict, Optional, List, Any
from pydantic import BaseModel, Field
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class ExperimentBase(BaseModel):
    name: str
    description: Optional[str] = None
    model_type: str
    status: str = "pending"
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    metrics: Optional[Dict[str, float]] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)

class ExperimentCreate(ExperimentBase):
    pass

class ExperimentUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    model_type: Optional[str] = None
    status: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    tags: Optional[List[str]] = None

class ExperimentInDB(ExperimentBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: PyObjectId
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class Experiment(ExperimentBase):
    id: str = Field(alias="_id")
    user_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
