from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime

class ExperimentBase(BaseModel):
    name: str
    model_type: str
    hyperparameters: Dict[str, Any]

class ExperimentCreate(ExperimentBase):
    pass

class ExperimentUpdate(BaseModel):
    name: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    status: Optional[str] = None

class ExperimentInDBBase(ExperimentBase):
    id: int
    metrics: Optional[Dict[str, Any]] = None
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    user_id: int

    class Config:
        from_attributes = True

class Experiment(ExperimentInDBBase):
    pass

class ModelBase(BaseModel):
    name: str
    version: str
    path: str
    accuracy: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class ModelCreate(ModelBase):
    experiment_id: int

class ModelUpdate(BaseModel):
    accuracy: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[str] = None

class ModelInDBBase(ModelBase):
    id: int
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    experiment_id: int

    class Config:
        from_attributes = True

class Model(ModelInDBBase):
    pass
