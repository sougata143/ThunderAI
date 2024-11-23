from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    is_public: Optional[bool] = False

class ProjectCreate(ProjectBase):
    user_id: str

class ProjectUpdate(ProjectBase):
    pass

class ProjectInDBBase(ProjectBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda dt: dt.isoformat()}

class Project(ProjectInDBBase):
    pass

class ProjectResponse(ProjectInDBBase):
    pass
