from typing import Optional
from pydantic import BaseModel

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenPayload(BaseModel):
    sub: str  # user id
    exp: int  # expiration timestamp
    type: str = "access"  # token type (access, refresh, etc.)
    scopes: list[str] = []  # token scopes/permissions
