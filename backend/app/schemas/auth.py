from pydantic import BaseModel, EmailStr

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenPayload(BaseModel):
    sub: str | None = None
    scopes: list[str] = []

class Login(BaseModel):
    email: EmailStr
    password: str
