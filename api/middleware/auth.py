from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from core.config import settings
from .base import BaseMiddleware
from starlette.responses import Response
from starlette.middleware.base import RequestResponseEndpoint

security = HTTPBearer()

class AuthMiddleware(BaseMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Allow OPTIONS requests to pass through for CORS preflight
        if request.method == "OPTIONS":
            return await call_next(request)

        # List of public endpoints that don't require authentication
        public_paths = [
            '/api/v1/token',
            '/api/v1/auth/login',
            '/api/v1/auth/register',
            '/api/v1/models/train',
            '/health',
            '/api/v1/models/ws',
            '/docs',           # Swagger UI
            '/redoc',         # ReDoc UI
            '/openapi.json'   # OpenAPI schema
        ]

        # Skip auth for public endpoints and docs
        if any(request.url.path.startswith(path) for path in public_paths):
            return await call_next(request)

        try:
            # Get authorization header
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                raise HTTPException(
                    status_code=401,
                    detail="Not authenticated",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Extract token from Bearer header
            scheme, token = auth_header.split()
            if scheme.lower() != 'bearer':
                raise HTTPException(
                    status_code=401,
                    detail="Invalid authentication scheme",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Verify token
            try:
                payload = jwt.decode(
                    token, 
                    settings.SECRET_KEY, 
                    algorithms=[settings.ALGORITHM]
                )
                # Add user info to request state
                request.state.user = payload
            except JWTError:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid token or expired token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            response = await call_next(request)
            return response

        except HTTPException as e:
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail},
                headers={"Access-Control-Allow-Origin": "*", **e.headers}
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"detail": str(e)},
                headers={"Access-Control-Allow-Origin": "*"}
            ) 