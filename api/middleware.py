from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import json

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            return Response(
                content=json.dumps({
                    "error": str(e),
                    "detail": "An internal server error occurred"
                }),
                status_code=500,
                media_type="application/json"
            ) 