from typing import Dict, Any, Optional, List
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
from ..monitoring.custom_metrics import MetricsCollector
from .security_service import SecurityService

class SecurityMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        security_service: SecurityService,
        excluded_paths: List[str] = None
    ):
        super().__init__(app)
        self.security_service = security_service
        self.excluded_paths = excluded_paths or ["/health", "/metrics"]
        self.metrics_collector = MetricsCollector()
        self.bearer = HTTPBearer()
    
    async def dispatch(
        self,
        request: Request,
        call_next: Any
    ) -> Any:
        start_time = time.time()
        path = request.url.path
        
        try:
            # Skip security checks for excluded paths
            if not any(path.startswith(excluded) for excluded in self.excluded_paths):
                # Rate limiting
                client_id = request.client.host
                if not self.security_service.check_rate_limit(client_id):
                    self.metrics_collector.record_security_metric(
                        "rate_limit_exceeded",
                        client_id
                    )
                    raise HTTPException(
                        status_code=429,
                        detail="Too many requests"
                    )
                
                # Token validation
                credentials: HTTPAuthorizationCredentials = await self.bearer(request)
                if not credentials:
                    raise HTTPException(
                        status_code=401,
                        detail="Missing authentication token"
                    )
                
                token = credentials.credentials
                payload = self.security_service.verify_token(token)
                
                # Add user info to request state
                request.state.user = payload
            
            response = await call_next(request)
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            
            return response
            
        except HTTPException as e:
            self.metrics_collector.record_security_metric(
                "security_error",
                str(e.detail)
            )
            raise
        except Exception as e:
            logging.error(f"Security middleware error: {str(e)}")
            self.metrics_collector.record_security_metric(
                "security_error",
                str(e)
            )
            raise HTTPException(
                status_code=500,
                detail="Internal server error"
            )
        finally:
            # Record request duration
            duration = time.time() - start_time
            self.metrics_collector.record_request_metric(
                path=path,
                duration=duration
            )

class RequestValidator:
    """Validate and sanitize incoming requests"""
    @staticmethod
    def validate_prediction_request(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate prediction request data"""
        required_fields = ["text", "model_id"]
        for field in required_fields:
            if field not in data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        # Sanitize input
        data["text"] = RequestValidator.sanitize_text(data["text"])
        return data
    
    @staticmethod
    def sanitize_text(text: str) -> str:
        """Sanitize input text"""
        # Remove potentially dangerous characters
        text = text.replace("<", "&lt;").replace(">", "&gt;")
        return text.strip()

class SecurityHeaders:
    """Security headers configuration"""
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline';",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        } 