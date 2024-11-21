import logging
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import threading
from ..monitoring.custom_metrics import MetricsCollector

class LoggingService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.log_dir = Path(config.get("LOG_DIR", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_collector = MetricsCollector()
        self.setup_logging()
        
        # Thread-local storage for request context
        self.context = threading.local()
    
    def setup_logging(self):
        """Configure logging handlers"""
        # Application logger
        app_handler = RotatingFileHandler(
            self.log_dir / "app.log",
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        app_handler.setFormatter(self._get_formatter())
        
        # Security logger
        security_handler = RotatingFileHandler(
            self.log_dir / "security.log",
            maxBytes=10485760,
            backupCount=5
        )
        security_handler.setFormatter(self._get_formatter())
        
        # Model logger
        model_handler = RotatingFileHandler(
            self.log_dir / "model.log",
            maxBytes=10485760,
            backupCount=5
        )
        model_handler.setFormatter(self._get_formatter())
        
        # Configure loggers
        logging.getLogger('app').addHandler(app_handler)
        logging.getLogger('security').addHandler(security_handler)
        logging.getLogger('model').addHandler(model_handler)
    
    def _get_formatter(self):
        """Create log formatter"""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def set_context(self, **kwargs):
        """Set context for current thread"""
        for key, value in kwargs.items():
            setattr(self.context, key, value)
    
    def clear_context(self):
        """Clear context for current thread"""
        self.context = threading.local()
    
    def _get_context(self) -> Dict[str, Any]:
        """Get context for current thread"""
        return {
            key: getattr(self.context, key)
            for key in dir(self.context)
            if not key.startswith('_')
        }
    
    def log_event(
        self,
        event_type: str,
        message: str,
        level: str = "INFO",
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log an event with context"""
        logger = logging.getLogger(event_type)
        log_level = getattr(logging, level.upper())
        
        # Prepare log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            "context": self._get_context(),
            "extra": extra or {}
        }
        
        # Log the event
        logger.log(log_level, json.dumps(log_entry))
        
        # Update metrics
        self.metrics_collector.record_logging_metric(
            event_type=event_type,
            level=level
        )
    
    def log_model_event(
        self,
        model_name: str,
        event_type: str,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log model-related events"""
        self.log_event(
            event_type="model",
            message=f"Model {model_name}: {event_type}",
            extra={
                "model_name": model_name,
                "event_type": event_type,
                "metrics": metrics or {},
                **(extra or {})
            }
        )
    
    def log_security_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        success: bool = True,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log security-related events"""
        self.log_event(
            event_type="security",
            message=f"Security event: {event_type}",
            level="WARNING" if not success else "INFO",
            extra={
                "event_type": event_type,
                "user_id": user_id,
                "success": success,
                **(extra or {})
            }
        ) 