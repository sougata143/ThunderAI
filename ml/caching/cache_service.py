from typing import Dict, Any, Optional, Union, Callable
import redis
import pickle
from functools import wraps
import time
import logging
from ..monitoring.custom_metrics import MetricsCollector

class CacheService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.get('REDIS_HOST', 'localhost'),
            port=config.get('REDIS_PORT', 6379),
            db=config.get('REDIS_DB', 0),
            password=config.get('REDIS_PASSWORD'),
            decode_responses=False
        )
        self.metrics_collector = MetricsCollector()
        self.default_ttl = config.get('CACHE_TTL', 3600)  # 1 hour default
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            start_time = time.time()
            value = self.redis_client.get(key)
            
            if value is not None:
                self.metrics_collector.record_cache_metric('hit', time.time() - start_time)
                return pickle.loads(value)
            
            self.metrics_collector.record_cache_metric('miss', time.time() - start_time)
            return None
        except Exception as e:
            logging.error(f"Cache get error: {str(e)}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache"""
        try:
            ttl = ttl or self.default_ttl
            serialized_value = pickle.dumps(value)
            return self.redis_client.setex(key, ttl, serialized_value)
        except Exception as e:
            logging.error(f"Cache set error: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logging.error(f"Cache delete error: {str(e)}")
            return False
    
    def clear(self, pattern: str = "*") -> bool:
        """Clear cache entries matching pattern"""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return bool(self.redis_client.delete(*keys))
            return True
        except Exception as e:
            logging.error(f"Cache clear error: {str(e)}")
            return False
    
    def cached(
        self,
        prefix: str,
        ttl: Optional[int] = None,
        key_builder: Optional[Callable] = None
    ):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Build cache key
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    cache_key = f"{prefix}:{hash(str(args) + str(kwargs))}"
                
                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result
            return wrapper
        return decorator

class ModelCache:
    """Specialized cache for model artifacts and predictions"""
    def __init__(self, cache_service: CacheService):
        self.cache_service = cache_service
        self.model_ttl = 24 * 3600  # 24 hours for model artifacts
        self.prediction_ttl = 3600  # 1 hour for predictions
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get model from cache"""
        return self.cache_service.get(f"model:{model_id}")
    
    def cache_model(self, model_id: str, model: Any):
        """Cache model artifacts"""
        return self.cache_service.set(
            f"model:{model_id}",
            model,
            self.model_ttl
        )
    
    def get_prediction(
        self,
        model_id: str,
        input_hash: str
    ) -> Optional[Any]:
        """Get cached prediction"""
        return self.cache_service.get(
            f"pred:{model_id}:{input_hash}"
        )
    
    def cache_prediction(
        self,
        model_id: str,
        input_hash: str,
        prediction: Any
    ):
        """Cache model prediction"""
        return self.cache_service.set(
            f"pred:{model_id}:{input_hash}",
            prediction,
            self.prediction_ttl
        )
    
    def invalidate_model(self, model_id: str):
        """Invalidate model cache"""
        self.cache_service.delete(f"model:{model_id}")
        self.cache_service.clear(f"pred:{model_id}:*")

class DistributedCache:
    """Distributed caching using Redis Cluster"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_cluster = redis.RedisCluster(
            startup_nodes=[
                {"host": host, "port": port}
                for host, port in config.get('REDIS_NODES', [])
            ],
            decode_responses=False,
            password=config.get('REDIS_PASSWORD')
        )
        self.metrics_collector = MetricsCollector()
    
    def get_or_set(
        self,
        key: str,
        value_generator: Callable,
        ttl: Optional[int] = None
    ) -> Any:
        """Get value from cache or generate and cache it"""
        try:
            # Try to get from cache
            value = self.redis_cluster.get(key)
            if value is not None:
                self.metrics_collector.record_cache_metric('hit')
                return pickle.loads(value)
            
            # Generate value
            value = value_generator()
            
            # Cache value
            serialized_value = pickle.dumps(value)
            if ttl:
                self.redis_cluster.setex(key, ttl, serialized_value)
            else:
                self.redis_cluster.set(key, serialized_value)
            
            self.metrics_collector.record_cache_metric('miss')
            return value
        except Exception as e:
            logging.error(f"Distributed cache error: {str(e)}")
            return value_generator() 