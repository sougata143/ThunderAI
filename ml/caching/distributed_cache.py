from typing import Dict, Any, Optional, List, Callable
import redis
from redis.cluster import RedisCluster
import pickle
import logging
import time
from functools import wraps
from ..monitoring.custom_metrics import MetricsCollector

class DistributedCacheConfig:
    def __init__(
        self,
        nodes: List[Dict[str, Any]],
        password: Optional[str] = None,
        default_ttl: int = 3600,
        max_retries: int = 3,
        retry_delay: float = 0.1
    ):
        self.nodes = nodes
        self.password = password
        self.default_ttl = default_ttl
        self.max_retries = max_retries
        self.retry_delay = retry_delay

class DistributedCacheService:
    def __init__(self, config: DistributedCacheConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.cluster = self._create_cluster()
        
    def _create_cluster(self) -> RedisCluster:
        """Create Redis cluster connection"""
        try:
            return RedisCluster(
                startup_nodes=self.config.nodes,
                password=self.config.password,
                decode_responses=False,
                skip_full_coverage_check=True
            )
        except Exception as e:
            logging.error(f"Failed to create Redis cluster: {str(e)}")
            raise
    
    def _retry_operation(self, operation: Callable) -> Any:
        """Retry operation with exponential backoff"""
        for attempt in range(self.config.max_retries):
            try:
                return operation()
            except redis.RedisClusterException as e:
                if attempt == self.config.max_retries - 1:
                    raise
                delay = self.config.retry_delay * (2 ** attempt)
                time.sleep(delay)
                continue
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            start_time = time.time()
            value = self._retry_operation(lambda: self.cluster.get(key))
            
            if value is not None:
                self.metrics_collector.record_cache_metric(
                    'hit',
                    time.time() - start_time
                )
                return pickle.loads(value)
            
            self.metrics_collector.record_cache_metric(
                'miss',
                time.time() - start_time
            )
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
            ttl = ttl or self.config.default_ttl
            serialized_value = pickle.dumps(value)
            return self._retry_operation(
                lambda: self.cluster.setex(key, ttl, serialized_value)
            )
        except Exception as e:
            logging.error(f"Cache set error: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            return bool(self._retry_operation(
                lambda: self.cluster.delete(key)
            ))
        except Exception as e:
            logging.error(f"Cache delete error: {str(e)}")
            return False
    
    def clear_pattern(self, pattern: str) -> bool:
        """Clear cache entries matching pattern"""
        try:
            keys = self._retry_operation(
                lambda: self.cluster.keys(pattern)
            )
            if keys:
                return bool(self._retry_operation(
                    lambda: self.cluster.delete(*keys)
                ))
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
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    cache_key = f"{prefix}:{hash(str(args) + str(kwargs))}"
                
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                result = await func(*args, **kwargs)
                self.set(cache_key, result, ttl)
                return result
            return wrapper
        return decorator 