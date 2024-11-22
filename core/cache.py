from typing import Any, Optional
from redis import Redis
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

class CacheManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = Redis.from_url(redis_url, encoding="utf8", decode_responses=True)
        
    def init_cache(self):
        FastAPICache.init(
            RedisBackend(self.redis_client),
            prefix="thunderai-cache"
        )
    
    async def get(self, key: str) -> Any:
        return await self.redis_client.get(key)
    
    async def set(self, key: str, value: Any, expire: Optional[int] = None):
        await self.redis_client.set(key, value, ex=expire)
    
    async def delete(self, key: str):
        await self.redis_client.delete(key)
    
    async def clear_all(self):
        await self.redis_client.flushall() 