from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis
from typing import Any

class CacheManager:
    def __init__(self, redis_url: str = "redis://localhost"):
        self.redis_url = redis_url
        self.redis = None

    async def init_cache(self):
        self.redis = aioredis.from_url(self.redis_url)
        FastAPICache.init(RedisBackend(self.redis), prefix="thunderai-cache")

    async def close(self):
        if self.redis:
            await self.redis.close()

    @staticmethod
    def cache_response(expire: int = 60):
        return cache(expire=expire)
    
    async def get(self, key: str) -> Any:
        return await FastAPICache.get(key)
    
    async def set(self, key: str, value: Any, expire: int = 60):
        await FastAPICache.set(key, value, expire=expire) 