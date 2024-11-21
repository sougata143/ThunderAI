from typing import Dict, Any, List, Optional, Callable
import asyncio
import logging
from datetime import datetime, timedelta
from ..monitoring.custom_metrics import MetricsCollector

class CacheWarmer:
    def __init__(
        self,
        cache_service: Any,
        warm_up_interval: int = 3600,
        max_concurrent_tasks: int = 10
    ):
        self.cache_service = cache_service
        self.warm_up_interval = warm_up_interval
        self.max_concurrent_tasks = max_concurrent_tasks
        self.metrics_collector = MetricsCollector()
        self.warm_up_tasks: Dict[str, Callable] = {}
        self.is_running = False
    
    def register_warm_up_task(
        self,
        task_id: str,
        task: Callable,
        args: Optional[List] = None,
        kwargs: Optional[Dict] = None
    ):
        """Register a cache warm-up task"""
        self.warm_up_tasks[task_id] = {
            'task': task,
            'args': args or [],
            'kwargs': kwargs or {}
        }
    
    async def execute_warm_up_task(
        self,
        task_id: str,
        task_info: Dict[str, Any]
    ):
        """Execute a single warm-up task"""
        try:
            start_time = datetime.now()
            result = await task_info['task'](
                *task_info['args'],
                **task_info['kwargs']
            )
            duration = (datetime.now() - start_time).total_seconds()
            
            self.metrics_collector.record_cache_metric(
                'warm_up_success',
                duration,
                {'task_id': task_id}
            )
            
            return result
        except Exception as e:
            logging.error(f"Cache warm-up task {task_id} failed: {str(e)}")
            self.metrics_collector.record_cache_metric(
                'warm_up_failure',
                0,
                {'task_id': task_id, 'error': str(e)}
            )
    
    async def warm_up_cache(self):
        """Execute all registered warm-up tasks"""
        tasks = []
        for task_id, task_info in self.warm_up_tasks.items():
            task = asyncio.create_task(
                self.execute_warm_up_task(task_id, task_info)
            )
            tasks.append(task)
            
            if len(tasks) >= self.max_concurrent_tasks:
                await asyncio.gather(*tasks)
                tasks = []
        
        if tasks:
            await asyncio.gather(*tasks)
    
    async def start(self):
        """Start periodic cache warming"""
        self.is_running = True
        while self.is_running:
            await self.warm_up_cache()
            await asyncio.sleep(self.warm_up_interval)
    
    def stop(self):
        """Stop periodic cache warming"""
        self.is_running = False 