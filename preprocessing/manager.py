from typing import Dict, Any, Union, List
import pandas as pd
import numpy as np
from .pipelines import create_preprocessing_pipeline
from ..monitoring.custom_metrics import MetricsCollector
import time

class PreprocessingManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipelines = {}
        self.metrics_collector = MetricsCollector()
        
    def get_pipeline(self, data_type: str) -> Pipeline:
        if data_type not in self.pipelines:
            if data_type == "text":
                self.pipelines[data_type] = create_preprocessing_pipeline(
                    clean_text=self.config.get("clean_text", True),
                    normalize=self.config.get("normalize", True),
                    extract_entities=self.config.get("extract_entities", False)
                )
            # Add more data types as needed
            
        return self.pipelines[data_type]
    
    def preprocess(
        self,
        data: Union[str, List[str], pd.DataFrame],
        data_type: str
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            pipeline = self.get_pipeline(data_type)
            
            if isinstance(data, str):
                data = [data]
            
            processed_data = pipeline.transform(data)
            
            processing_time = time.time() - start_time
            
            # Record metrics
            self.metrics_collector.record_preprocessing(
                data_type=data_type,
                processing_time=processing_time,
                data_size=len(data)
            )
            
            return {
                "processed_data": processed_data,
                "processing_time": processing_time,
                "status": "success"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            } 