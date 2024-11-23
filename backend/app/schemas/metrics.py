from typing import List, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class MetricEntry(BaseModel):
    timestamp: datetime
    train_accuracy: float
    val_accuracy: float
    train_loss: float
    val_loss: float
    f1_score: float
    precision: float
    recall: float

class MetricsSummary(BaseModel):
    avg_train_accuracy: float
    avg_val_accuracy: float
    avg_f1_score: float
    total_records: int

class MetricsOverview(BaseModel):
    metrics: List[MetricEntry]
    summary: MetricsSummary
