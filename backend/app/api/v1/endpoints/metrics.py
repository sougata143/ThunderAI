from datetime import datetime, timedelta
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorCollection

from ....api.deps import get_current_active_user
from ....db.mongodb import db
from ....schemas.user import User
from ....schemas.metrics import MetricsOverview

router = APIRouter()

@router.get("/overview", response_model=MetricsOverview)
async def get_metrics_overview(
    current_user: User = Depends(get_current_active_user)
):
    """
    Get metrics overview for the last 30 days
    """
    # Get metrics collection
    metrics_collection: AsyncIOMotorCollection = db.get_db()["metrics"]
    
    # Calculate date range
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)
    
    # Query metrics for current user
    cursor = metrics_collection.find({
        "user_id": current_user.id,
        "timestamp": {"$gte": start_date, "$lte": end_date}
    }).sort("timestamp", 1)
    
    # Process metrics
    metrics = []
    async for doc in cursor:
        metrics.append({
            "timestamp": doc["timestamp"],
            "train_accuracy": doc.get("train_accuracy", 0.0),
            "val_accuracy": doc.get("val_accuracy", 0.0),
            "train_loss": doc.get("train_loss", 0.0),
            "val_loss": doc.get("val_loss", 0.0),
            "f1_score": doc.get("f1_score", 0.0),
            "precision": doc.get("precision", 0.0),
            "recall": doc.get("recall", 0.0)
        })
    
    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="No metrics found for the specified time range"
        )
    
    # Calculate averages
    num_metrics = len(metrics)
    avg_train_acc = sum(m["train_accuracy"] for m in metrics) / num_metrics
    avg_val_acc = sum(m["val_accuracy"] for m in metrics) / num_metrics
    avg_f1 = sum(m["f1_score"] for m in metrics) / num_metrics
    
    return {
        "metrics": metrics,
        "summary": {
            "avg_train_accuracy": avg_train_acc,
            "avg_val_accuracy": avg_val_acc,
            "avg_f1_score": avg_f1,
            "total_records": num_metrics
        }
    }
