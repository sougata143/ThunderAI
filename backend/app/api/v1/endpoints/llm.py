from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from ....models.llm.model import (
    LLMModel,
    LLMModelBase,
    TextGenerationRequest,
    TextGenerationResponse,
    ModelMetrics
)
from ....services.llm.model_service import LLMService
from ....core.deps import get_db
from ....db.mongodb import AsyncIOMotorClient

router = APIRouter()

@router.post("/models/", response_model=str, status_code=status.HTTP_201_CREATED)
async def create_model(
    model: LLMModelBase,
    db: AsyncIOMotorClient = Depends(get_db)
):
    """Create a new LLM model."""
    try:
        model_service = LLMService(db)
        model_id = await model_service.create_model(model.dict())
        return model_id
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating model: {str(e)}"
        )

@router.get("/models/{model_id}", response_model=LLMModel)
async def get_model(
    model_id: str,
    db: AsyncIOMotorClient = Depends(get_db)
):
    """Get a specific model by ID."""
    model_service = LLMService(db)
    model = await model_service.get_model(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    return model

@router.get("/models/", response_model=List[LLMModel])
async def list_models(
    skip: int = 0,
    limit: int = 10,
    db: AsyncIOMotorClient = Depends(get_db)
):
    """List all models with pagination."""
    model_service = LLMService(db)
    return await model_service.list_models(skip=skip, limit=limit)

@router.post("/models/{model_id}/generate", response_model=TextGenerationResponse)
async def generate_text(
    model_id: str,
    request: TextGenerationRequest,
    db: AsyncIOMotorClient = Depends(get_db)
):
    """Generate text using a specific model."""
    model_service = LLMService(db)
    try:
        response = await model_service.generate_text(model_id, request)
        return TextGenerationResponse(**response)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating text: {str(e)}"
        )

@router.put("/models/{model_id}/metrics")
async def update_metrics(
    model_id: str,
    metrics: ModelMetrics,
    db: AsyncIOMotorClient = Depends(get_db)
):
    """Update model metrics."""
    model_service = LLMService(db)
    success = await model_service.update_metrics(model_id, metrics)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    return {"message": "Metrics updated successfully"}

@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: str,
    db: AsyncIOMotorClient = Depends(get_db)
):
    """Delete a model."""
    model_service = LLMService(db)
    success = await model_service.delete_model(model_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
