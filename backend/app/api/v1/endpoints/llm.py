from typing import List
from fastapi import APIRouter, HTTPException, Depends, status
from app.schemas.llm import (
    LLMModel, LLMModelCreate, LLMModelUpdate,
    GenerationRequest, GenerationResponse
)
from app.services.llm_service import LLMService
from app.api.deps import get_current_active_user
from app.schemas.user import User

router = APIRouter()
llm_service = LLMService()

@router.get("/models", response_model=List[LLMModel])
async def get_models(
    current_user: User = Depends(get_current_active_user),
    skip: int = 0,
    limit: int = 100
):
    """
    Retrieve all LLM models.
    """
    try:
        models = await llm_service.get_models(skip=skip, limit=limit)
        return models
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve models: {str(e)}"
        )

@router.post("/models", response_model=LLMModel)
async def create_model(
    model: LLMModelCreate,
    current_user: User = Depends(get_current_active_user)
):
    """
    Create a new LLM model.
    """
    try:
        created_model = await llm_service.create_model(model)
        return created_model
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create model: {str(e)}"
        )

@router.get("/models/{model_id}", response_model=LLMModel)
async def get_model(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get a specific LLM model by ID.
    """
    try:
        model = await llm_service.get_model(model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        return model
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model: {str(e)}"
        )

@router.put("/models/{model_id}", response_model=LLMModel)
async def update_model(
    model_id: str,
    model_update: LLMModelUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """
    Update a specific LLM model.
    """
    try:
        updated_model = await llm_service.update_model(model_id, model_update)
        if not updated_model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        return updated_model
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update model: {str(e)}"
        )

@router.delete("/models/{model_id}")
async def delete_model(
    model_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete a specific LLM model.
    """
    try:
        # Delete the model
        success = await llm_service.delete_model(model_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        return {"message": "Model deleted successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model: {str(e)}"
        )

@router.post("/models/{model_id}/generate", response_model=GenerationResponse)
async def generate_text(
    model_id: str,
    request: GenerationRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Generate text using a specific LLM model.
    """
    try:
        response = await llm_service.generate_text(model_id, request)
        return response
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate text: {str(e)}"
        )
