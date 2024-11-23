from typing import List, Optional
from datetime import datetime
from bson import ObjectId
from fastapi import HTTPException, status
from app.schemas.llm import (
    LLMModel, LLMModelCreate, LLMModelUpdate,
    GenerationRequest, GenerationResponse, GenerationParameters
)
from app.core.config import settings
from app.db.mongodb import db
from openai import OpenAI, OpenAIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.db_name = settings.MONGODB_DB_NAME
        self.collection_name = "llm_models"
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    async def _get_collection(self):
        database = db.get_db()
        return database[self.collection_name]

    async def get_models(self, skip: int = 0, limit: int = 100) -> List[LLMModel]:
        """
        Retrieve all LLM models with pagination.
        """
        try:
            collection = await self._get_collection()
            cursor = collection.find().skip(skip).limit(limit)
            models = []
            async for document in cursor:
                # Convert ObjectId to string before creating LLMModel
                if '_id' in document and isinstance(document['_id'], ObjectId):
                    document['_id'] = str(document['_id'])
                models.append(LLMModel(**document))
            return models
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve models: {str(e)}"
            )

    async def create_model(self, model: LLMModelCreate) -> LLMModel:
        """
        Create a new LLM model.
        """
        try:
            model_dict = model.dict()
            model_dict["created_at"] = datetime.utcnow()
            model_dict["_id"] = ObjectId()  # Create ObjectId directly
            
            collection = await self._get_collection()
            await collection.insert_one(model_dict)
            model_dict["_id"] = str(model_dict["_id"])  # Convert to string for response
            return LLMModel(**model_dict)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create model: {str(e)}"
            )

    async def get_model(self, model_id: str) -> Optional[LLMModel]:
        """
        Get a specific LLM model by ID.
        """
        try:
            collection = await self._get_collection()
            try:
                # Try to convert string ID to ObjectId
                object_id = ObjectId(model_id)
            except Exception:
                # If conversion fails, the ID is invalid
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Invalid model ID format"
                )

            # Query MongoDB with the ObjectId
            document = await collection.find_one({"_id": object_id})
            if document:
                document["_id"] = str(document["_id"])  # Convert ObjectId to string
                return LLMModel(**document)
            return None
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve model: {str(e)}"
            )

    async def update_model(self, model_id: str, model_update: LLMModelUpdate) -> Optional[LLMModel]:
        """
        Update a specific LLM model.
        """
        try:
            update_dict = model_update.dict(exclude_unset=True)
            if update_dict:
                update_dict["updated_at"] = datetime.utcnow()
                collection = await self._get_collection()
                # Convert string ID to ObjectId for MongoDB query
                result = await collection.update_one(
                    {"_id": ObjectId(model_id)},
                    {"$set": update_dict}
                )
                if result.modified_count:
                    document = await collection.find_one({"_id": ObjectId(model_id)})
                    return LLMModel(**document)
            return None
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update model: {str(e)}"
            )

    async def delete_model(self, model_id: str) -> bool:
        """
        Delete a specific LLM model.
        """
        try:
            collection = await self._get_collection()
            try:
                # Try to convert string ID to ObjectId
                object_id = ObjectId(model_id)
            except Exception:
                logger.error(f"Invalid model ID format: {model_id}")
                return False

            # First check if model exists
            model = await collection.find_one({"_id": object_id})
            if not model:
                logger.warning(f"No model found with ID: {model_id}")
                return False

            logger.info(f"Attempting to delete model with ID: {model_id}")
            result = await collection.delete_one({"_id": object_id})
            
            if result.deleted_count == 0:
                logger.warning(f"Model found but deletion failed for ID: {model_id}")
                return False
                
            logger.info(f"Successfully deleted model with ID: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete model: {str(e)}"
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=10),
        retry=retry_if_exception_type((OpenAIError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _call_openai_api(self, messages, parameters):
        """
        Call OpenAI API with retry logic for transient errors.
        """
        try:
            # Convert Pydantic model to dict and extract model name from metadata
            params_dict = parameters.dict()
            model_name = params_dict.get("model", "gpt-3.5-turbo")
            
            # Create API call parameters with only supported fields
            api_params = {
                "model": model_name,
                "messages": messages,
                "max_tokens": parameters.max_tokens,
                "temperature": parameters.temperature,
                "top_p": parameters.top_p,
                "frequency_penalty": parameters.frequency_penalty,
                "presence_penalty": parameters.presence_penalty,
            }
            
            # Add optional stop sequences if provided
            if parameters.stop:
                api_params["stop"] = parameters.stop
            
            response = self.client.chat.completions.create(**api_params)
            return response
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            if "rate limit" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later."
                )
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling OpenAI API: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error calling OpenAI API: {str(e)}"
            )

    async def generate_text(self, model_id: str, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text using the specified LLM model.
        """
        try:
            # Get model details
            model = await self.get_model(model_id)
            if not model:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Model not found"
                )

            # Use model parameters or defaults
            parameters = request.parameters or GenerationParameters()
            
            # Add model name to parameters from model metadata
            if model.metadata and "model_name" in model.metadata:
                parameters.model = model.metadata["model_name"]
            
            # Prepare messages
            messages = [{"role": "user", "content": request.prompt}]
            
            # Call OpenAI API with retry logic
            response = await self._call_openai_api(messages, parameters)

            # Extract generated text from response
            generated_text = response.choices[0].message.content if response.choices else ""

            # Create response object
            return GenerationResponse(
                model_id=model_id,
                generated_text=generated_text,
                prompt=request.prompt,
                parameters=parameters,
                usage=response.usage.dict() if response.usage else {},
                created_at=datetime.utcnow()
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in generate_text: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate text: {str(e)}"
            )
