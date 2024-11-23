import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from typing import Dict, Optional, Any, List
import time
import logging
from ...models.llm.model import LLMModel, LLMModelInDB, TextGenerationRequest, ModelMetrics
from motor.motor_asyncio import AsyncIOMotorDatabase
from bson import ObjectId

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = self.db.models
        self.models: Dict[str, Any] = {}  # Cache for loaded models
        self.tokenizers: Dict[str, Any] = {}  # Cache for tokenizers
        
    async def create_model(self, model_data: dict) -> str:
        """Create a new LLM model entry in the database."""
        try:
            model = LLMModelInDB(**model_data)
            result = await self.collection.insert_one(model.dict(by_alias=True, exclude_none=True))
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error creating model: {str(e)}")
            raise
    
    def _convert_model(self, model_data: dict) -> LLMModel:
        """Convert a model from DB format to API format."""
        model_id = str(model_data["_id"])
        del model_data["_id"]
        return LLMModel(id=model_id, **model_data)
    
    async def get_model(self, model_id: str) -> Optional[LLMModel]:
        """Retrieve a model by its ID."""
        model_data = await self.collection.find_one({"_id": ObjectId(model_id)})
        if model_data:
            return self._convert_model(model_data)
        return None
    
    async def list_models(self, skip: int = 0, limit: int = 10) -> List[LLMModel]:
        """List all models with pagination."""
        cursor = self.collection.find().skip(skip).limit(limit)
        models = await cursor.to_list(length=limit)
        return [self._convert_model(model) for model in models]
    
    def _load_model(self, model_id: str, model_name: str):
        """Load a model and tokenizer into memory."""
        if model_id not in self.models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                
                if torch.cuda.is_available():
                    model = model.to("cuda")
                
                self.models[model_id] = model
                self.tokenizers[model_id] = tokenizer
                
                logger.info(f"Loaded model {model_name} with ID {model_id}")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                raise
    
    async def generate_text(self, model_id: str, request: TextGenerationRequest) -> Dict[str, Any]:
        """Generate text using the specified model."""
        model_data = await self.get_model(model_id)
        if not model_data:
            raise ValueError(f"Model {model_id} not found")
        
        # Load model if not already loaded
        if model_id not in self.models:
            self._load_model(model_id, model_data.training_config.model_name)
        
        model = self.models[model_id]
        tokenizer = self.tokenizers[model_id]
        
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = tokenizer(request.prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate
            outputs = model.generate(
                **inputs,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            
            return {
                "generated_text": generated_text,
                "prompt": request.prompt,
                "model_id": model_id,
                "generation_time": generation_time
            }
            
        except Exception as e:
            logger.error(f"Error generating text with model {model_id}: {str(e)}")
            raise
    
    async def update_metrics(self, model_id: str, metrics: ModelMetrics) -> bool:
        """Update model metrics."""
        result = await self.collection.update_one(
            {"_id": ObjectId(model_id)},
            {"$set": {"metrics": metrics.dict()}}
        )
        return result.modified_count > 0
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        result = await self.collection.delete_one({"_id": ObjectId(model_id)})
        if model_id in self.models:
            del self.models[model_id]
            del self.tokenizers[model_id]
        return result.deleted_count > 0
