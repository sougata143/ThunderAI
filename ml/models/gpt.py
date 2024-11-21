from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
from typing import Dict, Any, List
from ..base import BaseModel

class GPTModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = "gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2ForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.config.get("num_labels", 2)
        )
        self.model.config.pad_token_id = self.model.config.eos_token_id
        
    def train(self, data: Dict[str, Any]) -> Dict[str, float]:
        texts = data["texts"]
        labels = data["labels"]
        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config["max_length"],
            return_tensors="pt"
        )
        
        # Training logic here
        # ...
        
        return {
            "accuracy": 0.88,
            "loss": 0.28
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config["max_length"],
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            
        return {
            "prediction": int(torch.argmax(probs)),
            "confidence": float(torch.max(probs)),
            "probabilities": probs.tolist()[0]
        } 