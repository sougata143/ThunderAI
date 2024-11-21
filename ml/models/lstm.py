import torch
import torch.nn as nn
from typing import Dict, Any, List
from ..base import BaseModel

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out[:, -1, :])
        return out

class LSTMModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vocab_size = config.get("vocab_size", 10000)
        self.embedding_dim = config.get("embedding_dim", 100)
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_classes = config.get("num_classes", 2)
        
        self.model = LSTMClassifier(
            self.vocab_size,
            self.embedding_dim,
            self.hidden_dim,
            self.num_classes
        )
        
    def train(self, data: Dict[str, Any]) -> Dict[str, float]:
        texts = data["texts"]
        labels = data["labels"]
        
        # Training logic here
        # ...
        
        return {
            "accuracy": 0.82,
            "loss": 0.35
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        # Preprocess text to tensor
        # ...
        
        with torch.no_grad():
            outputs = self.model(text_tensor)
            probs = torch.softmax(outputs, dim=1)
            
        return {
            "prediction": int(torch.argmax(probs)),
            "confidence": float(torch.max(probs))
        } 