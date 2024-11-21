import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
from ..base import BaseModel

class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        n_filters: int,
        filter_sizes: List[int],
        num_classes: int,
        dropout: float
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=n_filters,
                kernel_size=(fs, embedding_dim)
            )
            for fs in filter_sizes
        ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)
        
        conved = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

class CNNModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vocab_size = config.get("vocab_size", 10000)
        self.embedding_dim = config.get("embedding_dim", 300)
        self.n_filters = config.get("n_filters", 100)
        self.filter_sizes = config.get("filter_sizes", [3, 4, 5])
        self.num_classes = config.get("num_classes", 2)
        self.dropout = config.get("dropout", 0.5)
        
        self.model = TextCNN(
            self.vocab_size,
            self.embedding_dim,
            self.n_filters,
            self.filter_sizes,
            self.num_classes,
            self.dropout
        )
        
    def train(self, data: Dict[str, Any]) -> Dict[str, float]:
        # Training implementation
        pass
        
    def predict(self, text: str) -> Dict[str, Any]:
        # Prediction implementation
        pass 