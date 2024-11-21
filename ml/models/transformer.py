import torch
import torch.nn as nn
from typing import Dict, Any
from .base_model import BaseModel
import logging

logger = logging.getLogger(__name__)

class TransformerModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {
            "vocab_size": 30000,
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "intermediate_size": 3072,
            "max_position_embeddings": 512
        }
        super().__init__(config)
        
        # Model architecture
        self.embedding = nn.Embedding(
            config["vocab_size"], 
            config["hidden_size"]
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["hidden_size"],
            nhead=config["num_attention_heads"],
            dim_feedforward=config["intermediate_size"],
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config["num_hidden_layers"]
        )
        
        self.output = nn.Linear(config["hidden_size"], config["vocab_size"])
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.output(x)
        
    def train_step(self, batch):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(
            outputs.view(-1, self.config["vocab_size"]),
            targets.view(-1)
        )
        return {"loss": loss}
        
    def validate_step(self, batch):
        with torch.no_grad():
            return self.train_step(batch)