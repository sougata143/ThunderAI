import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from .base_model import BaseModel
import logging
import numpy as np
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class LSTMModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {
            "vocab_size": 30000,
            "embedding_dim": 300,
            "hidden_size": 512,
            "num_layers": 3,
            "dropout": 0.3,
            "bidirectional": True,
            "use_attention": True,
            "attention_heads": 8,
            "use_layer_norm": True,
            "use_residual": True,
            "use_highway": True,
            "highway_layers": 2
        }
        super().__init__(config)
        
        # Embedding layer
        self.embedding = nn.Embedding(
            config["vocab_size"],
            config["embedding_dim"],
            padding_idx=0
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config["embedding_dim"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"] if config["num_layers"] > 1 else 0,
            batch_first=True,
            bidirectional=config["bidirectional"]
        )
        
        # Calculate output dimension
        lstm_output_dim = config["hidden_size"] * (2 if config["bidirectional"] else 1)
        
        # Multi-head attention
        if config["use_attention"]:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=config["attention_heads"],
                dropout=config["dropout"]
            )
            
        # Layer normalization
        if config["use_layer_norm"]:
            self.layer_norm = nn.LayerNorm(lstm_output_dim)
            
        # Highway layers
        if config["use_highway"]:
            self.highway_layers = nn.ModuleList([
                Highway(lstm_output_dim) 
                for _ in range(config["highway_layers"])
            ])
            
        # Output layer
        self.output = nn.Linear(lstm_output_dim, config["vocab_size"])
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Apply attention if enabled
        if self.config["use_attention"]:
            attended_out, attention_weights = self.attention(
                lstm_out,
                lstm_out,
                lstm_out
            )
            
            # Residual connection if enabled
            if self.config["use_residual"]:
                lstm_out = lstm_out + attended_out
        
        # Apply layer normalization if enabled
        if self.config["use_layer_norm"]:
            lstm_out = self.layer_norm(lstm_out)
            
        # Apply highway layers if enabled
        if self.config["use_highway"]:
            for highway in self.highway_layers:
                lstm_out = highway(lstm_out)
                
        # Output projection
        output = self.output(lstm_out)
        
        return output, hidden
        
    def train_step(self, batch):
        """Enhanced training step with advanced loss computation"""
        inputs, targets = batch
        batch_size = inputs.size(0)
        
        # Forward pass
        outputs, _ = self(inputs)
        
        # Calculate loss with label smoothing
        smoothing = 0.1
        confidence = 1.0 - smoothing
        
        # Compute smoothed loss
        log_probs = F.log_softmax(outputs.view(-1, self.config["vocab_size"]), dim=-1)
        targets = targets.view(-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1))
        smooth_loss = -log_probs.sum(dim=-1, keepdim=True)
        
        loss = confidence * nll_loss + smoothing * smooth_loss / (self.config["vocab_size"] - 1)
        loss = loss.mean()
        
        # Calculate perplexity
        perplexity = torch.exp(loss)
        
        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=-1)
        accuracy = (predictions == targets).float().mean()
        
        return {
            "loss": loss,
            "perplexity": perplexity.item(),
            "accuracy": accuracy.item()
        }
        
    def validate_step(self, batch):
        """Validation step with additional metrics"""
        with torch.no_grad():
            metrics = self.train_step(batch)
            return metrics
            
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ):
        """Generate text with various decoding strategies"""
        self.eval()
        current_sequence = prompt.clone()
        
        with torch.no_grad():
            hidden = None
            
            for _ in range(max_length):
                # Get predictions
                output, hidden = self(current_sequence[:, -1:], hidden)
                logits = output[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                    
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                    
                # Sample from the filtered distribution
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                current_sequence = torch.cat([current_sequence, next_token], dim=1)
                
                # Stop if we predict the end token
                if next_token.item() == self.config.get("eos_token_id", 0):
                    break
                    
        return current_sequence

class Highway(nn.Module):
    """Highway layer for better gradient flow"""
    def __init__(self, size: int):
        super().__init__()
        self.transform = nn.Linear(size, size)
        self.gate = nn.Linear(size, size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transform = F.relu(self.transform(x))
        gate = torch.sigmoid(self.gate(x))
        return gate * transform + (1 - gate) * x