import torch
import torch.nn as nn
import math
from typing import Dict, Any, Optional
from .base_architecture import CustomArchitecture

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_linear(out)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        attended = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attended))
        
        # Feed forward
        fed_forward = self.feed_forward(x)
        x = self.norm2(x + self.dropout(fed_forward))
        
        return x

class CustomTransformerArchitecture(CustomArchitecture):
    def __init__(self, config: Dict[str, Any]):
        self.vocab_size = config.get('vocab_size', 30000)
        self.d_model = config.get('d_model', 512)
        self.num_heads = config.get('num_heads', 8)
        self.num_layers = config.get('num_layers', 6)
        self.d_ff = config.get('d_ff', 2048)
        self.dropout = config.get('dropout', 0.1)
        self.max_seq_length = config.get('max_seq_length', 512)
        self.num_classes = config.get('num_classes', 2)
        
        super().__init__(config)
    
    def build_layers(self):
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = self.create_positional_encoding()
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                self.d_model,
                self.num_heads,
                self.d_ff,
                self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        self.classifier = nn.Linear(self.d_model, self.num_classes)
        self.dropout = nn.Dropout(self.dropout)
    
    def create_positional_encoding(self) -> torch.Tensor:
        pos_encoding = torch.zeros(self.max_seq_length, self.d_model)
        position = torch.arange(0, self.max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)
        )
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add positional encoding to embeddings
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :x.size(1)].to(x.device)
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.classifier(x)
        
        return x 