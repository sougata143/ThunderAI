import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from .base_architecture import CustomArchitecture

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.dropout(x)

class CustomCNNArchitecture(CustomArchitecture):
    def __init__(self, config: Dict[str, Any]):
        self.vocab_size = config.get('vocab_size', 30000)
        self.embedding_dim = config.get('embedding_dim', 300)
        self.num_filters = config.get('num_filters', [128, 256, 512])
        self.kernel_sizes = config.get('kernel_sizes', [3, 4, 5])
        self.dropout = config.get('dropout', 0.1)
        self.num_classes = config.get('num_classes', 2)
        
        super().__init__(config)
    
    def build_layers(self):
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # Create parallel convolutional blocks
        self.conv_blocks = nn.ModuleList([
            ConvBlock(
                in_channels=self.embedding_dim,
                out_channels=n_filters,
                kernel_size=k_size,
                dropout=self.dropout
            )
            for n_filters, k_size in zip(self.num_filters, self.kernel_sizes)
        ])
        
        # Global max pooling
        self.global_maxpool = nn.AdaptiveMaxPool1d(1)
        
        # Fully connected layers
        total_filters = sum(self.num_filters)
        self.fc = nn.Sequential(
            nn.Linear(total_filters, total_filters // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(total_filters // 2, self.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding layer
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        
        # Parallel convolutions
        conv_outputs = []
        for conv_block in self.conv_blocks:
            conv_out = conv_block(x)
            pooled = self.global_maxpool(conv_out)
            conv_outputs.append(pooled)
        
        # Concatenate outputs
        x = torch.cat(conv_outputs, dim=1)
        x = x.squeeze(-1)
        
        # Classification
        return self.fc(x) 