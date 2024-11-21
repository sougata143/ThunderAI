import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from .base_architecture import CustomArchitecture

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, attention_size: int = None):
        super().__init__()
        if attention_size is None:
            attention_size = hidden_size
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: [batch_size, seq_len, hidden_size]
        attention_weights = self.attention(hidden_states)  # [batch_size, seq_len, 1]
        
        if mask is not None:
            attention_weights = attention_weights.masked_fill(
                mask.unsqueeze(-1) == 0,
                float('-inf')
            )
        
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.bmm(
            attention_weights.transpose(1, 2),
            hidden_states
        )  # [batch_size, 1, hidden_size]
        
        return attended.squeeze(1), attention_weights.squeeze(-1)

class CustomRNNArchitecture(CustomArchitecture):
    def __init__(self, config: Dict[str, Any]):
        self.vocab_size = config.get('vocab_size', 30000)
        self.embedding_dim = config.get('embedding_dim', 300)
        self.hidden_size = config.get('hidden_size', 512)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.1)
        self.bidirectional = config.get('bidirectional', True)
        self.rnn_type = config.get('rnn_type', 'lstm')
        self.attention_size = config.get('attention_size', None)
        self.num_classes = config.get('num_classes', 2)
        
        super().__init__(config)
    
    def build_layers(self):
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # RNN layer
        rnn_class = getattr(nn, self.rnn_type.upper())
        self.rnn = rnn_class(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional,
            batch_first=True
        )
        
        # Attention layer
        rnn_output_size = self.hidden_size * (2 if self.bidirectional else 1)
        self.attention = AttentionLayer(rnn_output_size, self.attention_size)
        
        # Output layer
        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_size, rnn_output_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(rnn_output_size // 2, self.num_classes)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embedding
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # Pack sequence if lengths are provided
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # RNN
        rnn_output, _ = self.rnn(x)
        
        # Unpack if necessary
        if lengths is not None:
            rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
                rnn_output, batch_first=True
            )
        
        # Create attention mask if lengths are provided
        mask = None
        if lengths is not None:
            max_len = rnn_output.size(1)
            mask = torch.arange(max_len).expand(len(lengths), max_len).to(lengths.device)
            mask = mask < lengths.unsqueeze(1)
        
        # Apply attention
        context, attention_weights = self.attention(rnn_output, mask)
        
        # Classification
        output = self.classifier(context)
        
        return output, attention_weights 