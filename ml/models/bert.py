import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
from typing import Dict, Any, List, Optional, Union
from .base_model import BaseModel
import logging
import numpy as np
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class BERTModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {
            "pretrained_model": "bert-base-uncased",
            "num_labels": 2,
            "dropout": 0.1,
            "max_length": 512,
            "gradient_checkpointing": True,
            "use_weighted_loss": True,
            "use_focal_loss": True,
            "focal_loss_gamma": 2.0,
            "use_layer_norm": True,
            "hidden_dropout": 0.2,
            "attention_dropout": 0.1
        }
        super().__init__(config)
        
        # Load pretrained model and config
        self.bert_config = BertConfig.from_pretrained(
            config["pretrained_model"],
            hidden_dropout_prob=config["hidden_dropout"],
            attention_probs_dropout_prob=config["attention_dropout"],
            output_hidden_states=True,
            output_attentions=True
        )
        
        self.bert = BertModel.from_pretrained(
            config["pretrained_model"],
            config=self.bert_config
        )
        
        if config.get("gradient_checkpointing", True):
            self.bert.gradient_checkpointing_enable()
            
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrained_model"])
        
        # Enhanced model architecture
        self.dropout = nn.Dropout(config["dropout"])
        self.layer_norm = nn.LayerNorm(self.bert.config.hidden_size)
        
        # Multi-head attention for better feature extraction
        self.attention = nn.MultiheadAttention(
            embed_dim=self.bert.config.hidden_size,
            num_heads=8,
            dropout=config["attention_dropout"]
        )
        
        # Multiple classification heads
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config["hidden_dropout"]),
            nn.Linear(self.bert.config.hidden_size // 2, config["num_labels"])
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier initialization"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            return_dict=return_dict
        )
        
        # Get sequence output and attention weights
        sequence_output = outputs.last_hidden_state
        attention_weights = outputs.attentions
        
        # Apply self-attention
        attended_output, _ = self.attention(
            sequence_output,
            sequence_output,
            sequence_output,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        # Pool the outputs
        pooled_output = self._pool_outputs(attended_output, attention_mask)
        
        # Apply layer normalization and dropout
        normalized_output = self.layer_norm(pooled_output)
        dropped_output = self.dropout(normalized_output)
        
        # Get logits
        logits = self.classifier(dropped_output)
        
        if return_dict:
            return {
                "logits": logits,
                "hidden_states": outputs.hidden_states,
                "attentions": attention_weights
            }
        return logits
        
    def _pool_outputs(self, sequence_output: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Advanced pooling with attention weights"""
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = torch.sum(sequence_output * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        
        # If no mask, use mean pooling
        return torch.mean(sequence_output, dim=1)
        
    def train_step(self, batch):
        """Enhanced training step with multiple loss functions"""
        input_ids, attention_mask, labels = batch
        
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        logits = outputs["logits"]
        
        # Calculate losses
        if self.config.get("use_focal_loss", True):
            loss = self._compute_focal_loss(logits, labels)
        else:
            loss = nn.CrossEntropyLoss()(logits, labels)
            
        if self.config.get("use_weighted_loss", True):
            # Add L2 regularization
            l2_lambda = 0.01
            l2_reg = torch.tensor(0., device=self.device)
            for param in self.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            
        # Calculate metrics
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
        
        # Calculate F1 score
        f1 = self._compute_f1_score(predictions, labels)
        
        return {
            "loss": loss,
            "accuracy": accuracy.item(),
            "f1_score": f1,
            "attention_weights": outputs["attentions"][-1].detach().mean().item()
        }
        
    def _compute_focal_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        """Compute focal loss for imbalanced datasets"""
        gamma = self.config.get("focal_loss_gamma", 2.0)
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
        return focal_loss
        
    def _compute_f1_score(self, predictions: torch.Tensor, labels: torch.Tensor):
        """Compute F1 score"""
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        
        true_positives = np.sum((predictions == 1) & (labels == 1))
        false_positives = np.sum((predictions == 1) & (labels == 0))
        false_negatives = np.sum((predictions == 0) & (labels == 1))
        
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1
        
    def validate_step(self, batch):
        """Validation step with additional metrics"""
        with torch.no_grad():
            metrics = self.train_step(batch)
            return metrics
        
    def preprocess(self, texts):
        """Preprocess input texts"""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config["max_length"],
            return_tensors="pt"
        )
        return encoded