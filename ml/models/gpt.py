import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config
from typing import Dict, Any, List, Optional, Union
from .base_model import BaseModel
import logging
import numpy as np
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class GPTModel(BaseModel):
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {
            "pretrained_model": "gpt2",
            "num_labels": 2,
            "dropout": 0.1,
            "max_length": 1024,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
            "num_return_sequences": 1,
            "use_cache": True,
            "gradient_checkpointing": True
        }
        super().__init__(config)
        
        # Load pretrained model and tokenizer
        self.gpt_config = GPT2Config.from_pretrained(
            config["pretrained_model"],
            output_hidden_states=True,
            output_attentions=True
        )
        
        self.gpt = GPT2Model.from_pretrained(
            config["pretrained_model"],
            config=self.gpt_config
        )
        
        if config.get("gradient_checkpointing", True):
            self.gpt.gradient_checkpointing_enable()
            
        self.tokenizer = GPT2Tokenizer.from_pretrained(config["pretrained_model"])
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Classification head
        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Linear(self.gpt.config.n_embd, config["num_labels"])
        
        # Additional layers for improved capabilities
        self.layer_norm = nn.LayerNorm(self.gpt.config.n_embd)
        self.attention_weights = nn.Linear(self.gpt.config.n_embd, 1)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        return_dict: bool = True
    ):
        outputs = self.gpt(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=self.config.get("use_cache", True),
            return_dict=return_dict
        )
        
        # Get hidden states and attention weights
        hidden_states = outputs.last_hidden_state
        attention_weights = outputs.attentions
        
        # Apply weighted attention
        attention_scores = self.attention_weights(hidden_states).squeeze(-1)
        attention_probs = F.softmax(attention_scores, dim=1)
        weighted_states = torch.bmm(
            attention_probs.unsqueeze(1),
            hidden_states
        ).squeeze(1)
        
        # Apply layer normalization and dropout
        normalized_states = self.layer_norm(weighted_states)
        dropped_states = self.dropout(normalized_states)
        
        # Get logits for classification
        logits = self.classifier(dropped_states)
        
        return logits, attention_probs if return_dict else (logits,)
        
    def generate_text(
        self,
        prompt: Union[str, List[str]],
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """Generate text based on prompt"""
        try:
            # Prepare inputs
            if isinstance(prompt, str):
                prompt = [prompt]
                
            encoded = self.tokenizer(
                prompt,
                padding=True,
                truncation=True,
                max_length=self.config["max_length"],
                return_tensors="pt"
            ).to(self.device)
            
            # Set generation parameters
            gen_kwargs = {
                "max_length": max_length or self.config["max_length"],
                "min_length": min_length,
                "temperature": self.config["temperature"],
                "top_k": self.config["top_k"],
                "top_p": self.config["top_p"],
                "repetition_penalty": self.config["repetition_penalty"],
                "num_return_sequences": self.config["num_return_sequences"],
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                **kwargs
            }
            
            # Generate text
            with torch.no_grad():
                output_sequences = self.gpt.generate(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                    **gen_kwargs
                )
            
            # Decode generated sequences
            generated_texts = []
            for sequence in output_sequences:
                text = self.tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                generated_texts.append(text)
                
            return generated_texts
            
        except Exception as e:
            logger.error(f"Text generation error: {str(e)}")
            raise
            
    def train_step(self, batch):
        """Enhanced training step with gradient clipping and loss scaling"""
        input_ids, attention_mask, labels = batch
        
        # Forward pass
        outputs, attention_probs = self(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Calculate primary loss
        classification_loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # Add attention diversity loss
        attention_diversity_loss = self._compute_attention_diversity_loss(attention_probs)
        
        # Combine losses
        total_loss = classification_loss + 0.1 * attention_diversity_loss
        
        # Calculate metrics
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean()
        
        # Apply gradient clipping
        if hasattr(self, 'clip_grad_norm'):
            torch.nn.utils.clip_grad_norm_(
                self.parameters(),
                self.config.get('max_grad_norm', 1.0)
            )
        
        return {
            "loss": total_loss,
            "accuracy": accuracy.item(),
            "classification_loss": classification_loss.item(),
            "attention_diversity_loss": attention_diversity_loss.item()
        }
        
    def _compute_attention_diversity_loss(self, attention_probs):
        """Compute loss to encourage diverse attention patterns"""
        # Calculate entropy of attention distributions
        entropy = -(attention_probs * torch.log(attention_probs + 1e-10)).sum(dim=1)
        # Return negative entropy to maximize diversity
        return -entropy.mean()
        
    def validate_step(self, batch):
        """Validation step with additional metrics"""
        with torch.no_grad():
            metrics = self.train_step(batch)
            
            # Add additional validation metrics
            outputs, attention_probs = self(
                input_ids=batch[0],
                attention_mask=batch[1]
            )
            predictions = torch.argmax(outputs, dim=1)
            
            # Calculate F1 score
            f1 = self._compute_f1_score(predictions, batch[2])
            metrics['f1_score'] = f1
            
            return metrics
            
    def _compute_f1_score(self, predictions, labels):
        """Compute F1 score for binary classification"""
        predictions = predictions.cpu().numpy()
        labels = labels.cpu().numpy()
        
        true_positives = np.sum((predictions == 1) & (labels == 1))
        false_positives = np.sum((predictions == 1) & (labels == 0))
        false_negatives = np.sum((predictions == 0) & (labels == 1))
        
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1
        
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