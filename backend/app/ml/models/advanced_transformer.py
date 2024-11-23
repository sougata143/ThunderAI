import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
import numpy as np
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    T5ForConditionalGeneration, GPT2LMHeadModel
)
from .base import BaseModel

class AdvancedTransformerModel(BaseModel):
    """Advanced Transformer model with multiple architectures and training strategies."""
    
    SUPPORTED_ARCHITECTURES = {
        'bert': 'bert-base-uncased',
        'roberta': 'roberta-base',
        't5': 't5-base',
        'gpt2': 'gpt2',
        'distilbert': 'distilbert-base-uncased',
        'xlnet': 'xlnet-base-cased'
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.architecture = config.get('architecture', 'bert')
        self.model_name = self.SUPPORTED_ARCHITECTURES.get(
            self.architecture,
            config.get('model_name', 'bert-base-uncased')
        )
        self.max_length = config.get('max_length', 512)
        self.num_labels = config.get('num_labels', 2)
        self.task_type = config.get('task_type', 'classification')
        
        # Advanced configurations
        self.gradient_checkpointing = config.get('gradient_checkpointing', False)
        self.use_mixed_precision = config.get('use_mixed_precision', True)
        self.use_dynamic_padding = config.get('use_dynamic_padding', True)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self.build_model()
        self.model.to(self.device)
        
        # Initialize mixed precision if enabled
        if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
    
    def build_model(self) -> nn.Module:
        """Build model based on architecture and task type."""
        if self.task_type == 'generation' and self.architecture == 't5':
            model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        elif self.task_type == 'generation' and self.architecture == 'gpt2':
            model = GPT2LMHeadModel.from_pretrained(self.model_name)
        else:
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_labels
            )
            base_model = AutoModel.from_pretrained(
                self.model_name,
                config=config
            )
            
            class AdvancedTransformerClassifier(nn.Module):
                def __init__(self, base_model, config):
                    super().__init__()
                    self.base_model = base_model
                    self.dropout = nn.Dropout(config.hidden_dropout_prob)
                    
                    # Multi-head attention for document-level features
                    self.doc_attention = nn.MultiheadAttention(
                        config.hidden_size,
                        num_heads=8,
                        dropout=0.1
                    )
                    
                    # Additional layers for better feature extraction
                    self.feature_layers = nn.Sequential(
                        nn.Linear(config.hidden_size, config.hidden_size),
                        nn.LayerNorm(config.hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    )
                    
                    # Output layers
                    self.classifier = nn.Linear(config.hidden_size, config.num_labels)
                
                def forward(self, input_ids, attention_mask, token_type_ids=None):
                    outputs = self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        output_hidden_states=True
                    )
                    
                    # Get hidden states from last 4 layers
                    last_hidden_states = outputs.hidden_states[-4:]
                    concatenated_states = torch.cat(
                        [state[:, 0] for state in last_hidden_states],
                        dim=-1
                    )
                    
                    # Apply document-level attention
                    doc_features = concatenated_states.view(
                        -1, 4, outputs.last_hidden_state.size(-1)
                    )
                    doc_features = doc_features.permute(1, 0, 2)
                    attended_features, _ = self.doc_attention(
                        doc_features, doc_features, doc_features
                    )
                    
                    # Pool attention outputs
                    pooled_features = attended_features.mean(dim=0)
                    
                    # Apply feature extraction layers
                    enhanced_features = self.feature_layers(pooled_features)
                    
                    # Final classification
                    logits = self.classifier(enhanced_features)
                    return logits
            
            model = AdvancedTransformerClassifier(base_model, config)
            
            # Enable gradient checkpointing if configured
            if self.gradient_checkpointing:
                model.base_model.gradient_checkpointing_enable()
        
        return model
    
    def preprocess_data(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Advanced preprocessing with dynamic padding and special tokens."""
        # Dynamic max length based on input if enabled
        if self.use_dynamic_padding:
            max_length = min(
                max(len(self.tokenizer.encode(text)) for text in texts) + 2,
                self.max_length
            )
        else:
            max_length = self.max_length
        
        # Advanced tokenization with special token handling
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt',
            return_token_type_ids=True,
            return_attention_mask=True,
            return_special_tokens_mask=True
        )
        
        # Move tensors to device
        return {k: v.to(self.device) for k, v in encodings.items()}
    
    def train_step(self, batch: Dict[str, torch.Tensor], labels: torch.Tensor,
                  optimizer: torch.optim.Optimizer) -> float:
        """Advanced training step with mixed precision and gradient clipping."""
        self.model.train()
        
        if self.use_mixed_precision and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                outputs = self.model(**batch)
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs, labels)
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Mixed precision optimizer step
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            outputs = self.model(**batch)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
        
        optimizer.zero_grad()
        return loss.item()
    
    def generate_text(self, prompt: str, max_length: Optional[int] = None) -> str:
        """Generate text using generative models (T5 or GPT-2)."""
        if self.task_type != 'generation':
            raise ValueError("Model not configured for text generation")
        
        self.model.eval()
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length or self.max_length,
                num_beams=4,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_attention_weights(self, text: str) -> Dict[str, torch.Tensor]:
        """Get attention weights for model interpretability."""
        self.model.eval()
        inputs = self.preprocess_data([text])
        
        with torch.no_grad():
            outputs = self.model.base_model(
                **inputs,
                output_attentions=True
            )
        
        # Average attention weights across all layers and heads
        attention_weights = torch.stack(outputs.attentions).mean(dim=(0, 1))
        return {
            'tokens': self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
            'attention_weights': attention_weights[0].cpu()
        }
