import torch
import torch.nn as nn
from typing import Dict, Any, List
import numpy as np
from transformers import AutoModel, AutoTokenizer

from .base import BaseModel

class TransformerModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model_name', 'bert-base-uncased')
        self.max_length = config.get('max_length', 512)
        self.num_labels = config.get('num_labels', 2)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = self.build_model()
        self.model.to(self.device)
    
    def build_model(self) -> nn.Module:
        base_model = AutoModel.from_pretrained(self.model_name)
        
        class TransformerClassifier(nn.Module):
            def __init__(self, base_model, num_labels):
                super().__init__()
                self.base_model = base_model
                self.dropout = nn.Dropout(0.1)
                self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)
            
            def forward(self, input_ids, attention_mask):
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                pooled_output = outputs.last_hidden_state[:, 0]
                pooled_output = self.dropout(pooled_output)
                logits = self.classifier(pooled_output)
                return logits
        
        return TransformerClassifier(base_model, self.num_labels)
    
    def preprocess_data(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize and prepare text data for the model."""
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encodings['input_ids'].to(self.device),
            'attention_mask': encodings['attention_mask'].to(self.device)
        }
    
    def postprocess_output(self, output: torch.Tensor) -> np.ndarray:
        """Convert logits to probabilities."""
        probs = torch.softmax(output, dim=1)
        return probs.cpu().detach().numpy()
    
    def train_step(self, batch: Dict[str, torch.Tensor], labels: torch.Tensor,
                  optimizer: torch.optim.Optimizer) -> float:
        """Perform one training step."""
        self.model.train()
        optimizer.zero_grad()
        
        outputs = self.model(**batch)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate the model on a dataset."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        loss_fn = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch, labels in dataloader:
                outputs = self.model(**batch)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        avg_loss = total_loss / len(dataloader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
