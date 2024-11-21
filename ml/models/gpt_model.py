import torch
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer, AdamW
from typing import Dict, Any, List, Optional
from ..base import BaseModel
import logging

class GPTClassifier(nn.Module):
    def __init__(self, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained('gpt2')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.gpt.config.n_embd, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled_output = hidden_states.mean(dim=1)  # Average pooling
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class GPTModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.num_labels = config.get('num_labels', 2)
        self.max_length = config.get('max_length', 1024)
        self.batch_size = config.get('batch_size', 8)
        self.learning_rate = config.get('learning_rate', 1e-5)
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPTClassifier(self.num_labels).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
    def train(self, data: Dict[str, Any]) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        texts = data['texts']
        labels = torch.tensor(data['labels']).to(self.device)
        
        # Tokenize and prepare input
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Forward pass
        outputs = self.model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Calculate metrics
        _, predictions = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(predictions == labels)
        total_predictions += labels.shape[0]
        total_loss += loss.item()
        
        accuracy = correct_predictions.float() / total_predictions
        
        return {
            'loss': total_loss,
            'accuracy': accuracy.item()
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        self.model.eval()
        
        # Tokenize input
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
        
        return {
            'prediction': prediction.item(),
            'confidence': confidence.item(),
            'probabilities': probabilities[0].tolist()
        }
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        
    def load(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config'] 