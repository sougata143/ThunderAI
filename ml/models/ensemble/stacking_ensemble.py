from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .base_ensemble import BaseEnsemble

class MetaLearner(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

class StackingEnsemble(BaseEnsemble):
    def __init__(
        self,
        models: List[Any],
        meta_learner: Optional[nn.Module] = None,
        num_classes: int = 2
    ):
        super().__init__(models)
        input_size = len(models) * num_classes
        self.meta_learner = meta_learner or MetaLearner(
            input_size=input_size,
            hidden_size=input_size * 2,
            num_classes=num_classes
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.meta_learner.to(self.device)
        self.optimizer = torch.optim.Adam(self.meta_learner.parameters())
        
    def train_meta_learner(
        self,
        validation_data: Dict[str, Any],
        epochs: int = 10,
        batch_size: int = 32
    ):
        # Get base model predictions
        base_predictions = []
        for model in self.models:
            preds = []
            for text in validation_data['texts']:
                pred = model.predict(text)
                preds.append(pred['probabilities'])
            base_predictions.append(preds)
        
        # Prepare training data
        X = torch.tensor(np.concatenate(base_predictions, axis=1), dtype=torch.float32)
        y = torch.tensor(validation_data['labels'], dtype=torch.long)
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train meta-learner
        criterion = nn.CrossEntropyLoss()
        self.meta_learner.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.meta_learner(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def aggregate_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Combine base model predictions
        probs = [pred['probabilities'] for pred in predictions]
        combined_probs = torch.tensor(np.concatenate(probs), dtype=torch.float32)
        combined_probs = combined_probs.unsqueeze(0).to(self.device)
        
        # Get meta-learner prediction
        self.meta_learner.eval()
        with torch.no_grad():
            meta_output = self.meta_learner(combined_probs)
            probabilities = torch.softmax(meta_output, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
        
        return {
            'prediction': prediction.item(),
            'confidence': confidence.item(),
            'probabilities': probabilities[0].tolist()
        } 