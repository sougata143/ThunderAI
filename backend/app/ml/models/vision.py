import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, Any, Union, List
import numpy as np
from PIL import Image

from .base import BaseModel

class VisionModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get('model_name', 'resnet50')
        self.num_classes = config.get('num_classes', 1000)
        self.pretrained = config.get('pretrained', True)
        self.fine_tune = config.get('fine_tune', True)
        
        # Standard image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.model = self.build_model()
        self.model.to(self.device)
    
    def build_model(self) -> nn.Module:
        # Load pre-trained model
        if self.model_name == 'resnet50':
            model = models.resnet50(pretrained=self.pretrained)
        elif self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(pretrained=self.pretrained)
        elif self.model_name == 'vit_b_16':
            model = models.vit_b_16(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Freeze or unfreeze layers based on fine_tune setting
        if not self.fine_tune:
            for param in model.parameters():
                param.requires_grad = False
        
        # Replace the final layer
        if isinstance(model, models.ResNet):
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
        elif isinstance(model, models.EfficientNet):
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, self.num_classes)
        elif isinstance(model, models.VisionTransformer):
            num_features = model.heads.head.in_features
            model.heads.head = nn.Linear(num_features, self.num_classes)
        
        return model
    
    def preprocess_data(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Preprocess image data for the model."""
        if isinstance(images, list):
            return torch.stack([self.transform(img) for img in images]).to(self.device)
        return self.transform(images).unsqueeze(0).to(self.device)
    
    def postprocess_output(self, output: torch.Tensor) -> np.ndarray:
        """Convert logits to probabilities."""
        probs = torch.softmax(output, dim=1)
        return probs.cpu().detach().numpy()
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor,
                  optimizer: torch.optim.Optimizer) -> float:
        """Perform one training step."""
        self.model.train()
        optimizer.zero_grad()
        
        outputs = self.model(images)
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
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
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
    
    def predict_top_k(self, image: Image.Image, k: int = 5) -> List[Dict[str, Union[int, float]]]:
        """Predict top-k classes for an image."""
        self.model.eval()
        with torch.no_grad():
            input_tensor = self.preprocess_data(image)
            output = self.model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            
            top_k_probs, top_k_indices = torch.topk(probabilities, k)
            
            results = []
            for i in range(k):
                results.append({
                    'class_id': top_k_indices[0][i].item(),
                    'probability': top_k_probs[0][i].item()
                })
            
            return results
