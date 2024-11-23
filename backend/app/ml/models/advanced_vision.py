import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, Any, Union, List, Tuple
import numpy as np
from PIL import Image
import timm
from torch.cuda.amp import autocast
from .base import BaseModel

class AdvancedVisionModel(BaseModel):
    """Advanced Vision model with state-of-the-art architectures and training strategies."""
    
    SUPPORTED_ARCHITECTURES = {
        # Standard architectures
        'resnet': 'resnet50',
        'efficientnet': 'tf_efficientnet_b0_ns',
        'vit': 'vit_base_patch16_224',
        # Advanced architectures
        'swin': 'swin_base_patch4_window7_224',
        'convnext': 'convnext_base',
        'deit': 'deit_base_patch16_224',
        'beit': 'beit_base_patch16_224'
    }
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.architecture = config.get('architecture', 'resnet')
        self.model_name = self.SUPPORTED_ARCHITECTURES.get(
            self.architecture,
            config.get('model_name', 'resnet50')
        )
        self.num_classes = config.get('num_classes', 1000)
        self.pretrained = config.get('pretrained', True)
        
        # Advanced configurations
        self.use_mixed_precision = config.get('use_mixed_precision', True)
        self.use_ema = config.get('use_ema', True)  # Exponential Moving Average
        self.use_augmix = config.get('use_augmix', True)
        self.use_cutmix = config.get('use_cutmix', True)
        self.cutmix_prob = config.get('cutmix_prob', 0.5)
        
        # Initialize advanced augmentations
        self.train_transform = self._create_advanced_transforms(train=True)
        self.eval_transform = self._create_advanced_transforms(train=False)
        
        # Build model
        self.model = self.build_model()
        self.model.to(self.device)
        
        # Initialize EMA if enabled
        if self.use_ema:
            self.ema_model = self._create_ema_model()
        
        # Initialize mixed precision if enabled
        if self.use_mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()
    
    def _create_advanced_transforms(self, train: bool = True) -> transforms.Compose:
        """Create advanced data augmentation pipeline."""
        if train:
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def _create_ema_model(self) -> nn.Module:
        """Create Exponential Moving Average model copy."""
        ema_model = type(self.model)(self.config)
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.to(self.device)
        ema_model.eval()
        return ema_model
    
    def _update_ema_model(self, decay: float = 0.999) -> None:
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, model_param in zip(
                self.ema_model.parameters(),
                self.model.parameters()
            ):
                ema_param.data.mul_(decay).add_(
                    model_param.data,
                    alpha=1 - decay
                )
    
    def build_model(self) -> nn.Module:
        """Build model using timm library for advanced architectures."""
        model = timm.create_model(
            self.model_name,
            pretrained=self.pretrained,
            num_classes=self.num_classes,
            drop_rate=0.1,
            drop_path_rate=0.1
        )
        
        if 'vit' in self.model_name or 'deit' in self.model_name:
            # Add advanced attention mechanisms for transformer models
            model.head = nn.Sequential(
                nn.LayerNorm(model.head.in_features),
                nn.Linear(model.head.in_features, 2 * model.head.in_features),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(2 * model.head.in_features, self.num_classes)
            )
        
        return model
    
    def _apply_cutmix(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix augmentation."""
        if np.random.random() > self.cutmix_prob:
            return images, labels
        
        batch_size = images.size(0)
        indices = torch.randperm(batch_size).to(self.device)
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]
        
        lam = np.random.beta(1.0, 1.0)
        
        image_h, image_w = images.shape[2:]
        cx = np.random.uniform(0, image_w)
        cy = np.random.uniform(0, image_h)
        w = image_w * np.sqrt(1 - lam)
        h = image_h * np.sqrt(1 - lam)
        x0 = int(np.round(max(cx - w / 2, 0)))
        x1 = int(np.round(min(cx + w / 2, image_w)))
        y0 = int(np.round(max(cy - h / 2, 0)))
        y1 = int(np.round(min(cy + h / 2, image_h)))
        
        images[:, :, y0:y1, x0:x1] = shuffled_images[:, :, y0:y1, x0:x1]
        
        # Adjust labels
        lam = 1 - ((x1 - x0) * (y1 - y0) / (image_h * image_w))
        return images, (labels, shuffled_labels, lam)
    
    def train_step(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Advanced training step with mixed precision and augmentations."""
        self.model.train()
        
        # Apply CutMix
        if self.use_cutmix:
            images, labels_info = self._apply_cutmix(images, labels)
        
        if self.use_mixed_precision and torch.cuda.is_available():
            with autocast():
                outputs = self.model(images)
                if self.use_cutmix and isinstance(labels_info, tuple):
                    labels, shuffled_labels, lam = labels_info
                    loss_fn = nn.CrossEntropyLoss()
                    loss = lam * loss_fn(outputs, labels) + \
                           (1 - lam) * loss_fn(outputs, shuffled_labels)
                else:
                    loss_fn = nn.CrossEntropyLoss()
                    loss = loss_fn(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            outputs = self.model(images)
            if self.use_cutmix and isinstance(labels_info, tuple):
                labels, shuffled_labels, lam = labels_info
                loss_fn = nn.CrossEntropyLoss()
                loss = lam * loss_fn(outputs, labels) + \
                       (1 - lam) * loss_fn(outputs, shuffled_labels)
            else:
                loss_fn = nn.CrossEntropyLoss()
                loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        
        # Update EMA model
        if self.use_ema:
            self._update_ema_model()
        
        return loss.item()
    
    def predict_with_cam(
        self,
        image: Image.Image
    ) -> Tuple[List[Dict[str, Union[int, float]]], np.ndarray]:
        """Predict with Class Activation Mapping."""
        self.model.eval()
        input_tensor = self.eval_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get model outputs
            features = []
            def hook_fn(module, input, output):
                features.append(output)
            
            # Register hook to get features
            if hasattr(self.model, 'features'):
                hook = self.model.features[-1].register_forward_hook(hook_fn)
            else:
                # For transformer models, get attention maps
                hook = self.model.blocks[-1].register_forward_hook(hook_fn)
            
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Get top predictions
            top_k_probs, top_k_indices = torch.topk(probabilities, k=5)
            predictions = [
                {
                    'class_id': idx.item(),
                    'probability': prob.item()
                }
                for idx, prob in zip(top_k_indices[0], top_k_probs[0])
            ]
            
            # Generate CAM
            features = features[0]
            weights = self.model.head[-1].weight
            cam = torch.einsum('ck,nchw->nhw', weights, features)
            cam = torch.relu(cam)
            cam = cam.cpu().numpy()
            
            # Normalize CAM
            cam = (cam - cam.min()) / (cam.max() - cam.min())
            
            hook.remove()
            
            return predictions, cam
