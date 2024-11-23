import torch
import numpy as np
from typing import Dict, Any, List, Union, Optional, Tuple
import shap
import lime
import lime.lime_text
import lime.lime_image
from captum.attr import (
    IntegratedGradients,
    LayerGradCam,
    DeepLift,
    NoiseTunnel,
    visualization
)
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from transformers import AutoTokenizer
import json
import logging

logger = logging.getLogger(__name__)

class ModelExplainer:
    """Advanced model explainability class supporting multiple interpretation techniques."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        task_type: str = 'classification',
        modality: str = 'text'
    ):
        self.model = model
        self.device = device
        self.task_type = task_type
        self.modality = modality
        
        # Initialize explainers based on modality
        if modality == 'text':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.text_explainer = lime.lime_text.LimeTextExplainer(
                class_names=['negative', 'positive']
            )
        elif modality == 'vision':
            self.image_explainer = lime.lime_image.LimeImageExplainer()
        
        # Initialize Captum explainers
        self.integrated_gradients = IntegratedGradients(self.model)
        self.deep_lift = DeepLift(self.model)
        self.grad_cam = LayerGradCam(self.model, self.model.layer4[-1])
    
    def explain_prediction(
        self,
        input_data: Union[str, Image.Image, torch.Tensor],
        target_class: Optional[int] = None,
        method: str = 'integrated_gradients',
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """Generate explanations using multiple interpretation techniques."""
        if self.modality == 'text':
            return self._explain_text(input_data, target_class, method, num_samples)
        elif self.modality == 'vision':
            return self._explain_image(input_data, target_class, method, num_samples)
        else:
            raise ValueError(f"Unsupported modality: {self.modality}")
    
    def _explain_text(
        self,
        text: str,
        target_class: Optional[int],
        method: str,
        num_samples: int
    ) -> Dict[str, Any]:
        """Generate explanations for text input."""
        explanations = {}
        
        # LIME explanation
        if method == 'lime':
            def predictor(texts):
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    return_tensors='pt'
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                return probs
            
            exp = self.text_explainer.explain_instance(
                text,
                predictor,
                num_features=10,
                num_samples=num_samples
            )
            explanations['lime'] = {
                'word_importance': exp.as_list(),
                'visualization': exp.as_html()
            }
        
        # Integrated Gradients explanation
        elif method == 'integrated_gradients':
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(self.device)
            
            attributions = self.integrated_gradients.attribute(
                inputs.input_ids,
                target=target_class,
                n_steps=50
            )
            
            # Convert token attributions to word importance
            tokens = self.tokenizer.convert_ids_to_tokens(
                inputs.input_ids[0].cpu().numpy()
            )
            word_importance = list(zip(
                tokens,
                attributions.sum(dim=-1)[0].cpu().numpy()
            ))
            
            explanations['integrated_gradients'] = {
                'word_importance': word_importance,
                'attributions': attributions.cpu().numpy().tolist()
            }
        
        # SHAP explanation
        elif method == 'shap':
            explainer = shap.Explainer(self.model)
            shap_values = explainer(text)
            explanations['shap'] = {
                'values': shap_values.values.tolist(),
                'base_values': shap_values.base_values.tolist(),
                'data': shap_values.data.tolist()
            }
        
        return explanations
    
    def _explain_image(
        self,
        image: Union[Image.Image, torch.Tensor],
        target_class: Optional[int],
        method: str,
        num_samples: int
    ) -> Dict[str, Any]:
        """Generate explanations for image input."""
        if isinstance(image, Image.Image):
            image_tensor = self._preprocess_image(image)
        else:
            image_tensor = image
        
        explanations = {}
        
        # Grad-CAM explanation
        if method == 'grad_cam':
            attribution = self.grad_cam.attribute(
                image_tensor,
                target=target_class
            )
            
            # Generate heatmap
            heatmap = self._generate_heatmap(
                attribution.cpu().numpy()[0],
                image if isinstance(image, Image.Image) else None
            )
            
            explanations['grad_cam'] = {
                'heatmap': heatmap.tolist(),
                'attribution': attribution.cpu().numpy().tolist()
            }
        
        # Integrated Gradients explanation
        elif method == 'integrated_gradients':
            attribution = self.integrated_gradients.attribute(
                image_tensor,
                target=target_class,
                n_steps=50
            )
            
            # Generate visualization
            vis = visualization.visualize_image_attr(
                attribution.cpu().numpy()[0].transpose(1, 2, 0),
                original_image=np.array(image) if isinstance(image, Image.Image) else None,
                method='heat_map',
                sign='all',
                show_colorbar=True
            )
            
            explanations['integrated_gradients'] = {
                'attribution': attribution.cpu().numpy().tolist(),
                'visualization': vis[0].tolist()
            }
        
        # LIME explanation
        elif method == 'lime':
            def predictor(images):
                batch = torch.stack([self._preprocess_image(img) for img in images])
                with torch.no_grad():
                    outputs = self.model(batch.to(self.device))
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                return probs
            
            exp = self.image_explainer.explain_instance(
                np.array(image),
                predictor,
                top_labels=5,
                hide_color=0,
                num_samples=num_samples
            )
            
            # Get explanation for target class
            target = target_class if target_class is not None else exp.top_labels[0]
            temp, mask = exp.get_image_and_mask(
                target,
                positive_only=True,
                num_features=10,
                hide_rest=False
            )
            
            explanations['lime'] = {
                'segments': exp.segments.tolist(),
                'explanation_map': mask.tolist()
            }
        
        return explanations
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input."""
        # Implement preprocessing based on model requirements
        # This is a placeholder implementation
        image = image.resize((224, 224))
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).float()
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def _generate_heatmap(
        self,
        attribution: np.ndarray,
        original_image: Optional[Image.Image] = None
    ) -> np.ndarray:
        """Generate heatmap visualization from attribution."""
        heatmap = cv2.applyColorMap(
            np.uint8(255 * attribution),
            cv2.COLORMAP_JET
        )
        
        if original_image is not None:
            original_array = np.array(original_image)
            heatmap = cv2.addWeighted(
                original_array,
                0.5,
                heatmap,
                0.5,
                0
            )
        
        return heatmap
    
    def get_feature_importance(
        self,
        input_data: Union[str, Image.Image, torch.Tensor],
        method: str = 'integrated_gradients'
    ) -> Dict[str, float]:
        """Get feature importance scores."""
        explanations = self.explain_prediction(input_data, method=method)
        
        if self.modality == 'text':
            if method == 'lime':
                return dict(explanations['lime']['word_importance'])
            elif method == 'integrated_gradients':
                return dict(explanations['integrated_gradients']['word_importance'])
        else:
            # For vision, return aggregated importance scores
            if method == 'grad_cam':
                attribution = np.array(explanations['grad_cam']['attribution'])
                return {
                    'spatial_importance': attribution.mean(axis=(0, 1)).tolist()
                }
        
        return {}
