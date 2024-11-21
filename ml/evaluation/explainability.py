from typing import Dict, Any, List, Union, Optional
import torch
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizer
import shap
import lime.lime_text
from captum.attr import (
    IntegratedGradients,
    LayerIntegratedGradients,
    NoiseTunnel,
    visualization
)
from ..monitoring.custom_metrics import MetricsCollector

class ExplainabilityService:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = 'cuda'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.metrics_collector = MetricsCollector()
        
        # Initialize explainers
        self.integrated_gradients = IntegratedGradients(self.model)
        self.layer_ig = LayerIntegratedGradients(
            self.model,
            self.model.embeddings
        )
        
        # Initialize LIME explainer
        self.lime_explainer = lime.lime_text.LimeTextExplainer(
            class_names=['negative', 'positive']
        )
    
    def explain_prediction(
        self,
        text: str,
        method: str = 'integrated_gradients',
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate explanation for a prediction"""
        if method == 'integrated_gradients':
            explanation = self._explain_integrated_gradients(text, target_class)
        elif method == 'lime':
            explanation = self._explain_lime(text)
        elif method == 'shap':
            explanation = self._explain_shap(text)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
        
        # Log metrics
        self.metrics_collector.record_explanation_metric(
            method=method,
            computation_time=explanation['computation_time']
        )
        
        return explanation
    
    def _explain_integrated_gradients(
        self,
        text: str,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate explanation using Integrated Gradients"""
        start_time = time.time()
        
        # Tokenize input
        encoded = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate baseline (zeros)
        baseline = torch.zeros_like(encoded['input_ids'])
        
        # Calculate attributions
        attributions = self.integrated_gradients.attribute(
            encoded['input_ids'],
            baseline,
            target=target_class,
            n_steps=50,
            internal_batch_size=32
        )
        
        # Process attributions
        word_attributions = []
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        
        for token, attr in zip(tokens, attributions[0]):
            if token not in self.tokenizer.all_special_tokens:
                word_attributions.append({
                    'token': token,
                    'attribution': float(attr.sum())
                })
        
        computation_time = time.time() - start_time
        
        return {
            'method': 'integrated_gradients',
            'word_attributions': word_attributions,
            'computation_time': computation_time
        }
    
    def _explain_lime(self, text: str) -> Dict[str, Any]:
        """Generate explanation using LIME"""
        start_time = time.time()
        
        def predict_proba(texts):
            encoded = self.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                probs = torch.softmax(outputs.logits, dim=1)
            
            return probs.cpu().numpy()
        
        # Generate LIME explanation
        exp = self.lime_explainer.explain_instance(
            text,
            predict_proba,
            num_features=10,
            num_samples=100
        )
        
        # Process explanation
        word_attributions = []
        for word, importance in exp.as_list():
            word_attributions.append({
                'token': word,
                'attribution': importance
            })
        
        computation_time = time.time() - start_time
        
        return {
            'method': 'lime',
            'word_attributions': word_attributions,
            'computation_time': computation_time
        }
    
    def _explain_shap(self, text: str) -> Dict[str, Any]:
        """Generate explanation using SHAP"""
        start_time = time.time()
        
        # Create a background dataset
        background = [""] * 100  # Empty strings as background
        
        # Initialize SHAP explainer
        explainer = shap.Explainer(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        # Calculate SHAP values
        shap_values = explainer([text])
        
        # Process SHAP values
        word_attributions = []
        for token, value in zip(shap_values.data[0], shap_values.values[0]):
            if token not in self.tokenizer.all_special_tokens:
                word_attributions.append({
                    'token': token,
                    'attribution': float(value.sum())
                })
        
        computation_time = time.time() - start_time
        
        return {
            'method': 'shap',
            'word_attributions': word_attributions,
            'computation_time': computation_time
        }
    
    def visualize_attributions(
        self,
        text: str,
        attributions: List[Dict[str, float]],
        method: str
    ) -> Dict[str, Any]:
        """Generate visualization for attributions"""
        # Normalize attributions
        values = [attr['attribution'] for attr in attributions]
        min_val, max_val = min(values), max(values)
        norm_values = [(v - min_val) / (max_val - min_val) for v in values]
        
        # Generate HTML visualization
        html_output = []
        for token, norm_val in zip(
            [attr['token'] for attr in attributions],
            norm_values
        ):
            color = self._get_color_for_value(norm_val)
            html_output.append(
                f'<span style="background-color: {color}">{token}</span>'
            )
        
        return {
            'html': ' '.join(html_output),
            'raw_attributions': attributions,
            'method': method
        }
    
    def _get_color_for_value(self, value: float) -> str:
        """Generate color for attribution value"""
        if value > 0.5:
            # Positive attribution - green
            intensity = int(255 * (value - 0.5) * 2)
            return f'rgba(0, {intensity}, 0, 0.3)'
        else:
            # Negative attribution - red
            intensity = int(255 * (0.5 - value) * 2)
            return f'rgba({intensity}, 0, 0, 0.3)' 