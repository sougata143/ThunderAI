from typing import List, Dict, Any, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from transformers import AutoTokenizer
import spacy
import re
from ..monitoring.custom_metrics import MetricsCollector

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        remove_urls: bool = True,
        remove_numbers: bool = True,
        remove_punctuation: bool = True,
        lowercase: bool = True,
        min_length: int = 3,
        language: str = 'en'
    ):
        self.remove_urls = remove_urls
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.min_length = min_length
        self.language = language
        
        # Load spaCy model
        self.nlp = spacy.load(f'{language}_core_web_sm')
        self.metrics_collector = MetricsCollector()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, texts: List[str]) -> List[str]:
        """Apply text preprocessing steps"""
        processed_texts = []
        
        for text in texts:
            if self.lowercase:
                text = text.lower()
            
            if self.remove_urls:
                text = re.sub(r'http\S+|www\S+|https\S+', '', text)
            
            if self.remove_numbers:
                text = re.sub(r'\d+', '', text)
            
            if self.remove_punctuation:
                text = re.sub(r'[^\w\s]', '', text)
            
            # SpaCy processing
            doc = self.nlp(text)
            tokens = [
                token.text for token in doc
                if not token.is_stop and len(token.text) >= self.min_length
            ]
            
            processed_text = ' '.join(tokens)
            if processed_text:
                processed_texts.append(processed_text)
        
        self.metrics_collector.record_preprocessing_metric(
            'processed_texts',
            len(processed_texts)
        )
        
        return processed_texts

class AdvancedTokenizer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.metrics_collector = MetricsCollector()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize texts using transformer tokenizer"""
        encoded = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        self.metrics_collector.record_preprocessing_metric(
            'tokenized_texts',
            len(texts)
        )
        
        return encoded

class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        scaler_type: str = 'standard',
        feature_range: tuple = (0, 1)
    ):
        self.scaler_type = scaler_type
        self.feature_range = feature_range
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler(feature_range=feature_range)
    
    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        return self.scaler.transform(X)

class EnhancedPreprocessingPipeline:
    def __init__(
        self,
        text_preprocessing_config: Dict[str, Any],
        tokenizer_config: Dict[str, Any],
        scaling_config: Optional[Dict[str, Any]] = None
    ):
        self.text_preprocessor = TextPreprocessor(**text_preprocessing_config)
        self.tokenizer = AdvancedTokenizer(**tokenizer_config)
        self.scaler = None
        
        if scaling_config:
            self.scaler = FeatureScaler(**scaling_config)
    
    def process(
        self,
        texts: List[str],
        additional_features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Process text data through the pipeline"""
        # Text preprocessing
        processed_texts = self.text_preprocessor.transform(texts)
        
        # Tokenization
        encoded = self.tokenizer.transform(processed_texts)
        
        result = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
        
        # Scale additional features if provided
        if additional_features is not None and self.scaler:
            if not hasattr(self.scaler, 'mean_'):
                self.scaler.fit(additional_features)
            
            scaled_features = self.scaler.transform(additional_features)
            result['additional_features'] = torch.tensor(scaled_features)
        
        return result 