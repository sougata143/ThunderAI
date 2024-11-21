from typing import Dict, Any, List, Optional, Union, Callable
import torch
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import PreTrainedTokenizer
import nltk
import re
from ..monitoring.custom_metrics import MetricsCollector

class BasePreprocessor(BaseEstimator, TransformerMixin):
    """Base class for all preprocessors"""
    def __init__(self):
        self.metrics_collector = MetricsCollector()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        raise NotImplementedError
    
    def log_metrics(self, metric_name: str, value: float):
        self.metrics_collector.record_preprocessing_metric(metric_name, value)

class TextCleaner(BasePreprocessor):
    """Clean and normalize text data"""
    def __init__(
        self,
        remove_urls: bool = True,
        remove_numbers: bool = True,
        remove_special_chars: bool = True,
        lowercase: bool = True,
        min_length: int = 3
    ):
        super().__init__()
        self.remove_urls = remove_urls
        self.remove_numbers = remove_numbers
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        self.min_length = min_length
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        self.number_pattern = re.compile(r'\d+')
        self.special_char_pattern = re.compile(r'[^\w\s]')
    
    def transform(self, texts: List[str]) -> List[str]:
        cleaned_texts = []
        for text in texts:
            if self.lowercase:
                text = text.lower()
            
            if self.remove_urls:
                text = self.url_pattern.sub('', text)
            
            if self.remove_numbers:
                text = self.number_pattern.sub('', text)
            
            if self.remove_special_chars:
                text = self.special_char_pattern.sub('', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            if len(text.split()) >= self.min_length:
                cleaned_texts.append(text)
            
        self.log_metrics('cleaned_texts', len(cleaned_texts))
        return cleaned_texts

class TokenizerWrapper(BasePreprocessor):
    """Wrapper for different tokenizers"""
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, str] = 'word',
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ):
        super().__init__()
        self.tokenizer_type = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        if isinstance(tokenizer, str):
            if tokenizer == 'word':
                self.tokenizer = nltk.word_tokenize
            else:
                raise ValueError(f"Unknown tokenizer type: {tokenizer}")
        else:
            self.tokenizer = tokenizer
    
    def transform(self, texts: List[str]) -> Union[List[List[str]], Dict[str, torch.Tensor]]:
        if isinstance(self.tokenizer, PreTrainedTokenizer):
            encoded = self.tokenizer(
                texts,
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
                return_tensors='pt'
            )
            self.log_metrics('avg_sequence_length', encoded['input_ids'].size(1))
            return encoded
        else:
            tokenized = [self.tokenizer(text) for text in texts]
            if self.max_length:
                tokenized = [tokens[:self.max_length] for tokens in tokenized]
            self.log_metrics('avg_sequence_length', np.mean([len(t) for t in tokenized]))
            return tokenized

class DataAugmenter(BasePreprocessor):
    """Augment text data with various techniques"""
    def __init__(
        self,
        techniques: List[str] = ['synonym_replacement'],
        aug_probability: float = 0.3,
        max_aug_per_text: int = 1
    ):
        super().__init__()
        self.techniques = techniques
        self.aug_probability = aug_probability
        self.max_aug_per_text = max_aug_per_text
        
        # Load resources
        if 'synonym_replacement' in techniques:
            nltk.download('wordnet')
            from nltk.corpus import wordnet
            self.wordnet = wordnet
    
    def transform(self, texts: List[str]) -> List[str]:
        augmented_texts = []
        for text in texts:
            if np.random.random() < self.aug_probability:
                for _ in range(self.max_aug_per_text):
                    aug_text = self._apply_augmentation(text)
                    augmented_texts.append(aug_text)
            augmented_texts.append(text)
        
        self.log_metrics('augmented_texts', len(augmented_texts) - len(texts))
        return augmented_texts
    
    def _apply_augmentation(self, text: str) -> str:
        if 'synonym_replacement' in self.techniques:
            return self._synonym_replacement(text)
        return text
    
    def _synonym_replacement(self, text: str) -> str:
        words = text.split()
        for i, word in enumerate(words):
            if np.random.random() < self.aug_probability:
                synonyms = []
                for syn in self.wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        synonyms.append(lemma.name())
                if synonyms:
                    words[i] = np.random.choice(synonyms)
        return ' '.join(words)

class PreprocessingPipeline:
    """Orchestrate multiple preprocessing steps"""
    def __init__(
        self,
        steps: List[BasePreprocessor],
        cache_dir: Optional[str] = None
    ):
        self.steps = steps
        self.cache_dir = cache_dir
        self.metrics_collector = MetricsCollector()
    
    def process(
        self,
        texts: List[str],
        cache_key: Optional[str] = None
    ) -> Union[List[str], Dict[str, torch.Tensor]]:
        """Process texts through the pipeline"""
        # Try to load from cache
        if cache_key and self.cache_dir:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
        
        processed = texts
        start_time = time.time()
        
        for step in self.steps:
            processed = step.transform(processed)
        
        processing_time = time.time() - start_time
        self.metrics_collector.record_preprocessing_metric(
            'processing_time',
            processing_time
        )
        
        # Cache results
        if cache_key and self.cache_dir:
            self._save_to_cache(cache_key, processed)
        
        return processed
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        if self.cache_dir:
            cache_path = Path(self.cache_dir) / f"{cache_key}.pt"
            if cache_path.exists():
                return torch.load(cache_path)
        return None
    
    def _save_to_cache(self, cache_key: str, data: Any):
        if self.cache_dir:
            cache_path = Path(self.cache_dir) / f"{cache_key}.pt"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, cache_path) 