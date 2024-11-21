from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import logging
from ..monitoring.custom_metrics import MetricsCollector

@dataclass
class DataQualityMetrics:
    missing_values: Dict[str, float]
    duplicates: int
    class_distribution: Dict[str, int]
    text_length_stats: Dict[str, float]
    vocabulary_size: int
    unique_tokens: int
    special_chars_ratio: float

class DataValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.min_text_length = config.get('min_text_length', 10)
        self.max_text_length = config.get('max_text_length', 512)
        self.min_unique_tokens = config.get('min_unique_tokens', 5)
        self.max_special_chars_ratio = config.get('max_special_chars_ratio', 0.3)
    
    def validate_dataset(
        self,
        texts: List[str],
        labels: List[Any]
    ) -> Dict[str, bool]:
        """Validate dataset quality"""
        quality_metrics = self.compute_quality_metrics(texts, labels)
        
        validations = {
            'text_length': all(
                self.min_text_length <= len(text.split()) <= self.max_text_length
                for text in texts
            ),
            'unique_tokens': quality_metrics.unique_tokens >= self.min_unique_tokens,
            'special_chars': quality_metrics.special_chars_ratio <= self.max_special_chars_ratio,
            'class_balance': self._check_class_balance(quality_metrics.class_distribution),
            'duplicates': quality_metrics.duplicates == 0,
            'missing_values': all(
                ratio == 0 for ratio in quality_metrics.missing_values.values()
            )
        }
        
        # Log validation results
        for check, result in validations.items():
            self.metrics_collector.record_validation_metric(
                check_name=check,
                success=result
            )
        
        return validations
    
    def compute_quality_metrics(
        self,
        texts: List[str],
        labels: List[Any]
    ) -> DataQualityMetrics:
        """Compute dataset quality metrics"""
        # Check missing values
        missing_values = {
            'texts': sum(1 for t in texts if not t) / len(texts),
            'labels': sum(1 for l in labels if l is None) / len(labels)
        }
        
        # Check duplicates
        duplicates = len(texts) - len(set(texts))
        
        # Compute class distribution
        class_distribution = {}
        for label in labels:
            class_distribution[str(label)] = class_distribution.get(str(label), 0) + 1
        
        # Compute text length statistics
        text_lengths = [len(text.split()) for text in texts]
        text_length_stats = {
            'mean': np.mean(text_lengths),
            'std': np.std(text_lengths),
            'min': min(text_lengths),
            'max': max(text_lengths)
        }
        
        # Compute vocabulary statistics
        all_tokens = [token for text in texts for token in text.split()]
        vocabulary_size = len(set(all_tokens))
        unique_tokens = len(set(all_tokens))
        
        # Compute special characters ratio
        special_chars = sum(
            sum(not c.isalnum() and not c.isspace() for c in text)
            for text in texts
        )
        total_chars = sum(len(text) for text in texts)
        special_chars_ratio = special_chars / total_chars if total_chars > 0 else 0
        
        return DataQualityMetrics(
            missing_values=missing_values,
            duplicates=duplicates,
            class_distribution=class_distribution,
            text_length_stats=text_length_stats,
            vocabulary_size=vocabulary_size,
            unique_tokens=unique_tokens,
            special_chars_ratio=special_chars_ratio
        )
    
    def _check_class_balance(self, class_distribution: Dict[str, int]) -> bool:
        """Check if classes are reasonably balanced"""
        if not class_distribution:
            return False
        
        counts = list(class_distribution.values())
        max_count = max(counts)
        min_count = min(counts)
        
        # Check if imbalance ratio is within acceptable range
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        return imbalance_ratio <= self.config.get('max_class_imbalance_ratio', 10.0)
    
    def split_dataset(
        self,
        texts: List[str],
        labels: List[Any],
        test_size: float = 0.2,
        validation_size: float = 0.1,
        stratify: bool = True
    ) -> Dict[str, Any]:
        """Split dataset into train, validation, and test sets"""
        # First split: train + validation vs test
        stratify_labels = labels if stratify else None
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts,
            labels,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=42
        )
        
        # Second split: train vs validation
        if validation_size > 0:
            val_size_adjusted = validation_size / (1 - test_size)
            stratify_labels = train_val_labels if stratify else None
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_val_texts,
                train_val_labels,
                test_size=val_size_adjusted,
                stratify=stratify_labels,
                random_state=42
            )
            
            return {
                'train': {'texts': train_texts, 'labels': train_labels},
                'validation': {'texts': val_texts, 'labels': val_labels},
                'test': {'texts': test_texts, 'labels': test_labels}
            }
        
        return {
            'train': {'texts': train_val_texts, 'labels': train_val_labels},
            'test': {'texts': test_texts, 'labels': test_labels}
        } 