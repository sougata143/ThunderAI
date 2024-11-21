from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import spacy
from collections import Counter
import logging
from ..monitoring.custom_metrics import MetricsCollector

@dataclass
class ValidationConfig:
    min_samples: int = 1000
    min_text_length: int = 10
    max_text_length: int = 512
    min_unique_words: int = 5
    max_missing_ratio: float = 0.1
    min_class_samples: int = 100
    class_balance_threshold: float = 0.1
    language: str = 'en'
    allowed_languages: List[str] = None
    custom_validators: Dict[str, callable] = None

@dataclass
class ValidationReport:
    is_valid: bool
    validation_scores: Dict[str, float]
    issues: List[str]
    statistics: Dict[str, Any]
    recommendations: List[str]

class DataValidator:
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize custom validators
        self.validators = {
            'sample_size': self._validate_sample_size,
            'text_length': self._validate_text_length,
            'vocabulary': self._validate_vocabulary,
            'missing_values': self._validate_missing_values,
            'class_distribution': self._validate_class_distribution,
            'language': self._validate_language,
            'data_quality': self._validate_data_quality
        }
        
        # Add custom validators
        if config.custom_validators:
            self.validators.update(config.custom_validators)
    
    def validate_dataset(
        self,
        texts: List[str],
        labels: Optional[List[Any]] = None,
        validation_types: Optional[List[str]] = None
    ) -> ValidationReport:
        """Validate dataset against all configured criteria"""
        if validation_types is None:
            validation_types = list(self.validators.keys())
        
        validation_scores = {}
        issues = []
        statistics = {}
        recommendations = []
        
        try:
            # Run all specified validations
            for validation_type in validation_types:
                if validation_type in self.validators:
                    score, issue, stats = self.validators[validation_type](texts, labels)
                    validation_scores[validation_type] = score
                    
                    if issue:
                        issues.append(issue)
                        recommendations.append(
                            self._get_recommendation(validation_type, stats)
                        )
                    
                    statistics.update(stats)
            
            # Calculate overall validity
            is_valid = all(
                score >= 0.7 for score in validation_scores.values()
            )
            
            # Record metrics
            self._record_validation_metrics(validation_scores)
            
            return ValidationReport(
                is_valid=is_valid,
                validation_scores=validation_scores,
                issues=issues,
                statistics=statistics,
                recommendations=recommendations
            )
            
        except Exception as e:
            logging.error(f"Error in data validation: {str(e)}")
            raise
    
    def _validate_sample_size(
        self,
        texts: List[str],
        labels: Optional[List[Any]]
    ) -> tuple:
        """Validate dataset size"""
        n_samples = len(texts)
        score = min(1.0, n_samples / self.config.min_samples)
        issue = None if score >= 0.7 else f"Insufficient samples: {n_samples}"
        stats = {"total_samples": n_samples}
        return score, issue, stats
    
    def _validate_text_length(
        self,
        texts: List[str],
        labels: Optional[List[Any]]
    ) -> tuple:
        """Validate text length distribution"""
        lengths = [len(text.split()) for text in texts]
        valid_lengths = [
            l for l in lengths
            if self.config.min_text_length <= l <= self.config.max_text_length
        ]
        
        score = len(valid_lengths) / len(texts)
        issue = None if score >= 0.7 else "Text length requirements not met"
        
        stats = {
            "avg_length": np.mean(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths)
        }
        return score, issue, stats
    
    def _validate_vocabulary(
        self,
        texts: List[str],
        labels: Optional[List[Any]]
    ) -> tuple:
        """Validate vocabulary richness"""
        all_words = []
        unique_words_per_text = []
        
        for text in texts:
            words = text.lower().split()
            unique_words = set(words)
            all_words.extend(words)
            unique_words_per_text.append(len(unique_words))
        
        vocab_size = len(set(all_words))
        avg_unique_words = np.mean(unique_words_per_text)
        
        score = min(1.0, avg_unique_words / self.config.min_unique_words)
        issue = None if score >= 0.7 else "Low vocabulary diversity"
        
        stats = {
            "vocabulary_size": vocab_size,
            "avg_unique_words": avg_unique_words
        }
        return score, issue, stats
    
    def _validate_missing_values(
        self,
        texts: List[str],
        labels: Optional[List[Any]]
    ) -> tuple:
        """Validate missing or empty values"""
        empty_texts = sum(1 for t in texts if not t.strip())
        missing_ratio = empty_texts / len(texts)
        
        score = 1.0 - (missing_ratio / self.config.max_missing_ratio)
        issue = None if score >= 0.7 else "High ratio of missing values"
        
        stats = {
            "missing_ratio": missing_ratio,
            "empty_texts": empty_texts
        }
        return score, issue, stats
    
    def _validate_class_distribution(
        self,
        texts: List[str],
        labels: Optional[List[Any]]
    ) -> tuple:
        """Validate class distribution and balance"""
        if not labels:
            return 1.0, None, {}
        
        class_counts = Counter(labels)
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        
        balance_ratio = min_count / max_count
        sufficient_samples = min_count >= self.config.min_class_samples
        
        score = balance_ratio if sufficient_samples else 0.0
        issue = None if score >= self.config.class_balance_threshold else "Class imbalance"
        
        stats = {
            "class_distribution": dict(class_counts),
            "balance_ratio": balance_ratio
        }
        return score, issue, stats
    
    def _validate_language(
        self,
        texts: List[str],
        labels: Optional[List[Any]]
    ) -> tuple:
        """Validate text language"""
        if not self.config.allowed_languages:
            return 1.0, None, {}
        
        languages = []
        for text in texts:
            doc = self.nlp(text[:100])  # Process first 100 chars for efficiency
            languages.append(doc.lang_)
        
        valid_langs = sum(
            1 for lang in languages
            if lang in self.config.allowed_languages
        )
        score = valid_langs / len(texts)
        
        issue = None if score >= 0.7 else "Mixed or invalid languages"
        stats = {"language_distribution": Counter(languages)}
        return score, issue, stats
    
    def _validate_data_quality(
        self,
        texts: List[str],
        labels: Optional[List[Any]]
    ) -> tuple:
        """Validate overall data quality"""
        quality_scores = []
        
        for text in texts:
            # Check for common quality issues
            doc = self.nlp(text)
            
            # Calculate quality score based on multiple factors
            score = sum([
                len(doc) > 0,  # Not empty
                any(ent.label_ for ent in doc.ents),  # Contains entities
                any(sent.root.dep_ == 'ROOT' for sent in doc.sents),  # Valid syntax
                doc.sentiment > 0  # Has sentiment
            ]) / 4.0
            
            quality_scores.append(score)
        
        avg_quality = np.mean(quality_scores)
        issue = None if avg_quality >= 0.7 else "Low data quality"
        
        stats = {
            "avg_quality_score": avg_quality,
            "quality_distribution": np.percentile(quality_scores, [25, 50, 75])
        }
        return avg_quality, issue, stats
    
    def _get_recommendation(
        self,
        validation_type: str,
        stats: Dict[str, Any]
    ) -> str:
        """Generate recommendations based on validation issues"""
        recommendations = {
            'sample_size': "Consider collecting more data samples",
            'text_length': "Filter or preprocess texts to meet length requirements",
            'vocabulary': "Enhance text diversity and vocabulary richness",
            'missing_values': "Handle or remove missing values",
            'class_distribution': "Apply class balancing techniques",
            'language': "Filter texts by language or use language-specific models",
            'data_quality': "Improve text quality through preprocessing"
        }
        return recommendations.get(validation_type, "Review and improve data quality")
    
    def _record_validation_metrics(self, validation_scores: Dict[str, float]):
        """Record validation metrics"""
        for validation_type, score in validation_scores.items():
            self.metrics_collector.record_preprocessing_metric(
                f'validation_{validation_type}',
                score
            ) 