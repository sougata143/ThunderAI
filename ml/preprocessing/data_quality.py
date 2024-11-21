from typing import List, Dict, Any, Optional, Union
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging
from dataclasses import dataclass
from ..monitoring.custom_metrics import MetricsCollector

@dataclass
class DataQualityReport:
    """Data quality report containing validation results and statistics"""
    total_samples: int
    missing_values: Dict[str, float]
    duplicates: int
    class_distribution: Dict[str, float]
    feature_statistics: Dict[str, Dict[str, float]]
    data_drift: Optional[Dict[str, float]] = None
    validation_status: Dict[str, bool] = None
    quality_score: float = 0.0

class DataQualityValidator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.label_encoder = LabelEncoder()
        
        # Quality thresholds
        self.missing_threshold = config.get('missing_threshold', 0.1)
        self.duplicate_threshold = config.get('duplicate_threshold', 0.05)
        self.class_imbalance_threshold = config.get('class_imbalance_threshold', 0.8)
        self.min_samples_per_class = config.get('min_samples_per_class', 100)
        
    def validate_dataset(
        self,
        texts: List[str],
        labels: Optional[List[Any]] = None,
        reference_data: Optional[Dict[str, Any]] = None
    ) -> DataQualityReport:
        """Validate dataset quality and generate report"""
        try:
            # Basic statistics
            total_samples = len(texts)
            
            # Check missing values
            missing_values = self._check_missing_values(texts, labels)
            
            # Check duplicates
            duplicates = self._check_duplicates(texts)
            
            # Check class distribution if labels provided
            class_distribution = {}
            if labels is not None:
                class_distribution = self._analyze_class_distribution(labels)
            
            # Calculate feature statistics
            feature_stats = self._calculate_feature_statistics(texts)
            
            # Check data drift if reference data provided
            data_drift = None
            if reference_data:
                data_drift = self._check_data_drift(texts, reference_data)
            
            # Validate against thresholds
            validation_status = self._validate_thresholds(
                missing_values,
                duplicates / total_samples if total_samples > 0 else 1,
                class_distribution
            )
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(validation_status)
            
            # Create report
            report = DataQualityReport(
                total_samples=total_samples,
                missing_values=missing_values,
                duplicates=duplicates,
                class_distribution=class_distribution,
                feature_statistics=feature_stats,
                data_drift=data_drift,
                validation_status=validation_status,
                quality_score=quality_score
            )
            
            # Log metrics
            self._log_quality_metrics(report)
            
            return report
            
        except Exception as e:
            logging.error(f"Error in data validation: {str(e)}")
            raise
    
    def _check_missing_values(
        self,
        texts: List[str],
        labels: Optional[List[Any]] = None
    ) -> Dict[str, float]:
        """Check for missing values in texts and labels"""
        total = len(texts)
        missing = {
            'texts': sum(1 for t in texts if not t) / total if total > 0 else 1.0
        }
        
        if labels is not None:
            missing['labels'] = sum(1 for l in labels if l is None) / total if total > 0 else 1.0
        
        return missing
    
    def _check_duplicates(self, texts: List[str]) -> int:
        """Check for duplicate texts"""
        return len(texts) - len(set(texts))
    
    def _analyze_class_distribution(self, labels: List[Any]) -> Dict[str, float]:
        """Analyze class distribution and imbalance"""
        total = len(labels)
        counter = Counter(labels)
        return {str(k): v/total for k, v in counter.items()}
    
    def _calculate_feature_statistics(self, texts: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate text feature statistics"""
        lengths = [len(text.split()) for text in texts]
        unique_words = [len(set(text.lower().split())) for text in texts]
        
        return {
            'text_length': {
                'mean': np.mean(lengths),
                'std': np.std(lengths),
                'min': min(lengths),
                'max': max(lengths)
            },
            'vocabulary_size': {
                'mean': np.mean(unique_words),
                'std': np.std(unique_words),
                'min': min(unique_words),
                'max': max(unique_words)
            }
        }
    
    def _check_data_drift(
        self,
        texts: List[str],
        reference_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Check for data drift between current and reference data"""
        current_stats = self._calculate_feature_statistics(texts)
        ref_stats = self._calculate_feature_statistics(reference_data['texts'])
        
        drift_metrics = {}
        for feature in current_stats:
            drift = abs(
                current_stats[feature]['mean'] - ref_stats[feature]['mean']
            ) / ref_stats[feature]['std'] if ref_stats[feature]['std'] > 0 else 0
            drift_metrics[feature] = drift
        
        return drift_metrics
    
    def _validate_thresholds(
        self,
        missing_values: Dict[str, float],
        duplicate_ratio: float,
        class_distribution: Dict[str, float]
    ) -> Dict[str, bool]:
        """Validate data quality against thresholds"""
        validations = {
            'missing_values': all(v <= self.missing_threshold for v in missing_values.values()),
            'duplicates': duplicate_ratio <= self.duplicate_threshold
        }
        
        if class_distribution:
            # Check class imbalance
            min_class_ratio = min(class_distribution.values())
            max_class_ratio = max(class_distribution.values())
            validations['class_balance'] = min_class_ratio / max_class_ratio >= self.class_imbalance_threshold
            
            # Check minimum samples per class
            total_samples = sum(class_distribution.values())
            validations['min_samples'] = all(
                v * total_samples >= self.min_samples_per_class
                for v in class_distribution.values()
            )
        
        return validations
    
    def _calculate_quality_score(self, validation_status: Dict[str, bool]) -> float:
        """Calculate overall data quality score"""
        if not validation_status:
            return 0.0
        
        # Weight different validation aspects
        weights = {
            'missing_values': 0.3,
            'duplicates': 0.2,
            'class_balance': 0.3,
            'min_samples': 0.2
        }
        
        score = sum(
            weights.get(k, 0) * float(v)
            for k, v in validation_status.items()
        ) / sum(weights.get(k, 0) for k in validation_status.keys())
        
        return score
    
    def _log_quality_metrics(self, report: DataQualityReport):
        """Log data quality metrics"""
        self.metrics_collector.record_preprocessing_metric(
            'data_quality_score',
            report.quality_score
        )
        
        for check, status in report.validation_status.items():
            self.metrics_collector.record_preprocessing_metric(
                f'validation_{check}',
                float(status)
            ) 