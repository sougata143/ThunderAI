from typing import List, Dict, Any, Optional, Union
import numpy as np
from .data_validation import DataValidator
from .feature_engineering import FeatureExtractor
from .tokenization import TokenizerFactory
from .enhanced_pipeline import EnhancedPreprocessingPipeline
from .augmentation import AdvancedTextAugmenter
from ..monitoring.custom_metrics import MetricsCollector
import logging

class PreprocessingOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        
        # Initialize components
        self.validator = DataValidator(config.get('validation_config', {}))
        self.feature_extractor = FeatureExtractor(config.get('feature_config', {}))
        self.tokenizer = TokenizerFactory.create_tokenizer(
            config.get('tokenizer_type', 'pretrained'),
            config.get('tokenizer_config', {})
        )
        self.augmenter = AdvancedTextAugmenter(config.get('augmentation_config', {}))
        self.pipeline = EnhancedPreprocessingPipeline(
            config.get('pipeline_config', {})
        )
    
    async def process_data(
        self,
        texts: List[str],
        labels: Optional[List[Any]] = None,
        augment: bool = False,
        extract_features: bool = True
    ) -> Dict[str, Any]:
        """Process data through the complete pipeline"""
        try:
            start_time = time.time()
            
            # Validate data
            if labels is not None:
                validation_results = self.validator.validate_dataset(texts, labels)
                if not all(validation_results.values()):
                    logging.warning(f"Data validation issues: {validation_results}")
            
            # Augment data if requested
            if augment:
                augmented_data = self.augmenter.augment(
                    texts,
                    techniques=self.config.get('augmentation_techniques', ['synonym'])
                )
                texts.extend(augmented_data)
                if labels is not None:
                    labels.extend(labels * len(augmented_data))
            
            # Extract features
            features = {}
            if extract_features:
                features = {
                    'statistical': self.feature_extractor.extract_statistical_features(texts),
                    'semantic': self.feature_extractor.extract_semantic_features(
                        texts,
                        method=self.config.get('semantic_method', 'transformer')
                    ),
                    'sentiment': self.feature_extractor.extract_sentiment_features(texts)
                }
            
            # Tokenize texts
            tokenized = self.tokenizer.encode(
                texts,
                padding=True,
                truncation=True
            )
            
            # Process through enhanced pipeline
            processed = self.pipeline.process(texts)
            
            # Combine all outputs
            result = {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'features': features,
                'processed_text': processed,
                'labels': labels if labels is not None else None
            }
            
            # Record metrics
            processing_time = time.time() - start_time
            self.metrics_collector.record_preprocessing_metric(
                'total_processing_time',
                processing_time
            )
            
            return result
            
        except Exception as e:
            logging.error(f"Error in preprocessing pipeline: {str(e)}")
            self.metrics_collector.record_preprocessing_metric(
                'preprocessing_error',
                1
            )
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        return self.feature_extractor.get_feature_importance()
    
    def save_pipeline_state(self, path: str):
        """Save pipeline state and configurations"""
        state = {
            'config': self.config,
            'tokenizer_state': self.tokenizer.save_pretrained(path),
            'feature_extractor_state': self.feature_extractor.save_state(path)
        }
        torch.save(state, f"{path}/pipeline_state.pt")
    
    def load_pipeline_state(self, path: str):
        """Load pipeline state and configurations"""
        state = torch.load(f"{path}/pipeline_state.pt")
        self.config = state['config']
        self.tokenizer = TokenizerFactory.from_pretrained(path)
        self.feature_extractor.load_state(path)

class PreprocessingPipelineBuilder:
    """Builder for creating preprocessing pipelines"""
    def __init__(self):
        self.config = {}
        self.components = []
    
    def add_validation(
        self,
        validation_config: Dict[str, Any]
    ) -> 'PreprocessingPipelineBuilder':
        self.config['validation_config'] = validation_config
        self.components.append('validation')
        return self
    
    def add_augmentation(
        self,
        augmentation_config: Dict[str, Any]
    ) -> 'PreprocessingPipelineBuilder':
        self.config['augmentation_config'] = augmentation_config
        self.components.append('augmentation')
        return self
    
    def add_feature_extraction(
        self,
        feature_config: Dict[str, Any]
    ) -> 'PreprocessingPipelineBuilder':
        self.config['feature_config'] = feature_config
        self.components.append('feature_extraction')
        return self
    
    def add_tokenization(
        self,
        tokenizer_config: Dict[str, Any]
    ) -> 'PreprocessingPipelineBuilder':
        self.config['tokenizer_config'] = tokenizer_config
        self.components.append('tokenization')
        return self
    
    def build(self) -> PreprocessingOrchestrator:
        """Build and return preprocessing orchestrator"""
        if not self.components:
            raise ValueError("No components added to the pipeline")
        
        return PreprocessingOrchestrator(self.config) 