from typing import List, Dict, Any, Optional, Union
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as naflow
import numpy as np
from transformers import AutoTokenizer
import torch
from ..monitoring.custom_metrics import MetricsCollector

class TextAugmenter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.get('model_name', 'bert-base-uncased')
        )
        
        # Initialize augmenters
        self.augmenters = {
            'synonym': naw.SynonymAug(aug_min=1, aug_max=3),
            'back_translation': naw.BackTranslationAug(
                from_model_name='facebook/wmt19-en-de',
                to_model_name='facebook/wmt19-de-en'
            ),
            'contextual': naw.ContextualWordEmbsAug(
                model_path=config.get('model_name', 'bert-base-uncased'),
                action="substitute"
            ),
            'random': naw.RandomWordAug(action="swap"),
            'spelling': nac.KeyboardAug(aug_char_min=1, aug_char_max=2),
            'sentence': nas.AbstSummAug(model_path='facebook/bart-large-cnn')
        }
    
    def augment(
        self,
        texts: Union[str, List[str]],
        techniques: List[str] = ['synonym'],
        num_augmentations: int = 1
    ) -> List[str]:
        """Augment text using specified techniques"""
        if isinstance(texts, str):
            texts = [texts]
        
        augmented_texts = []
        for text in texts:
            text_augmentations = []
            
            for technique in techniques:
                if technique not in self.augmenters:
                    continue
                
                try:
                    augmenter = self.augmenters[technique]
                    augmented = augmenter.augment(text, n=num_augmentations)
                    text_augmentations.extend(augmented)
                    
                    # Record metrics
                    self.metrics_collector.record_augmentation_metric(
                        technique=technique,
                        success=True
                    )
                except Exception as e:
                    self.metrics_collector.record_augmentation_metric(
                        technique=technique,
                        success=False
                    )
                    continue
            
            augmented_texts.extend(text_augmentations)
        
        return augmented_texts
    
    def create_augmentation_flow(
        self,
        techniques: List[str],
        probabilities: Optional[List[float]] = None
    ) -> naflow.Sequential:
        """Create a sequential augmentation flow"""
        if not probabilities:
            probabilities = [1.0] * len(techniques)
        
        flow_augmenters = []
        for technique, prob in zip(techniques, probabilities):
            if technique in self.augmenters:
                flow_augmenters.append(
                    (self.augmenters[technique], prob)
                )
        
        return naflow.Sequential(flow_augmenters)
    
    def generate_contrastive_pairs(
        self,
        texts: Union[str, List[str]],
        technique: str = 'contextual',
        num_pairs: int = 1
    ) -> List[Dict[str, str]]:
        """Generate contrastive pairs for contrastive learning"""
        if isinstance(texts, str):
            texts = [texts]
        
        pairs = []
        augmenter = self.augmenters.get(technique)
        if not augmenter:
            return pairs
        
        for text in texts:
            augmented = self.augment(text, [technique], num_pairs)
            pairs.extend([
                {'anchor': text, 'positive': aug}
                for aug in augmented
            ])
        
        return pairs

class DataAugmentationPipeline:
    def __init__(
        self,
        augmenter: TextAugmenter,
        techniques: List[str],
        num_augmentations: int = 1
    ):
        self.augmenter = augmenter
        self.techniques = techniques
        self.num_augmentations = num_augmentations
    
    def augment_dataset(
        self,
        texts: List[str],
        labels: List[Any]
    ) -> Dict[str, List]:
        """Augment entire dataset while preserving labels"""
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(texts, labels):
            # Augment text
            augmentations = self.augmenter.augment(
                text,
                self.techniques,
                self.num_augmentations
            )
            
            # Add augmentations and corresponding labels
            augmented_texts.extend(augmentations)
            augmented_labels.extend([label] * len(augmentations))
        
        # Combine original and augmented data
        all_texts = texts + augmented_texts
        all_labels = labels + augmented_labels
        
        return {
            'texts': all_texts,
            'labels': all_labels
        }
    
    def generate_balanced_augmentations(
        self,
        texts: List[str],
        labels: List[Any]
    ) -> Dict[str, List]:
        """Generate augmentations to balance class distribution"""
        # Count samples per class
        class_counts = {}
        for label in labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Find majority class count
        max_count = max(class_counts.values())
        
        augmented_texts = []
        augmented_labels = []
        
        # Augment minority classes
        for label in class_counts:
            if class_counts[label] < max_count:
                # Get indices for current class
                indices = [i for i, l in enumerate(labels) if l == label]
                needed = max_count - class_counts[label]
                
                # Augment samples until balanced
                while len(augmented_texts) < needed:
                    idx = np.random.choice(indices)
                    augmentations = self.augmenter.augment(
                        texts[idx],
                        self.techniques,
                        1
                    )
                    augmented_texts.extend(augmentations)
                    augmented_labels.extend([label] * len(augmentations))
        
        # Combine original and augmented data
        all_texts = texts + augmented_texts
        all_labels = labels + augmented_labels
        
        return {
            'texts': all_texts,
            'labels': all_labels
        } 