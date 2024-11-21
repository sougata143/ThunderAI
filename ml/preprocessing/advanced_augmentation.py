from typing import List, Dict, Any, Optional, Union
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np
from ..monitoring.custom_metrics import MetricsCollector

class AdvancedTextAugmenter:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        
        # Initialize T5 for paraphrasing
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
        
        # Advanced augmenters
        self.augmenters = {
            'eda': self._create_eda_augmenter(),
            'paraphrase': self._create_paraphrase_augmenter(),
            'mixup': self._create_mixup_augmenter(),
            'multilingual': self._create_multilingual_augmenter(),
            'entity_replacement': self._create_entity_replacement_augmenter()
        }
    
    def _create_eda_augmenter(self):
        """Create Easy Data Augmentation (EDA) augmenter"""
        return {
            'synonym': naw.SynonymAug(aug_min=1, aug_max=3),
            'insert': naw.WordInsertAug(aug_min=1, aug_max=3),
            'swap': naw.RandomWordAug(action="swap"),
            'delete': naw.RandomWordAug(action="delete")
        }
    
    def _create_paraphrase_augmenter(self):
        """Create paraphrase-based augmenter using T5"""
        def paraphrase_text(text: str, num_return_sequences: int = 1) -> List[str]:
            prefix = "paraphrase: "
            input_ids = self.t5_tokenizer.encode(
                prefix + text,
                return_tensors='pt',
                max_length=512,
                truncation=True
            )
            
            outputs = self.t5_model.generate(
                input_ids,
                max_length=512,
                num_return_sequences=num_return_sequences,
                num_beams=num_return_sequences * 2,
                temperature=0.7,
                do_sample=True
            )
            
            return [
                self.t5_tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
        
        return paraphrase_text
    
    def _create_mixup_augmenter(self):
        """Create text mixup augmenter"""
        def mixup_texts(texts: List[str], alpha: float = 0.2) -> List[str]:
            if len(texts) < 2:
                return texts
            
            mixed_texts = []
            for i in range(len(texts)):
                # Sample another text
                j = np.random.choice([k for k in range(len(texts)) if k != i])
                
                # Generate mixup weight
                lam = np.random.beta(alpha, alpha)
                
                # Split texts into words
                words1 = texts[i].split()
                words2 = texts[j].split()
                
                # Mix words based on lambda
                num_words = max(1, int(lam * len(words1) + (1 - lam) * len(words2)))
                mixed_words = (
                    words1[:int(lam * len(words1))] +
                    words2[:int((1 - lam) * len(words2))]
                )[:num_words]
                
                mixed_texts.append(' '.join(mixed_words))
            
            return mixed_texts
        
        return mixup_texts
    
    def _create_multilingual_augmenter(self):
        """Create multilingual augmentation using back-translation"""
        return naw.BackTranslationAug(
            from_model_name='Helsinki-NLP/opus-mt-en-de',
            to_model_name='Helsinki-NLP/opus-mt-de-en',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def _create_entity_replacement_augmenter(self):
        """Create entity-based replacement augmenter"""
        import spacy
        nlp = spacy.load('en_core_web_sm')
        
        def replace_entities(text: str, num_replacements: int = 1) -> List[str]:
            doc = nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            if not entities:
                return [text]
            
            augmented_texts = []
            for _ in range(num_replacements):
                new_text = text
                for ent_text, ent_label in entities:
                    if np.random.random() < 0.5:
                        # Replace with similar entity from predefined lists
                        replacement = self._get_similar_entity(ent_label)
                        if replacement:
                            new_text = new_text.replace(ent_text, replacement)
                augmented_texts.append(new_text)
            
            return augmented_texts
        
        return replace_entities
    
    def _get_similar_entity(self, entity_type: str) -> Optional[str]:
        """Get similar entity based on entity type"""
        entity_examples = {
            'PERSON': ['John', 'Mary', 'David', 'Sarah'],
            'ORG': ['Google', 'Microsoft', 'Apple', 'Amazon'],
            'GPE': ['London', 'Paris', 'New York', 'Tokyo'],
            'DATE': ['Monday', 'January', '2024', 'yesterday']
        }
        
        if entity_type in entity_examples:
            return np.random.choice(entity_examples[entity_type])
        return None
    
    def augment(
        self,
        texts: Union[str, List[str]],
        techniques: List[str] = ['eda'],
        num_augmentations: int = 1
    ) -> List[str]:
        """Apply multiple augmentation techniques"""
        if isinstance(texts, str):
            texts = [texts]
        
        augmented_texts = []
        for text in texts:
            text_augmentations = []
            
            for technique in techniques:
                try:
                    if technique == 'eda':
                        # Apply all EDA techniques
                        for eda_aug in self.augmenters['eda'].values():
                            augmented = eda_aug.augment(text, n=num_augmentations)
                            text_augmentations.extend(augmented)
                    
                    elif technique == 'paraphrase':
                        augmented = self.augmenters['paraphrase'](
                            text,
                            num_augmentations
                        )
                        text_augmentations.extend(augmented)
                    
                    elif technique == 'mixup':
                        augmented = self.augmenters['mixup'](
                            [text] * num_augmentations
                        )
                        text_augmentations.extend(augmented)
                    
                    elif technique == 'multilingual':
                        augmented = self.augmenters['multilingual'].augment(
                            text,
                            n=num_augmentations
                        )
                        text_augmentations.extend(augmented)
                    
                    elif technique == 'entity_replacement':
                        augmented = self.augmenters['entity_replacement'](
                            text,
                            num_augmentations
                        )
                        text_augmentations.extend(augmented)
                    
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