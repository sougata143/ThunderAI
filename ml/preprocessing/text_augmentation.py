from typing import List, Dict, Any, Optional, Union
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.sentence as nas
import torch
from transformers import MarianMTModel, MarianTokenizer
import random
import spacy
from ..monitoring.custom_metrics import MetricsCollector

class TextAugmentationService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize augmenters
        self.augmenters = {
            'synonym': self._init_synonym_augmenter(),
            'back_translation': self._init_back_translation_augmenter(),
            'contextual': self._init_contextual_augmenter(),
            'keyboard': self._init_keyboard_augmenter(),
            'random': self._init_random_augmenter(),
            'sentence': self._init_sentence_augmenter()
        }
        
        # Load translation models if needed
        if config.get('use_back_translation', True):
            self.translation_models = self._load_translation_models()
    
    def _init_synonym_augmenter(self):
        """Initialize synonym replacement augmenter"""
        return naw.SynonymAug(
            aug_min=1,
            aug_max=self.config.get('max_synonym_replacements', 3)
        )
    
    def _init_back_translation_augmenter(self):
        """Initialize back translation augmenter"""
        return naw.BackTranslationAug(
            from_model_name='Helsinki-NLP/opus-mt-en-de',
            to_model_name='Helsinki-NLP/opus-mt-de-en',
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def _init_contextual_augmenter(self):
        """Initialize contextual word embeddings augmenter"""
        return naw.ContextualWordEmbsAug(
            model_path=self.config.get('contextual_model', 'bert-base-uncased'),
            action="substitute"
        )
    
    def _init_keyboard_augmenter(self):
        """Initialize keyboard augmenter for typos"""
        return nac.KeyboardAug(
            aug_char_min=1,
            aug_char_max=2,
            include_special_char=False
        )
    
    def _init_random_augmenter(self):
        """Initialize random word augmenter"""
        return naw.RandomWordAug(
            action="swap",
            aug_min=1,
            aug_max=self.config.get('max_word_swaps', 3)
        )
    
    def _init_sentence_augmenter(self):
        """Initialize sentence augmenter"""
        return nas.AbstSummAug(
            model_path='facebook/bart-large-cnn'
        )
    
    def _load_translation_models(self) -> Dict[str, Any]:
        """Load translation models for multiple languages"""
        languages = self.config.get('translation_languages', ['de', 'fr', 'es'])
        models = {}
        
        for lang in languages:
            try:
                forward_model = f'Helsinki-NLP/opus-mt-en-{lang}'
                backward_model = f'Helsinki-NLP/opus-mt-{lang}-en'
                
                models[lang] = {
                    'forward': {
                        'model': MarianMTModel.from_pretrained(forward_model),
                        'tokenizer': MarianTokenizer.from_pretrained(forward_model)
                    },
                    'backward': {
                        'model': MarianMTModel.from_pretrained(backward_model),
                        'tokenizer': MarianTokenizer.from_pretrained(backward_model)
                    }
                }
                
                if torch.cuda.is_available():
                    models[lang]['forward']['model'] = models[lang]['forward']['model'].cuda()
                    models[lang]['backward']['model'] = models[lang]['backward']['model'].cuda()
                    
            except Exception as e:
                print(f"Error loading translation models for {lang}: {str(e)}")
                continue
        
        return models
    
    def augment_text(
        self,
        text: str,
        techniques: List[str] = None,
        num_augmentations: int = 1
    ) -> List[str]:
        """Apply multiple augmentation techniques to text"""
        if techniques is None:
            techniques = list(self.augmenters.keys())
        
        augmented_texts = []
        for technique in techniques:
            try:
                if technique in self.augmenters:
                    augmented = self.augmenters[technique].augment(
                        text,
                        n=num_augmentations
                    )
                    augmented_texts.extend(augmented)
                    
                    # Record metrics
                    self.metrics_collector.record_preprocessing_metric(
                        f'augmentation_{technique}',
                        len(augmented)
                    )
            except Exception as e:
                print(f"Error applying {technique} augmentation: {str(e)}")
                continue
        
        return augmented_texts
    
    def back_translate(
        self,
        text: str,
        target_lang: str = 'de',
        num_variants: int = 1
    ) -> List[str]:
        """Perform back translation augmentation"""
        if target_lang not in self.translation_models:
            raise ValueError(f"Translation models not loaded for language: {target_lang}")
        
        forward_model = self.translation_models[target_lang]['forward']['model']
        forward_tokenizer = self.translation_models[target_lang]['forward']['tokenizer']
        backward_model = self.translation_models[target_lang]['backward']['model']
        backward_tokenizer = self.translation_models[target_lang]['backward']['tokenizer']
        
        # Translate to target language
        inputs = forward_tokenizer([text], return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        translated = forward_model.generate(**inputs, num_return_sequences=num_variants)
        translated_texts = forward_tokenizer.batch_decode(translated, skip_special_tokens=True)
        
        # Translate back to English
        back_translated = []
        for translated_text in translated_texts:
            inputs = backward_tokenizer([translated_text], return_tensors="pt", padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            outputs = backward_model.generate(**inputs)
            back_translated.extend(
                backward_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            )
        
        return back_translated
    
    def create_adversarial_examples(
        self,
        text: str,
        num_examples: int = 1
    ) -> List[str]:
        """Create adversarial examples by combining multiple techniques"""
        adversarial_examples = []
        
        for _ in range(num_examples):
            # Apply random combination of augmentation techniques
            techniques = random.sample(
                list(self.augmenters.keys()),
                k=random.randint(1, 3)
            )
            
            augmented = self.augment_text(text, techniques, num_augmentations=1)
            adversarial_examples.extend(augmented)
        
        return adversarial_examples
    
    def augment_batch(
        self,
        texts: List[str],
        techniques: List[str] = None,
        num_augmentations: int = 1
    ) -> List[str]:
        """Augment a batch of texts"""
        augmented_texts = []
        for text in texts:
            augmented = self.augment_text(text, techniques, num_augmentations)
            augmented_texts.extend(augmented)
        return augmented_texts 