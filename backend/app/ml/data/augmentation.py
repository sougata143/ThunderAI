import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from PIL import Image
import albumentations as A
from transformers import AutoTokenizer
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.char as nac
import nlpaug.flow as naf
from copy import deepcopy
import random
import logging

logger = logging.getLogger(__name__)

class AdvancedAugmentation:
    """Advanced data augmentation for both text and vision tasks."""
    
    def __init__(
        self,
        modality: str = 'text',
        task_type: str = 'classification',
        config: Optional[Dict[str, Any]] = None
    ):
        self.modality = modality
        self.task_type = task_type
        self.config = config or {}
        
        if modality == 'text':
            self._setup_text_augmenters()
        elif modality == 'vision':
            self._setup_vision_augmenters()
        else:
            raise ValueError(f"Unsupported modality: {modality}")
    
    def _setup_text_augmenters(self) -> None:
        """Setup text augmentation pipeline."""
        # Contextual word embeddings augmenter
        self.contextual_augmenter = naw.ContextualWordEmbsAug(
            model_path='bert-base-uncased',
            action="substitute",
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Synonym augmenter
        self.synonym_augmenter = naw.SynonymAug(
            aug_src='wordnet',
            lang='eng'
        )
        
        # Back translation augmenter
        self.back_translation_augmenter = naw.BackTranslationAug(
            from_model_name='facebook/wmt19-en-de',
            to_model_name='facebook/wmt19-de-en'
        )
        
        # Keyboard augmenter for typos
        self.keyboard_augmenter = nac.KeyboardAug(
            aug_char_min=1,
            aug_char_max=2,
            aug_word_min=1,
            aug_word_max=2
        )
    
    def _setup_vision_augmenters(self) -> None:
        """Setup vision augmentation pipeline using albumentations."""
        # Basic augmentations
        self.basic_transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ])
        
        # Advanced augmentations
        self.advanced_transform = A.Compose([
            A.RandomSizedCrop(
                min_max_height=(128, 256),
                height=256,
                width=256,
                p=0.5
            ),
            A.OneOf([
                A.RandomShadow(p=0.5),
                A.RandomSunFlare(p=0.5),
            ], p=0.3),
            A.OneOf([
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    p=0.5
                ),
                A.GridDropout(p=0.5),
            ], p=0.3),
            A.OneOf([
                A.ElasticTransform(p=0.5),
                A.IAAPerspective(p=0.5),
            ], p=0.3),
            A.OneOf([
                A.ChannelShuffle(p=0.5),
                A.RGBShift(p=0.5),
            ], p=0.3),
        ])
        
        # CutMix and MixUp augmentations
        self.use_cutmix = self.config.get('use_cutmix', True)
        self.use_mixup = self.config.get('use_mixup', True)
        self.mix_prob = self.config.get('mix_prob', 0.5)
    
    def augment_text(
        self,
        texts: List[str],
        num_augmentations: int = 1,
        methods: Optional[List[str]] = None
    ) -> List[str]:
        """Apply text augmentation with multiple methods."""
        if self.modality != 'text':
            raise ValueError("Text augmentation not supported for this modality")
        
        methods = methods or ['contextual', 'synonym', 'back_translation', 'keyboard']
        augmented_texts = []
        
        for text in texts:
            text_augmentations = []
            
            for _ in range(num_augmentations):
                method = random.choice(methods)
                
                if method == 'contextual':
                    aug_text = self.contextual_augmenter.augment(text)[0]
                elif method == 'synonym':
                    aug_text = self.synonym_augmenter.augment(text)[0]
                elif method == 'back_translation':
                    aug_text = self.back_translation_augmenter.augment(text)[0]
                elif method == 'keyboard':
                    aug_text = self.keyboard_augmenter.augment(text)[0]
                else:
                    continue
                
                text_augmentations.append(aug_text)
            
            augmented_texts.extend(text_augmentations)
        
        return augmented_texts
    
    def augment_image(
        self,
        image: np.ndarray,
        num_augmentations: int = 1,
        severity: str = 'basic'
    ) -> List[np.ndarray]:
        """Apply vision augmentation with varying severity levels."""
        if self.modality != 'vision':
            raise ValueError("Image augmentation not supported for this modality")
        
        augmented_images = []
        transform = self.basic_transform if severity == 'basic' else self.advanced_transform
        
        for _ in range(num_augmentations):
            augmented = transform(image=image)['image']
            augmented_images.append(augmented)
        
        return augmented_images
    
    def mixup_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply MixUp augmentation to a batch of images."""
        if random.random() > self.mix_prob or not self.use_mixup:
            return images, labels
        
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]
        
        lam = np.random.beta(alpha, alpha)
        mixed_images = lam * images + (1 - lam) * shuffled_images
        mixed_labels = (lam * labels + (1 - lam) * shuffled_labels)
        
        return mixed_images, mixed_labels
    
    def cutmix_batch(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        alpha: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix augmentation to a batch of images."""
        if random.random() > self.mix_prob or not self.use_cutmix:
            return images, labels
        
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]
        
        lam = np.random.beta(alpha, alpha)
        
        # Generate random box
        cut_rat = np.sqrt(1.0 - lam)
        h, w = images.size()[-2:]
        cut_h = int(h * cut_rat)
        cut_w = int(w * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply CutMix
        images[:, :, bby1:bby2, bbx1:bbx2] = \
            shuffled_images[:, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        mixed_labels = lam * labels + (1 - lam) * shuffled_labels
        
        return images, mixed_labels
    
    def create_augmented_dataset(
        self,
        dataset: Dataset,
        num_augmentations: int = 1,
        methods: Optional[List[str]] = None
    ) -> Dataset:
        """Create an augmented version of the entire dataset."""
        augmented_dataset = deepcopy(dataset)
        
        if self.modality == 'text':
            if hasattr(dataset, 'texts'):
                augmented_texts = self.augment_text(
                    dataset.texts,
                    num_augmentations,
                    methods
                )
                augmented_dataset.texts.extend(augmented_texts)
                if hasattr(dataset, 'labels'):
                    augmented_dataset.labels.extend(
                        dataset.labels * num_augmentations
                    )
        else:
            if hasattr(dataset, 'images'):
                augmented_images = []
                for image in dataset.images:
                    augmented = self.augment_image(
                        image,
                        num_augmentations
                    )
                    augmented_images.extend(augmented)
                augmented_dataset.images.extend(augmented_images)
                if hasattr(dataset, 'labels'):
                    augmented_dataset.labels.extend(
                        dataset.labels * num_augmentations
                    )
        
        return augmented_dataset
