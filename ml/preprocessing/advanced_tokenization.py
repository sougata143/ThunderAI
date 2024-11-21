from typing import List, Dict, Any, Optional, Union
import torch
from transformers import PreTrainedTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import sentencepiece as spm
from collections import Counter
import regex as re
from dataclasses import dataclass
from ..monitoring.custom_metrics import MetricsCollector

@dataclass
class TokenizationConfig:
    """Configuration for tokenization"""
    vocab_size: int = 30000
    min_frequency: int = 2
    special_tokens: List[str] = None
    max_length: int = 512
    tokenizer_type: str = 'wordpiece'  # ['wordpiece', 'bpe', 'unigram', 'custom']
    add_prefix_space: bool = True
    strip_accents: bool = True
    lowercase: bool = True
    handle_chinese_chars: bool = True
    split_on_whitespace: bool = True

class AdvancedTokenizer:
    def __init__(self, config: TokenizationConfig):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.tokenizer = None
        self.special_tokens = config.special_tokens or [
            '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'
        ]
        
        # Initialize tokenizer based on type
        if config.tokenizer_type == 'wordpiece':
            self._init_wordpiece_tokenizer()
        elif config.tokenizer_type == 'bpe':
            self._init_bpe_tokenizer()
        elif config.tokenizer_type == 'unigram':
            self._init_unigram_tokenizer()
        elif config.tokenizer_type == 'custom':
            self._init_custom_tokenizer()
    
    def _init_wordpiece_tokenizer(self):
        """Initialize WordPiece tokenizer"""
        self.tokenizer = Tokenizer(models.WordPiece(
            vocab={},
            unk_token='[UNK]'
        ))
        
        # Add pre-tokenization
        self.tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        
        # Add post-processing
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single='[CLS] $A [SEP]',
            pair='[CLS] $A [SEP] $B [SEP]',
            special_tokens=[
                ('[CLS]', 1),
                ('[SEP]', 2),
            ],
        )
    
    def _init_bpe_tokenizer(self):
        """Initialize BPE tokenizer"""
        self.tokenizer = Tokenizer(models.BPE(
            vocab={},
            merges=[],
            unk_token='[UNK]'
        ))
        
        # Add pre-tokenization
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=self.config.add_prefix_space
        )
        
        # Add decoder
        self.tokenizer.decoder = decoders.ByteLevel()
    
    def _init_unigram_tokenizer(self):
        """Initialize Unigram tokenizer using SentencePiece"""
        self.sp_model = spm.SentencePieceProcessor()
        self.tokenizer = self.sp_model
    
    def _init_custom_tokenizer(self):
        """Initialize custom tokenizer with specific rules"""
        self.tokenizer = Tokenizer(models.WordLevel(
            vocab={},
            unk_token='[UNK]'
        ))
        
        # Custom pre-tokenization rules
        if self.config.split_on_whitespace:
            self.tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        
        # Add normalization
        normalizers = []
        if self.config.lowercase:
            normalizers.append(normalizers.Lowercase())
        if self.config.strip_accents:
            normalizers.append(normalizers.StripAccents())
        if self.config.handle_chinese_chars:
            normalizers.append(normalizers.BertNormalizer())
        
        self.tokenizer.normalizer = normalizers.Sequence(normalizers)
    
    def train(self, texts: List[str]):
        """Train tokenizer on corpus"""
        if self.config.tokenizer_type == 'unigram':
            self._train_unigram(texts)
        else:
            trainer = trainers.WordPieceTrainer(
                vocab_size=self.config.vocab_size,
                min_frequency=self.config.min_frequency,
                special_tokens=self.special_tokens
            )
            self.tokenizer.train_from_iterator(texts, trainer=trainer)
        
        # Log metrics
        self.metrics_collector.record_preprocessing_metric(
            'vocabulary_size',
            self.get_vocab_size()
        )
    
    def _train_unigram(self, texts: str):
        """Train Unigram model using SentencePiece"""
        # Write texts to temporary file
        with open('temp_corpus.txt', 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        # Train SentencePiece model
        spm.SentencePieceTrainer.train(
            input='temp_corpus.txt',
            model_prefix='unigram',
            vocab_size=self.config.vocab_size,
            character_coverage=0.9995,
            model_type='unigram',
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3
        )
        
        # Load trained model
        self.sp_model.load('unigram.model')
    
    def encode(
        self,
        texts: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Encode texts to token IDs"""
        if isinstance(texts, str):
            texts = [texts]
        
        max_len = max_length or self.config.max_length
        
        if self.config.tokenizer_type == 'unigram':
            # Handle SentencePiece encoding
            encoded = []
            attention_masks = []
            
            for text in texts:
                ids = self.sp_model.encode_as_ids(text)
                if truncation:
                    ids = ids[:max_len]
                encoded.append(ids)
                attention_masks.append([1] * len(ids))
            
            # Pad sequences
            if padding:
                max_len = max(len(ids) for ids in encoded)
                encoded = [ids + [0] * (max_len - len(ids)) for ids in encoded]
                attention_masks = [mask + [0] * (max_len - len(mask))
                                 for mask in attention_masks]
        else:
            # Handle Huggingface tokenizers encoding
            encoding = self.tokenizer.encode_batch(
                texts,
                padding=padding,
                truncation=truncation,
                max_length=max_len
            )
            
            encoded = [e.ids for e in encoding]
            attention_masks = [e.attention_mask for e in encoding]
        
        return {
            'input_ids': torch.tensor(encoded),
            'attention_mask': torch.tensor(attention_masks)
        }
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode token IDs back to text"""
        if self.config.tokenizer_type == 'unigram':
            return [self.sp_model.decode_ids(ids.tolist())
                    for ids in token_ids]
        else:
            return self.tokenizer.decode_batch(
                token_ids.tolist(),
                skip_special_tokens=skip_special_tokens
            )
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.config.tokenizer_type == 'unigram':
            return self.sp_model.get_piece_size()
        else:
            return self.tokenizer.get_vocab_size()
    
    def save(self, path: str):
        """Save tokenizer to disk"""
        if self.config.tokenizer_type == 'unigram':
            self.sp_model.save(f"{path}/tokenizer.model")
        else:
            self.tokenizer.save(f"{path}/tokenizer.json")
    
    def load(self, path: str):
        """Load tokenizer from disk"""
        if self.config.tokenizer_type == 'unigram':
            self.sp_model.load(f"{path}/tokenizer.model")
        else:
            self.tokenizer = Tokenizer.from_file(f"{path}/tokenizer.json") 