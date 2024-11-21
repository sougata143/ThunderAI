from typing import List, Dict, Any, Optional, Union
import torch
from transformers import PreTrainedTokenizer, AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
import sentencepiece as spm
from collections import Counter
import numpy as np
from ..monitoring.custom_metrics import MetricsCollector

class CustomTokenizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.vocab_size = config.get('vocab_size', 30000)
        self.min_freq = config.get('min_freq', 2)
        self.special_tokens = config.get('special_tokens', ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
        self.max_length = config.get('max_length', 512)
        
        self.tokenizer = None
        self.vocab = None
        self.token2idx = None
        self.idx2token = None
    
    def train(self, texts: List[str]):
        """Train tokenizer on corpus"""
        # Build vocabulary
        counter = Counter()
        for text in texts:
            tokens = self._basic_tokenize(text)
            counter.update(tokens)
        
        # Filter by frequency and create vocab
        vocab = self.special_tokens.copy()
        for token, count in counter.most_common():
            if count >= self.min_freq and len(vocab) < self.vocab_size:
                vocab.append(token)
        
        self.vocab = vocab
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        
        # Log metrics
        self.metrics_collector.record_preprocessing_metric(
            'vocabulary_size',
            len(self.vocab)
        )
    
    def _basic_tokenize(self, text: str) -> List[str]:
        """Basic tokenization"""
        return text.lower().split()
    
    def encode(
        self,
        texts: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Encode texts to token IDs"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize and convert to IDs
        encoded = []
        attention_masks = []
        
        for text in texts:
            tokens = self._basic_tokenize(text)
            if truncation:
                tokens = tokens[:self.max_length-2]  # Account for [CLS] and [SEP]
            
            # Add special tokens
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            
            # Convert to IDs
            ids = [self.token2idx.get(token, self.token2idx['[UNK]']) 
                  for token in tokens]
            mask = [1] * len(ids)
            
            encoded.append(ids)
            attention_masks.append(mask)
        
        # Pad sequences
        if padding:
            max_len = max(len(ids) for ids in encoded)
            encoded = [ids + [self.token2idx['[PAD]']] * (max_len - len(ids))
                      for ids in encoded]
            attention_masks = [mask + [0] * (max_len - len(mask))
                             for mask in attention_masks]
        
        return {
            'input_ids': torch.tensor(encoded),
            'attention_mask': torch.tensor(attention_masks)
        }
    
    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs back to text"""
        decoded = []
        for ids in token_ids:
            tokens = [self.idx2token[idx.item()] for idx in ids
                     if idx.item() in self.idx2token]
            # Remove special tokens and join
            tokens = [t for t in tokens if t not in self.special_tokens]
            decoded.append(' '.join(tokens))
        return decoded

class SubwordTokenizer:
    """Implements subword tokenization using SentencePiece"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.model_prefix = config.get('model_prefix', 'subword')
        self.vocab_size = config.get('vocab_size', 8000)
        self.character_coverage = config.get('character_coverage', 0.9995)
        self.model_type = config.get('model_type', 'unigram')  # or 'bpe', 'char', 'word'
        
        self.sp_model = None
    
    def train(self, texts: List[str]):
        """Train SentencePiece model"""
        # Write texts to temporary file
        with open('temp_corpus.txt', 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        # Train SentencePiece model
        spm.SentencePieceTrainer.train(
            input='temp_corpus.txt',
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=self.character_coverage,
            model_type=self.model_type,
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3
        )
        
        # Load trained model
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(f'{self.model_prefix}.model')
        
        # Log metrics
        self.metrics_collector.record_preprocessing_metric(
            'subword_vocab_size',
            self.sp_model.get_piece_size()
        )
    
    def encode(
        self,
        texts: Union[str, List[str]],
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Encode texts using SentencePiece model"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Encode texts
        encoded = []
        for text in texts:
            ids = self.sp_model.encode_as_ids(text)
            if truncation:
                ids = ids[:self.config.get('max_length', 512)]
            encoded.append(ids)
        
        # Pad sequences
        if padding:
            max_len = max(len(ids) for ids in encoded)
            encoded = [ids + [self.sp_model.pad_id()] * (max_len - len(ids))
                      for ids in encoded]
            attention_masks = [[1] * len(ids) + [0] * (max_len - len(ids))
                             for ids in encoded]
        else:
            attention_masks = [[1] * len(ids) for ids in encoded]
        
        return {
            'input_ids': torch.tensor(encoded),
            'attention_mask': torch.tensor(attention_masks)
        }
    
    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """Decode token IDs using SentencePiece model"""
        decoded = []
        for ids in token_ids:
            text = self.sp_model.decode_ids(ids.tolist())
            decoded.append(text)
        return decoded

class TokenizerFactory:
    """Factory for creating different types of tokenizers"""
    @staticmethod
    def create_tokenizer(tokenizer_type: str, config: Dict[str, Any]) -> Union[CustomTokenizer, SubwordTokenizer, PreTrainedTokenizer]:
        if tokenizer_type == 'custom':
            return CustomTokenizer(config)
        elif tokenizer_type == 'subword':
            return SubwordTokenizer(config)
        elif tokenizer_type == 'pretrained':
            return AutoTokenizer.from_pretrained(
                config.get('model_name', 'bert-base-uncased')
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {tokenizer_type}") 