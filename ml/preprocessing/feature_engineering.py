from typing import List, Dict, Any, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
import spacy
from textblob import TextBlob
import torch
from transformers import AutoModel, AutoTokenizer
from ..monitoring.custom_metrics import MetricsCollector

class FeatureExtractor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize feature extractors
        self.tfidf = TfidfVectorizer(
            max_features=config.get('max_features', 1000),
            ngram_range=config.get('ngram_range', (1, 3))
        )
        
        # Initialize dimensionality reduction
        self.svd = TruncatedSVD(
            n_components=config.get('n_components', 100)
        )
        
        # Initialize topic modeling
        self.lda = LatentDirichletAllocation(
            n_components=config.get('n_topics', 10),
            random_state=42
        )
        
        # Initialize transformer model for embeddings
        if config.get('use_transformer_embeddings', True):
            model_name = config.get('transformer_model', 'bert-base-uncased')
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            if torch.cuda.is_available():
                self.model = self.model.cuda()
    
    def extract_statistical_features(self, texts: List[str]) -> np.ndarray:
        """Extract statistical features from texts"""
        features = []
        for text in texts:
            doc = self.nlp(text)
            
            # Basic statistics
            stats = {
                'length': len(text),
                'word_count': len(doc),
                'avg_word_length': np.mean([len(token.text) for token in doc]),
                'sentence_count': len(list(doc.sents)),
                'unique_words': len(set(token.text.lower() for token in doc))
            }
            
            # POS tag distributions
            pos_counts = {}
            for token in doc:
                pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
            
            # Entity counts
            entity_counts = {}
            for ent in doc.ents:
                entity_counts[ent.label_] = entity_counts.get(ent.label_, 0) + 1
            
            # Combine features
            feature_vector = [
                stats['length'],
                stats['word_count'],
                stats['avg_word_length'],
                stats['sentence_count'],
                stats['unique_words']
            ]
            
            # Add POS and entity features
            feature_vector.extend(pos_counts.values())
            feature_vector.extend(entity_counts.values())
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_semantic_features(
        self,
        texts: List[str],
        method: str = 'transformer'
    ) -> np.ndarray:
        """Extract semantic features using different methods"""
        if method == 'transformer':
            return self._get_transformer_embeddings(texts)
        elif method == 'tfidf':
            return self._get_tfidf_features(texts)
        elif method == 'topics':
            return self._get_topic_features(texts)
        else:
            raise ValueError(f"Unknown semantic feature method: {method}")
    
    def _get_transformer_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings from transformer model"""
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            
            if torch.cuda.is_available():
                encoded = {k: v.cuda() for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use [CLS] token embeddings
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _get_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Get TF-IDF features with dimensionality reduction"""
        tfidf_matrix = self.tfidf.fit_transform(texts)
        return self.svd.fit_transform(tfidf_matrix)
    
    def _get_topic_features(self, texts: List[str]) -> np.ndarray:
        """Get topic distribution features"""
        tfidf_matrix = self.tfidf.fit_transform(texts)
        return self.lda.fit_transform(tfidf_matrix)
    
    def extract_sentiment_features(self, texts: List[str]) -> np.ndarray:
        """Extract sentiment-related features"""
        features = []
        for text in texts:
            blob = TextBlob(text)
            
            # Get sentiment scores
            sentiment_features = [
                blob.sentiment.polarity,
                blob.sentiment.subjectivity,
                len([s for s in blob.sentences if s.sentiment.polarity > 0]) / len(blob.sentences),
                len([s for s in blob.sentences if s.sentiment.polarity < 0]) / len(blob.sentences)
            ]
            
            features.append(sentiment_features)
        
        return np.array(features)
    
    def combine_features(
        self,
        texts: List[str],
        feature_types: List[str] = ['statistical', 'semantic', 'sentiment']
    ) -> np.ndarray:
        """Combine multiple types of features"""
        feature_matrices = []
        
        if 'statistical' in feature_types:
            statistical = self.extract_statistical_features(texts)
            feature_matrices.append(statistical)
        
        if 'semantic' in feature_types:
            semantic = self.extract_semantic_features(texts)
            feature_matrices.append(semantic)
        
        if 'sentiment' in feature_types:
            sentiment = self.extract_sentiment_features(texts)
            feature_matrices.append(sentiment)
        
        # Concatenate all features
        combined = np.hstack(feature_matrices)
        
        # Log metrics
        self.metrics_collector.record_preprocessing_metric(
            'feature_dimension',
            combined.shape[1]
        )
        
        return combined 