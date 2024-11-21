from typing import List, Dict, Any, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF
import torch
from transformers import AutoModel, AutoTokenizer
from gensim.models import Word2Vec, FastText
import spacy
from textblob import TextBlob
from ..monitoring.custom_metrics import MetricsCollector

class AdvancedFeatureExtractor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize embeddings
        self.word2vec = None
        self.fasttext = None
        self.transformer = None
        self.tokenizer = None
        
        if config.get('use_word2vec', False):
            self._init_word2vec()
        if config.get('use_fasttext', False):
            self._init_fasttext()
        if config.get('use_transformer', True):
            self._init_transformer()
    
    def _init_word2vec(self):
        """Initialize Word2Vec model"""
        self.word2vec = Word2Vec(
            vector_size=self.config.get('embedding_dim', 300),
            window=self.config.get('window_size', 5),
            min_count=self.config.get('min_count', 1)
        )
    
    def _init_fasttext(self):
        """Initialize FastText model"""
        self.fasttext = FastText(
            vector_size=self.config.get('embedding_dim', 300),
            window=self.config.get('window_size', 5),
            min_count=self.config.get('min_count', 1)
        )
    
    def _init_transformer(self):
        """Initialize transformer model"""
        model_name = self.config.get('transformer_model', 'bert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.transformer = self.transformer.cuda()
    
    def extract_linguistic_features(self, texts: List[str]) -> np.ndarray:
        """Extract advanced linguistic features"""
        features = []
        for text in texts:
            doc = self.nlp(text)
            blob = TextBlob(text)
            
            # Dependency parsing features
            dep_features = {
                'num_deps': len([token for token in doc if token.dep_ != '']),
                'num_roots': len([token for token in doc if token.dep_ == 'ROOT']),
                'avg_dep_length': np.mean([token.head.i - token.i for token in doc]) if len(doc) > 0 else 0
            }
            
            # Syntactic features
            syntax_features = {
                'num_noun_phrases': len(list(doc.noun_chunks)),
                'avg_noun_phrase_length': np.mean([len(np) for np in doc.noun_chunks]) if len(list(doc.noun_chunks)) > 0 else 0,
                'sentence_complexity': len([token for token in doc if token.dep_ in ['ccomp', 'xcomp', 'advcl']])
            }
            
            # Semantic features
            semantic_features = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'num_entities': len(doc.ents),
                'unique_entities': len(set([ent.label_ for ent in doc.ents]))
            }
            
            # Combine all features
            feature_vector = list(dep_features.values()) + \
                           list(syntax_features.values()) + \
                           list(semantic_features.values())
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_contextual_embeddings(
        self,
        texts: List[str],
        pooling: str = 'mean'
    ) -> np.ndarray:
        """Extract contextual embeddings using transformer"""
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
                outputs = self.transformer(**encoded)
                if pooling == 'mean':
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                elif pooling == 'cls':
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                
                embeddings.extend(batch_embeddings.cpu().numpy())
        
        return np.array(embeddings)
    
    def extract_topic_features(
        self,
        texts: List[str],
        method: str = 'nmf',
        n_topics: int = 10
    ) -> np.ndarray:
        """Extract topic-based features"""
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            max_features=self.config.get('max_features', 1000),
            ngram_range=(1, self.config.get('max_ngram', 3))
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Apply topic modeling
        if method == 'nmf':
            model = NMF(n_components=n_topics, random_state=42)
        else:
            model = TruncatedSVD(n_components=n_topics, random_state=42)
        
        topic_features = model.fit_transform(tfidf_matrix)
        return topic_features
    
    def extract_word_embeddings(
        self,
        texts: List[str],
        method: str = 'word2vec'
    ) -> np.ndarray:
        """Extract word embeddings using Word2Vec or FastText"""
        embeddings = []
        model = self.word2vec if method == 'word2vec' else self.fasttext
        
        for text in texts:
            words = text.split()
            word_vectors = []
            for word in words:
                try:
                    vector = model.wv[word]
                    word_vectors.append(vector)
                except KeyError:
                    continue
            
            if word_vectors:
                # Average word vectors
                embedding = np.mean(word_vectors, axis=0)
            else:
                embedding = np.zeros(model.vector_size)
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def combine_features(
        self,
        texts: List[str],
        feature_types: List[str] = ['linguistic', 'contextual', 'topic']
    ) -> np.ndarray:
        """Combine multiple types of features"""
        feature_matrices = []
        
        for feature_type in feature_types:
            if feature_type == 'linguistic':
                features = self.extract_linguistic_features(texts)
            elif feature_type == 'contextual':
                features = self.extract_contextual_embeddings(texts)
            elif feature_type == 'topic':
                features = self.extract_topic_features(texts)
            elif feature_type in ['word2vec', 'fasttext']:
                features = self.extract_word_embeddings(texts, method=feature_type)
            
            feature_matrices.append(features)
        
        # Concatenate all features
        combined = np.hstack(feature_matrices)
        
        # Log metrics
        self.metrics_collector.record_preprocessing_metric(
            'feature_dimension',
            combined.shape[1]
        )
        
        return combined 