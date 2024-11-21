from typing import List, Dict, Any, Optional, Union
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
import spacy
from gensim.models import Word2Vec, FastText, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from ..monitoring.custom_metrics import MetricsCollector

class FeatureExtractionService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.nlp = spacy.load('en_core_web_sm')
        
        # Initialize feature extractors
        self.extractors = {
            'transformer': self._init_transformer_extractor(),
            'tfidf': self._init_tfidf_extractor(),
            'word2vec': self._init_word2vec_extractor(),
            'fasttext': self._init_fasttext_extractor(),
            'doc2vec': self._init_doc2vec_extractor(),
            'lda': self._init_lda_extractor()
        }
    
    def _init_transformer_extractor(self) -> Dict[str, Any]:
        """Initialize transformer-based feature extractor"""
        model_name = self.config.get('transformer_model', 'bert-base-uncased')
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        return {
            'model': model,
            'tokenizer': tokenizer
        }
    
    def _init_tfidf_extractor(self) -> TfidfVectorizer:
        """Initialize TF-IDF vectorizer"""
        return TfidfVectorizer(
            max_features=self.config.get('max_features', 10000),
            ngram_range=self.config.get('ngram_range', (1, 3)),
            min_df=self.config.get('min_df', 2)
        )
    
    def _init_word2vec_extractor(self) -> Word2Vec:
        """Initialize Word2Vec model"""
        return Word2Vec(
            vector_size=self.config.get('embedding_dim', 300),
            window=self.config.get('window_size', 5),
            min_count=self.config.get('min_count', 1)
        )
    
    def _init_fasttext_extractor(self) -> FastText:
        """Initialize FastText model"""
        return FastText(
            vector_size=self.config.get('embedding_dim', 300),
            window=self.config.get('window_size', 5),
            min_count=self.config.get('min_count', 1)
        )
    
    def _init_doc2vec_extractor(self) -> Doc2Vec:
        """Initialize Doc2Vec model"""
        return Doc2Vec(
            vector_size=self.config.get('embedding_dim', 300),
            min_count=self.config.get('min_count', 2),
            epochs=self.config.get('epochs', 20)
        )
    
    def _init_lda_extractor(self) -> Dict[str, Any]:
        """Initialize LDA topic model"""
        return {
            'vectorizer': TfidfVectorizer(
                max_features=self.config.get('max_features', 10000)
            ),
            'model': LatentDirichletAllocation(
                n_components=self.config.get('n_topics', 10),
                random_state=42
            )
        }
    
    def extract_features(
        self,
        texts: List[str],
        methods: List[str] = None,
        combine: bool = True
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Extract features using specified methods"""
        if methods is None:
            methods = ['transformer']
        
        features = {}
        for method in methods:
            if method in self.extractors:
                try:
                    if method == 'transformer':
                        features[method] = self._extract_transformer_features(texts)
                    elif method == 'tfidf':
                        features[method] = self._extract_tfidf_features(texts)
                    elif method in ['word2vec', 'fasttext']:
                        features[method] = self._extract_word_embeddings(texts, method)
                    elif method == 'doc2vec':
                        features[method] = self._extract_doc_embeddings(texts)
                    elif method == 'lda':
                        features[method] = self._extract_topic_features(texts)
                    
                    # Record metrics
                    self.metrics_collector.record_preprocessing_metric(
                        f'feature_extraction_{method}',
                        features[method].shape[1]
                    )
                except Exception as e:
                    print(f"Error extracting {method} features: {str(e)}")
                    continue
        
        if combine and features:
            return np.hstack([feat for feat in features.values()])
        return features
    
    def _extract_transformer_features(self, texts: List[str]) -> np.ndarray:
        """Extract features using transformer model"""
        model = self.extractors['transformer']['model']
        tokenizer = self.extractors['transformer']['tokenizer']
        
        features = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embeddings
                batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                features.extend(batch_features)
        
        return np.array(features)
    
    def _extract_tfidf_features(self, texts: List[str]) -> np.ndarray:
        """Extract TF-IDF features"""
        vectorizer = self.extractors['tfidf']
        features = vectorizer.fit_transform(texts)
        
        # Optionally reduce dimensionality
        if self.config.get('reduce_dim', False):
            svd = TruncatedSVD(
                n_components=self.config.get('n_components', 100)
            )
            features = svd.fit_transform(features)
        
        return features.toarray() if not isinstance(features, np.ndarray) else features
    
    def _extract_word_embeddings(
        self,
        texts: List[str],
        method: str
    ) -> np.ndarray:
        """Extract word embeddings using Word2Vec or FastText"""
        model = self.extractors[method]
        
        # Train model if not trained
        if not model.wv.vectors.shape[0]:
            sentences = [text.split() for text in texts]
            model.build_vocab(sentences)
            model.train(
                sentences,
                total_examples=model.corpus_count,
                epochs=model.epochs
            )
        
        # Average word vectors for each text
        features = []
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
                features.append(np.mean(word_vectors, axis=0))
            else:
                features.append(np.zeros(model.vector_size))
        
        return np.array(features)
    
    def _extract_doc_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract document embeddings using Doc2Vec"""
        model = self.extractors['doc2vec']
        
        # Train model if not trained
        if not model.dv.vectors.shape[0]:
            tagged_docs = [
                TaggedDocument(text.split(), [i])
                for i, text in enumerate(texts)
            ]
            model.build_vocab(tagged_docs)
            model.train(
                tagged_docs,
                total_examples=model.corpus_count,
                epochs=model.epochs
            )
        
        # Infer vectors for texts
        features = []
        for text in texts:
            vector = model.infer_vector(text.split())
            features.append(vector)
        
        return np.array(features)
    
    def _extract_topic_features(self, texts: List[str]) -> np.ndarray:
        """Extract topic features using LDA"""
        vectorizer = self.extractors['lda']['vectorizer']
        lda_model = self.extractors['lda']['model']
        
        # Transform texts to document-term matrix
        dtm = vectorizer.fit_transform(texts)
        
        # Extract topic distributions
        features = lda_model.fit_transform(dtm)
        return features 