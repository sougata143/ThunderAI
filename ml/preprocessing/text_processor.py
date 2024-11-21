from typing import List, Dict, Any, Optional, Union
import re
import unicodedata
import spacy
from bs4 import BeautifulSoup
import emoji
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from ..monitoring.custom_metrics import MetricsCollector

class TextProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.nlp = spacy.load('en_core_web_sm')
        self.stop_words = set(stopwords.words('english'))
        
        # Compile regex patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+')
        self.number_pattern = re.compile(r'\d+')
        self.special_char_pattern = re.compile(r'[^\w\s]')
    
    def process_text(
        self,
        text: str,
        steps: List[str] = None
    ) -> str:
        """Process text through specified preprocessing steps"""
        if steps is None:
            steps = self.config.get('default_steps', ['clean', 'normalize'])
        
        processed_text = text
        for step in steps:
            if hasattr(self, f"_{step}"):
                processed_text = getattr(self, f"_{step}")(processed_text)
        
        # Log metrics
        self.metrics_collector.record_preprocessing_metric(
            'text_processing_steps',
            len(steps)
        )
        
        return processed_text
    
    def _clean(self, text: str) -> str:
        """Clean text by removing unwanted elements"""
        # Remove HTML tags
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove emails
        text = self.email_pattern.sub(' ', text)
        
        # Remove emojis
        text = emoji.replace_emoji(text, '')
        
        # Remove special characters
        if self.config.get('remove_special_chars', True):
            text = self.special_char_pattern.sub(' ', text)
        
        # Remove numbers
        if self.config.get('remove_numbers', True):
            text = self.number_pattern.sub(' ', text)
        
        return text.strip()
    
    def _normalize(self, text: str) -> str:
        """Normalize text"""
        # Convert to lowercase
        if self.config.get('lowercase', True):
            text = text.lower()
        
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _lemmatize(self, text: str) -> str:
        """Lemmatize text using spaCy"""
        doc = self.nlp(text)
        return ' '.join([token.lemma_ for token in doc])
    
    def _remove_stopwords(self, text: str) -> str:
        """Remove stop words"""
        words = word_tokenize(text)
        return ' '.join([word for word in words if word.lower() not in self.stop_words])
    
    def _expand_contractions(self, text: str) -> str:
        """Expand contractions (e.g., don't -> do not)"""
        # Add your contraction mapping here
        contractions = {
            "n't": " not",
            "'ll": " will",
            "'re": " are",
            "'ve": " have",
            "'m": " am",
            "'d": " would"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        return text
    
    def _fix_spelling(self, text: str) -> str:
        """Fix spelling using TextBlob"""
        return str(TextBlob(text).correct())
    
    def process_batch(
        self,
        texts: List[str],
        steps: List[str] = None,
        n_jobs: int = -1
    ) -> List[str]:
        """Process a batch of texts in parallel"""
        from joblib import Parallel, delayed
        
        if n_jobs == -1:
            n_jobs = self.config.get('n_jobs', 1)
        
        processed_texts = Parallel(n_jobs=n_jobs)(
            delayed(self.process_text)(text, steps)
            for text in texts
        )
        
        return processed_texts
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text"""
        doc = self.nlp(text)
        blob = TextBlob(text)
        
        metadata = {
            'length': len(text),
            'word_count': len(doc),
            'sentence_count': len(list(doc.sents)),
            'entities': [(ent.text, ent.label_) for ent in doc.ents],
            'sentiment': {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            },
            'language': doc.lang_
        }
        
        return metadata
    
    def get_text_quality_score(self, text: str) -> float:
        """Calculate text quality score based on various metrics"""
        doc = self.nlp(text)
        
        # Calculate various quality metrics
        metrics = {
            'spelling_errors': len(TextBlob(text).correct().split()) - len(text.split()),
            'grammar_score': sum(1 for token in doc if token.dep_ != ''),
            'vocabulary_richness': len(set(token.text.lower() for token in doc)) / len(doc),
            'sentence_complexity': sum(len(sent.split()) for sent in sent_tokenize(text)) / len(sent_tokenize(text))
        }
        
        # Combine metrics into a single score (0-1)
        weights = {
            'spelling_errors': -0.3,
            'grammar_score': 0.3,
            'vocabulary_richness': 0.2,
            'sentence_complexity': 0.2
        }
        
        score = sum(
            weights[metric] * value
            for metric, value in metrics.items()
        )
        
        return max(0, min(1, (score + 1) / 2))  # Normalize to 0-1 