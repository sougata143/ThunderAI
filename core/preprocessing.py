import pandas as pd
import numpy as np
from typing import Union, List, Dict
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

class DataPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))
        
    def process_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Process pandas DataFrame containing text data"""
        df['processed_text'] = df[text_column].apply(self.clean_text)
        df['tokens'] = df['processed_text'].apply(self.tokenize)
        return df
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data"""
        doc = self.nlp(text.lower())
        cleaned_tokens = [
            token.text for token in doc 
            if not token.is_stop and not token.is_punct
        ]
        return " ".join(cleaned_tokens)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text using NLTK"""
        return word_tokenize(text)
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create text embeddings using SpaCy"""
        return np.array([self.nlp(text).vector for text in texts]) 