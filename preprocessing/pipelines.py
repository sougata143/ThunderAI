from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import spacy
import re

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, remove_urls: bool = True, remove_numbers: bool = True):
        self.remove_urls = remove_urls
        self.remove_numbers = remove_numbers
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: List[str]) -> List[str]:
        cleaned_texts = []
        for text in X:
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            if self.remove_urls:
                text = re.sub(r'http\S+|www\S+|https\S+', '', text)
                
            # Remove numbers
            if self.remove_numbers:
                text = re.sub(r'\d+', '', text)
                
            # Remove special characters
            text = re.sub(r'[^\w\s]', '', text)
            
            cleaned_texts.append(text)
            
        return cleaned_texts

class TextNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, lemmatize: bool = True):
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer() if lemmatize else None
        self.stop_words = set(stopwords.words('english'))
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: List[str]) -> List[str]:
        normalized_texts = []
        for text in X:
            tokens = word_tokenize(text)
            
            # Remove stop words and lemmatize
            if self.lemmatize:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                         if token not in self.stop_words]
            else:
                tokens = [token for token in tokens if token not in self.stop_words]
                
            normalized_texts.append(' '.join(tokens))
            
        return normalized_texts

class SpacyEntityExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, model: str = "en_core_web_sm"):
        self.nlp = spacy.load(model)
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X: List[str]) -> List[Dict[str, List[str]]]:
        entities = []
        for text in X:
            doc = self.nlp(text)
            text_entities = {
                'ORG': [],
                'PERSON': [],
                'GPE': [],
                'DATE': []
            }
            
            for ent in doc.ents:
                if ent.label_ in text_entities:
                    text_entities[ent.label_].append(ent.text)
                    
            entities.append(text_entities)
            
        return entities

def create_preprocessing_pipeline(
    clean_text: bool = True,
    normalize: bool = True,
    extract_entities: bool = False
) -> Pipeline:
    steps = []
    
    if clean_text:
        steps.append(('cleaner', TextCleaner()))
    
    if normalize:
        steps.append(('normalizer', TextNormalizer()))
    
    if extract_entities:
        steps.append(('entity_extractor', SpacyEntityExtractor()))
    
    return Pipeline(steps) 