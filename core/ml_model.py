import numpy as np
from typing import Union, List
import spacy
import nltk
from transformers import pipeline

class ThunderAIModel:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
    def preprocess_text(self, text: str) -> List[str]:
        # Process text using both SpaCy and NLTK
        doc = self.nlp(text)
        tokens = nltk.word_tokenize(text)
        return [token.lower_ for token in doc], tokens
    
    def analyze_sentiment(self, text: str) -> dict:
        return self.sentiment_analyzer(text)[0]