import torch
import tensorflow as tf
import numpy as np
from typing import Union, List
import spacy
import nltk
from transformers import pipeline

class ThunderAIModel:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.pytorch_model = self._init_pytorch_model()
        self.tensorflow_model = self._init_tensorflow_model()
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        
    def _init_pytorch_model(self):
        # Initialize PyTorch model
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        )
        return model
    
    def _init_tensorflow_model(self):
        # Initialize TensorFlow model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu')
        ])
        return model
    
    def preprocess_text(self, text: str) -> List[str]:
        # Process text using both SpaCy and NLTK
        doc = self.nlp(text)
        tokens = nltk.word_tokenize(text)
        return [token.lower_ for token in doc], tokens
    
    def analyze_sentiment(self, text: str) -> dict:
        return self.sentiment_analyzer(text)[0] 