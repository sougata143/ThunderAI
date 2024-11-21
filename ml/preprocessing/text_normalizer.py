from typing import List, Dict, Any, Optional, Union
import re
import unicodedata
from num2words import num2words
import inflect
from nltk.tokenize import word_tokenize
from ..monitoring.custom_metrics import MetricsCollector

class TextNormalizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.inflect_engine = inflect.engine()
        
        # Compile regex patterns
        self.number_pattern = re.compile(r'(?:\d*\.)?\d+')
        self.ordinal_pattern = re.compile(r'\d+(st|nd|rd|th)')
        self.time_pattern = re.compile(r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?')
        self.date_pattern = re.compile(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}')
        self.money_pattern = re.compile(r'[$€£¥]?\d+(?:\.\d{2})?')
        
        # Load custom mappings
        self.custom_mappings = config.get('custom_mappings', {})
        self.abbreviations = config.get('abbreviations', {
            'mr.': 'mister',
            'mrs.': 'missus',
            'dr.': 'doctor',
            'prof.': 'professor',
            'etc.': 'etcetera'
        })
    
    def normalize_text(
        self,
        text: str,
        normalize_numbers: bool = True,
        normalize_dates: bool = True,
        normalize_times: bool = True,
        normalize_money: bool = True,
        expand_abbreviations: bool = True,
        custom_normalize: bool = True
    ) -> str:
        """Apply various normalization techniques to text"""
        if normalize_numbers:
            text = self._normalize_numbers(text)
        
        if normalize_dates:
            text = self._normalize_dates(text)
        
        if normalize_times:
            text = self._normalize_times(text)
        
        if normalize_money:
            text = self._normalize_money(text)
        
        if expand_abbreviations:
            text = self._expand_abbreviations(text)
        
        if custom_normalize:
            text = self._apply_custom_normalization(text)
        
        # Record metrics
        self.metrics_collector.record_preprocessing_metric(
            'text_normalization',
            len(text.split())
        )
        
        return text
    
    def _normalize_numbers(self, text: str) -> str:
        """Convert numbers to words"""
        def replace_number(match):
            num = match.group()
            try:
                # Handle ordinals
                if self.ordinal_pattern.match(num):
                    num = num[:-2]  # Remove ordinal suffix
                    return self.inflect_engine.number_to_words(
                        self.inflect_engine.ordinal(int(float(num)))
                    )
                # Handle regular numbers
                return num2words(float(num))
            except:
                return num
        
        return self.number_pattern.sub(replace_number, text)
    
    def _normalize_dates(self, text: str) -> str:
        """Convert dates to a standard format"""
        def replace_date(match):
            date = match.group()
            try:
                parts = re.split(r'[-/]', date)
                if len(parts) == 3:
                    month, day, year = parts
                    # Convert to words
                    month_word = self.inflect_engine.number_to_words(int(month))
                    day_word = self.inflect_engine.ordinal(int(day))
                    return f"{month_word} {day_word} {year}"
            except:
                pass
            return date
        
        return self.date_pattern.sub(replace_date, text)
    
    def _normalize_times(self, text: str) -> str:
        """Convert times to words"""
        def replace_time(match):
            time = match.group()
            try:
                # Split time into components
                parts = time.lower().replace(':', ' ').replace('am', ' am').replace('pm', ' pm').split()
                hour = int(parts[0])
                minute = int(parts[1])
                period = parts[2] if len(parts) > 2 else ''
                
                # Convert to words
                hour_word = self.inflect_engine.number_to_words(hour)
                minute_word = self.inflect_engine.number_to_words(minute) if minute > 0 else ''
                
                if minute_word:
                    return f"{hour_word} {minute_word} {period}".strip()
                return f"{hour_word} {period}".strip()
            except:
                return time
        
        return self.time_pattern.sub(replace_time, text)
    
    def _normalize_money(self, text: str) -> str:
        """Convert money amounts to words"""
        def replace_money(match):
            amount = match.group()
            try:
                # Remove currency symbol and convert to float
                numeric_amount = float(re.sub(r'[^\d.]', '', amount))
                return num2words(numeric_amount, to='currency')
            except:
                return amount
        
        return self.money_pattern.sub(replace_money, text)
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations"""
        words = word_tokenize(text)
        expanded = []
        
        for word in words:
            lower_word = word.lower()
            if lower_word in self.abbreviations:
                expanded.append(self.abbreviations[lower_word])
            else:
                expanded.append(word)
        
        return ' '.join(expanded)
    
    def _apply_custom_normalization(self, text: str) -> str:
        """Apply custom normalization rules"""
        for pattern, replacement in self.custom_mappings.items():
            text = re.sub(pattern, replacement, text)
        return text
    
    def normalize_batch(
        self,
        texts: List[str],
        **kwargs
    ) -> List[str]:
        """Normalize a batch of texts"""
        return [self.normalize_text(text, **kwargs) for text in texts]
    
    def add_custom_mapping(self, pattern: str, replacement: str):
        """Add a custom normalization mapping"""
        self.custom_mappings[pattern] = replacement
    
    def add_abbreviation(self, abbreviation: str, expansion: str):
        """Add a new abbreviation mapping"""
        self.abbreviations[abbreviation.lower()] = expansion.lower() 