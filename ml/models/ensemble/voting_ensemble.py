from typing import Dict, Any, List
import numpy as np
from collections import Counter
from .base_ensemble import BaseEnsemble

class VotingEnsemble(BaseEnsemble):
    def __init__(self, models: List[Any], weights: List[float] = None, voting: str = 'hard'):
        super().__init__(models, weights)
        self.voting = voting
    
    def aggregate_predictions(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.voting == 'hard':
            return self._hard_voting(predictions)
        else:
            return self._soft_voting(predictions)
    
    def _hard_voting(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Get predicted classes with weights
        weighted_predictions = []
        for pred, weight in zip(predictions, self.weights):
            weighted_predictions.extend([pred['prediction']] * int(weight * 100))
        
        # Get most common prediction
        final_prediction = Counter(weighted_predictions).most_common(1)[0][0]
        
        # Calculate confidence as ratio of votes
        confidence = Counter(weighted_predictions)[final_prediction] / len(weighted_predictions)
        
        return {
            'prediction': final_prediction,
            'confidence': confidence
        }
    
    def _soft_voting(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Weight and average probabilities
        weighted_probs = np.zeros_like(predictions[0]['probabilities'])
        for pred, weight in zip(predictions, self.weights):
            weighted_probs += np.array(pred['probabilities']) * weight
            
        final_prediction = int(np.argmax(weighted_probs))
        confidence = float(np.max(weighted_probs))
        
        return {
            'prediction': final_prediction,
            'confidence': confidence,
            'probabilities': weighted_probs.tolist()
        } 