from typing import Dict, Any, List
from .voting_ensemble import VotingEnsemble
from .stacking_ensemble import StackingEnsemble
from ..base import BaseModel

class EnsembleFactory:
    @staticmethod
    def create_ensemble(
        ensemble_type: str,
        models: List[BaseModel],
        config: Dict[str, Any]
    ) -> BaseModel:
        """Create an ensemble model based on type and configuration"""
        if ensemble_type == 'voting':
            return VotingEnsemble(
                models=models,
                weights=config.get('weights'),
                voting=config.get('voting', 'soft')
            )
        elif ensemble_type == 'stacking':
            return StackingEnsemble(
                models=models,
                meta_learner=config.get('meta_learner'),
                num_classes=config.get('num_classes', 2)
            )
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
    
    @staticmethod
    def load_ensemble(
        ensemble_type: str,
        path: str,
        model_configs: List[Dict[str, Any]]
    ) -> BaseModel:
        """Load an ensemble model from disk"""
        # Load base models
        models = []
        for config in model_configs:
            model = BaseModel.from_config(config)
            models.append(model)
        
        # Create and load ensemble
        ensemble = EnsembleFactory.create_ensemble(
            ensemble_type=ensemble_type,
            models=models,
            config={}
        )
        ensemble.load(path)
        
        return ensemble 