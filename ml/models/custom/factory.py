from typing import Dict, Any, Type, List
from .base_architecture import CustomArchitecture, CustomModelWrapper
from .transformer import CustomTransformerArchitecture
from .cnn import CustomCNNArchitecture
from .rnn import CustomRNNArchitecture

class ModelFactory:
    _architectures = {
        'transformer': CustomTransformerArchitecture,
        'cnn': CustomCNNArchitecture,
        'rnn': CustomRNNArchitecture
    }
    
    @classmethod
    def register_architecture(
        cls,
        name: str,
        architecture_class: Type[CustomArchitecture]
    ):
        """Register a new architecture"""
        cls._architectures[name] = architecture_class
    
    @classmethod
    def create_model(
        cls,
        architecture: str,
        config: Dict[str, Any]
    ) -> CustomModelWrapper:
        """Create a model with the specified architecture"""
        if architecture not in cls._architectures:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        architecture_class = cls._architectures[architecture]
        return CustomModelWrapper(architecture_class, config)
    
    @classmethod
    def get_available_architectures(cls) -> List[str]:
        """Get list of available architectures"""
        return list(cls._architectures.keys()) 