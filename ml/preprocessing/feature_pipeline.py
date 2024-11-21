from typing import List, Dict, Any, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from ..monitoring.custom_metrics import MetricsCollector

class FeaturePipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        self.pipeline = self._build_pipeline()
    
    def _build_pipeline(self) -> Pipeline:
        """Build the feature engineering pipeline"""
        numeric_features = self.config.get('numeric_features', [])
        categorical_features = self.config.get('categorical_features', [])
        text_features = self.config.get('text_features', [])
        
        numeric_transformer = Pipeline(
            steps=[
                ('scaler', StandardScaler())
            ]
        )
        
        categorical_transformer = Pipeline(
            steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]
        )
        
        text_transformer = Pipeline(
            steps=[
                ('vectorizer', TfidfVectorizer(
                    max_features=self.config.get('max_features', 1000)
                )),
                ('svd', TruncatedSVD(
                    n_components=self.config.get('n_components', 100)
                ))
            ]
        )
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('text', text_transformer, text_features)
            ]
        )
        
        return Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selector', FeatureSelector(
                n_features=self.config.get('n_features', 100)
            ))
        ])
    
    def fit_transform(
        self,
        data: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Fit and transform the data"""
        try:
            start_time = time.time()
            transformed = self.pipeline.fit_transform(data)
            
            # Record metrics
            self.metrics_collector.record_preprocessing_metric(
                'feature_pipeline_time',
                time.time() - start_time
            )
            
            return transformed
        except Exception as e:
            logging.error(f"Error in feature pipeline: {str(e)}")
            self.metrics_collector.record_preprocessing_metric(
                'feature_pipeline_error',
                1
            )
            raise
    
    def transform(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """Transform new data"""
        return self.pipeline.transform(data)
    
    def get_feature_names(self) -> List[str]:
        """Get names of the engineered features"""
        return self.pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    def save(self, path: str):
        """Save pipeline state"""
        joblib.dump(self.pipeline, f"{path}/feature_pipeline.joblib")
    
    def load(self, path: str):
        """Load pipeline state"""
        self.pipeline = joblib.load(f"{path}/feature_pipeline.joblib")

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selection transformer"""
    def __init__(self, n_features: int = 100):
        self.n_features = n_features
        self.selected_features_ = None
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit the feature selector"""
        # Implement feature selection logic (e.g., using mutual information)
        self.selected_features_ = np.argsort(
            np.var(X, axis=0)
        )[-self.n_features:]
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data by selecting features"""
        if self.selected_features_ is None:
            raise ValueError("Transformer not fitted")
        return X[:, self.selected_features_] 