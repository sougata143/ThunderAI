from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PerformanceVisualizer:
    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str] = None,
        normalize: bool = True,
        title: str = 'Confusion Matrix'
    ) -> go.Figure:
        """Create an interactive confusion matrix plot"""
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=labels or [f'Class {i}' for i in range(cm.shape[1])],
            y=labels or [f'Class {i}' for i in range(cm.shape[0])],
            hoverongaps=False,
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=600,
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_roc_curve(
        fpr: np.ndarray,
        tpr: np.ndarray,
        roc_auc: float,
        title: str = 'ROC Curve'
    ) -> go.Figure:
        """Create an interactive ROC curve plot"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {roc_auc:.2f})'
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash'),
            name='Random'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=600
        )
        
        return fig
    
    @staticmethod
    def plot_training_history(
        history: Dict[str, List[float]],
        title: str = 'Training History'
    ) -> go.Figure:
        """Create an interactive training history plot"""
        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=('Loss', 'Metrics')
        )
        
        # Plot loss
        if 'loss' in history:
            fig.add_trace(
                go.Scatter(
                    y=history['loss'],
                    name='Training Loss',
                    mode='lines'
                ),
                row=1,
                col=1
            )
        
        if 'val_loss' in history:
            fig.add_trace(
                go.Scatter(
                    y=history['val_loss'],
                    name='Validation Loss',
                    mode='lines'
                ),
                row=1,
                col=1
            )
        
        # Plot other metrics
        for metric in history:
            if metric not in ['loss', 'val_loss']:
                fig.add_trace(
                    go.Scatter(
                        y=history[metric],
                        name=metric,
                        mode='lines'
                    ),
                    row=2,
                    col=1
                )
        
        fig.update_layout(
            height=800,
            title_text=title,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def plot_feature_importance(
        features: List[str],
        importance: np.ndarray,
        title: str = 'Feature Importance'
    ) -> go.Figure:
        """Create an interactive feature importance plot"""
        # Sort features by importance
        sorted_idx = np.argsort(importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        fig = go.Figure(data=go.Bar(
            x=importance[sorted_idx],
            y=[features[i] for i in sorted_idx],
            orientation='h'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance',
            yaxis_title='Features',
            height=max(400, len(features) * 20),
            width=800
        )
        
        return fig 