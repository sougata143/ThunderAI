import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Any, List

class ModelPerformanceVisualizer:
    def create_metrics_dashboard(self, metrics: Dict[str, float]) -> go.Figure:
        """Create comprehensive metrics visualization dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Accuracy & Loss', 'Precision-Recall',
                'Confusion Matrix', 'ROC Curve'
            )
        )

        # Accuracy & Loss plot
        fig.add_trace(
            go.Scatter(
                y=metrics['accuracy_history'],
                name='Accuracy',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                y=metrics['loss_history'],
                name='Loss',
                line=dict(color='red')
            ),
            row=1, col=1
        )

        # Precision-Recall curve
        fig.add_trace(
            go.Scatter(
                x=metrics['recall'],
                y=metrics['precision'],
                name='Precision-Recall',
                fill='tozeroy'
            ),
            row=1, col=2
        )

        # Confusion Matrix heatmap
        fig.add_trace(
            go.Heatmap(
                z=metrics['confusion_matrix'],
                text=metrics['confusion_matrix'],
                texttemplate="%{text}",
                textfont={"size": 20},
                colorscale='Blues'
            ),
            row=2, col=1
        )

        # ROC curve
        fig.add_trace(
            go.Scatter(
                x=metrics['fpr'],
                y=metrics['tpr'],
                name=f'ROC (AUC = {metrics["auc"]:.3f})',
                fill='tozeroy'
            ),
            row=2, col=2
        )

        fig.update_layout(height=800, title_text="Model Performance Dashboard")
        return fig 