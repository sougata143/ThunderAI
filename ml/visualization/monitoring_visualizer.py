import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

class MonitoringVisualizer:
    def create_monitoring_dashboard(self, monitoring_data: Dict[str, Any]) -> go.Figure:
        """Create real-time monitoring visualization dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Prediction Latency', 'Request Rate',
                'Error Rate', 'Resource Usage',
                'Prediction Distribution', 'System Health'
            )
        )

        # Prediction Latency
        fig.add_trace(
            go.Scatter(
                x=monitoring_data['timestamps'],
                y=monitoring_data['latencies'],
                name='Latency',
                line=dict(color='blue')
            ),
            row=1, col=1
        )

        # Request Rate
        fig.add_trace(
            go.Bar(
                x=monitoring_data['timestamps'],
                y=monitoring_data['request_rate'],
                name='Requests/sec'
            ),
            row=1, col=2
        )

        # Error Rate
        fig.add_trace(
            go.Scatter(
                x=monitoring_data['timestamps'],
                y=monitoring_data['error_rate'],
                name='Error Rate',
                line=dict(color='red')
            ),
            row=2, col=1
        )

        # Resource Usage
        fig.add_trace(
            go.Scatter(
                x=monitoring_data['timestamps'],
                y=monitoring_data['memory_usage'],
                name='Memory',
                line=dict(color='purple')
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=monitoring_data['timestamps'],
                y=monitoring_data['cpu_usage'],
                name='CPU',
                line=dict(color='orange')
            ),
            row=2, col=2
        )

        # Prediction Distribution
        fig.add_trace(
            go.Pie(
                labels=list(monitoring_data['prediction_dist'].keys()),
                values=list(monitoring_data['prediction_dist'].values())
            ),
            row=3, col=1
        )

        # System Health
        health_metrics = monitoring_data['system_health']
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=health_metrics['health_score'],
                gauge={'axis': {'range': [0, 100]}},
                title={'text': "System Health"}
            ),
            row=3, col=2
        )

        fig.update_layout(height=1200, title_text="Model Monitoring Dashboard")
        return fig 