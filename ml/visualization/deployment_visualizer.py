import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

class DeploymentVisualizer:
    def create_deployment_dashboard(self, deployment_data: Dict[str, Any]) -> go.Figure:
        """Create deployment status visualization dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Deployment Status', 'Version History',
                'Performance Comparison', 'Resource Utilization'
            )
        )

        # Deployment Status timeline
        stages = deployment_data['deployment_stages']
        fig.add_trace(
            go.Timeline(
                x=[stage['start_time'] for stage in stages],
                y=[stage['name'] for stage in stages],
                mode='markers+lines',
                marker=dict(
                    color=[
                        'green' if stage['status'] == 'completed'
                        else 'red' if stage['status'] == 'failed'
                        else 'yellow'
                        for stage in stages
                    ]
                )
            ),
            row=1, col=1
        )

        # Version History
        versions = deployment_data['version_history']
        fig.add_trace(
            go.Scatter(
                x=[v['timestamp'] for v in versions],
                y=[v['version'] for v in versions],
                mode='markers+lines',
                name='Version History'
            ),
            row=1, col=2
        )

        # Performance Comparison
        perf_comparison = deployment_data['performance_comparison']
        fig.add_trace(
            go.Bar(
                x=['Previous', 'Current'],
                y=[
                    perf_comparison['previous_version']['accuracy'],
                    perf_comparison['current_version']['accuracy']
                ],
                name='Accuracy Comparison'
            ),
            row=2, col=1
        )

        # Resource Utilization
        resources = deployment_data['resource_utilization']
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=resources['cpu_usage'],
                title={'text': "CPU Usage (%)"},
                gauge={'axis': {'range': [0, 100]}}
            ),
            row=2, col=2
        )

        fig.update_layout(height=800, title_text="Deployment Status Dashboard")
        return fig 