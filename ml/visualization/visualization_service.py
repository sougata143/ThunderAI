from typing import Dict, Any
import plotly.graph_objects as go
from .performance_visualizer import ModelPerformanceVisualizer
from .test_visualizer import TestResultsVisualizer
from .monitoring_visualizer import MonitoringVisualizer
from .deployment_visualizer import DeploymentVisualizer

class VisualizationService:
    def __init__(self):
        self.performance_visualizer = ModelPerformanceVisualizer()
        self.test_visualizer = TestResultsVisualizer()
        self.monitoring_visualizer = MonitoringVisualizer()
        self.deployment_visualizer = DeploymentVisualizer()

    def create_comprehensive_dashboard(
        self,
        metrics: Dict[str, Any],
        test_results: Dict[str, Any],
        monitoring_data: Dict[str, Any],
        deployment_data: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """Create comprehensive visualization dashboard"""
        return {
            'performance': self.performance_visualizer.create_metrics_dashboard(metrics),
            'testing': self.test_visualizer.create_test_dashboard(test_results),
            'monitoring': self.monitoring_visualizer.create_monitoring_dashboard(monitoring_data),
            'deployment': self.deployment_visualizer.create_deployment_dashboard(deployment_data)
        }

    def save_visualizations(
        self,
        visualizations: Dict[str, go.Figure],
        output_dir: str
    ):
        """Save visualizations to files"""
        for name, fig in visualizations.items():
            fig.write_html(f"{output_dir}/{name}_dashboard.html")
            fig.write_image(f"{output_dir}/{name}_dashboard.png") 