import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

class TestResultsVisualizer:
    def create_test_dashboard(self, test_results: Dict[str, Any]) -> go.Figure:
        """Create test results visualization dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Test Pass/Fail', 'Performance Tests',
                'Error Distribution', 'Resource Usage'
            )
        )

        # Test Pass/Fail pie chart
        passed = sum(1 for t in test_results['functional_tests'] if t['passed'])
        failed = len(test_results['functional_tests']) - passed
        fig.add_trace(
            go.Pie(
                labels=['Passed', 'Failed'],
                values=[passed, failed],
                marker=dict(colors=['green', 'red'])
            ),
            row=1, col=1
        )

        # Performance test results
        perf_metrics = test_results['performance_tests']
        fig.add_trace(
            go.Bar(
                x=['Latency', 'Throughput', 'Memory', 'CPU'],
                y=[
                    perf_metrics['avg_latency'],
                    perf_metrics['throughput'],
                    perf_metrics['memory_usage'],
                    perf_metrics['cpu_usage']
                ]
            ),
            row=1, col=2
        )

        # Error distribution
        error_types = test_results.get('error_distribution', {})
        fig.add_trace(
            go.Bar(
                x=list(error_types.keys()),
                y=list(error_types.values())
            ),
            row=2, col=1
        )

        # Resource usage over time
        fig.add_trace(
            go.Scatter(
                y=test_results['resource_usage']['memory'],
                name='Memory',
                line=dict(color='blue')
            ),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(
                y=test_results['resource_usage']['cpu'],
                name='CPU',
                line=dict(color='red')
            ),
            row=2, col=2
        )

        fig.update_layout(height=800, title_text="Test Results Dashboard")
        return fig 