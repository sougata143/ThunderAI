from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .analytics_service import WorkflowAnalytics
from ..monitoring.custom_metrics import MetricsCollector

class WorkflowVisualizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analytics = WorkflowAnalytics(config)
        self.metrics_collector = MetricsCollector()
    
    def create_workflow_dashboard(
        self,
        workflow_id: str,
        execution_data: Dict[str, Any]
    ) -> Dict[str, go.Figure]:
        """Create comprehensive workflow visualization dashboard"""
        return {
            'execution_graph': self.create_execution_graph(execution_data),
            'performance_metrics': self.create_performance_visualization(execution_data),
            'resource_usage': self.create_resource_visualization(execution_data),
            'bottleneck_analysis': self.create_bottleneck_visualization(execution_data),
            'optimization_impact': self.create_optimization_visualization(execution_data)
        }
    
    def create_execution_graph(
        self,
        execution_data: Dict[str, Any]
    ) -> go.Figure:
        """Create interactive execution graph visualization"""
        G = nx.DiGraph()
        
        # Add nodes and edges
        for step_name, step_data in execution_data['steps'].items():
            G.add_node(
                step_name,
                status=step_data['status'],
                duration=step_data.get('duration', 0),
                resources=step_data.get('resource_usage', {})
            )
            
            for dep in step_data.get('dependencies', []):
                G.add_edge(dep, step_name)
        
        # Calculate layout
        pos = nx.spring_layout(G)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
        )
        
        # Add nodes
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_text = []
        
        color_map = {
            'completed': '#2ecc71',
            'running': '#3498db',
            'failed': '#e74c3c',
            'pending': '#95a5a6'
        }
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node]
            node_colors.append(color_map.get(node_data['status'], '#95a5a6'))
            node_sizes.append(30)
            
            # Create hover text
            text = (
                f"Step: {node}<br>"
                f"Status: {node_data['status']}<br>"
                f"Duration: {node_data['duration']:.2f}s"
            )
            
            if node_data['resources']:
                text += "<br>Resources:<br>"
                text += "<br>".join(
                    f"{k}: {v:.2f}"
                    for k, v in node_data['resources'].items()
                )
            
            node_text.append(text)
        
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=2)
                )
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Workflow Execution Graph',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_performance_visualization(
        self,
        execution_data: Dict[str, Any]
    ) -> go.Figure:
        """Create performance metrics visualization"""
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Step Durations',
                'Success Rate',
                'Error Distribution',
                'Resource Efficiency'
            )
        )
        
        # Add step durations bar chart
        step_names = []
        durations = []
        colors = []
        
        for step_name, step_data in execution_data['steps'].items():
            step_names.append(step_name)
            durations.append(step_data.get('duration', 0))
            colors.append('#2ecc71' if step_data['status'] == 'completed' else '#e74c3c')
        
        fig.add_trace(
            go.Bar(
                x=step_names,
                y=durations,
                marker_color=colors,
                name='Duration'
            ),
            row=1,
            col=1
        )
        
        # Add success rate gauge
        success_rate = sum(
            1 for step in execution_data['steps'].values()
            if step['status'] == 'completed'
        ) / len(execution_data['steps'])
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=success_rate * 100,
                gauge={'axis': {'range': [0, 100]}},
                title={'text': "Success Rate (%)"}
            ),
            row=1,
            col=2
        )
        
        # Add error distribution pie chart
        error_counts = {}
        for step_data in execution_data['steps'].values():
            if step_data['status'] == 'failed':
                error_type = step_data.get('error', {}).get('type', 'unknown')
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        if error_counts:
            fig.add_trace(
                go.Pie(
                    labels=list(error_counts.keys()),
                    values=list(error_counts.values()),
                    name='Error Distribution'
                ),
                row=2,
                col=1
            )
        
        # Add resource efficiency scatter plot
        cpu_usage = []
        memory_usage = []
        step_labels = []
        
        for step_name, step_data in execution_data['steps'].items():
            resources = step_data.get('resource_usage', {})
            if 'cpu' in resources and 'memory' in resources:
                cpu_usage.append(resources['cpu'])
                memory_usage.append(resources['memory'])
                step_labels.append(step_name)
        
        fig.add_trace(
            go.Scatter(
                x=cpu_usage,
                y=memory_usage,
                mode='markers+text',
                text=step_labels,
                name='Resource Usage'
            ),
            row=2,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Workflow Performance Analysis"
        )
        
        return fig
    
    def create_resource_visualization(
        self,
        execution_data: Dict[str, Any]
    ) -> go.Figure:
        """Create resource usage visualization"""
        # Implementation for resource visualization
        pass
    
    def create_bottleneck_visualization(
        self,
        execution_data: Dict[str, Any]
    ) -> go.Figure:
        """Create bottleneck analysis visualization"""
        # Implementation for bottleneck visualization
        pass
    
    def create_optimization_visualization(
        self,
        execution_data: Dict[str, Any]
    ) -> go.Figure:
        """Create optimization impact visualization"""
        # Implementation for optimization visualization
        pass 