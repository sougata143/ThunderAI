from typing import Dict, Any, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class AdvancedWorkflowVisualizer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.metrics_history = {}
        self.layout_cache = {}
    
    def create_timeline_visualization(
        self,
        workflow_data: Dict[str, Any]
    ) -> go.Figure:
        """Create a Gantt chart timeline visualization"""
        steps = workflow_data['steps']
        
        # Prepare timeline data
        tasks = []
        start_times = []
        end_times = []
        durations = []
        colors = []
        
        color_map = {
            'completed': 'rgb(44, 160, 44)',
            'failed': 'rgb(214, 39, 40)',
            'running': 'rgb(255, 127, 14)',
            'pending': 'rgb(31, 119, 180)'
        }
        
        for step_name, step_data in steps.items():
            tasks.append(step_name)
            start_time = step_data.get('start_time', workflow_data['start_time'])
            end_time = step_data.get('end_time', datetime.utcnow())
            
            start_times.append(start_time)
            end_times.append(end_time)
            durations.append((end_time - start_time).total_seconds())
            colors.append(color_map.get(step_data.get('status', 'pending')))
        
        # Create Gantt chart
        fig = go.Figure()
        
        for idx, task in enumerate(tasks):
            fig.add_trace(go.Bar(
                x=[durations[idx]],
                y=[task],
                orientation='h',
                marker=dict(color=colors[idx]),
                customdata=[[
                    start_times[idx].strftime('%Y-%m-%d %H:%M:%S'),
                    end_times[idx].strftime('%Y-%m-%d %H:%M:%S'),
                    f"{durations[idx]:.2f}s"
                ]],
                hovertemplate=(
                    "Task: %{y}<br>"
                    "Start: %{customdata[0]}<br>"
                    "End: %{customdata[1]}<br>"
                    "Duration: %{customdata[2]}<br>"
                )
            ))
        
        fig.update_layout(
            title='Workflow Timeline',
            xaxis_title='Duration (seconds)',
            showlegend=False,
            height=max(300, len(tasks) * 40),
            barmode='overlay'
        )
        
        return fig
    
    def create_metrics_visualization(
        self,
        workflow_metrics: Dict[str, Any]
    ) -> go.Figure:
        """Create a comprehensive metrics visualization"""
        # Create subplots for different metrics
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                'Step Durations',
                'Resource Usage',
                'Success Rate',
                'Custom Metrics'
            )
        )
        
        # Step durations bar chart
        step_names = list(workflow_metrics['step_durations'].keys())
        durations = list(workflow_metrics['step_durations'].values())
        
        fig.add_trace(
            go.Bar(
                x=step_names,
                y=durations,
                name='Duration'
            ),
            row=1,
            col=1
        )
        
        # Resource usage line chart
        resource_metrics = workflow_metrics['resource_usage']
        for metric, values in resource_metrics.items():
            if isinstance(values, list):
                fig.add_trace(
                    go.Scatter(
                        y=values,
                        name=metric,
                        mode='lines'
                    ),
                    row=1,
                    col=2
                )
        
        # Success rate gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=workflow_metrics['success_rate'] * 100,
                gauge={'axis': {'range': [0, 100]}},
                title={'text': "Success Rate (%)"}
            ),
            row=2,
            col=1
        )
        
        # Custom metrics
        custom_metrics = workflow_metrics.get('custom_metrics', {})
        metric_names = list(custom_metrics.keys())
        metric_values = list(custom_metrics.values())
        
        fig.add_trace(
            go.Bar(
                x=metric_names,
                y=metric_values,
                name='Custom Metrics'
            ),
            row=2,
            col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Workflow Metrics Dashboard"
        )
        
        return fig
    
    def create_dependency_graph(
        self,
        workflow_data: Dict[str, Any],
        layout_type: str = 'spring'
    ) -> go.Figure:
        """Create an interactive dependency graph visualization"""
        # Build graph
        self.graph.clear()
        steps = workflow_data['steps']
        
        # Add nodes
        for step_name, step_data in steps.items():
            self.graph.add_node(
                step_name,
                status=step_data.get('status', 'pending'),
                duration=step_data.get('duration', 0),
                metrics=step_data.get('metrics', {})
            )
        
        # Add edges based on dependencies
        for step_name, step_data in steps.items():
            if 'dependencies' in step_data:
                for dep in step_data['dependencies']:
                    self.graph.add_edge(dep, step_name)
        
        # Calculate layout
        if layout_type == 'spring':
            layout = nx.spring_layout(self.graph)
        elif layout_type == 'circular':
            layout = nx.circular_layout(self.graph)
        else:
            layout = nx.kamada_kawai_layout(self.graph)
        
        # Create visualization
        edge_trace, node_trace = self._create_graph_traces(layout)
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Workflow Dependency Graph',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                annotations=[
                    dict(
                        text="",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0,
                        y=-0.1
                    )
                ]
            )
        )
        
        return fig
    
    def _create_graph_traces(
        self,
        layout: Dict[str, np.ndarray]
    ) -> tuple:
        """Create edge and node traces for graph visualization"""
        # Create edges
        edge_x = []
        edge_y = []
        
        for edge in self.graph.edges():
            x0, y0 = layout[edge[0]]
            x1, y1 = layout[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        color_map = {
            'completed': '#2ca02c',
            'failed': '#d62728',
            'running': '#ff7f0e',
            'pending': '#1f77b4'
        }
        
        for node in self.graph.nodes():
            x, y = layout[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = self.graph.nodes[node]
            text = (
                f"Step: {node}<br>"
                f"Status: {node_data['status']}<br>"
                f"Duration: {node_data['duration']:.2f}s"
            )
            
            if node_data['metrics']:
                text += "<br>Metrics:<br>"
                text += "<br>".join(
                    f"{k}: {v:.4f}"
                    for k, v in node_data['metrics'].items()
                )
            
            node_text.append(text)
            node_color.append(color_map.get(node_data['status'], '#888'))
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                showscale=False,
                color=node_color,
                size=20,
                line=dict(width=2)
            )
        )
        
        return edge_trace, node_trace 