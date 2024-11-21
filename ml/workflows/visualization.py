from typing import Dict, Any, List, Optional
import networkx as nx
import plotly.graph_objects as go
from datetime import datetime
import json

class WorkflowVisualizer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.layout = None
    
    def create_workflow_graph(
        self,
        steps: List[Dict[str, Any]],
        dependencies: Dict[str, List[str]]
    ):
        """Create a directed graph representation of the workflow"""
        # Clear existing graph
        self.graph.clear()
        
        # Add nodes (steps)
        for step in steps:
            self.graph.add_node(
                step['name'],
                status=step.get('status', 'pending'),
                duration=step.get('duration', 0),
                start_time=step.get('start_time'),
                end_time=step.get('end_time')
            )
        
        # Add edges (dependencies)
        for step, deps in dependencies.items():
            for dep in deps:
                self.graph.add_edge(dep, step)
        
        # Calculate layout
        self.layout = nx.spring_layout(self.graph)
    
    def generate_workflow_visualization(
        self,
        format: str = 'plotly'
    ) -> Any:
        """Generate workflow visualization"""
        if format == 'plotly':
            return self._generate_plotly_visualization()
        else:
            raise ValueError(f"Unsupported visualization format: {format}")
    
    def _generate_plotly_visualization(self) -> go.Figure:
        """Generate interactive Plotly visualization"""
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = self.layout[edge[0]]
            x1, y1 = self.layout[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        color_map = {
            'pending': '#1f77b4',
            'running': '#ff7f0e',
            'completed': '#2ca02c',
            'failed': '#d62728'
        }
        
        for node in self.graph.nodes():
            x, y = self.layout[node]
            node_x.append(x)
            node_y.append(y)
            
            # Create node text
            node_data = self.graph.nodes[node]
            text = f"Step: {node}<br>"
            text += f"Status: {node_data['status']}<br>"
            if node_data.get('duration'):
                text += f"Duration: {node_data['duration']:.2f}s<br>"
            if node_data.get('start_time'):
                text += f"Start: {node_data['start_time'].strftime('%H:%M:%S')}<br>"
            if node_data.get('end_time'):
                text += f"End: {node_data['end_time'].strftime('%H:%M:%S')}"
            
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
                line_width=2
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title='Workflow Execution Graph',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        return fig
    
    def export_visualization(
        self,
        fig: go.Figure,
        filename: str,
        format: str = 'html'
    ):
        """Export visualization to file"""
        if format == 'html':
            fig.write_html(filename)
        elif format == 'png':
            fig.write_image(filename)
        elif format == 'json':
            with open(filename, 'w') as f:
                json.dump(fig.to_dict(), f)
        else:
            raise ValueError(f"Unsupported export format: {format}") 