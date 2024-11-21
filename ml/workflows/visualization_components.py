from typing import Dict, Any, List, Optional
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta

class WorkflowVisualizer:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.layout_cache = {}
    
    def create_workflow_graph(
        self,
        workflow_data: Dict[str, Any]
    ) -> go.Figure:
        """Create interactive workflow graph visualization"""
        # Build graph
        self.graph.clear()
        
        # Add nodes for each step
        for step in workflow_data['steps']:
            self.graph.add_node(
                step['name'],
                status=step.get('status', 'pending'),
                duration=step.get('duration', 0),
                resources=step.get('resources', {}),
                metrics=step.get('metrics', {})
            )
            
            # Add edges for dependencies
            for dep in step.get('dependencies', []):
                self.graph.add_edge(dep, step['name'])
        
        # Calculate layout
        layout = nx.spring_layout(self.graph)
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        for edge in self.graph.edges():
            x0, y0 = layout[edge[0]]
            x1, y1 = layout[edge[1]]
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
        
        for node in self.graph.nodes():
            x, y = layout[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = self.graph.nodes[node]
            node_colors.append(color_map.get(node_data['status'], '#95a5a6'))
            node_sizes.append(30)
            
            # Create hover text
            hover_text = [
                f"Step: {node}",
                f"Status: {node_data['status']}",
                f"Duration: {node_data['duration']:.2f}s"
            ]
            
            if node_data['metrics']:
                hover_text.extend([
                    f"{k}: {v:.4f}"
                    for k, v in node_data['metrics'].items()
                ])
            
            node_text.append('<br>'.join(hover_text))
        
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
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_timeline_view(
        self,
        workflow_data: Dict[str, Any]
    ) -> go.Figure:
        """Create Gantt chart timeline visualization"""
        steps = []
        for step in workflow_data['steps']:
            start_time = step.get('start_time', workflow_data['start_time'])
            end_time = step.get('end_time', datetime.utcnow())
            
            steps.append({
                'Task': step['name'],
                'Start': start_time,
                'Finish': end_time,
                'Status': step.get('status', 'pending')
            })
        
        df = pd.DataFrame(steps)
        
        color_map = {
            'completed': '#2ecc71',
            'running': '#3498db',
            'failed': '#e74c3c',
            'pending': '#95a5a6'
        }
        
        fig = go.Figure()
        
        for status in color_map:
            df_status = df[df['Status'] == status]
            if not df_status.empty:
                fig.add_trace(
                    go.Bar(
                        name=status,
                        x=[(finish - start).total_seconds() / 3600
                           for start, finish in zip(df_status['Start'], df_status['Finish'])],
                        y=df_status['Task'],
                        orientation='h',
                        marker=dict(color=color_map[status]),
                        customdata=df_status[['Start', 'Finish']].values,
                        hovertemplate=(
                            "Task: %{y}<br>"
                            "Duration: %{x:.2f} hours<br>"
                            "Start: %{customdata[0]}<br>"
                            "End: %{customdata[1]}<br>"
                        )
                    )
                )
        
        fig.update_layout(
            title='Workflow Timeline',
            barmode='stack',
            height=max(300, len(steps) * 40),
            showlegend=True,
            xaxis_title='Duration (hours)',
            yaxis=dict(
                title='',
                autorange="reversed"
            )
        )
        
        return fig 