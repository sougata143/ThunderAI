from typing import Dict, Any, List, Optional, Union
import numpy as np
from datetime import datetime, timedelta
from .analytics_service import WorkflowAnalytics
from ..monitoring.custom_metrics import MetricsCollector
import logging

class WorkflowOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.analytics = WorkflowAnalytics(config)
        self.metrics_collector = MetricsCollector()
        
        # Optimization parameters
        self.optimization_window = config.get('optimization_window', 7)  # days
        self.min_executions = config.get('min_executions', 10)
        self.resource_threshold = config.get('resource_threshold', 0.8)
        self.performance_threshold = config.get('performance_threshold', 0.9)
    
    def optimize_workflow(
        self,
        workflow_id: str,
        execution_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate workflow optimization recommendations"""
        try:
            # Analyze historical performance
            performance_analysis = self._analyze_performance(execution_history)
            
            # Generate optimization strategies
            optimizations = {
                'resource_allocation': self._optimize_resources(performance_analysis),
                'step_ordering': self._optimize_step_order(performance_analysis),
                'parallelization': self._identify_parallelization(performance_analysis),
                'caching': self._optimize_caching(performance_analysis)
            }
            
            # Calculate potential improvements
            improvements = self._calculate_improvements(optimizations)
            
            # Record optimization metrics
            self._record_optimization_metrics(workflow_id, optimizations)
            
            return {
                'optimizations': optimizations,
                'improvements': improvements,
                'analysis': performance_analysis
            }
            
        except Exception as e:
            logging.error(f"Error optimizing workflow: {str(e)}")
            raise
    
    def _analyze_performance(
        self,
        execution_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze workflow performance patterns"""
        analysis = {
            'step_durations': {},
            'resource_usage': {},
            'dependencies': {},
            'bottlenecks': []
        }
        
        for execution in execution_history:
            for step, data in execution['steps'].items():
                # Analyze step durations
                if step not in analysis['step_durations']:
                    analysis['step_durations'][step] = []
                analysis['step_durations'][step].append(data['duration'])
                
                # Analyze resource usage
                if step not in analysis['resource_usage']:
                    analysis['resource_usage'][step] = {
                        'cpu': [],
                        'memory': [],
                        'gpu': []
                    }
                for resource, usage in data.get('resource_usage', {}).items():
                    analysis['resource_usage'][step][resource].append(usage)
                
                # Analyze dependencies
                if 'dependencies' in data:
                    analysis['dependencies'][step] = data['dependencies']
        
        # Identify bottlenecks
        analysis['bottlenecks'] = self._identify_bottlenecks(analysis)
        
        return analysis
    
    def _optimize_resources(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize resource allocation"""
        recommendations = {}
        
        for step, resource_usage in analysis['resource_usage'].items():
            step_recommendations = {}
            
            for resource, usage in resource_usage.items():
                if usage:  # If we have usage data
                    mean_usage = np.mean(usage)
                    p95_usage = np.percentile(usage, 95)
                    
                    if p95_usage > self.resource_threshold:
                        step_recommendations[resource] = {
                            'action': 'increase',
                            'current_usage': mean_usage,
                            'recommended': p95_usage * 1.2  # 20% buffer
                        }
                    elif mean_usage < self.resource_threshold * 0.5:
                        step_recommendations[resource] = {
                            'action': 'decrease',
                            'current_usage': mean_usage,
                            'recommended': p95_usage * 1.5  # Conservative decrease
                        }
            
            if step_recommendations:
                recommendations[step] = step_recommendations
        
        return recommendations
    
    def _optimize_step_order(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize workflow step ordering"""
        import networkx as nx
        
        # Create dependency graph
        G = nx.DiGraph()
        for step, deps in analysis['dependencies'].items():
            for dep in deps:
                G.add_edge(dep, step)
        
        # Find critical path
        critical_path = nx.dag_longest_path(G, weight='duration')
        
        # Generate step ordering recommendations
        recommendations = {
            'critical_path': critical_path,
            'parallel_groups': self._identify_parallel_groups(G),
            'reordering_suggestions': self._suggest_reordering(G, analysis)
        }
        
        return recommendations
    
    def _identify_parallelization(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Identify parallelization opportunities"""
        opportunities = {}
        
        # Find independent steps
        dependency_graph = analysis['dependencies']
        for step in analysis['step_durations']:
            if step not in dependency_graph or not dependency_graph[step]:
                opportunities[step] = {
                    'type': 'independent',
                    'potential_speedup': analysis['step_durations'][step][-1]
                }
        
        # Find data-parallel opportunities
        for step, durations in analysis['step_durations'].items():
            if np.mean(durations) > np.median(durations) * 1.5:
                opportunities[step] = {
                    'type': 'data_parallel',
                    'potential_speedup': np.mean(durations) * 0.6  # Estimated 40% improvement
                }
        
        return opportunities
    
    def _optimize_caching(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize workflow caching strategy"""
        cache_recommendations = {}
        
        for step, durations in analysis['step_durations'].items():
            mean_duration = np.mean(durations)
            
            if mean_duration > 60:  # Cache steps taking more than 1 minute
                cache_recommendations[step] = {
                    'strategy': 'result_cache',
                    'ttl': self._calculate_cache_ttl(mean_duration),
                    'potential_savings': mean_duration
                }
        
        return cache_recommendations
    
    def _identify_bottlenecks(
        self,
        analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify workflow bottlenecks"""
        bottlenecks = []
        
        for step, durations in analysis['step_durations'].items():
            mean_duration = np.mean(durations)
            p95_duration = np.percentile(durations, 95)
            
            if p95_duration > mean_duration * 2:
                bottlenecks.append({
                    'step': step,
                    'type': 'performance',
                    'severity': 'high',
                    'metric': {
                        'mean': mean_duration,
                        'p95': p95_duration
                    }
                })
            
            # Check resource bottlenecks
            resource_usage = analysis['resource_usage'].get(step, {})
            for resource, usage in resource_usage.items():
                if usage and np.mean(usage) > self.resource_threshold:
                    bottlenecks.append({
                        'step': step,
                        'type': 'resource',
                        'resource': resource,
                        'severity': 'medium',
                        'metric': {
                            'mean_usage': np.mean(usage),
                            'p95_usage': np.percentile(usage, 95)
                        }
                    })
        
        return bottlenecks
    
    def _calculate_improvements(
        self,
        optimizations: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate potential improvements from optimizations"""
        improvements = {
            'execution_time': 0.0,
            'resource_efficiency': 0.0,
            'cost_savings': 0.0
        }
        
        # Calculate execution time improvement
        if 'parallelization' in optimizations:
            parallel_savings = sum(
                opp['potential_speedup']
                for opp in optimizations['parallelization'].values()
            )
            improvements['execution_time'] += parallel_savings
        
        # Calculate resource efficiency improvement
        if 'resource_allocation' in optimizations:
            resource_savings = sum(
                1 - rec['recommended'] / rec['current_usage']
                for step in optimizations['resource_allocation'].values()
                for rec in step.values()
                if rec['action'] == 'decrease'
            )
            improvements['resource_efficiency'] = resource_savings
        
        # Calculate cost savings
        improvements['cost_savings'] = (
            improvements['execution_time'] * 0.001 +  # Estimated cost per second
            improvements['resource_efficiency'] * 0.01  # Estimated cost per resource unit
        )
        
        return improvements
    
    def _record_optimization_metrics(
        self,
        workflow_id: str,
        optimizations: Dict[str, Any]
    ):
        """Record optimization-related metrics"""
        self.metrics_collector.record_workflow_metric(
            workflow_id=workflow_id,
            metric_name='optimization_suggestions',
            value=len(optimizations)
        )
        
        for opt_type, opt_data in optimizations.items():
            self.metrics_collector.record_workflow_metric(
                workflow_id=workflow_id,
                metric_name=f'optimization_{opt_type}',
                value=len(opt_data)
            ) 