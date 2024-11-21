from typing import Dict, Any, List, Optional, Union
import numpy as np
from datetime import datetime
from sklearn.model_selection import ParameterGrid
import logging
from ..monitoring.custom_metrics import MetricsCollector

class WorkflowOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_collector = MetricsCollector()
        
        # Initialize optimization parameters
        self.param_space = config.get('param_space', {})
        self.optimization_metric = config.get('optimization_metric', 'duration')
        self.max_trials = config.get('max_trials', 10)
        self.exploration_rate = config.get('exploration_rate', 0.2)
    
    def optimize_workflow(
        self,
        workflow_template: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Optimize workflow configuration"""
        try:
            # Analyze historical performance
            performance_data = self._analyze_historical_performance(
                historical_data
            )
            
            # Generate parameter combinations
            param_grid = ParameterGrid(self.param_space)
            
            # Find optimal configuration
            optimal_config = self._find_optimal_configuration(
                param_grid,
                performance_data
            )
            
            # Record optimization metrics
            self._record_optimization_metrics(optimal_config)
            
            return optimal_config
            
        except Exception as e:
            logging.error(f"Error optimizing workflow: {str(e)}")
            raise
    
    def _analyze_historical_performance(
        self,
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze historical workflow performance"""
        performance_data = {
            'configurations': [],
            'metrics': []
        }
        
        for execution in historical_data:
            if execution['status'] == 'completed':
                performance_data['configurations'].append(
                    execution['config']
                )
                performance_data['metrics'].append({
                    'duration': execution['duration'],
                    'success_rate': execution['success_rate'],
                    'resource_usage': execution.get('resource_usage', {})
                })
        
        return performance_data
    
    def _find_optimal_configuration(
        self,
        param_grid: ParameterGrid,
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Find optimal configuration using exploration and exploitation"""
        best_config = None
        best_score = float('inf')
        
        # Exploration phase
        exploration_trials = int(self.max_trials * self.exploration_rate)
        for _ in range(exploration_trials):
            config = self._sample_random_config(param_grid)
            score = self._evaluate_configuration(
                config,
                performance_data
            )
            
            if score < best_score:
                best_score = score
                best_config = config
        
        # Exploitation phase
        remaining_trials = self.max_trials - exploration_trials
        for _ in range(remaining_trials):
            config = self._refine_configuration(
                best_config,
                param_grid,
                performance_data
            )
            score = self._evaluate_configuration(
                config,
                performance_data
            )
            
            if score < best_score:
                best_score = score
                best_config = config
        
        return best_config
    
    def _sample_random_config(
        self,
        param_grid: ParameterGrid
    ) -> Dict[str, Any]:
        """Sample random configuration from parameter space"""
        all_configs = list(param_grid)
        return np.random.choice(all_configs)
    
    def _refine_configuration(
        self,
        base_config: Dict[str, Any],
        param_grid: ParameterGrid,
        performance_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Refine configuration based on local search"""
        refined_config = base_config.copy()
        
        for param in base_config:
            if param in self.param_space:
                param_values = self.param_space[param]
                current_idx = param_values.index(base_config[param])
                
                # Try neighboring values
                neighbors = []
                if current_idx > 0:
                    neighbors.append(param_values[current_idx - 1])
                if current_idx < len(param_values) - 1:
                    neighbors.append(param_values[current_idx + 1])
                
                for value in neighbors:
                    refined_config[param] = value
                    score = self._evaluate_configuration(
                        refined_config,
                        performance_data
                    )
                    
                    if score < self._evaluate_configuration(
                        base_config,
                        performance_data
                    ):
                        return refined_config
        
        return base_config
    
    def _evaluate_configuration(
        self,
        config: Dict[str, Any],
        performance_data: Dict[str, Any]
    ) -> float:
        """Evaluate configuration based on historical performance"""
        similar_executions = self._find_similar_executions(
            config,
            performance_data
        )
        
        if not similar_executions:
            return float('inf')
        
        # Calculate score based on optimization metric
        if self.optimization_metric == 'duration':
            return np.mean([
                execution['duration']
                for execution in similar_executions
            ])
        elif self.optimization_metric == 'resource_usage':
            return np.mean([
                sum(execution['resource_usage'].values())
                for execution in similar_executions
            ])
        else:
            return -np.mean([
                execution['success_rate']
                for execution in similar_executions
            ])
    
    def _find_similar_executions(
        self,
        config: Dict[str, Any],
        performance_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find historical executions with similar configurations"""
        similar_executions = []
        
        for i, historical_config in enumerate(
            performance_data['configurations']
        ):
            if self._is_similar_config(config, historical_config):
                similar_executions.append(
                    performance_data['metrics'][i]
                )
        
        return similar_executions
    
    def _is_similar_config(
        self,
        config1: Dict[str, Any],
        config2: Dict[str, Any]
    ) -> bool:
        """Check if two configurations are similar"""
        similarity_threshold = self.config.get('similarity_threshold', 0.8)
        common_params = set(config1.keys()) & set(config2.keys())
        
        if not common_params:
            return False
        
        matches = sum(
            config1[param] == config2[param]
            for param in common_params
        )
        
        return matches / len(common_params) >= similarity_threshold
    
    def _record_optimization_metrics(
        self,
        optimal_config: Dict[str, Any]
    ):
        """Record optimization metrics"""
        self.metrics_collector.record_workflow_metric(
            'optimization_completed',
            1,
            {'config': str(optimal_config)}
        ) 