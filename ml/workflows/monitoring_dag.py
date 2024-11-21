from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.time_delta import TimeDeltaSensor
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

from ..monitoring.model_performance_monitor import ModelPerformanceMonitor
from ..retraining.orchestrator import RetrainingOrchestrator
from ..monitoring.custom_metrics import MetricsCollector
from ..versioning.model_registry import ModelRegistry

# Initialize services
model_registry = ModelRegistry()
metrics_collector = MetricsCollector()
performance_monitor = ModelPerformanceMonitor({})
retraining_orchestrator = RetrainingOrchestrator({})

default_args = {
    'owner': 'thunderai',
    'depends_on_past': False,
    'email': ['alerts@thunderai.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def check_model_performance(**context) -> Dict[str, Any]:
    """Check model performance and trigger retraining if needed"""
    try:
        model_id = context['model_id']
        current_metrics = performance_monitor.get_current_metrics(model_id)
        
        # Check if retraining is needed
        needs_retraining = retraining_orchestrator.check_retraining_triggers(
            model_id,
            current_metrics
        )
        
        context['task_instance'].xcom_push(
            key='needs_retraining',
            value=needs_retraining
        )
        
        return current_metrics
        
    except Exception as e:
        logging.error(f"Error checking model performance: {str(e)}")
        raise

def trigger_retraining(**context) -> None:
    """Trigger model retraining if needed"""
    try:
        needs_retraining = context['task_instance'].xcom_pull(
            key='needs_retraining',
            task_ids='check_performance'
        )
        
        if needs_retraining:
            model_id = context['model_id']
            retraining_orchestrator.run_retraining_pipeline(
                model_id,
                context['task_instance'].xcom_pull(
                    key='return_value',
                    task_ids='check_performance'
                )
            )
            
    except Exception as e:
        logging.error(f"Error triggering retraining: {str(e)}")
        raise

def update_monitoring_dashboards(**context) -> None:
    """Update monitoring dashboards with latest metrics"""
    try:
        model_id = context['model_id']
        metrics = context['task_instance'].xcom_pull(
            key='return_value',
            task_ids='check_performance'
        )
        
        # Update Grafana dashboards
        metrics_collector.update_dashboard_metrics(model_id, metrics)
        
    except Exception as e:
        logging.error(f"Error updating dashboards: {str(e)}")
        raise

def send_monitoring_alerts(**context) -> None:
    """Send alerts based on monitoring thresholds"""
    try:
        metrics = context['task_instance'].xcom_pull(
            key='return_value',
            task_ids='check_performance'
        )
        
        # Check alert thresholds and send notifications
        performance_monitor.check_and_send_alerts(metrics)
        
    except Exception as e:
        logging.error(f"Error sending alerts: {str(e)}")
        raise

# Create DAG for each monitored model
for model_info in model_registry.list_models():
    model_id = model_info['id']
    
    dag = DAG(
        f'model_monitoring_{model_id}',
        default_args=default_args,
        description=f'Monitoring DAG for model {model_id}',
        schedule_interval=timedelta(hours=1),
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['monitoring', 'ml', model_id],
    )
    
    with dag:
        # Wait sensor
        wait_for_interval = TimeDeltaSensor(
            task_id='wait_for_interval',
            delta=timedelta(hours=1)
        )
        
        # Check model performance
        check_performance = PythonOperator(
            task_id='check_performance',
            python_callable=check_model_performance,
            op_kwargs={'model_id': model_id},
            provide_context=True,
        )
        
        # Trigger retraining if needed
        trigger_model_retraining = PythonOperator(
            task_id='trigger_retraining',
            python_callable=trigger_retraining,
            op_kwargs={'model_id': model_id},
            provide_context=True,
        )
        
        # Update monitoring dashboards
        update_dashboards = PythonOperator(
            task_id='update_dashboards',
            python_callable=update_monitoring_dashboards,
            op_kwargs={'model_id': model_id},
            provide_context=True,
        )
        
        # Send monitoring alerts
        send_alerts = PythonOperator(
            task_id='send_alerts',
            python_callable=send_monitoring_alerts,
            op_kwargs={'model_id': model_id},
            provide_context=True,
        )
        
        # Define task dependencies
        wait_for_interval >> check_performance >> [
            trigger_model_retraining,
            update_dashboards,
            send_alerts
        ]
        
    # Add DAG to global namespace
    globals()[f'monitoring_dag_{model_id}'] = dag 