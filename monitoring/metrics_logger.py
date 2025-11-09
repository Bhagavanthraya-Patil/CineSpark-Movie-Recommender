import mlflow
from typing import Dict, Any
from datetime import datetime

class MetricsLogger:
    def __init__(self):
        mlflow.set_tracking_uri("databricks")
        
    def log_training_metrics(self, metrics: Dict[str, float]):
        with mlflow.start_run():
            mlflow.log_metrics(metrics)
    
    def log_inference_metrics(self, predictions: Any, actuals: Any):
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'prediction_count': len(predictions),
            'avg_response_time': 0.1  # Replace with actual timing
        }
        self.log_training_metrics(metrics)
    
    def log_system_metrics(self, cpu_usage: float, memory_usage: float):
        metrics = {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'timestamp': datetime.now().isoformat()
        }
        self.log_training_metrics(metrics)
