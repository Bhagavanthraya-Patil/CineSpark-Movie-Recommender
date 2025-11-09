"""
MLflow tracking utilities for RagFlix recommendation models.
Handles experiment tracking, model logging, and registry operations.
"""

import mlflow
import mlflow.spark
from mlflow.tracking import MlflowClient
from typing import Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowTracker:
    """Manages MLflow experiment tracking and model registry."""
    
    def __init__(
        self,
        experiment_name: str = "ragflix-recommendations",
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (defaults to Databricks)
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created experiment: {experiment_name}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"Could not set experiment: {e}")
            experiment_id = None
        
        self.experiment_name = experiment_name
        self.experiment_id = experiment_id
        self.client = MlflowClient()
    
    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLflow run."""
        mlflow.set_experiment(self.experiment_name)
        return mlflow.start_run(run_name=run_name)
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        mlflow.log_params(params)
        logger.info(f"Logged parameters: {params}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        mlflow.log_metrics(metrics, step=step)
        logger.info(f"Logged metrics: {metrics}")
    
    def log_model(self, model, artifact_path: str = "model", **kwargs):
        """
        Log a model to MLflow.
        
        Args:
            model: Model object (PySpark, PyTorch, etc.)
            artifact_path: Path within run artifacts
            **kwargs: Additional arguments for model logging
        """
        if hasattr(model, 'save'):
            # PySpark model
            mlflow.spark.log_model(model, artifact_path, **kwargs)
        else:
            # Generic model (use appropriate flavor)
            mlflow.sklearn.log_model(model, artifact_path, **kwargs)
        
        logger.info(f"Logged model to {artifact_path}")
    
    def register_model(
        self,
        model_uri: str,
        model_name: str = "ragflix-recommendation-model"
    ):
        """
        Register a model in MLflow Model Registry.
        
        Args:
            model_uri: URI of the model (e.g., "runs:/run_id/model")
            model_name: Name for the registered model
        """
        try:
            mv = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=mlflow.active_run().info.run_id if mlflow.active_run() else None
            )
            logger.info(f"Registered model: {model_name} version {mv.version}")
            return mv
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def get_production_model(self, model_name: str = "ragflix-recommendation-model"):
        """
        Get the production model from registry.
        
        Args:
            model_name: Name of the registered model
            
        Returns:
            Model version in Production stage
        """
        try:
            model_version = self.client.get_latest_versions(
                model_name,
                stages=["Production"]
            )[0]
            logger.info(f"Retrieved production model: version {model_version.version}")
            return model_version
        except IndexError:
            logger.warning("No production model found")
            return None
    
    def promote_to_production(
        self,
        model_name: str = "ragflix-recommendation-model",
        version: Optional[int] = None
    ):
        """
        Promote a model version to Production stage.
        
        Args:
            model_name: Name of the registered model
            version: Version to promote (defaults to latest)
        """
        if version is None:
            # Get latest version
            versions = self.client.get_latest_versions(model_name, stages=["None"])
            if not versions:
                raise ValueError(f"No versions found for {model_name}")
            version = versions[0].version
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        logger.info(f"Promoted {model_name} version {version} to Production")


# Example usage
if __name__ == "__main__":
    tracker = MLflowTracker()
    
    with tracker.start_run(run_name="test-run"):
        tracker.log_params({
            "alpha": 0.1,
            "rank": 10,
            "iterations": 20
        })
        tracker.log_metrics({
            "rmse": 0.85,
            "mae": 0.65,
            "precision_at_10": 0.42
        })

