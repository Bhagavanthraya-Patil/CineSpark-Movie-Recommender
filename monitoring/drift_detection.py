"""
Data drift detection using Evidently AI for RagFlix recommendation system.
Monitors feature distributions and model performance.
"""

from typing import Dict, Any, Optional
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data drift in features and model performance."""
    
    def __init__(
        self,
        reference_data_path: Optional[str] = None,
        current_data_path: Optional[str] = None
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data_path: Path to reference dataset
            current_data_path: Path to current dataset
        """
        self.reference_data_path = reference_data_path
        self.current_data_path = current_data_path
        
        try:
            from evidently import ColumnMapping
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report
            
            self.ColumnMapping = ColumnMapping
            self.DataDriftPreset = DataDriftPreset
            self.Report = Report
            self.evidently_available = True
            logger.info("Evidently AI initialized")
        except ImportError:
            self.evidently_available = False
            logger.warning("Evidently AI not installed, using placeholder")
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from path (Delta, CSV, etc.)."""
        if data_path.endswith(".csv"):
            return pd.read_csv(data_path)
        elif "delta" in data_path.lower():
            # Load from Delta Lake
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            df = spark.read.format("delta").load(data_path)
            return df.toPandas()
        else:
            raise ValueError(f"Unsupported data format: {data_path}")
    
    def detect_drift(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        current_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            
        Returns:
            Drift detection report
        """
        if not self.evidently_available:
            return {
                "has_drift": False,
                "message": "Evidently AI not available",
                "drift_score": 0.0
            }
        
        # Load data if paths provided
        if reference_data is None and self.reference_data_path:
            reference_data = self.load_data(self.reference_data_path)
        
        if current_data is None and self.current_data_path:
            current_data = self.load_data(self.current_data_path)
        
        if reference_data is None or current_data is None:
            raise ValueError("Reference and current data must be provided")
        
        # Define column mapping
        column_mapping = self.ColumnMapping()
        column_mapping.numerical_features = [
            col for col in reference_data.columns
            if reference_data[col].dtype in ['int64', 'float64']
        ]
        column_mapping.categorical_features = [
            col for col in reference_data.columns
            if reference_data[col].dtype == 'object'
        ]
        
        # Generate drift report
        data_drift_report = self.Report(metrics=[self.DataDriftPreset()])
        data_drift_report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )
        
        # Extract drift metrics
        metrics = data_drift_report.as_dict()["metrics"]
        
        # Check for drift
        has_drift = False
        drift_score = 0.0
        
        for metric in metrics:
            if "dataset_drift" in metric:
                has_drift = metric["dataset_drift"]["value"]
                drift_score = metric.get("dataset_drift_score", {}).get("value", 0.0)
                break
        
        report = {
            "has_drift": has_drift,
            "drift_score": drift_score,
            "metrics": metrics,
            "report_html": data_drift_report.get_html()
        }
        
        if has_drift:
            logger.warning(f"Data drift detected! Score: {drift_score}")
        else:
            logger.info("No data drift detected")
        
        return report
    
    def detect_model_drift(
        self,
        reference_predictions: pd.DataFrame,
        current_predictions: pd.DataFrame,
        reference_target: Optional[pd.Series] = None,
        current_target: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Detect model performance drift.
        
        Args:
            reference_predictions: Reference model predictions
            current_predictions: Current model predictions
            reference_target: Reference actual values
            current_target: Current actual values
            
        Returns:
            Model drift report
        """
        if not self.evidently_available:
            return {
                "has_drift": False,
                "message": "Evidently AI not available"
            }
        
        try:
            from evidently.metric_preset import ClassificationPerformancePreset
            
            # Create report
            model_performance_report = self.Report(
                metrics=[ClassificationPerformancePreset()]
            )
            
            # Prepare data
            reference_data = reference_predictions.copy()
            current_data = current_predictions.copy()
            
            if reference_target is not None:
                reference_data["target"] = reference_target
            if current_target is not None:
                current_data["target"] = current_target
            
            # Run report
            model_performance_report.run(
                reference_data=reference_data,
                current_data=current_data
            )
            
            metrics = model_performance_report.as_dict()["metrics"]
            
            return {
                "has_drift": True,  # Would need to check specific metrics
                "metrics": metrics,
                "report_html": model_performance_report.get_html()
            }
        
        except Exception as e:
            logger.error(f"Error detecting model drift: {e}")
            return {
                "has_drift": False,
                "error": str(e)
            }


# Example usage
if __name__ == "__main__":
    detector = DriftDetector()
    
    # Example with sample data
    import numpy as np
    
    reference_data = pd.DataFrame({
        "user_id": range(100),
        "avg_rating": np.random.normal(3.5, 0.5, 100),
        "total_events": np.random.poisson(50, 100)
    })
    
    current_data = pd.DataFrame({
        "user_id": range(100),
        "avg_rating": np.random.normal(4.0, 0.6, 100),  # Shifted distribution
        "total_events": np.random.poisson(60, 100)  # Different distribution
    })
    
    report = detector.detect_drift(reference_data, current_data)
    print(f"Drift detected: {report['has_drift']}")
    print(f"Drift score: {report['drift_score']}")

