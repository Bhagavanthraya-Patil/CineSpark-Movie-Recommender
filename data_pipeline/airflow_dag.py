"""
Airflow DAG for orchestrating RagFlix MLOps pipeline.
Schedules daily feature refresh, model retraining, and deployment.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.databricks.operators.databricks import DatabricksSubmitRunOperator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'ragflix-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# DAG definition
dag = DAG(
    'ragflix_mlops_pipeline',
    default_args=default_args,
    description='RagFlix MLOps pipeline: feature refresh, training, deployment',
    schedule_interval='@daily',  # Run daily at midnight
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'recommendation', 'ragflix'],
)

# Task 1: Refresh feature store
def refresh_feature_store(**context):
    """Refresh user and movie features from Delta Lake."""
    from feature_store.user_features import UserFeatureStore
    from feature_store.movie_features import MovieFeatureStore
    
    logger.info("Refreshing feature store...")
    
    user_fs = UserFeatureStore()
    user_fs.refresh_features()
    
    movie_fs = MovieFeatureStore()
    movie_fs.refresh_features()
    
    logger.info("Feature store refresh completed")
    return "Feature store refreshed successfully"

refresh_features_task = PythonOperator(
    task_id='refresh_feature_store',
    python_callable=refresh_feature_store,
    dag=dag,
)

# Task 2: Run model training on Databricks
train_model_task = DatabricksSubmitRunOperator(
    task_id='train_recommendation_model',
    databricks_conn_id='databricks_default',
    existing_cluster_id='{{ var.value.databricks_cluster_id }}',
    notebook_task={
        'notebook_path': '/Workspace/ragflix/model_pipeline/als_training',
    },
    dag=dag,
)

# Task 3: Evaluate model and check metrics
def evaluate_model(**context):
    """Evaluate model performance and check if it meets deployment criteria."""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    
    # Get latest experiment run
    experiment = client.get_experiment_by_name("ragflix-recommendations")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("No runs found in experiment")
    
    latest_run = runs[0]
    metrics = latest_run.data.metrics
    
    # Check if model meets criteria (example: RMSE < 1.0)
    rmse = metrics.get('rmse', float('inf'))
    if rmse >= 1.0:
        raise ValueError(f"Model RMSE {rmse} does not meet deployment criteria (< 1.0)")
    
    logger.info(f"Model evaluation passed: RMSE={rmse}")
    return f"Model evaluation passed: RMSE={rmse}"

evaluate_model_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

# Task 4: Register model in MLflow
def register_model(**context):
    """Register the best model to MLflow Model Registry."""
    import mlflow
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    
    # Get latest run
    experiment = client.get_experiment_by_name("ragflix-recommendations")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],  # Best model (lowest RMSE)
        max_results=1
    )
    
    if not runs:
        raise ValueError("No runs found")
    
    best_run = runs[0]
    
    # Register model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mv = client.create_model_version(
        name="ragflix-recommendation-model",
        source=model_uri,
        run_id=best_run.info.run_id
    )
    
    logger.info(f"Model registered: {mv.name} version {mv.version}")
    return f"Model registered: version {mv.version}"

register_model_task = PythonOperator(
    task_id='register_model',
    python_callable=register_model,
    dag=dag,
)

# Task 5: Promote model to production
def promote_to_production(**context):
    """Promote model to production stage."""
    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    
    # Get latest model version
    model_name = "ragflix-recommendation-model"
    latest_version = client.get_latest_versions(model_name, stages=["None"])[0]
    
    # Transition to production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version.version,
        stage="Production"
    )
    
    logger.info(f"Model {latest_version.version} promoted to Production")
    return f"Model {latest_version.version} promoted to Production"

promote_model_task = PythonOperator(
    task_id='promote_to_production',
    python_callable=promote_to_production,
    dag=dag,
)

# Task 6: Run drift detection
def run_drift_detection(**context):
    """Run data drift detection using Evidently AI."""
    from monitoring.drift_detection import DriftDetector
    
    detector = DriftDetector()
    report = detector.detect_drift()
    
    if report.has_drift:
        logger.warning("Data drift detected! Consider retraining model.")
        # Could trigger alert or retraining
    else:
        logger.info("No data drift detected")
    
    return report

drift_detection_task = PythonOperator(
    task_id='drift_detection',
    python_callable=run_drift_detection,
    dag=dag,
)

# Task 7: Deploy model (trigger API restart)
deploy_model_task = BashOperator(
    task_id='deploy_model',
    bash_command='''
    # Restart API service to load new model
    # This could be a Kubernetes rollout, Docker restart, etc.
    echo "Deploying model to production API..."
    # kubectl rollout restart deployment/ragflix-api
    # or docker-compose restart api
    ''',
    dag=dag,
)

# Define task dependencies
refresh_features_task >> train_model_task >> evaluate_model_task
evaluate_model_task >> register_model_task >> promote_model_task
promote_model_task >> drift_detection_task >> deploy_model_task

