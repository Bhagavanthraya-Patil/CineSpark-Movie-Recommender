"""
ALS (Alternating Least Squares) collaborative filtering model training.
Trains recommendation model on MovieLens data using PySpark MLlib.
"""

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, split, explode
import mlflow
from model_pipeline.mlflow_tracking import MLflowTracker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_als_model(
    ratings_path: str = "/dbfs/mnt/ragflix/delta/ratings",
    train_test_split: float = 0.8,
    rank: int = 10,
    max_iter: int = 10,
    reg_param: float = 0.1,
    alpha: float = 1.0,
    cold_start_strategy: str = "drop"
):
    """
    Train ALS collaborative filtering model.
    
    Args:
        ratings_path: Path to ratings Delta table
        train_test_split: Train/test split ratio
        rank: Number of latent factors
        max_iter: Maximum iterations
        reg_param: Regularization parameter
        alpha: Alpha parameter for implicit feedback
        cold_start_strategy: Strategy for handling cold start
        
    Returns:
        Trained ALS model
    """
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("ALSTraining") \
        .getOrCreate()
    
    logger.info("Loading ratings data...")
    
    # Load ratings data
    ratings_df = spark.read.format("delta").load(ratings_path)
    
    # Ensure required columns exist
    ratings_df = ratings_df.select(
        col("user_id").cast("int"),
        col("movie_id").cast("int"),
        col("rating").cast("float")
    ).filter(
        col("user_id").isNotNull() &
        col("movie_id").isNotNull() &
        col("rating").isNotNull()
    )
    
    logger.info(f"Loaded {ratings_df.count()} ratings")
    
    # Split into train and test
    train_df, test_df = ratings_df.randomSplit([train_test_split, 1 - train_test_split])
    
    logger.info(f"Train: {train_df.count()}, Test: {test_df.count()}")
    
    # Initialize MLflow tracker
    tracker = MLflowTracker()
    
    with tracker.start_run(run_name=f"ALS_rank{rank}_iter{max_iter}"):
        # Log hyperparameters
        tracker.log_params({
            "rank": rank,
            "max_iter": max_iter,
            "reg_param": reg_param,
            "alpha": alpha,
            "cold_start_strategy": cold_start_strategy,
            "train_test_split": train_test_split
        })
        
        # Initialize ALS model
        als = ALS(
            rank=rank,
            maxIter=max_iter,
            regParam=reg_param,
            alpha=alpha,
            userCol="user_id",
            itemCol="movie_id",
            ratingCol="rating",
            coldStartStrategy=cold_start_strategy,
            implicitPrefs=False  # Explicit ratings
        )
        
        logger.info("Training ALS model...")
        
        # Train model
        model = als.fit(train_df)
        
        logger.info("Model training completed")
        
        # Evaluate on test set
        logger.info("Evaluating model...")
        predictions = model.transform(test_df)
        
        # Remove NaN predictions (cold start)
        predictions = predictions.filter(col("prediction").isNotNull())
        
        # Calculate metrics
        evaluator_rmse = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )
        rmse = evaluator_rmse.evaluate(predictions)
        
        evaluator_mae = RegressionEvaluator(
            metricName="mae",
            labelCol="rating",
            predictionCol="prediction"
        )
        mae = evaluator_mae.evaluate(predictions)
        
        # Calculate precision@k (top 10 recommendations)
        # This is a simplified version
        user_recs = model.recommendForAllUsers(10)
        precision_at_10 = calculate_precision_at_k(user_recs, test_df, k=10)
        
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "precision_at_10": precision_at_10
        }
        
        logger.info(f"Metrics: {metrics}")
        
        # Log metrics
        tracker.log_metrics(metrics)
        
        # Log model
        tracker.log_model(model, "als_model")
        
        logger.info("Model logged to MLflow")
        
        return model


def calculate_precision_at_k(user_recs_df, test_df, k=10):
    """
    Calculate precision@k metric.
    
    Args:
        user_recs_df: DataFrame with recommendations
        test_df: Test set with actual ratings
        k: Number of recommendations to consider
        
    Returns:
        Precision@k score
    """
    # This is a simplified implementation
    # In production, you'd want a more robust calculation
    
    # Get top-k recommendations per user
    recommendations = user_recs_df.select(
        col("user_id"),
        explode(col("recommendations")).alias("rec")
    ).select(
        col("user_id"),
        col("rec.movie_id").alias("movie_id"),
        col("rec.rating").alias("predicted_rating")
    )
    
    # Join with test set to find relevant items (rating >= 4)
    relevant_items = test_df.filter(col("rating") >= 4.0)
    
    # Calculate precision
    hits = recommendations.join(
        relevant_items,
        on=["user_id", "movie_id"],
        how="inner"
    ).count()
    
    total_recs = recommendations.count()
    
    if total_recs == 0:
        return 0.0
    
    precision = hits / total_recs
    return precision


# Main execution
if __name__ == "__main__":
    # Train with default parameters
    model = train_als_model()
    
    # Or train with custom parameters
    # model = train_als_model(
    #     rank=20,
    #     max_iter=15,
    #     reg_param=0.05
    # )

