"""
User feature engineering and storage for RagFlix recommendation system.
Computes user preferences, watch history, and behavioral features.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, max as spark_max, min as spark_min,
    collect_list, when, sum as spark_sum, datediff, current_date
)
from databricks import feature_store
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserFeatureStore:
    """Manages user feature computation and storage in Databricks Feature Store."""
    
    def __init__(
        self,
        delta_path: str = "/dbfs/mnt/ragflix/delta",
        feature_store_name: str = "ragflix_feature_store"
    ):
        """
        Initialize user feature store.
        
        Args:
            delta_path: Path to Delta Lake tables
            feature_store_name: Name of the feature store
        """
        self.spark = SparkSession.builder \
            .appName("UserFeatureStore") \
            .getOrCreate()
        
        self.delta_path = delta_path
        self.fs = feature_store.FeatureStoreClient()
        self.feature_store_name = feature_store_name
        
        logger.info("UserFeatureStore initialized")
    
    def compute_user_features(self):
        """
        Compute user features from events and ratings.
        
        Returns:
            DataFrame with user features
        """
        # Read raw events
        events_df = self.spark.read.format("delta").load(
            f"{self.delta_path}/raw_events"
        )
        
        # Read ratings (assuming MovieLens format)
        ratings_df = self.spark.read.format("delta").load(
            f"{self.delta_path}/ratings"
        )
        
        # User activity features
        user_activity = events_df \
            .groupBy("user_id") \
            .agg(
                count("*").alias("total_events"),
                count(when(col("event_type") == "view", 1)).alias("total_views"),
                count(when(col("event_type") == "rating", 1)).alias("total_ratings"),
                count(when(col("event_type") == "click", 1)).alias("total_clicks"),
                spark_max("event_timestamp").alias("last_activity_date"),
                spark_min("event_timestamp").alias("first_activity_date")
            ) \
            .withColumn(
                "days_since_last_activity",
                datediff(current_date(), col("last_activity_date"))
            ) \
            .withColumn(
                "user_tenure_days",
                datediff(col("last_activity_date"), col("first_activity_date"))
            )
        
        # User rating statistics
        user_ratings = ratings_df \
            .groupBy("user_id") \
            .agg(
                avg("rating").alias("avg_rating"),
                spark_max("rating").alias("max_rating"),
                spark_min("rating").alias("min_rating"),
                count("*").alias("num_ratings"),
                spark_sum("rating").alias("total_rating_sum")
            )
        
        # User genre preferences (if genre data available)
        # This would require joining with movie metadata
        # For now, we'll create a placeholder
        
        # Combine all user features
        user_features = user_activity \
            .join(user_ratings, on="user_id", how="outer") \
            .fillna(0)  # Fill nulls with 0
        
        logger.info(f"Computed features for {user_features.count()} users")
        return user_features
    
    def compute_user_movie_interaction_features(self):
        """
        Compute user-movie interaction features.
        
        Returns:
            DataFrame with user-movie features
        """
        events_df = self.spark.read.format("delta").load(
            f"{self.delta_path}/raw_events"
        )
        
        ratings_df = self.spark.read.format("delta").load(
            f"{self.delta_path}/ratings"
        )
        
        # User-movie interaction history
        interactions = events_df \
            .groupBy("user_id", "movie_id") \
            .agg(
                count("*").alias("interaction_count"),
                count(when(col("event_type") == "view", 1)).alias("view_count"),
                spark_max("event_timestamp").alias("last_interaction")
            )
        
        # User-movie ratings
        user_movie_ratings = ratings_df \
            .select("user_id", "movie_id", "rating", "timestamp") \
            .withColumnRenamed("timestamp", "rating_timestamp")
        
        # Combine interactions and ratings
        user_movie_features = interactions \
            .join(user_movie_ratings, on=["user_id", "movie_id"], how="outer")
        
        return user_movie_features
    
    def write_to_feature_store(self, features_df, table_name: str = "user_features"):
        """
        Write features to Databricks Feature Store.
        
        Args:
            features_df: DataFrame with features
            table_name: Name of the feature table
        """
        try:
            # Create feature table
            self.fs.create_table(
                name=f"{self.feature_store_name}.{table_name}",
                primary_keys=["user_id"],
                df=features_df,
                description="User features for recommendation system"
            )
            logger.info(f"Features written to {table_name}")
        except Exception as e:
            # Table might already exist, try to write
            logger.warning(f"Table might exist: {e}, attempting to write...")
            self.fs.write_table(
                name=f"{self.feature_store_name}.{table_name}",
                df=features_df,
                mode="merge"
            )
            logger.info(f"Features merged to {table_name}")
    
    def refresh_features(self):
        """Refresh all user features and write to feature store."""
        logger.info("Refreshing user features...")
        
        # Compute features
        user_features = self.compute_user_features()
        
        # Write to feature store
        self.write_to_feature_store(user_features, "user_features")
        
        # Compute and write user-movie features
        user_movie_features = self.compute_user_movie_interaction_features()
        self.write_to_feature_store(
            user_movie_features,
            "user_movie_features"
        )
        
        logger.info("User features refreshed successfully")


# Example usage
if __name__ == "__main__":
    user_fs = UserFeatureStore()
    user_fs.refresh_features()

