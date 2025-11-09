"""
Movie feature engineering and storage for RagFlix recommendation system.
Computes movie metadata, popularity, and content-based features.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, avg, max as spark_max, min as spark_min,
    collect_list, when, sum as spark_sum, explode, split, countDistinct
)
from databricks import feature_store
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieFeatureStore:
    """Manages movie feature computation and storage in Databricks Feature Store."""
    
    def __init__(
        self,
        delta_path: str = "/dbfs/mnt/ragflix/delta",
        feature_store_name: str = "ragflix_feature_store"
    ):
        """
        Initialize movie feature store.
        
        Args:
            delta_path: Path to Delta Lake tables
            feature_store_name: Name of the feature store
        """
        self.spark = SparkSession.builder \
            .appName("MovieFeatureStore") \
            .getOrCreate()
        
        self.delta_path = delta_path
        self.fs = feature_store.FeatureStoreClient()
        self.feature_store_name = feature_store_name
        
        logger.info("MovieFeatureStore initialized")
    
    def compute_movie_popularity_features(self):
        """
        Compute movie popularity features from events and ratings.
        
        Returns:
            DataFrame with movie popularity features
        """
        # Read raw events
        events_df = self.spark.read.format("delta").load(
            f"{self.delta_path}/raw_events"
        )
        
        # Read ratings
        ratings_df = self.spark.read.format("delta").load(
            f"{self.delta_path}/ratings"
        )
        
        # Movie activity features
        movie_activity = events_df \
            .groupBy("movie_id") \
            .agg(
                count("*").alias("total_events"),
                count(when(col("event_type") == "view", 1)).alias("total_views"),
                count(when(col("event_type") == "rating", 1)).alias("total_ratings"),
                count(when(col("event_type") == "click", 1)).alias("total_clicks"),
                countDistinct("user_id").alias("unique_users"),
                spark_max("event_timestamp").alias("last_activity_date")
            )
        
        # Movie rating statistics
        movie_ratings = ratings_df \
            .groupBy("movie_id") \
            .agg(
                avg("rating").alias("avg_rating"),
                spark_max("rating").alias("max_rating"),
                spark_min("rating").alias("min_rating"),
                count("*").alias("num_ratings"),
                spark_sum("rating").alias("total_rating_sum")
            )
        
        # Combine popularity features
        movie_popularity = movie_activity \
            .join(movie_ratings, on="movie_id", how="outer") \
            .fillna(0)
        
        return movie_popularity
    
    def compute_movie_content_features(self):
        """
        Compute content-based features from movie metadata.
        Assumes movies table has: movie_id, title, genres, year, etc.
        
        Returns:
            DataFrame with movie content features
        """
        # Read movie metadata
        movies_df = self.spark.read.format("delta").load(
            f"{self.delta_path}/movies"
        )
        
        # Extract genre features (one-hot encoding)
        # Assuming genres are pipe-separated: "Action|Adventure|Sci-Fi"
        genre_list = movies_df \
            .select("movie_id", explode(split(col("genres"), "\\|")).alias("genre")) \
            .groupBy("movie_id") \
            .agg(collect_list("genre").alias("genres"))
        
        # Join with original movies data
        movie_content = movies_df \
            .join(genre_list, on="movie_id", how="left") \
            .select(
                "movie_id",
                "title",
                "genres",
                col("year").alias("release_year"),
                # Add more metadata fields as needed
            )
        
        return movie_content
    
    def compute_movie_temporal_features(self):
        """
        Compute temporal features (trending, recency, etc.).
        
        Returns:
            DataFrame with temporal features
        """
        events_df = self.spark.read.format("delta").load(
            f"{self.delta_path}/raw_events"
        )
        
        # Recent activity (last 7 days, 30 days)
        from pyspark.sql.functions import datediff, current_date, to_date
        
        events_with_date = events_df \
            .withColumn("event_date", to_date(col("event_timestamp")))
        
        # Activity in last 7 days
        recent_7d = events_with_date \
            .filter(datediff(current_date(), col("event_date")) <= 7) \
            .groupBy("movie_id") \
            .agg(count("*").alias("events_7d"))
        
        # Activity in last 30 days
        recent_30d = events_with_date \
            .filter(datediff(current_date(), col("event_date")) <= 30) \
            .groupBy("movie_id") \
            .agg(count("*").alias("events_30d"))
        
        # Combine temporal features
        temporal_features = recent_7d \
            .join(recent_30d, on="movie_id", how="outer") \
            .fillna(0)
        
        return temporal_features
    
    def write_to_feature_store(self, features_df, table_name: str = "movie_features"):
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
                primary_keys=["movie_id"],
                df=features_df,
                description="Movie features for recommendation system"
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
        """Refresh all movie features and write to feature store."""
        logger.info("Refreshing movie features...")
        
        # Compute popularity features
        popularity_features = self.compute_movie_popularity_features()
        
        # Compute content features
        content_features = self.compute_movie_content_features()
        
        # Compute temporal features
        temporal_features = self.compute_movie_temporal_features()
        
        # Combine all features
        movie_features = popularity_features \
            .join(content_features, on="movie_id", how="outer") \
            .join(temporal_features, on="movie_id", how="outer")
        
        # Write to feature store
        self.write_to_feature_store(movie_features, "movie_features")
        
        logger.info("Movie features refreshed successfully")


# Example usage
if __name__ == "__main__":
    movie_fs = MovieFeatureStore()
    movie_fs.refresh_features()

