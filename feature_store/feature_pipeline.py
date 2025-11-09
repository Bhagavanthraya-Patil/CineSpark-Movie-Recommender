from databricks.feature_store import FeatureStoreClient
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

def create_user_features(spark: SparkSession):
    events_df = spark.table("events")
    
    user_features = events_df.groupBy("user_id").agg(
        F.count("*").alias("total_events"),
        F.avg("rating").alias("avg_rating"),
        F.collect_set("genre").alias("genre_preferences")
    )
    
    fs = FeatureStoreClient()
    
    fs.create_table(
        name="user_features",
        primary_keys=["user_id"],
        df=user_features,
        description="User level features for recommendation"
    )

def create_movie_features(spark: SparkSession):
    movies_df = spark.table("movies")
    
    movie_features = movies_df.groupBy("movie_id").agg(
        F.avg("rating").alias("avg_rating"),
        F.count("*").alias("popularity")
    )
    
    fs = FeatureStoreClient()
    
    fs.create_table(
        name="movie_features",
        primary_keys=["movie_id"],
        df=movie_features,
        description="Movie level features for recommendation"
    )

def main():
    spark = SparkSession.builder.getOrCreate()
    create_user_features(spark)
    create_movie_features(spark)

if __name__ == "__main__":
    main()
