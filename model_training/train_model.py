import mlflow
import mlflow.pyfunc
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from typing import Dict, Any

def create_spark_session():
    return SparkSession.builder \
        .appName("MovieRecommenderTraining") \
        .getOrCreate()

class MovieRecommender(mlflow.pyfunc.PythonModel):
    def __init__(self, model_params: Dict[str, Any]):
        self.model_params = model_params
        self.als_model = None

    def fit(self, ratings_df):
        als = ALS(
            userCol="user_id",
            itemCol="movie_id",
            ratingCol="rating",
            **self.model_params
        )
        self.als_model = als.fit(ratings_df)

    def predict(self, context, model_input):
        return self.als_model.transform(model_input)

def main():
    spark = create_spark_session()
    
    # Load training data
    ratings_df = spark.table("ratings")
    
    with mlflow.start_run():
        model_params = {
            "maxIter": 10,
            "regParam": 0.1,
            "rank": 50
        }
        
        recommender = MovieRecommender(model_params)
        recommender.fit(ratings_df)
        
        # Log parameters and model
        mlflow.log_params(model_params)
        mlflow.pyfunc.log_model(
            "movie_recommender",
            python_model=recommender
        )

if __name__ == "__main__":
    main()
