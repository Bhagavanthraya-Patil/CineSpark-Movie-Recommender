import pytest
from model_training.train_model import MovieRecommender

def test_movie_recommender_init():
    params = {"maxIter": 10, "regParam": 0.1, "rank": 50}
    model = MovieRecommender(params)
    assert model.model_params == params

def test_model_prediction_shape():
    # Add test for prediction shape
    pass

def test_model_feature_importance():
    # Add test for feature importance
    pass
