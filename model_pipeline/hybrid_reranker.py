"""
Hybrid recommendation reranker combining ALS collaborative filtering
with content-based features and sentence embeddings.
"""

import numpy as np
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, explode
from sentence_transformers import SentenceTransformer
import mlflow
import mlflow.pyfunc
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridReranker(mlflow.pyfunc.PythonModel):
    """Hybrid recommendation model combining ALS + content-based + embeddings."""
    
    def __init__(
        self,
        als_model: ALSModel,
        sentence_model_name: str = "all-MiniLM-L6-v2",
        content_weight: float = 0.3,
        embedding_weight: float = 0.2,
        collaborative_weight: float = 0.5
    ):
        """
        Initialize hybrid reranker.
        
        Args:
            als_model: Trained ALS collaborative filtering model
            sentence_model_name: Sentence transformer model name
            content_weight: Weight for content-based features
            embedding_weight: Weight for semantic embeddings
            collaborative_weight: Weight for collaborative filtering
        """
        self.als_model = als_model
        self.sentence_model = SentenceTransformer(sentence_model_name)
        self.content_weight = content_weight
        self.embedding_weight = embedding_weight
        self.collaborative_weight = collaborative_weight
        
        logger.info("HybridReranker initialized")
    
    def get_als_predictions(self, user_id: int, movie_ids: List[int], spark):
        """Get predictions from ALS model."""
        from pyspark.sql import Row
        
        # Create user-movie pairs
        user_movie_pairs = spark.createDataFrame([
            Row(user_id=user_id, movie_id=movie_id)
            for movie_id in movie_ids
        ])
        
        # Get predictions
        predictions = self.als_model.transform(user_movie_pairs)
        predictions_dict = {
            row.movie_id: row.prediction
            for row in predictions.collect()
        }
        
        return predictions_dict
    
    def get_content_similarity(
        self,
        user_genres: List[str],
        movie_genres: List[str]
    ) -> float:
        """Compute content-based similarity from genres."""
        if not user_genres or not movie_genres:
            return 0.0
        
        # Jaccard similarity
        user_set = set(user_genres)
        movie_set = set(movie_genres)
        intersection = user_set & movie_set
        union = user_set | movie_set
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def get_semantic_similarity(
        self,
        user_preferences: str,
        movie_description: str
    ) -> float:
        """Compute semantic similarity using sentence embeddings."""
        if not user_preferences or not movie_description:
            return 0.0
        
        # Encode texts
        user_embedding = self.sentence_model.encode(user_preferences)
        movie_embedding = self.sentence_model.encode(movie_description)
        
        # Cosine similarity
        similarity = np.dot(user_embedding, movie_embedding) / (
            np.linalg.norm(user_embedding) * np.linalg.norm(movie_embedding)
        )
        
        return float(similarity)
    
    def rerank(
        self,
        user_id: int,
        candidate_movies: List[Dict[str, Any]],
        spark,
        user_genres: List[str] = None,
        user_preferences: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate movies using hybrid approach.
        
        Args:
            user_id: User identifier
            candidate_movies: List of candidate movies with metadata
            spark: SparkSession
            user_genres: User's preferred genres
            user_preferences: Text description of user preferences
            
        Returns:
            Reranked list of movies with scores
        """
        movie_ids = [m["movie_id"] for m in candidate_movies]
        
        # Get ALS predictions
        als_scores = self.get_als_predictions(user_id, movie_ids, spark)
        
        # Compute hybrid scores
        reranked = []
        for movie in candidate_movies:
            movie_id = movie["movie_id"]
            
            # Normalize ALS score (0-1)
            als_score = als_scores.get(movie_id, 0.0)
            als_normalized = max(0.0, min(1.0, (als_score + 1) / 2))  # Assuming ratings 1-5
            
            # Content similarity
            movie_genres = movie.get("genres", [])
            content_score = self.get_content_similarity(
                user_genres or [],
                movie_genres
            )
            
            # Semantic similarity
            movie_description = movie.get("description", "")
            semantic_score = self.get_semantic_similarity(
                user_preferences,
                movie_description
            )
            
            # Weighted combination
            hybrid_score = (
                self.collaborative_weight * als_normalized +
                self.content_weight * content_score +
                self.embedding_weight * semantic_score
            )
            
            reranked.append({
                **movie,
                "hybrid_score": hybrid_score,
                "als_score": als_score,
                "content_score": content_score,
                "semantic_score": semantic_score
            })
        
        # Sort by hybrid score
        reranked.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        return reranked
    
    def predict(self, context, model_input):
        """
        MLflow PyFunc interface for predictions.
        
        Args:
            context: MLflow context
            model_input: Input DataFrame with user_id and movie_ids
        """
        # This would need to be implemented based on MLflow PyFunc requirements
        # For now, return placeholder
        return model_input


# Example usage
if __name__ == "__main__":
    from model_pipeline.mlflow_tracking import MLflowTracker
    
    # This would be called after training ALS model
    # tracker = MLflowTracker()
    # reranker = HybridReranker(als_model=trained_als_model)
    # tracker.log_model(reranker, "hybrid_reranker")

