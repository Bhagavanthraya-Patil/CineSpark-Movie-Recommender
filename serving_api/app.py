"""
FastAPI serving API for RagFlix recommendation system.
Provides endpoints for recommendations, search, and chat.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import mlflow
from mlflow.tracking import MlflowClient
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RagFlix API",
    description="Movie recommendation API with RAG chatbot",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MLflow client
mlflow_client = MlflowClient()

# Global model cache
model_cache = {}


# Pydantic models
class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User identifier")
    num_recommendations: int = Field(10, ge=1, le=100, description="Number of recommendations")
    include_metadata: bool = Field(True, description="Include movie metadata")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(10, ge=1, le=50, description="Maximum results")


class ChatRequest(BaseModel):
    user_id: int = Field(..., description="User identifier")
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")


class MovieResponse(BaseModel):
    movie_id: int
    title: str
    genres: List[str]
    description: Optional[str] = None
    poster_url: Optional[str] = None
    rating: Optional[float] = None
    score: Optional[float] = None


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieResponse]
    model_version: Optional[str] = None


# Model loading
def load_production_model():
    """Load the production model from MLflow registry."""
    global model_cache
    
    if "production_model" in model_cache:
        return model_cache["production_model"]
    
    try:
        model_name = "ragflix-recommendation-model"
        model_version = mlflow_client.get_latest_versions(
            model_name,
            stages=["Production"]
        )[0]
        
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pyfunc.load_model(model_uri)
        
        model_cache["production_model"] = model
        model_cache["model_version"] = model_version.version
        
        logger.info(f"Loaded production model: version {model_version.version}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ragflix-api"}


# Recommendations endpoint
@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get movie recommendations for a user.
    
    Args:
        request: Recommendation request with user_id and parameters
        
    Returns:
        List of recommended movies
    """
    try:
        model = load_production_model()
        
        # Generate recommendations
        # This is a placeholder - actual implementation depends on model type
        recommendations = model.predict({
            "user_id": request.user_id,
            "num_recommendations": request.num_recommendations
        })
        
        # Format response
        movie_responses = [
            MovieResponse(
                movie_id=rec.get("movie_id"),
                title=rec.get("title", "Unknown"),
                genres=rec.get("genres", []),
                description=rec.get("description"),
                poster_url=rec.get("poster_url"),
                rating=rec.get("rating"),
                score=rec.get("score")
            )
            for rec in recommendations
        ]
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=movie_responses,
            model_version=model_cache.get("model_version")
        )
    
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Search endpoint
@app.post("/api/search")
async def search_movies(request: SearchRequest):
    """
    Search movies using semantic search via RAG.
    
    Args:
        request: Search request with query
        
    Returns:
        List of matching movies
    """
    try:
        # Import RAG retriever
        from rag_agent.retriever import RAGRetriever
        
        retriever = RAGRetriever()
        results = retriever.search(request.query, limit=request.limit)
        
        movie_responses = [
            MovieResponse(
                movie_id=result.get("movie_id"),
                title=result.get("title", "Unknown"),
                genres=result.get("genres", []),
                description=result.get("description"),
                poster_url=result.get("poster_url"),
                rating=result.get("rating"),
                score=result.get("score")
            )
            for result in results
        ]
        
        return {"query": request.query, "results": movie_responses}
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Chat endpoint
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Chat with RAG-powered recommendation agent.
    
    Args:
        request: Chat request with user message
        
    Returns:
        Agent response with recommendations
    """
    try:
        from rag_agent.chatbot_agent import ChatbotAgent
        
        agent = ChatbotAgent()
        response = agent.chat(
            user_id=request.user_id,
            message=request.message,
            conversation_id=request.conversation_id
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get movie details
@app.get("/api/movies/{movie_id}")
async def get_movie(movie_id: int):
    """Get details for a specific movie."""
    # This would query your movie database
    # Placeholder implementation
    return {
        "movie_id": movie_id,
        "title": "Sample Movie",
        "genres": ["Action", "Adventure"],
        "description": "A sample movie description"
    }


# User history
@app.get("/api/users/{user_id}/history")
async def get_user_history(user_id: int, limit: int = 20):
    """Get user's watch history."""
    # This would query your events database
    return {"user_id": user_id, "history": []}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

