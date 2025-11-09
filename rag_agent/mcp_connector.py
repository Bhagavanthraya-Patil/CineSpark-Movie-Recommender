"""
MCP (Model Context Protocol) connector for integrating Databricks models
with LLM-based chatbot responses.
"""

from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPConnector:
    """Connects recommendation models with LLM context."""
    
    def __init__(
        self,
        databricks_host: Optional[str] = None,
        databricks_token: Optional[str] = None
    ):
        """
        Initialize MCP connector.
        
        Args:
            databricks_host: Databricks workspace host
            databricks_token: Databricks API token
        """
        self.databricks_host = databricks_host
        self.databricks_token = databricks_token
        
        # Initialize Databricks client if credentials provided
        if databricks_host and databricks_token:
            try:
                from databricks import sql
                self.connection = sql.connect(
                    server_hostname=databricks_host,
                    http_path="/sql/1.0/warehouses/your-warehouse-id",
                    access_token=databricks_token
                )
                logger.info("Databricks connection established")
            except Exception as e:
                logger.warning(f"Could not connect to Databricks: {e}")
                self.connection = None
        else:
            self.connection = None
            logger.info("MCP connector initialized without Databricks connection")
    
    def get_user_context(self, user_id: int) -> Dict[str, Any]:
        """
        Retrieve user context from Databricks Feature Store.
        
        Args:
            user_id: User identifier
            
        Returns:
            User context dictionary
        """
        if not self.connection:
            # Return placeholder context
            return {
                "user_id": user_id,
                "preferred_genres": [],
                "watch_history": [],
                "average_rating": 0.0
            }
        
        try:
            cursor = self.connection.cursor()
            
            # Query user features
            query = f"""
            SELECT * FROM ragflix_feature_store.user_features
            WHERE user_id = {user_id}
            """
            
            cursor.execute(query)
            result = cursor.fetchone()
            
            if result:
                return {
                    "user_id": user_id,
                    "total_events": result.get("total_events", 0),
                    "avg_rating": result.get("avg_rating", 0.0),
                    "preferred_genres": result.get("genres", []),
                    "watch_history": result.get("watch_history", [])
                }
            else:
                return {"user_id": user_id}
        
        except Exception as e:
            logger.error(f"Error fetching user context: {e}")
            return {"user_id": user_id}
    
    def get_movie_context(self, movie_ids: List[int]) -> List[Dict[str, Any]]:
        """
        Retrieve movie context from Databricks.
        
        Args:
            movie_ids: List of movie identifiers
            
        Returns:
            List of movie context dictionaries
        """
        if not self.connection or not movie_ids:
            return []
        
        try:
            cursor = self.connection.cursor()
            
            ids_str = ",".join(map(str, movie_ids))
            query = f"""
            SELECT * FROM ragflix_feature_store.movie_features
            WHERE movie_id IN ({ids_str})
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            return [
                {
                    "movie_id": row.get("movie_id"),
                    "title": row.get("title"),
                    "genres": row.get("genres", []),
                    "avg_rating": row.get("avg_rating", 0.0),
                    "description": row.get("description")
                }
                for row in results
            ]
        
        except Exception as e:
            logger.error(f"Error fetching movie context: {e}")
            return []
    
    def format_context_for_llm(
        self,
        user_context: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        movie_context: List[Dict[str, Any]]
    ) -> str:
        """
        Format context for LLM prompt.
        
        Args:
            user_context: User context dictionary
            recommendations: List of recommended movies
            movie_context: List of movie context dictionaries
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        # User context
        context_parts.append(f"User ID: {user_context.get('user_id')}")
        if user_context.get("preferred_genres"):
            context_parts.append(
                f"Preferred genres: {', '.join(user_context['preferred_genres'])}"
            )
        if user_context.get("avg_rating"):
            context_parts.append(
                f"Average rating given: {user_context['avg_rating']:.1f}"
            )
        
        # Recommendations
        if recommendations:
            context_parts.append("\nRecommended movies:")
            for i, rec in enumerate(recommendations[:5], 1):
                movie_id = rec.get("movie_id")
                movie_info = next(
                    (m for m in movie_context if m.get("movie_id") == movie_id),
                    {}
                )
                title = movie_info.get("title", rec.get("title", "Unknown"))
                genres = movie_info.get("genres", rec.get("genres", []))
                context_parts.append(
                    f"{i}. {title} (Genres: {', '.join(genres)})"
                )
        
        return "\n".join(context_parts)
    
    def close(self):
        """Close Databricks connection."""
        if self.connection:
            self.connection.close()
            logger.info("Databricks connection closed")


# Example usage
if __name__ == "__main__":
    connector = MCPConnector()
    
    user_context = connector.get_user_context(user_id=1)
    print(f"User context: {user_context}")

