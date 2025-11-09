"""
RAG-powered chatbot agent for conversational movie recommendations.
Integrates LLM with recommendation models and vector search.
"""

from typing import Dict, Any, Optional, List
from rag_agent.retriever import RAGRetriever
from rag_agent.mcp_connector import MCPConnector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatbotAgent:
    """Conversational AI agent for movie recommendations."""
    
    def __init__(
        self,
        llm_provider: str = "openai",  # or "anthropic", "local"
        model_name: str = "gpt-3.5-turbo",
        use_rag: bool = True
    ):
        """
        Initialize chatbot agent.
        
        Args:
            llm_provider: LLM provider name
            model_name: Model name to use
            use_rag: Whether to use RAG retrieval
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.use_rag = use_rag
        
        # Initialize components
        self.retriever = RAGRetriever() if use_rag else None
        self.mcp_connector = MCPConnector()
        
        # Initialize LLM
        self._init_llm()
        
        logger.info(f"ChatbotAgent initialized with {llm_provider}")
    
    def _init_llm(self):
        """Initialize LLM client."""
        if self.llm_provider == "openai":
            try:
                from openai import OpenAI
                self.llm_client = OpenAI()
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI not installed, using placeholder")
                self.llm_client = None
        elif self.llm_provider == "anthropic":
            try:
                from anthropic import Anthropic
                self.llm_client = Anthropic()
                logger.info("Anthropic client initialized")
            except ImportError:
                logger.warning("Anthropic not installed, using placeholder")
                self.llm_client = None
        else:
            # Local model or placeholder
            self.llm_client = None
            logger.info("Using placeholder LLM")
    
    def _generate_llm_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate response from LLM.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            
        Returns:
            LLM response text
        """
        if not self.llm_client:
            # Placeholder response
            return "I understand you're looking for movie recommendations. Based on your preferences, I'd suggest checking out our recommended movies section."
        
        system_prompt = system_prompt or """You are a helpful movie recommendation assistant for RagFlix.
        Provide friendly, personalized movie recommendations based on user preferences and context.
        Explain why you're recommending specific movies. Be conversational and engaging."""
        
        if self.llm_provider == "openai":
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        
        elif self.llm_provider == "anthropic":
            response = self.llm_client.messages.create(
                model=self.model_name,
                max_tokens=500,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        return "LLM response placeholder"
    
    def _extract_intent(self, message: str) -> Dict[str, Any]:
        """
        Extract user intent from message.
        
        Args:
            message: User message
            
        Returns:
            Intent dictionary
        """
        message_lower = message.lower()
        
        intent = {
            "type": "general",
            "genres": [],
            "keywords": []
        }
        
        # Simple keyword-based intent extraction
        if any(word in message_lower for word in ["recommend", "suggest", "what should"]):
            intent["type"] = "recommendation"
        
        if any(word in message_lower for word in ["search", "find", "looking for"]):
            intent["type"] = "search"
        
        # Extract genres
        genres = ["action", "comedy", "drama", "horror", "sci-fi", "romance", "thriller"]
        for genre in genres:
            if genre in message_lower:
                intent["genres"].append(genre)
        
        return intent
    
    def chat(
        self,
        user_id: int,
        message: str,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process chat message and generate response.
        
        Args:
            user_id: User identifier
            message: User message
            conversation_id: Optional conversation ID for context
            
        Returns:
            Response dictionary with text and recommendations
        """
        # Extract intent
        intent = self._extract_intent(message)
        
        # Get user context via MCP
        user_context = self.mcp_connector.get_user_context(user_id)
        
        # Retrieve relevant movies if using RAG
        relevant_movies = []
        if self.use_rag and self.retriever:
            if intent["type"] == "search":
                # Semantic search
                search_results = self.retriever.search(message, limit=5)
                relevant_movies = search_results
            else:
                # General retrieval based on user preferences
                query = f"{message} {', '.join(user_context.get('preferred_genres', []))}"
                search_results = self.retriever.search(query, limit=5)
                relevant_movies = search_results
        
        # Get movie context
        movie_ids = [m.get("movie_id") for m in relevant_movies if m.get("movie_id")]
        movie_context = self.mcp_connector.get_movie_context(movie_ids)
        
        # Format context for LLM
        context = self.mcp_connector.format_context_for_llm(
            user_context,
            relevant_movies,
            movie_context
        )
        
        # Build prompt
        prompt = f"""User message: {message}

Context:
{context}

Please provide a helpful response with movie recommendations."""
        
        # Generate LLM response
        response_text = self._generate_llm_response(prompt)
        
        return {
            "response": response_text,
            "recommendations": relevant_movies[:5],
            "intent": intent,
            "conversation_id": conversation_id
        }


# Example usage
if __name__ == "__main__":
    agent = ChatbotAgent()
    
    response = agent.chat(
        user_id=1,
        message="I'm looking for sci-fi movies with space battles"
    )
    
    print(f"Response: {response['response']}")
    print(f"Recommendations: {len(response['recommendations'])} movies")

