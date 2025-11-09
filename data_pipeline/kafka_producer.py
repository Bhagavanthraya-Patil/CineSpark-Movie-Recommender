"""
Kafka Producer for streaming user events to Databricks Delta Lake.
Handles movie views, ratings, and user interactions.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserEventProducer:
    """Produces user interaction events to Kafka for real-time processing."""
    
    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "user-events",
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    ):
        """
        Initialize Kafka producer.
        
        Args:
            bootstrap_servers: Kafka broker addresses
            topic: Topic name for user events
            value_serializer: Function to serialize message values
        """
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=value_serializer,
            acks='all',
            retries=3,
            max_in_flight_requests_per_connection=1
        )
        self.topic = topic
        logger.info(f"Kafka producer initialized for topic: {topic}")
    
    def send_event(
        self,
        user_id: int,
        movie_id: int,
        event_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send a user event to Kafka.
        
        Args:
            user_id: User identifier
            movie_id: Movie identifier
            event_type: Type of event (view, rating, click, etc.)
            metadata: Additional event metadata
            
        Returns:
            True if event sent successfully, False otherwise
        """
        event = {
            "user_id": user_id,
            "movie_id": movie_id,
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        try:
            future = self.producer.send(self.topic, value=event)
            record_metadata = future.get(timeout=10)
            logger.info(
                f"Event sent to topic={record_metadata.topic}, "
                f"partition={record_metadata.partition}, "
                f"offset={record_metadata.offset}"
            )
            return True
        except KafkaError as e:
            logger.error(f"Failed to send event: {e}")
            return False
    
    def send_rating(self, user_id: int, movie_id: int, rating: float) -> bool:
        """Send a movie rating event."""
        return self.send_event(
            user_id=user_id,
            movie_id=movie_id,
            event_type="rating",
            metadata={"rating": rating}
        )
    
    def send_view(self, user_id: int, movie_id: int, watch_time: float = 0.0) -> bool:
        """Send a movie view/watch event."""
        return self.send_event(
            user_id=user_id,
            movie_id=movie_id,
            event_type="view",
            metadata={"watch_time": watch_time}
        )
    
    def send_click(self, user_id: int, movie_id: int, click_type: str = "poster") -> bool:
        """Send a click/interaction event."""
        return self.send_event(
            user_id=user_id,
            movie_id=movie_id,
            event_type="click",
            metadata={"click_type": click_type}
        )
    
    def close(self):
        """Close the producer connection."""
        self.producer.close()
        logger.info("Kafka producer closed")


# Example usage
if __name__ == "__main__":
    producer = UserEventProducer()
    
    # Simulate some events
    producer.send_rating(user_id=1, movie_id=100, rating=4.5)
    producer.send_view(user_id=1, movie_id=100, watch_time=120.5)
    producer.send_click(user_id=1, movie_id=200, click_type="poster")
    
    producer.close()

