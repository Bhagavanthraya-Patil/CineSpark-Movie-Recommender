"""
PySpark streaming job to ingest Kafka events into Databricks Delta Lake.
Processes user events in real-time and writes to Delta tables.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, window, count, avg, max as spark_max,
    current_timestamp, lit, struct
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    DoubleType, TimestampType, MapType
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SparkStreamIngest:
    """Streams Kafka events to Delta Lake using PySpark."""
    
    def __init__(
        self,
        kafka_bootstrap_servers: str = "localhost:9092",
        kafka_topic: str = "user-events",
        delta_path: str = "/dbfs/mnt/ragflix/events",
        checkpoint_path: str = "/dbfs/mnt/ragflix/checkpoints"
    ):
        """
        Initialize Spark streaming session.
        
        Args:
            kafka_bootstrap_servers: Kafka broker addresses
            kafka_topic: Kafka topic to read from
            delta_path: Path for Delta Lake storage
            checkpoint_path: Path for Spark checkpoint
        """
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.kafka_topic = kafka_topic
        self.delta_path = delta_path
        self.checkpoint_path = checkpoint_path
        
        # Initialize Spark session
        self.spark = SparkSession.builder \
            .appName("RagFlixStreamIngest") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()
        
        logger.info("Spark session initialized")
    
    def get_event_schema(self) -> StructType:
        """Define schema for user events."""
        return StructType([
            StructField("user_id", IntegerType(), True),
            StructField("movie_id", IntegerType(), True),
            StructField("event_type", StringType(), True),
            StructField("timestamp", StringType(), True),
            StructField("metadata", MapType(StringType(), StringType()), True)
        ])
    
    def read_kafka_stream(self):
        """Read stream from Kafka."""
        return self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_bootstrap_servers) \
            .option("subscribe", self.kafka_topic) \
            .option("startingOffsets", "latest") \
            .load()
    
    def parse_events(self, kafka_df):
        """Parse JSON events from Kafka."""
        schema = self.get_event_schema()
        
        return kafka_df \
            .select(
                col("key").cast("string").alias("kafka_key"),
                from_json(col("value").cast("string"), schema).alias("data"),
                col("timestamp").alias("kafka_timestamp")
            ) \
            .select(
                "kafka_key",
                "kafka_timestamp",
                "data.user_id",
                "data.movie_id",
                "data.event_type",
                col("data.timestamp").alias("event_timestamp"),
                "data.metadata"
            ) \
            .withColumn("processed_at", current_timestamp())
    
    def write_to_delta(self, df, table_name: str, mode: str = "append"):
        """
        Write streaming DataFrame to Delta Lake.
        
        Args:
            df: Streaming DataFrame
            table_name: Name of the Delta table
            mode: Write mode (append, overwrite, etc.)
        """
        delta_table_path = f"{self.delta_path}/{table_name}"
        
        return df.writeStream \
            .format("delta") \
            .outputMode(mode) \
            .option("checkpointLocation", f"{self.checkpoint_path}/{table_name}") \
            .option("path", delta_table_path) \
            .trigger(processingTime='10 seconds') \
            .start()
    
    def aggregate_events(self, events_df):
        """Aggregate events by user and movie for analytics."""
        return events_df \
            .withWatermark("event_timestamp", "1 hour") \
            .groupBy(
                window(col("event_timestamp"), "1 hour"),
                col("user_id"),
                col("movie_id"),
                col("event_type")
            ) \
            .agg(
                count("*").alias("event_count"),
                spark_max("event_timestamp").alias("last_event_time")
            )
    
    def run_streaming_job(self):
        """Main streaming job execution."""
        logger.info("Starting Kafka to Delta Lake streaming job")
        
        # Read from Kafka
        kafka_df = self.read_kafka_stream()
        
        # Parse events
        events_df = self.parse_events(kafka_df)
        
        # Write raw events to Delta
        raw_stream = self.write_to_delta(events_df, "raw_events")
        
        # Aggregate events
        aggregated_df = self.aggregate_events(events_df)
        
        # Write aggregated events to Delta
        agg_stream = self.write_to_delta(aggregated_df, "aggregated_events")
        
        logger.info("Streaming jobs started")
        
        # Wait for termination
        raw_stream.awaitTermination()
        agg_stream.awaitTermination()


# Example usage
if __name__ == "__main__":
    ingest = SparkStreamIngest()
    ingest.run_streaming_job()

