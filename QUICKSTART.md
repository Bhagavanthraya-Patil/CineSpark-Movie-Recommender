# 🚀 RagFlix Quick Start Guide

## Prerequisites

- Python 3.9+
- Node.js 18+
- Databricks account (or local Spark setup)
- Kafka (optional, for streaming)
- PostgreSQL/Vector DB (for RAG)

## Installation Steps

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd ragflix-mlops-platform

# Run setup script
# On Linux/Mac:
bash setup.sh

# On Windows:
powershell -ExecutionPolicy Bypass -File setup.ps1
```

### 2. Configure Environment

Edit `.env` file with your credentials:

```bash
# Databricks
DATABRICKS_HOST=your-databricks-host.cloud.databricks.com
DATABRICKS_TOKEN=your-token
DATABRICKS_CLUSTER_ID=your-cluster-id

# Kafka (if using)
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# LLM (optional)
OPENAI_API_KEY=your-openai-key

# Vector DB (optional)
PINECONE_API_KEY=your-pinecone-key
```

### 3. Prepare Data

```bash
# Download MovieLens dataset
# Place ratings.csv and movies.csv in data/raw/

# In Databricks, create Delta tables:
# - ratings: user_id, movie_id, rating, timestamp
# - movies: movie_id, title, genres, year
```

### 4. Start Services

#### Terminal 1: API Server
```bash
cd serving_api
uvicorn app:app --reload --port 8000
```

#### Terminal 2: UI
```bash
cd ui
npm run dev
```

#### Terminal 3: Kafka Producer (optional)
```bash
python -m data_pipeline.kafka_producer
```

### 5. Train Model

```bash
# In Databricks notebook or locally:
python -m model_pipeline.als_training

# Or use the Jupyter notebook:
jupyter notebook model_pipeline/als_training.ipynb
```

### 6. Access Application

- **UI**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## Testing

```bash
# Test API endpoints
curl http://localhost:8000/health

# Test recommendations
curl -X POST http://localhost:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "num_recommendations": 10}'

# Test search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "sci-fi movies", "limit": 5}'
```

## Common Issues

### Issue: Databricks connection fails
**Solution**: Verify your token and host in `.env` file

### Issue: Kafka connection error
**Solution**: Ensure Kafka is running: `docker-compose up kafka` or use local Kafka

### Issue: UI can't connect to API
**Solution**: Check API is running on port 8000 and CORS is configured

### Issue: Model not found
**Solution**: Train model first using `als_training.py` or notebook

## Next Steps

1. **Load MovieLens data** into Databricks Delta tables
2. **Train initial model** using the training notebook
3. **Set up Airflow** for automated retraining
4. **Configure vector database** for RAG search
5. **Deploy to production** using Docker/Kubernetes

## Architecture Overview

```
UI (React) → FastAPI → MCP → Databricks/MLflow
                    ↓
                 RAG Agent → Vector DB
                    ↓
                 Kafka → Delta Lake
```

For detailed architecture, see [README.md](README.md)

