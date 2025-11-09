#!/bin/bash

# RagFlix MLOps Platform Setup Script

echo "🎬 Setting up RagFlix MLOps Platform..."

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Setup UI
echo "Setting up React UI..."
cd ui
npm install
cd ..

# Create necessary directories
echo "Creating directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p logs
mkdir -p mlruns

# Set up environment variables template
if [ ! -f .env ]; then
    echo "Creating .env template..."
    cat > .env << EOF
# Databricks Configuration
DATABRICKS_HOST=your-databricks-host
DATABRICKS_TOKEN=your-databricks-token
DATABRICKS_CLUSTER_ID=your-cluster-id

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=user-events

# MLflow Configuration
MLFLOW_TRACKING_URI=databricks
MLFLOW_EXPERIMENT_NAME=ragflix-recommendations

# OpenAI/LLM Configuration (optional)
OPENAI_API_KEY=your-openai-key

# Vector Database (Pinecone)
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=your-pinecone-env

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
EOF
    echo ".env file created. Please update with your credentials."
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Update .env file with your credentials"
echo "2. Start Kafka (if running locally)"
echo "3. Configure Databricks connection"
echo "4. Run: python -m data_pipeline.kafka_producer (to test Kafka)"
echo "5. Run: uvicorn serving_api.app:app --reload (to start API)"
echo "6. Run: cd ui && npm run dev (to start UI)"

