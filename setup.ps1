# RagFlix MLOps Platform Setup Script (PowerShell)

Write-Host "🎬 Setting up RagFlix MLOps Platform..." -ForegroundColor Green

# Create virtual environment
Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

# Setup UI
Write-Host "Setting up React UI..." -ForegroundColor Yellow
cd ui
npm install
cd ..

# Create necessary directories
Write-Host "Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\raw"
New-Item -ItemType Directory -Force -Path "data\processed"
New-Item -ItemType Directory -Force -Path "logs"
New-Item -ItemType Directory -Force -Path "mlruns"

# Set up environment variables template
if (-not (Test-Path .env)) {
    Write-Host "Creating .env template..." -ForegroundColor Yellow
    @"
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
"@ | Out-File -FilePath .env -Encoding utf8
    Write-Host ".env file created. Please update with your credentials." -ForegroundColor Yellow
}

Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "1. Update .env file with your credentials"
Write-Host "2. Start Kafka (if running locally)"
Write-Host "3. Configure Databricks connection"
Write-Host "4. Run: python -m data_pipeline.kafka_producer (to test Kafka)"
Write-Host "5. Run: uvicorn serving_api.app:app --reload (to start API)"
Write-Host "6. Run: cd ui && npm run dev (to start UI)"

