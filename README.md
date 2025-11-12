# 🎬 RagFlix MLOps Platform

A Netflix-style movie recommendation platform with integrated Big Data processing, AI/ML recommendation pipelines, conversational AI via RAG and MCP, and a responsive Netflix-clone UI.

## 🏗️ Architecture

```
          ┌──────────────────────────────┐
          │  Netflix-style Web UI        │
          │  (React + Tailwind + RAG Chat)│
          └──────────┬───────────────────┘
                     │ REST / WebSocket
                     ▼
          ┌───────────────────────────┐
          │ FastAPI Gateway / API     │
          │ (auth, search, recs, chat)│
          └──────────┬────────────────┘
                     │
                     ▼
       ┌──────────────────────────────┐
       │  MCP Orchestrator            │
       │  (manages LLM + Databricks + RAG)│
       └──────────┬───────────────────┘
                  │
      ┌───────────┴─────────────┐
      ▼                         ▼
 ┌───────────────┐        ┌────────────────────┐
 │ Databricks ML │        │  RAG Chat Engine   │
 │  (PySpark + MLflow)│   │ LangChain + Vector DB│
 └───────────────┘        └────────────────────┘
          │
          ▼
 ┌────────────────────────────┐
 │ Kafka Stream + Delta Lake │
 │ (user events → training)  │
 └────────────────────────────┘
```

## 🧩 Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI Layer** | React + Vite + Tailwind CSS + Framer Motion |
| **API Gateway** | FastAPI + Uvicorn |
| **MCP Layer** | Model Context Protocol Python SDK |
| **Data Ingestion** | Databricks + PySpark + Kafka Connect |
| **Feature Engineering** | PySpark + Databricks Feature Store |
| **Model Training** | MLflow + PyTorch + LightGBM |
| **Model Registry & Serving** | MLflow Registry + FastAPI |
| **Chatbot / RAG** | LangChain + OpenAI/Llama + Pinecone |
| **MLOps Automation** | Airflow + GitHub Actions + Evidently AI |

## 📦 Project Structure

```
ragflix-mlops-platform/
├── data_pipeline/
│   ├── kafka_producer.py
│   ├── spark_stream_ingest.py
│   └── airflow_dag.py
├── feature_store/
│   ├── user_features.py
│   └── movie_features.py
├── model_pipeline/
│   ├── als_training.ipynb
│   ├── hybrid_reranker.py
│   └── mlflow_tracking.py
├── serving_api/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
├── rag_agent/
│   ├── retriever.py
│   ├── mcp_connector.py
│   └── chatbot_agent.py
├── ui/
│   ├── src/
│   │   ├── pages/
│   │   ├── components/
│   │   └── assets/
│   ├── package.json
│   └── vite.config.js
├── monitoring/
│   ├── drift_detection.py
│   └── mlflow_metrics_dashboard.ipynb
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- Databricks account (or local Spark)
- Kafka (or Docker for local setup)
- PostgreSQL/Vector DB for RAG

### Installation

1. **Clone and setup Python environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Setup Databricks:**
```bash
# Configure Databricks credentials
export DATABRICKS_HOST="your-databricks-host"
export DATABRICKS_TOKEN="your-token"
```

3. **Setup UI:**
```bash
cd ui
npm install
npm run dev
```

4. **Run API server:**
```bash
cd serving_api
uvicorn app:app --reload
```

### Run commands (Windows — exact commands)

If you tried `npm start` from the project root and saw:
```
npm ERR! enoent Could not read package.json: Error: ENOENT: no such file or directory, open 'C:\...final-year-project\package.json'
```
that means you ran npm in the wrong folder. The frontend's package.json lives in the frontend directory. Use these exact commands:

1. Frontend (dev server)
```powershell
cd "C:\Users\bhaga\OneDrive\Documents\workspace\Project\final-year-project\frontend"
npm install
npm start
# Open http://localhost:3000
```

2. Backend (optional, start before frontend if the UI proxies /api to it)
```powershell
cd "C:\Users\bhaga\OneDrive\Documents\workspace\Project\final-year-project\serving_api"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
# Backend will be available at http://localhost:8000
```

3. Run both (two terminals)
- Terminal A: start backend (step 2)
- Terminal B: start frontend (step 1)

### Run commands (macOS / Linux)
1. Frontend:
```bash
cd ~/path/to/final-year-project/frontend
npm install
npm start
# http://localhost:3000
```
2. Backend:
```bash
cd ~/path/to/final-year-project/serving_api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```

### Quick troubleshooting
- If you see ECONNREFUSED for `/favicon.ico` or other static files, ensure `public/favicon.svg` exists and you did not set a global `"proxy"` in frontend/package.json. Prefer `src/setupProxy.js` to proxy only `/api` routes to the backend.
- If `Cannot find module './App'` or similar, ensure `frontend/src/App.tsx` and `frontend/src/index.tsx` exist.
- If node/npm complains, remove node_modules and run `npm install` again.

## 📊 Data Flow

1. **Data Ingestion:** MovieLens + Kafka events → Databricks Delta tables
2. **Feature Engineering:** PySpark jobs aggregate watch history + metadata
3. **Model Training:** MLflow runs ALS + embedding models
4. **Model Registry:** Best model promoted to "Production"
5. **Model Serving:** FastAPI loads registered model → predicts top-N movies
6. **Chatbot Query:** MCP fetches user context → RAG retrieval → LLM response
7. **Feedback Loop:** User clicks → Kafka stream → retraining trigger

## 🎨 Features

- **Netflix-style UI:** Trending, Recommended, Continue Watching sections
- **Semantic Search:** RAG-powered movie search
- **AI Chatbot:** Conversational movie recommendations
- **Real-time Recommendations:** Kafka-powered event streaming
- **MLOps Pipeline:** Automated training, deployment, and monitoring

## 📝 License

MIT

