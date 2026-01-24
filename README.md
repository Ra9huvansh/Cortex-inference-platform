# Cortex Inference Platform

The **Cortex Inference Platform** is a robust, production-ready text generation API service built with Python and FastAPI. It provides a scalable architecture for serving Large Language Models (LLMs) like GPT-2, complete with user authentication, a credit-based billing system, and comprehensive observability via Prometheus metrics.

## 🚀 Features

*   **Text Generation**: Serve open-source models (GPT-2, DistilGPT-2) for text completion tasks.
*   **User Management**: Built-in user registration and API key management.
*   **Credit System**: Token-based billing system where users spend credits per generated token.
*   **Observability**: Integrated Prometheus metrics (`/metrics`) for monitoring latency, request outcomes, and queue depth.
*   **Resilience**: Request queuing, backpressure handling, and rate limiting to protect the service under load.
*   **Safety**: Input validation, context window checks, and error handling.
*   **Docker Ready**: containerized for easy deployment.

## 🛠️ Tech Stack

*   **Language**: Python 3.10+
*   **Framework**: FastAPI / Uvicorn
*   **ML Engine**: PyTorch & Hugging Face Transformers
*   **Database**: SQLite (WAL mode for concurrency)
*   **Monitoring**: Prometheus Client

## 📋 Prerequisites

*   Docker (recommended)
*   OR Python 3.10+ and `pip`

## ⚡ Getting Started

### Option 1: Docker (Recommended)

1.  **Build the container:**
    ```bash
    docker build -t model-service .
    ```

2.  **Run the service:**
    ```bash
    docker run -p 8000:8000 \
      -v $(pwd)/usage.db:/app/usage.db \
      model-service
    ```
    This mounts `usage.db` to persist user data and credits.

### Option 2: Local Development

1.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Start the server:**
    ```bash
    ./start.sh
    # OR directly:
    uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
    ```

The server will start at `http://localhost:8000`.

## 📖 Usage Guide

For detailed API documentation, see [API.md](API.md).
For operational guides and debugging, see [OPERATOR.md](OPERATOR.md).

### 1. Register a New User
You need an API key to use the service. Register to get one (and receive 20 free credits).

```bash
curl -X POST "http://localhost:8000/register?username=myuser"
```

**Response:**
```json
{
  "username": "myuser",
  "api_key": "YOUR_GENERATED_API_KEY",
  "credits": 20.0
}
```

### 2. Generate Text
Use your API key to generate text.

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_GENERATED_API_KEY" \
  -d '{
    "model": "gpt2",
    "prompt": "The future of AI is",
    "max_tokens": 50
  }'
```

### 3. Check Balance
See your remaining credits.

```bash
curl http://localhost:8000/me \
  -H "x-api-key: YOUR_GENERATED_API_KEY"
```

### 4. Health & Metrics
*   **Health Check**: `GET /health` - Checks if the service is running.
*   **Readiness Check**: `GET /ready` - Checks if the service is ready to accept traffic (DB loaded, models ready).
*   **Metrics**: `GET /metrics` - Prometheus metrics for scraping.

## ⚙️ Configuration

The service can be configured via environment variables:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `ADMIN_API_KEY` | Key for admin-only operations (like sensitive debugging) | `change-me-in-production-please` |

Internal configuration (pricing, rate limits) can be found in `app/server.py`.

## 🧪 Testing

To run a simple load test (burst of 50 requests), use the provided script:

```bash
./burst.sh
```
*Note: You may need to update the API key in `burst.sh` to a valid one you registered.*

## 📂 Project Structure

```
.
├── app/
│   ├── server.py       # Main application logic
│   └── static/         # Static assets
├── burst.sh            # Load testing script
├── start.sh            # Startup script
├── Dockerfile          # Docker build instructions
├── requirements.txt    # Python dependencies
├── API.md              # API documentation
└── OPERATOR.md         # Operational runbook
```
