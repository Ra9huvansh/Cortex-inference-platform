import os
import time
import torch
import sqlite3
import secrets

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse, FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from prometheus_client import Counter, Histogram, generate_latest

# -------------------------------------------------
# Device setup
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

CREDITS_PER_1000_TOKENS = 0.02   # ₹0.02 / $0.0002 per 1k tokens

# -------------------------------------------------
# Model registry (model name -> HF id)
# -------------------------------------------------
MODEL_REGISTRY = {
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2",
    # "phi-2": "microsoft/phi-2"
}

DEFAULT_MODEL = "gpt2"
loaded_models = {}


def get_model(model_name: str):
    """Load and cache a model by name."""
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not supported")

    if model_name not in loaded_models:
        print(f"[MODEL LOAD] Loading {model_name} on {device}...")
        tok = AutoTokenizer.from_pretrained(MODEL_REGISTRY[model_name])
        mdl = AutoModelForCausalLM.from_pretrained(MODEL_REGISTRY[model_name]).to(device)
        loaded_models[model_name] = (tok, mdl)

    return loaded_models[model_name]


# -------------------------------------------------
# SQLite setup
# -------------------------------------------------
conn = sqlite3.connect("usage.db", check_same_thread=False)

# logs table
conn.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT,
    model TEXT,
    prompt TEXT,
    completion TEXT,
    latency REAL,
    tokens INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

# users table
conn.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    api_key TEXT UNIQUE
)
""")

conn.commit()


def log_request(user, model, prompt, completion, latency, tokens):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO logs (user, model, prompt, completion, latency, tokens) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user, model, prompt, completion, latency, tokens),
    )
    conn.commit()
    cur.close()


# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")


class GenerateRequest(BaseModel):
    model: str = DEFAULT_MODEL
    prompt: str
    max_tokens: int = 60
    temperature: float = 0.7
    top_p: float = 0.9


def create_api_key():
    return secrets.token_hex(16)  # 32-char random key


async def verify_api_key(x_api_key: str = Header(None)):
    """Return username if API key is valid."""
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

    cur = conn.cursor()
    cur.execute("SELECT username FROM users WHERE api_key=?", (x_api_key,))
    row = cur.fetchone()
    cur.close()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

    return row[0]  # username


# -------------------------------------------------
# Prometheus metrics
# -------------------------------------------------
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Inference Requests",
    ["model"],
)
REQUEST_LATENCY = Histogram(
    "inference_latency_seconds",
    "Latency per request",
    ["model"],
)


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.post("/register")
def register(username: str):
    """Create a user + API key."""
    api_key = create_api_key()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (username, api_key, credits) VALUES (?, ?, ?)",
        (username, api_key, 20.0),  # 20 initial free-credits
    )
    conn.commit()
    cur.close()
    print(f"[REGISTER] user={username} key={api_key}")
    return {"username": username, "api_key": api_key, "credits": 20.0}


@app.get("/")
def root():
    return {"status": "ok", "models": list(MODEL_REGISTRY.keys())}


@app.get("/portal")
def portal():
    """Serve landing page (optional shortcut)."""
    return FileResponse("app/static/index.html")


@app.get("/models")
def list_models():
    return {"models": list(MODEL_REGISTRY.keys())}


@app.get("/me")
def me(user=Depends(verify_api_key)):
    """Check current user from API key."""
    return {"user": user}


@app.post("/generate")
async def generate(req: GenerateRequest, user=Depends(verify_api_key)):
    cur = conn.cursor()
    cur.execute("SELECT credits FROM users WHERE username=?", (user,))
    bal=cur.fetchone()[0]
    cur.close()

    if bal <= 0 :
        raise HTTPException(status_code=402, detail="Not enough credits")

    tokenizer, model = get_model(req.model)

    start = time.time()
    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
        )

    total_tokens = len(outputs[0])
    credits_used = (total_tokens / 1000) * CREDITS_PER_1000_TOKENS

    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET credits = credits - ? WHERE username=?",
        (credits_used, user),
    )
    conn.commit()
    cur.close()

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elapsed = time.time() - start

    REQUEST_COUNT.labels(req.model).inc()
    REQUEST_LATENCY.labels(req.model).observe(elapsed)

    log_request(user, req.model, req.prompt, text, elapsed, total_tokens)

    return {
        "completion": text,
        "latency_sec": elapsed,
        "tokens": total_tokens,
        "model": req.model,
        "user": user,
        "remaining_credits": bal - credits_used,
    }


@app.get("/dashboard")
def dashboard():
    """Global logs (admin view, JSON)"""
    cur = conn.cursor()
    cur.execute(
        "SELECT id, user, prompt, completion, latency, tokens, timestamp "
        "FROM logs ORDER BY timestamp DESC LIMIT 20"
    )
    rows = cur.fetchall()
    cur.close()
    return {"logs": rows}


@app.get("/usage/stats")
def usage_stats():
    """High-level stats."""
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM logs")
    total_requests = cur.fetchone()[0]

    cur.execute("SELECT COUNT(DISTINCT user) FROM logs")
    unique_users = cur.fetchone()[0]

    cur.execute("SELECT AVG(latency) FROM logs")
    avg_latency = cur.fetchone()[0]

    cur.execute("SELECT model, COUNT(*) FROM logs GROUP BY model")
    model_counts = dict(cur.fetchall())

    cur.execute("SELECT SUM(tokens) FROM logs")
    total_tokens = cur.fetchone()[0]

    cur.close()
    return {
        "total_requests": total_requests,
        "unique_users": unique_users,
        "avg_latency_sec": avg_latency,
        "model_usage": model_counts,
        "total_tokens": total_tokens,
    }


@app.get("/metrics")
def metrics():
    """Prometheus scrape endpoint."""
    return PlainTextResponse(generate_latest(), media_type="text/plain")


@app.get("/user/dashboard")
def user_dashboard(user=Depends(verify_api_key)):
    """Logs for *this* user only."""
    cur = conn.cursor()
    cur.execute(
        "SELECT id, prompt, completion, latency, tokens, timestamp "
        "FROM logs WHERE user=? ORDER BY timestamp DESC LIMIT 20",
        (user,),
    )
    rows = cur.fetchall()
    cur.close()
    return {"user": user, "logs": rows}

@app.post("/admin/add_credits")
def add_credits(username: str, amount: float):
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET credits = credits + ? WHERE username=?",
        (amount, username),
    )
    conn.commit()
    cur.close()
    return {"status": "ok", "username": username, "added": amount}


