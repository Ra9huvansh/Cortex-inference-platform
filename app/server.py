import os
import time
import torch
import sqlite3
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------------------------
# Device setup
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# Model registry (model name -> HF id)
# You can add more here later.
# -------------------------------------------------
MODEL_REGISTRY = {
    "gpt2": "gpt2",
    "distilgpt2": "distilgpt2",
    "phi-2": "microsoft/phi-2"  # enable later if RAM allows
}

DEFAULT_MODEL = "gpt2"

# Cache for loaded models so we don't reload every time
loaded_models: dict[str, tuple[AutoTokenizer, AutoModelForCausalLM]] = {}


def get_model(model_name: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Return (tokenizer, model) for given model_name, loading and caching if needed."""
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not supported")

    if model_name not in loaded_models:
        hf_id = MODEL_REGISTRY[model_name]
        print(f"[MODEL LOAD] Loading model '{model_name}' from '{hf_id}' on {device}")
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float32,  # CPU-friendly; adjust later for GPU
        ).to(device)
        loaded_models[model_name] = (tokenizer, model)

    return loaded_models[model_name]


# -------------------------------------------------
# SQLite logging setup
# -------------------------------------------------
conn = sqlite3.connect("usage.db", check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user TEXT,
    prompt TEXT,
    completion TEXT,
    latency REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

def log_request(user: str, prompt: str, completion: str, latency: float):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO logs (user, prompt, completion, latency) VALUES (?, ?, ?, ?)",
        (user, prompt, completion, latency),
    )
    conn.commit()
    cur.close()

# -------------------------------------------------
# FastAPI app + static files
# -------------------------------------------------
app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

# -------------------------------------------------
# API key auth
# -------------------------------------------------
API_KEYS = {
    "test-key-123": "test-user",  # key : username / user_id
}


class GenerateRequest(BaseModel):
    model: str = DEFAULT_MODEL
    prompt: str
    max_tokens: int = 60
    temperature: float = 0.7
    top_p: float = 0.9


async def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return API_KEYS[x_api_key]


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "default_model": DEFAULT_MODEL,
        "available_models": list(MODEL_REGISTRY.keys()),
    }


@app.get("/models")
def list_models():
    """List all available model names."""
    return {"available_models": list(MODEL_REGISTRY.keys())}


@app.post("/generate")
async def generate(req: GenerateRequest, user=Depends(verify_api_key)):

    # pick the correct model
    tokenizer, model = get_model(req.model)

    start = time.time()

    inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=req.top_p,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elapsed = time.time() - start
    print(
        f"[REQUEST] user={user} model={req.model} "
        f"len={len(req.prompt)} tokens={req.max_tokens} time={elapsed:.2f}s"
    )

    log_request(user, req.prompt, text, elapsed)

    return {
        "completion": text,
        "latency_sec": elapsed,
        "model": req.model,
        "user": user,
    }


@app.get("/dashboard")
def dashboard():
    cur = conn.cursor()
    cur.execute(
        "SELECT id, user, prompt, completion, latency, timestamp "
        "FROM logs ORDER BY timestamp DESC LIMIT 20"
    )
    rows = cur.fetchall()
    cur.close()
    return {"logs": rows}

