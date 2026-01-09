import os
import time
import torch
import sqlite3
import secrets
import signal
import json
import tempfile
import shutil
import asyncio
from contextlib import contextmanager
from uuid import uuid4
from datetime import datetime

from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from collections import defaultdict, deque


# In-memory per-process rate limiting (MVP)
rate_limit_store = defaultdict(deque)

# -------------------------------------------------
# Device setup
# -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# Credits pricing
CREDITS_PER_1000_TOKENS = 2   # e.g. ₹0.02 / $0.0002 per 1k tokens

# Admin authentication
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "change-me-in-production-please")

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

# GPT-2 context limit is 1024 tokens
MAX_PROMPT_CHARS = 3000      
MAX_NEW_TOKENS = 256
MIN_NEW_TOKENS = 1
MAX_CONTEXT_TOKENS = 1024  # GPT-2 context window

RATE_LIMIT_WINDOW_SEC = 60
RATE_LIMIT_MAX_REQUESTS = 100

# For model load and inference
MIN_FREE_DISK_MB = 100  # adjust if needed

# Backpressure - Cap concurrent inferences
MAX_INFLIGHT_INFERENCES = 3
inference_semaphore = asyncio.Semaphore(MAX_INFLIGHT_INFERENCES)

# -------------------------------------------------
# TASK 2: Structured Logging (FIXED)
# -------------------------------------------------
def log_structured(request_id: str, event: str, **kwargs):
    """Emit structured JSON logs. Only log fields that have values."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "request_id": request_id,
        "event": event,
    }
    # Only add fields that are not None
    for key, value in kwargs.items():
        if value is not None:
            log_entry[key] = value
    
    print(json.dumps(log_entry), flush=True)


# -------------------------------------------------
# TASK 1: Standardized Error Helper with request_id
# -------------------------------------------------
def api_error(request_id: str, code: str, message: str, status: int):
    """Standardized error response format with request_id."""
    raise HTTPException(
        status_code=status,
        detail={
            "request_id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
    )


def get_model(model_name: str, request_id: str = None):

    # TEMPORARY: Simulate random model crashes for testing
    # import random
    # if random.random() < 0.3:
    #     raise RuntimeError("Simulated model crash")
    
    """Load and cache a model by name."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not supported")

    if model_name not in loaded_models:
        # Structured model load logging
        if request_id:
            log_structured(request_id, "MODEL_LOAD", model=model_name, device=device)
        
        # Silence transformers warnings by setting verbosity
        import logging
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        tok = AutoTokenizer.from_pretrained(MODEL_REGISTRY[model_name])
        mdl = AutoModelForCausalLM.from_pretrained(MODEL_REGISTRY[model_name]).to(device)
        loaded_models[model_name] = (tok, mdl)

    return loaded_models[model_name]


# -------------------------------------------------
# SQLite setup - Thread-safe connection management
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "usage.db"))

print("USING DB FILE:", DB_PATH, flush=True)


@contextmanager
def get_db():
    """Create a thread-safe DB connection per request."""
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30,
        isolation_level=None
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    try:
        yield conn
    finally:
        conn.close()


# Initialize database schema
with get_db() as conn:
    # logs table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT,
        user TEXT,
        model TEXT,
        prompt TEXT,
        completion TEXT,
        latency REAL,
        tokens INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Add request_id column if it doesn't exist (migration)
    try:
        conn.execute("ALTER TABLE logs ADD COLUMN request_id TEXT")
        print("[MIGRATION] Added request_id column to logs table")
    except sqlite3.OperationalError:
        # Column already exists
        pass

    # users table
    conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        api_key TEXT UNIQUE,
        credits REAL DEFAULT 0,
        created_at DATETIME,
        last_used_at DATETIME
    )
    """)
    conn.commit()


def log_request(conn, request_id, user, model, prompt, completion, latency, tokens):
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO logs (request_id, user, model, prompt, completion, latency, tokens) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (request_id, user, model, prompt, completion, latency, tokens),
    )
    conn.commit()
    cur.close()


def update_last_used(conn, username: str):
    """Update last_used_at timestamp for a user."""
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET last_used_at = CURRENT_TIMESTAMP WHERE username=?",
        (username,)
    )
    conn.commit()
    cur.close()


def validate_username(username: str) -> str:
    """Validate username format."""
    if not username or len(username) < 3 or len(username) > 50:
        raise ValueError("Username must be 3-50 characters")
    if not username.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Username must be alphanumeric (with _ or - allowed)")
    return username.lower()


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
        raise ValueError("API key is required")

    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT username FROM users WHERE api_key=?", (x_api_key,))
            row = cur.fetchone()
            cur.close()
    except sqlite3.OperationalError:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "code": "DB_UNAVAILABLE",
                    "message": "Database unavailable"
                }
            }
        )

    if not row:
        raise ValueError("Invalid API key")

    return row[0]  # username


async def verify_admin(x_admin_key: str = Header(None)):
    """Verify admin API key."""
    if x_admin_key is None or x_admin_key != ADMIN_API_KEY:
        raise ValueError("Admin access required")
    return True


def cleanup_rate_limits():
    """Remove stale entries from rate limit store."""
    cutoff = time.time() - RATE_LIMIT_WINDOW_SEC
    for user in list(rate_limit_store.keys()):
        rate_limit_store[user] = deque(
            [t for t in rate_limit_store[user] if t > cutoff]
        )
        if not rate_limit_store[user]:
            del rate_limit_store[user]


# -------------------------------------------------
# Prometheus metrics
# -------------------------------------------------
REQUEST_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency per model",
    ["model"],
    buckets=(0.1, 0.3, 0.5, 1, 2, 5, 10)
)

REQUEST_OUTCOME = Counter(
    "inference_outcome_total",
    "Total inference requests by outcome",
    ["model", "status", "reason"]
)

CREDITS_SPENT = Counter(
    "credits_spent_total",
    "Total credits spent by user",
    ["user"]
)

REQUEST_REJECTED = Counter(
    "inference_requests_rejected_total",
    "Requests rejected before inference",
    ["reason"]
)

BACKPRESSURE_REJECTED = Counter(
    "backpressure_rejected_total",
    "Requests rejected due to backpressure"
)

# Queue infrastructure
REQUEST_QUEUE = asyncio.Queue()  # UNBOUNDED

QUEUE_DEPTH = Gauge(
    "queue_depth",
    "Number of requests waiting in inference queue"
)

QUEUE_ENQUEUE_TOTAL = Counter(
    "queue_enqueue_total",
    "Total requests enqueued"
)

QUEUE_DEQUEUE_TOTAL = Counter(
    "queue_dequeue_total",
    "Total requests dequeued"
)

QUEUE_WAIT_SECONDS = Histogram(
    "queue_wait_seconds",
    "Time spent waiting in queue before processing",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60)
)

# -------------------------------------------------
# Middleware for request tracking
# -------------------------------------------------
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request_id to all requests and handle errors uniformly."""
    request_id = f"req_{uuid4().hex[:12]}"
    request.state.request_id = request_id
    
    start_time = time.time()
    user = "anonymous"
    
    # Try to extract user from header
    try:
        api_key = request.headers.get("x-api-key")
        if api_key:
            with get_db() as conn:
                cur = conn.cursor()
                cur.execute("SELECT username FROM users WHERE api_key=?", (api_key,))
                row = cur.fetchone()
                if row:
                    user = row[0]
                cur.close()
    except sqlite3.OperationalError:
        # DB unavailable during auth - let it fail later with proper error
        pass
    except:
        pass
    
    # Log request start - ONLY known fields
    log_structured(
        request_id,
        "REQ_START",
        user=user,
        endpoint=request.url.path
    )
    
    try:
        response = await call_next(request)
        latency_ms = (time.time() - start_time) * 1000
        
        # Log request end
        log_structured(
            request_id,
            "REQ_END",
            user=user,
            endpoint=request.url.path,
            status=response.status_code,
            latency_ms=round(latency_ms, 2)
        )
        
        return response
    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        
        # Determine error details
        if isinstance(e, HTTPException):
            status = e.status_code
            detail = e.detail
        else:
            status = 500
            detail = {
                "request_id": request_id,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": str(e)
                }
            }
        
        # Log error
        log_structured(
            request_id,
            "REQ_ERROR",
            user=user,
            endpoint=request.url.path,
            status=status,
            latency_ms=round(latency_ms, 2),
            error=str(e)
        )
        
        return JSONResponse(status_code=status, content=detail)


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.post("/register")
def register(username: str, request: Request):
    """Create a user + API key + initial credits."""
    request_id = request.state.request_id
    
    try:
        username = validate_username(username)
    except ValueError as e:
        api_error(request_id, "INVALID_USERNAME", str(e), 400)
    
    api_key = create_api_key()
    
    with get_db() as conn:
        cur = conn.cursor()

        try:
            cur.execute(
                """
                INSERT INTO users (username, api_key, credits, created_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (username, api_key, 20.0),
            )
            conn.commit()

        except sqlite3.IntegrityError:
            # username already exists -> fetch existing key + credits
            cur.execute("SELECT api_key, credits FROM users WHERE username=?", (username,))
            row = cur.fetchone()
            cur.close()
            if not row:
                api_error(request_id, "USER_CREATE_FAILED", "User creation failed", 500)
            existing_key, existing_credits = row
            return {
                "request_id": request_id,
                "username": username,
                "api_key": existing_key,
                "credits": existing_credits,
                "note": "User already existed, returning existing key.",
            }

        cur.close()
    
    log_structured(request_id, "USER_REGISTERED", username=username)
    return {
        "request_id": request_id,
        "username": username,
        "api_key": api_key,
        "credits": 20.0
    }


@app.on_event("startup")
async def start_workers():
    asyncio.create_task(inference_worker())
    log_structured("startup", "WORKER_STARTED", workers=1)


@app.get("/")
def root(request: Request):
    return {
        "request_id": request.state.request_id,
        "status": "ok",
        "models": list(MODEL_REGISTRY.keys())
    }


@app.get("/portal")
def portal():
    """Serve landing page (optional shortcut)."""
    return FileResponse("app/static/index.html")


@app.get("/models")
def list_models(request: Request):
    return {
        "request_id": request.state.request_id,
        "models": list(MODEL_REGISTRY.keys())
    }


@app.get("/me")
def me(user=Depends(verify_api_key), request: Request = None):
    """Check current user + credits from API key."""
    request_id = request.state.request_id
    
    with get_db() as conn:
        update_last_used(conn, user)
        
        cur = conn.cursor()
        cur.execute("SELECT credits FROM users WHERE username=?", (user,))
        row = cur.fetchone()
        cur.close()

    credits = row[0] if row else None
    return {
        "request_id": request_id,
        "user": user,
        "credits": credits
    }


async def inference_worker():
    """Background worker that processes queue."""
    while True:
        item = await REQUEST_QUEUE.get()
        dequeue_time = time.time()

        QUEUE_DEQUEUE_TOTAL.inc()
        QUEUE_DEPTH.set(REQUEST_QUEUE.qsize())

        wait_time = dequeue_time - item["enqueue_time"]
        QUEUE_WAIT_SECONDS.observe(wait_time)

        request_id = item["request_id"]
        user = item["user"]
        req = item["req"]

        log_structured(
            request_id,
            "QUEUE_DEQUEUED",
            user=user,
            wait_seconds=round(wait_time, 2)
        )

        try:
            # Semaphore is acquired HERE
            await asyncio.wait_for(
                inference_semaphore.acquire(), 
                timeout=0.001
            )
            acquired = True
            
            try:
                await run_inference(request_id, user, req)
            finally:
                if acquired:
                    inference_semaphore.release()
        except asyncio.TimeoutError:
            # Backpressure - shouldn't happen if queue is slower than workers
            log_structured(request_id, "WORKER_BACKPRESSURE", user=user)
        except Exception as e:
            log_structured(request_id, "WORKER_ERROR", error=str(e))
        finally:
            REQUEST_QUEUE.task_done()


@app.post("/generate")
async def generate(req: GenerateRequest, user=Depends(verify_api_key), request: Request = None):
    
    request_id = request.state.request_id

    # -------- 1. Request validation (BEFORE semaphore) --------
    if not req.prompt or not req.prompt.strip():
        REQUEST_OUTCOME.labels(model=req.model, status="client_error", reason="empty_prompt").inc()
        REQUEST_REJECTED.labels(reason="invalid_input").inc()
        log_structured(request_id, "REQ_VALIDATION_FAIL", reason="empty_prompt", model=req.model)
        api_error(request_id, "EMPTY_PROMPT", "Prompt cannot be empty", 400)

    if len(req.prompt) > MAX_PROMPT_CHARS:
        REQUEST_OUTCOME.labels(model=req.model, status="client_error", reason="prompt_too_long").inc()
        REQUEST_REJECTED.labels(reason="invalid_input").inc()
        log_structured(request_id, "REQ_VALIDATION_FAIL", reason="prompt_too_long", model=req.model)
        api_error(request_id, "PROMPT_TOO_LONG", f"Prompt too long (max {MAX_PROMPT_CHARS} characters)", 413)

    if req.max_tokens < MIN_NEW_TOKENS or req.max_tokens > MAX_NEW_TOKENS:
        REQUEST_OUTCOME.labels(model=req.model, status="client_error", reason="invalid_max_tokens").inc()
        REQUEST_REJECTED.labels(reason="invalid_input").inc()
        log_structured(request_id, "REQ_VALIDATION_FAIL", reason="invalid_max_tokens", model=req.model)
        api_error(request_id, "INVALID_MAX_TOKENS", f"max_tokens must be between {MIN_NEW_TOKENS} and {MAX_NEW_TOKENS}", 400)


    # -------- 2. Rate limiting --------
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SEC

    timestamps = rate_limit_store[user]
    while timestamps and timestamps[0] < window_start:
        timestamps.popleft()

    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        REQUEST_OUTCOME.labels(model=req.model, status="rate_limited", reason="rate_limit_exceeded").inc()
        REQUEST_REJECTED.labels(reason="rate_limited").inc()
        log_structured(
            request_id, 
            "RATE_LIMIT_HIT",
            user=user,
            limit=f"{RATE_LIMIT_MAX_REQUESTS}/{RATE_LIMIT_WINDOW_SEC}s",
            model=req.model
        )
        api_error(request_id, "RATE_LIMIT_EXCEEDED", "Rate limit exceeded. Try again later.", 429)

    timestamps.append(now)
    
    if len(rate_limit_store) > 1000:
        cleanup_rate_limits()

    # # -------- 3. BACKPRESSURE: Try to acquire semaphore --------
    # acquired = False
    # try:
    #     await asyncio.wait_for(inference_semaphore.acquire(), timeout=0.001)
    #     acquired = True
    # except asyncio.TimeoutError:
    #     # Server at capacity - reject immediately
    #     BACKPRESSURE_REJECTED.inc()
    #     REQUEST_OUTCOME.labels(model=req.model, status="backpressure", reason="server_at_capacity").inc()
    #     REQUEST_REJECTED.labels(reason="backpressure").inc()
        
    #     log_structured(
    #         request_id, 
    #         "BACKPRESSURE_HIT", 
    #         user=user, 
    #         model=req.model, 
    #         max_inflight=MAX_INFLIGHT_INFERENCES
    #     )
    #     api_error(request_id, "SERVER_BUSY", "Server at capacity. Try again later.", 503)

    # log_structured(
    #     request_id,
    #     "WORKER_ID",
    #     pid=os.getpid(),
    #     user=user,
    #     model=req.model
    # )

    # # Semaphore acquired - log it
    # log_structured(request_id, "SEMAPHORE_ACQUIRED", user=user, model=req.model)

    # SLOW_MODE = False  # toggle this
    # if SLOW_MODE:
    #     await asyncio.sleep(30)  # ← REMOVE THIS BEFORE PRODUCTION

    # try:
    #     # -------- 4. Load model --------
    #     try:
    #         tokenizer, model = get_model(req.model, request_id)
    #     except ValueError as e:
    #         REQUEST_OUTCOME.labels(model=req.model, status="client_error", reason="invalid_model").inc()
    #         REQUEST_REJECTED.labels(reason="invalid_input").inc()
    #         log_structured(request_id, "MODEL_LOAD_FAIL", reason="invalid_model", model=req.model)
    #         api_error(request_id, "INVALID_MODEL", str(e), 400)
    #     except Exception as e:
    #         REQUEST_OUTCOME.labels(model=req.model, status="server_error", reason="model_load_failed").inc()
    #         log_structured(request_id, "MODEL_LOAD_FAIL", reason="load_error", model=req.model, error=str(e))
    #         api_error(request_id, "MODEL_LOAD_FAILED", "Model load failed", 500)

    #     # -------- 5. Context window check --------
    #     try:
    #         prompt_inputs = tokenizer(req.prompt, return_tensors="pt")
    #         prompt_tokens = prompt_inputs["input_ids"].shape[1]
            
    #         if prompt_tokens + req.max_tokens > MAX_CONTEXT_TOKENS:
    #             REQUEST_OUTCOME.labels(model=req.model, status="client_error", reason="context_overflow").inc()
    #             REQUEST_REJECTED.labels(reason="invalid_input").inc()
    #             log_structured(
    #                 request_id, 
    #                 "REQ_VALIDATION_FAIL",
    #                 reason="context_overflow",
    #                 prompt_tokens=prompt_tokens,
    #                 max_tokens=req.max_tokens,
    #                 limit=MAX_CONTEXT_TOKENS,
    #                 model=req.model
    #             )
    #             api_error(
    #                 request_id,
    #                 "CONTEXT_OVERFLOW",
    #                 f"Prompt ({prompt_tokens} tokens) + max_tokens ({req.max_tokens}) exceeds model context limit ({MAX_CONTEXT_TOKENS})",
    #                 400
    #             )
    #     except Exception as e:
    #         REQUEST_OUTCOME.labels(model=req.model, status="server_error", reason="tokenization_failed").inc()
    #         log_structured(request_id, "REQ_VALIDATION_FAIL", reason="tokenization_failed", model=req.model, error=str(e))
    #         api_error(request_id, "TOKENIZATION_FAILED", "Failed to process prompt", 500)

    #     # -------- 6. Run inference (semaphore is held) --------
    #     inference_start = time.time()
        
    #     try:
    #         inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

    #         loop = asyncio.get_running_loop()
    #         outputs = await loop.run_in_executor(
    #             None,
    #             lambda: model.generate(
    #                 **inputs,
    #                 max_new_tokens=req.max_tokens,
    #                 temperature=temperature,
    #                 top_p=top_p,
    #                 do_sample=True,
    #             )
    #         )
    #     except Exception as e:
    #         REQUEST_OUTCOME.labels(model=req.model, status="server_error", reason="inference_failed").inc()
    #         log_structured(request_id, "INFERENCE_FAIL", model=req.model, error=str(e))
    #         api_error(request_id, "INFERENCE_FAILED", "Inference failed", 500)
                
    #     inference_latency = time.time() - inference_start

    #     # -------- 7. Token accounting --------
    #     prompt_tokens = inputs["input_ids"].shape[1]
    #     completion_tokens = outputs[0].shape[0] - prompt_tokens
    #     total_tokens = prompt_tokens + completion_tokens

    #     credits_used = (completion_tokens / 1000.0) * CREDITS_PER_1000_TOKENS


    #     # -------- 8. Billing (AFTER successful inference) --------
    #     with get_db() as conn:
    #         cur = conn.cursor()
    #         try:
    #             cur.execute("BEGIN IMMEDIATE")

    #             cur.execute("SELECT credits FROM users WHERE username=?", (user,))
    #             row = cur.fetchone()
                
    #             if not row:
    #                 conn.rollback()
    #                 REQUEST_OUTCOME.labels(model=req.model, status="server_error", reason="user_not_found").inc()
    #                 REQUEST_REJECTED.labels(reason="auth_failed").inc()
    #                 log_structured(request_id, "BILLING_FAIL", reason="user_not_found", user=user, model=req.model)
    #                 api_error(request_id, "USER_NOT_FOUND", "User not found", 401)
                
    #             bal = row[0]

    #             if bal < credits_used:
    #                 conn.rollback()
    #                 REQUEST_OUTCOME.labels(model=req.model, status="client_error", reason="insufficient_credits").inc()
    #                 REQUEST_REJECTED.labels(reason="no_credits").inc()
    #                 log_structured(
    #                     request_id,
    #                     "BILLING_FAIL",
    #                     reason="insufficient_credits",
    #                     user=user,
    #                     balance=bal,
    #                     required=credits_used,
    #                     model=req.model
    #                 )
    #                 api_error(request_id, "INSUFFICIENT_CREDITS", "Not enough credits", 402)

    #             cur.execute(
    #                 "UPDATE users SET credits = credits - ? WHERE username=?",
    #                 (credits_used, user),
    #             )

    #             cur.execute(
    #                 "UPDATE users SET last_used_at = CURRENT_TIMESTAMP WHERE username=?",
    #                 (user,),
    #             )

    #             conn.commit()
                
    #             cur.execute("SELECT credits FROM users WHERE username=?", (user,))
    #             new_balance = cur.fetchone()[0]

    #         except HTTPException:
    #             raise
    #         except sqlite3.OperationalError as e:
    #             # DB-specific failures: locked, unavailable, corrupted
    #             conn.rollback()
    #             REQUEST_OUTCOME.labels(model=req.model, status="dependency_error", reason="db_unavailable").inc()
    #             log_structured(request_id, "BILLING_FAIL", reason="db_unavailable", user=user, model=req.model, error=str(e))
    #             api_error(request_id, "DB_UNAVAILABLE", "Database unavailable", 503)
    #         except Exception as e:
    #             # Other unexpected failures
    #             conn.rollback()
    #             REQUEST_OUTCOME.labels(model=req.model, status="server_error", reason="billing_failed").inc()
    #             log_structured(request_id, "BILLING_FAIL", reason="transaction_failed", user=user, model=req.model, error=str(e))
    #             api_error(request_id, "BILLING_FAILED", "Billing failure", 500)
    #         finally:
    #             cur.close()

    #     # -------- 9. Success metrics and logs --------
    #     text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     total_latency = time.time() - inference_start

    #     REQUEST_LATENCY.labels(req.model).observe(total_latency)
    #     REQUEST_OUTCOME.labels(model=req.model, status="success", reason="ok").inc()

    #     log_structured(
    #         request_id,
    #         "INFERENCE_SUCCESS",
    #         user=user,
    #         model=req.model,
    #         latency_ms=round(inference_latency * 1000, 2),
    #         prompt_tokens=prompt_tokens,
    #         completion_tokens=completion_tokens,
    #         credits_used=round(credits_used, 4)
    #     )

    #     with get_db() as conn:
    #         log_request(
    #             conn=conn,
    #             request_id=request_id,
    #             user=user,
    #             model=req.model,
    #             prompt=req.prompt,
    #             completion=text,
    #             latency=total_latency,
    #             tokens=completion_tokens,
    #         )

    #     CREDITS_SPENT.labels(user=user).inc(credits_used)

    #     return {
    #         "id": request_id,
    #         "completion": text,
    #         "usage": {
    #             "prompt_tokens": prompt_tokens,
    #             "completion_tokens": completion_tokens,
    #             "total_tokens": total_tokens
    #         },
    #         "billing": {
    #             "credits_used": credits_used,
    #             "remaining_credits": new_balance
    #         },
    #         "model": req.model,
    #         "latency_sec": total_latency
    #     }
        
    # finally:
    #     # CRITICAL: Always release semaphore
    #     if acquired:
    #         inference_semaphore.release()
    #         log_structured(request_id, "SEMAPHORE_RELEASED", user=user, model=req.model)

    # NEW: Enqueue instead of processing
    enqueue_time = time.time()
    
    await REQUEST_QUEUE.put({
        "request_id": request_id,
        "user": user,
        "req": req,
        "enqueue_time": enqueue_time
    })

    QUEUE_ENQUEUE_TOTAL.inc()
    QUEUE_DEPTH.set(REQUEST_QUEUE.qsize())

    log_structured(
        request_id,
        "REQUEST_ENQUEUED",
        user=user,
        queue_depth=REQUEST_QUEUE.qsize()
    )

    # Return immediately!
    return JSONResponse(
        status_code=202,
        content={
            "request_id": request_id,
            "status": "queued",
            "queue_position": REQUEST_QUEUE.qsize()
        }
    )

async def run_inference(request_id: str, user: str, req: GenerateRequest):
    """Process one inference request (moved from /generate)."""
    # Copy-paste from line 481 onwards:
    # - Model loading
    # - Context check
    # - Inference
    # - Billing
    # - Logging
    # (Everything that's currently in the try/finally block)

    # Clamp temperature and top_p
    temperature = max(0.0, min(2.0, req.temperature))
    top_p = max(0.01, min(1.0, req.top_p))
    
    if temperature != req.temperature or top_p != req.top_p:
        log_structured(
            request_id, 
            "PARAM_CLAMPED",
            original_temp=req.temperature if temperature != req.temperature else None,
            clamped_temp=temperature if temperature != req.temperature else None,
            original_top_p=req.top_p if top_p != req.top_p else None,
            clamped_top_p=top_p if top_p != req.top_p else None
        )

    SLOW_MODE = False  # toggle this
    if SLOW_MODE:
        await asyncio.sleep(30)  # ← REMOVE THIS BEFORE PRODUCTION

    # -------- 4. Load model --------
    try:
        tokenizer, model = get_model(req.model, request_id)
    except ValueError as e:
        REQUEST_OUTCOME.labels(model=req.model, status="client_error", reason="invalid_model").inc()
        REQUEST_REJECTED.labels(reason="invalid_input").inc()
        log_structured(request_id, "MODEL_LOAD_FAIL", reason="invalid_model", model=req.model)
        api_error(request_id, "INVALID_MODEL", str(e), 400)
    except Exception as e:
        REQUEST_OUTCOME.labels(model=req.model, status="server_error", reason="model_load_failed").inc()
        log_structured(request_id, "MODEL_LOAD_FAIL", reason="load_error", model=req.model, error=str(e))
        api_error(request_id, "MODEL_LOAD_FAILED", "Model load failed", 500)

    # -------- 5. Context window check --------
    try:
        prompt_inputs = tokenizer(req.prompt, return_tensors="pt")
        prompt_tokens = prompt_inputs["input_ids"].shape[1]
        
        if prompt_tokens + req.max_tokens > MAX_CONTEXT_TOKENS:
            REQUEST_OUTCOME.labels(model=req.model, status="client_error", reason="context_overflow").inc()
            REQUEST_REJECTED.labels(reason="invalid_input").inc()
            log_structured(
                request_id, 
                "REQ_VALIDATION_FAIL",
                reason="context_overflow",
                prompt_tokens=prompt_tokens,
                max_tokens=req.max_tokens,
                limit=MAX_CONTEXT_TOKENS,
                model=req.model
            )
            api_error(
                request_id,
                "CONTEXT_OVERFLOW",
                f"Prompt ({prompt_tokens} tokens) + max_tokens ({req.max_tokens}) exceeds model context limit ({MAX_CONTEXT_TOKENS})",
                400
            )
    except Exception as e:
        REQUEST_OUTCOME.labels(model=req.model, status="server_error", reason="tokenization_failed").inc()
        log_structured(request_id, "REQ_VALIDATION_FAIL", reason="tokenization_failed", model=req.model, error=str(e))
        api_error(request_id, "TOKENIZATION_FAILED", "Failed to process prompt", 500)

    # -------- 6. Run inference (semaphore is held) --------
    inference_start = time.time()
    
    try:
        inputs = tokenizer(req.prompt, return_tensors="pt").to(device)

        loop = asyncio.get_running_loop()
        outputs = await loop.run_in_executor(
            None,
            lambda: model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
        )
    except Exception as e:
        REQUEST_OUTCOME.labels(model=req.model, status="server_error", reason="inference_failed").inc()
        log_structured(request_id, "INFERENCE_FAIL", model=req.model, error=str(e))
        api_error(request_id, "INFERENCE_FAILED", "Inference failed", 500)
            
    inference_latency = time.time() - inference_start

    # -------- 7. Token accounting --------
    prompt_tokens = inputs["input_ids"].shape[1]
    completion_tokens = outputs[0].shape[0] - prompt_tokens
    total_tokens = prompt_tokens + completion_tokens

    credits_used = (completion_tokens / 1000.0) * CREDITS_PER_1000_TOKENS


    # -------- 8. Billing (AFTER successful inference) --------
    with get_db() as conn:
        cur = conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")

            cur.execute("SELECT credits FROM users WHERE username=?", (user,))
            row = cur.fetchone()
            
            if not row:
                conn.rollback()
                REQUEST_OUTCOME.labels(model=req.model, status="server_error", reason="user_not_found").inc()
                REQUEST_REJECTED.labels(reason="auth_failed").inc()
                log_structured(request_id, "BILLING_FAIL", reason="user_not_found", user=user, model=req.model)
                api_error(request_id, "USER_NOT_FOUND", "User not found", 401)
            
            bal = row[0]

            if bal < credits_used:
                conn.rollback()
                REQUEST_OUTCOME.labels(model=req.model, status="client_error", reason="insufficient_credits").inc()
                REQUEST_REJECTED.labels(reason="no_credits").inc()
                log_structured(
                    request_id,
                    "BILLING_FAIL",
                    reason="insufficient_credits",
                    user=user,
                    balance=bal,
                    required=credits_used,
                    model=req.model
                )
                api_error(request_id, "INSUFFICIENT_CREDITS", "Not enough credits", 402)

            cur.execute(
                "UPDATE users SET credits = credits - ? WHERE username=?",
                (credits_used, user),
            )

            cur.execute(
                "UPDATE users SET last_used_at = CURRENT_TIMESTAMP WHERE username=?",
                (user,),
            )

            conn.commit()
            
            cur.execute("SELECT credits FROM users WHERE username=?", (user,))
            new_balance = cur.fetchone()[0]

        except HTTPException:
            raise
        except sqlite3.OperationalError as e:
            # DB-specific failures: locked, unavailable, corrupted
            conn.rollback()
            REQUEST_OUTCOME.labels(model=req.model, status="dependency_error", reason="db_unavailable").inc()
            log_structured(request_id, "BILLING_FAIL", reason="db_unavailable", user=user, model=req.model, error=str(e))
            api_error(request_id, "DB_UNAVAILABLE", "Database unavailable", 503)
        except Exception as e:
            # Other unexpected failures
            conn.rollback()
            REQUEST_OUTCOME.labels(model=req.model, status="server_error", reason="billing_failed").inc()
            log_structured(request_id, "BILLING_FAIL", reason="transaction_failed", user=user, model=req.model, error=str(e))
            api_error(request_id, "BILLING_FAILED", "Billing failure", 500)
        finally:
            cur.close()

    # -------- 9. Success metrics and logs --------
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    total_latency = time.time() - inference_start

    REQUEST_LATENCY.labels(req.model).observe(total_latency)
    REQUEST_OUTCOME.labels(model=req.model, status="success", reason="ok").inc()

    log_structured(
        request_id,
        "INFERENCE_SUCCESS",
        user=user,
        model=req.model,
        latency_ms=round(inference_latency * 1000, 2),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        credits_used=round(credits_used, 4)
    )

    with get_db() as conn:
        log_request(
            conn=conn,
            request_id=request_id,
            user=user,
            model=req.model,
            prompt=req.prompt,
            completion=text,
            latency=total_latency,
            tokens=completion_tokens,
        )

    CREDITS_SPENT.labels(user=user).inc(credits_used)

    return {
        "id": request_id,
        "completion": text,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        },
        "billing": {
            "credits_used": credits_used,
            "remaining_credits": new_balance
        },
        "model": req.model,
        "latency_sec": total_latency
    }

# --------------------------------------------------------
# TASK 3: Health Endpoint That Means Something with Ready
# --------------------------------------------------------
@app.get("/health")
def health(request: Request):
    request_id = request.state.request_id
    checks = {}

    # ---------- DB CHECK (read-only) ----------
    try:
        with get_db() as conn:
            conn.execute("SELECT 1")
        checks["db"] = "ok"
    except Exception as e:
        checks["db"] = f"fail: {str(e)}"

    # ---------- MODEL REGISTRY CHECK ----------
    try:
        checks["models_loaded"] = list(loaded_models.keys())
        checks["model_registry"] = "ok" if loaded_models else "empty"
    except Exception as e:
        checks["model_registry"] = f"fail: {str(e)}"

    # ---------- DISK CHECK ----------
    try:
        st = shutil.disk_usage("/")
        free_mb = st.free // (1024 * 1024)
        checks["disk"] = f"ok ({free_mb}MB free)"
    except Exception as e:
        checks["disk"] = f"fail: {str(e)}"

    return {
        "status": "ok",
        "pid": os.getpid(),
        "request_id": request_id,
        "checks": checks,
        "device": device,
    }


@app.get("/ready")
def ready(request: Request):
    request_id = request.state.request_id

    # Calculate in-flight requests safely
    inflight = MAX_INFLIGHT_INFERENCES - inference_semaphore._value

    if inflight >= MAX_INFLIGHT_INFERENCES:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "reason": "at_capacity",
                "inflight": inflight,
                "max_inflight": MAX_INFLIGHT_INFERENCES,
                "request_id": request_id,
            }
        )

    return {
        "status": "ready",
        "inflight": inflight,
        "max_inflight": MAX_INFLIGHT_INFERENCES,
        "request_id": request_id,
    }


@app.get("/dashboard")
def dashboard(_: bool = Depends(verify_admin), request: Request = None):
    """Global logs (admin view, JSON)."""
    request_id = request.state.request_id
    
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, request_id, user, prompt, completion, latency, tokens, timestamp "
            "FROM logs ORDER BY timestamp DESC LIMIT 20"
        )
        rows = cur.fetchall()
        cur.close()
    return {
        "request_id": request_id,
        "logs": rows
    }


@app.get("/usage/stats")
def usage_stats(_: bool = Depends(verify_admin), request: Request = None):
    """High-level stats."""
    request_id = request.state.request_id
    
    with get_db() as conn:
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
        "request_id": request_id,
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
def user_dashboard(user=Depends(verify_api_key), request: Request = None):
    """Logs for *this* user only."""
    request_id = request.state.request_id
    
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, request_id, prompt, completion, latency, tokens, timestamp "
            "FROM logs WHERE user=? ORDER BY timestamp DESC LIMIT 20",
            (user,),
        )
        rows = cur.fetchall()
        cur.close()
    return {
        "request_id": request_id,
        "user": user,
        "logs": rows
    }


@app.post("/admin/add_credits")
def add_credits(username: str, amount: float, _: bool = Depends(verify_admin), request: Request = None):
    """Manual top-up endpoint (admin only)."""
    request_id = request.state.request_id
    
    try:
        username = validate_username(username)
    except ValueError as e:
        api_error(request_id, "INVALID_USERNAME", str(e), 400)
    
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET credits = credits + ? WHERE username=?",
            (amount, username),
        )
        conn.commit()

        cur.execute("SELECT credits FROM users WHERE username=?", (username,))
        row = cur.fetchone()
        cur.close()

    new_credits = row[0] if row else None
    log_structured(request_id, "CREDITS_ADDED", username=username, amount=amount, new_balance=new_credits)
    
    return {
        "request_id": request_id,
        "status": "ok",
        "username": username,
        "added": amount,
        "new_credits": new_credits
    }


@app.post("/keys/rotate")
def rotate_key(user=Depends(verify_api_key), request: Request = None):
    request_id = request.state.request_id
    new_key = create_api_key()

    with get_db() as conn:
        cur = conn.cursor()
        try:
            cur.execute("BEGIN IMMEDIATE")

            # Ensure user still exists
            cur.execute(
                "SELECT api_key FROM users WHERE username=?",
                (user,)
            )
            row = cur.fetchone()

            if not row:
                conn.rollback()
                api_error(request_id, "INVALID_API_KEY", "Invalid API key", 401)

            # Replace API key
            cur.execute(
                "UPDATE users SET api_key=? WHERE username=?",
                (new_key, user)
            )

            update_last_used(conn, user)

            conn.commit()

        except HTTPException:
            raise
        except Exception as e:
            conn.rollback()
            log_structured(request_id, "KEY_ROTATION_FAIL", user=user, error=str(e))
            api_error(request_id, "KEY_ROTATION_FAILED", "Key rotation failed", 500)
        finally:
            cur.close()

    log_structured(request_id, "KEY_ROTATED", user=user)
    return {
        "request_id": request_id,
        "old_key_revoked": True,
        "new_api_key": new_key
    }


@app.get("/keys/info")
def key_info(user=Depends(verify_api_key), request: Request = None):
    request_id = request.state.request_id
    
    with get_db() as conn:
        update_last_used(conn, user)
        
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT username, created_at, last_used_at FROM users WHERE username=?",
                (user,)
            )
            row = cur.fetchone()

            if not row:
                api_error(request_id, "INVALID_API_KEY", "Invalid API key", 401)

            return {
                "request_id": request_id,
                "username": row[0],
                "created_at": row[1],
                "last_used_at": row[2],
            }

        finally:
            cur.close()


# -------------------------------------------------
# Graceful shutdown
# -------------------------------------------------
def shutdown_handler(signum, frame):
    """Clean up models and close DB on shutdown."""
    # Use structured logging for shutdown
    shutdown_id = f"shutdown_{uuid4().hex[:8]}"
    log_structured(shutdown_id, "SERVER_SHUTDOWN", models_loaded=len(loaded_models))
    
    for model_name in list(loaded_models.keys()):
        del loaded_models[model_name]
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    log_structured(shutdown_id, "SHUTDOWN_COMPLETE")
    exit(0)


signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)