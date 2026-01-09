# Inference API

## Authentication

All authenticated endpoints require an API key.

**Header:**
```
x-api-key: <api_key>
```

---

## POST `/register`

Create a new user and issue an API key.

### Request

**Query parameters:**
- `username` (string, required)

### Response (200)

```json
{
  "username": "string",
  "api_key": "string",
  "credits": 20.0
}
```

### Errors

- **500:** Registration failed

---

## POST `/generate`

Run text generation using a supported model.

### Headers

- `x-api-key` (string, required)

### Request Body

```json
{
  "model": "gpt2",
  "prompt": "string",
  "max_tokens": 50,
  "temperature": 0.7,
  "top_p": 0.9
}
```

### Response (200)

```json
{
  "completion": "string",
  "latency_sec": 0.12,
  "prompt_tokens": 10,
  "completion_tokens": 50,
  "total_tokens": 60,
  "credits_used": 0.001,
  "remaining_credits": 19.999,
  "model": "gpt2",
  "user": "username"
}
```

### Errors

- **400:** Invalid request parameters
- **401:** Invalid or missing API key
- **402:** Not enough credits
- **413:** Prompt too long
- **429:** Rate limit exceeded
- **500:** Inference failed

---

## GET `/health`

Check service health and readiness.

### Response (200)

```json
{
  "status": "ok",
  "device": "cpu",
  "models_loaded": ["gpt2"],
  "rate_limit": {
    "window_sec": 60,
    "max_requests": 10
  }
}
```

### Errors

- None

---

## GET `/models`

List available models.

### Response (200)

```json
{
  "models": ["gpt2", "distilgpt2"]
}
```

### Errors

- None

---

## Examples

### Register

```bash
curl -X POST "http://localhost:8000/register?username=testuser"
```

### Generate

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "x-api-key: YOUR_API_KEY" \
  -d '{
    "prompt": "hello",
    "max_tokens": 20
  }'
```

### Health

```bash
curl http://localhost:8000/health
```

### Models

```bash
curl http://localhost:8000/models
```

---