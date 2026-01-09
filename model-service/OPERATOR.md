# Cortex Inference Platform — Operator Guide

This document explains how to operate, debug, and recover the service under pressure.

---

## 1. How to Start the Service

### Local (Docker)

```bash
docker build -t model-service .
docker run -p 8000:8000 \
  -v $(pwd)/usage.db:/app/usage.db \
  model-service
```

**Expected output:**
- `USING DB FILE: /app/usage.db`
- `Uvicorn running on http://0.0.0.0:8000`

---

## 2. Verify Service Health

### Liveness (Process Alive)

```bash
curl -i http://localhost:8000/health
```

**Expected:**
- HTTP 200
- **Means:** process is running

⚠️ `/health` does NOT guarantee the service can handle traffic.

### Readiness (Service Usable)

```bash
curl -i http://localhost:8000/ready
```

**Expected:**
- HTTP 200 → safe to receive traffic
- HTTP 503 → remove from load balancer

**Checks performed:**
- Database writable
- At least one model loadable
- Sufficient disk space

---

## 3. Debug: "Credits Missing"

### Step 1 — Check user balance

```bash
curl http://localhost:8000/me \
  -H "x-api-key: USER_API_KEY"
```

### Step 2 — Check logs

Look for:
- `BILLING_FAIL`
- `REQ_END (402)`

### Step 3 — Check metrics

```bash
curl http://localhost:8000/metrics | grep credits_spent_total
```

**If credits dropped:**
- Billing worked

**If not:**
- Request failed before billing

---

## 4. Debug: "Model Is Slow"

### Step 1 — Check latency

```bash
curl http://localhost:8000/metrics | grep inference_latency_seconds
```

**If latency high but errors low:**
- Resource pressure
- Cold start
- CPU-only inference

### Step 2 — Check logs

Look for:
- Long gap between `REQ_START` and `INFERENCE_SUCCESS`

**This is performance, not correctness.**

---

## 5. Debug: "Requests Failing"

### Identify failure type

| Symptom | Likely Cause |
|---------|--------------|
| 400 | Client error |
| 402 | Credits exhausted |
| 429 | Rate limiting |
| 500 | Server failure |
| 503 | Not ready (DB / disk / model) |

### Confirm via metrics

```bash
curl http://localhost:8000/metrics | grep inference_outcome_total
```

---

## 6. Rotate API Keys Safely

### Rotate key

```bash
curl -X POST http://localhost:8000/keys/rotate \
  -H "x-api-key: OLD_KEY"
```

**Response includes:**
- New API key
- Old key revoked immediately

**No downtime required.**

---

## 7. Database Recovery

### Database locked / corrupted

1. Stop container
2. Backup DB
   ```bash
   cp usage.db usage.db.bak
   ```
3. Restart service
4. Verify `/ready`

---

## 8. Safe Failure Rules (Golden Rules)

- `/health` must lie less than logs
- `/ready` must fail before `/generate`
- Billing must be atomic
- No inference on invalid requests
- Metrics must explain behavior faster than logs

---

## 9. Emergency Checklist

**At 3 a.m., do this in order:**

1. `/health`
2. `/ready`
3. Metrics (`inference_outcome_total`)
4. Logs (single `request_id`)
5. Restart only if `/ready` stays red

---

**End of Operator Guide**