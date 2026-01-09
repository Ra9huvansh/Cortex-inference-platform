#!/bin/bash

API_KEY="2014d75517a8f6adccfbb477c9f344d1"
URL="http://localhost:8000/generate"

echo "=== Burst start $(date) ==="

for i in {1..50}; do
  curl -s -o /dev/null -w "%{http_code}\n" \
    -X POST $URL \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -d '{ "prompt": "hello", "max_tokens": 10 }' &
done

wait

echo "=== Burst end $(date) ==="
