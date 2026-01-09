FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install CPU-only PyTorch from official CPU wheel index
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.4.0

# Then install the rest of the deps from PyPI
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]



