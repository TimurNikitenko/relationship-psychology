FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
# We install CPU-only PyTorch first to significantly reduce image size
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch>=2.0.0 --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download and cache the Hugging Face embedding model (cointegrated/rubert-tiny2)
# This prevents downloading it during container startup on the VPS
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2'); AutoModel.from_pretrained('cointegrated/rubert-tiny2')"

# Copy application files
COPY alembic.ini .
COPY migrations/ ./migrations/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY entrypoint.sh .

# Ensure entrypoint is executable
RUN chmod +x entrypoint.sh

# Run the entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
