# ──────────────────────────────────────────────────────────
# Email Triage OpenEnv — Dockerfile
# Compatible with HuggingFace Spaces (Docker SDK)
# Port: 7860 (required by HF Spaces)
# ──────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="openenv-submission"
LABEL description="Email Triage OpenEnv — AI agent environment"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY models.py .
COPY tasks.py .
COPY environment.py .
COPY server.py .
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
