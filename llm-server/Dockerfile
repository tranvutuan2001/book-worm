# Multi-stage Dockerfile for production-grade FastAPI LLM server
# - Build wheels in a builder stage to avoid compilation in the final image
# - Keep final image minimal, run as non-root, mount large model files at runtime

FROM python:3.11-slim-bullseye AS builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    gcc \
    g++ \
    libffi-dev \
    pkg-config \
    libopenblas-dev \
    libomp-dev \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /wheels

# Copy dependency files and build wheels so final image doesn't need build deps
COPY requirements.txt ./
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip wheel --wheel-dir /wheels -r requirements.txt


FROM python:3.11-slim-bullseye
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_DIR=/models

# Install minimal runtime OS deps needed by some libraries (OpenMP, BLAS)
RUN apt update && apt install -y --no-install-recommends \
    libgomp1 \
    libopenblas3 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create a non-root user to run the app
RUN groupadd -r app && useradd -r -g app -d /app -s /sbin/nologin app

WORKDIR /app

# Copy application files
COPY --chown=app:app . /app
# Copy wheels from builder and install from them (avoid building in final stage)
COPY --from=builder /wheels /wheels

RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-index --find-links=/wheels -r /app/requirements.txt && \
    rm -rf /wheels ~/.cache/pip

# Expose the application port
EXPOSE ${PORT}

# Create app log dir and ensure permissions
RUN mkdir -p /app/log && chown -R app:app /app

# Healthcheck for container orchestrators
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://127.0.0.1:${PORT}/health || exit 1

USER app

# Use Gunicorn + Uvicorn workers for production (adjust workers/threads as needed)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--workers", "2", "--threads", "4", "--bind", "0.0.0.0:8000", "main:app"]
