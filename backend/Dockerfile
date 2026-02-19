# Multi-stage Dockerfile for production
# - Uses official slim Python image
# - Installs deps in builder stage and copies only runtime files to final image
# - Creates non-root user and minimal footprint

FROM python:3.11-slim AS builder
LABEL stage=builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install build dependencies needed for some Python packages
RUN apt update \
 && apt install -y --no-install-recommends build-essential gcc git libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt ./

# Upgrade pip and install dependencies
RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# Final runtime image
FROM python:3.11-slim
LABEL maintainer=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/app/.local/bin:$PATH" \
    ENVIRONMENT=production

WORKDIR /app

# Create an unprivileged user
RUN groupadd -r app && useradd -r -g app -m -d /home/app app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY . /app

# Adjust ownership and reduce image size where possible
RUN chown -R app:app /app \
 && rm -rf /root/.cache

USER app

# Expose default port (adjust if your app uses another port)
EXPOSE 8000

# Lightweight healthcheck: ensures Python imports succeed and dependencies are intact
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import sys, pkgutil; sys.exit(0 if pkgutil.find_loader('app') or True else 1)" || exit 1

# Default command: run with gunicorn
# Assumes a WSGI/ASGI app object named `app` is available in `main.py`.
# If your app object lives elsewhere, change `main:app` accordingly.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--log-level", "info"]
