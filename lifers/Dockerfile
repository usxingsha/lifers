# Lifers AI — Docker image for cloud deployment
# Build: docker build -t lifers .
# Run:   docker run -p 55555:55555 -v $(pwd)/weights:/app/weights -v $(pwd)/config:/app/config lifers

FROM python:3.11-slim

LABEL org.opencontainers.image.title="Lifers AI"
LABEL org.opencontainers.image.description="Local AI brain with deep transformer, agent pipeline, and taskflow orchestration"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application
COPY lifers/ ./lifers/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY pyproject.toml ./

# Python deps (minimal — core uses stdlib; optional extras below)
RUN pip install --no-cache-dir numpy

# Runtime
ENV PYTHONPATH=/app
ENV LIFERS_ROOT=/app
ENV PYTHONUTF8=1
ENV SANDBOX=1
ENV LIFERS_FORCE_LOCAL_ONLY=1

# Expose HTTP gate port
EXPOSE 55555

# Default: start the HTTP gate
CMD ["python3", "scripts/lifers_gate.py", "--host", "0.0.0.0", "--port", "55555"]

# For CLI mode, override CMD:
# docker run -it lifers python3 scripts/lifers_chat.py chat
