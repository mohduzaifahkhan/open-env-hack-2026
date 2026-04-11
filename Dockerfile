# Copyright (c) 2026 TRIBUNAL Team.
# Smart Factory Assembly - Dockerfile for HuggingFace Spaces.

FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 1. COPY EVERYTHING FIRST (This fixes the cache miss!)
COPY . .

# 2. NOW INSTALL (So pip can see your /server folder)
RUN pip install --no-cache-dir .

# Expose the HuggingFace Spaces port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]