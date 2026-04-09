FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000
WORKDIR /app
ENV PYTHONUNBUFFERED=1
RUN pip install --no-cache-dir uv
COPY pyproject.toml requirements.txt ./
COPY uv.lock* ./
RUN if [ -f uv.lock ]; then \
        uv sync --system; \
    else \
        uv pip install --system -r requirements.txt; \
    fi
COPY . .
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy everything first
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the mandatory Hugging Face port
EXPOSE 7860

# Run the server on port 7860
# ... (rest of your Dockerfile) ...
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]