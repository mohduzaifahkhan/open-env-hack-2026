FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml requirements.txt ./
COPY uv.lock* ./
RUN if [ -f uv.lock ]; then \
        uv sync --system; \
    else \
        uv pip install --system -r requirements.txt; \
    fi
COPY . .
CMD ["openenv-server", "start", "--config", "openenv.yaml", "--host", "0.0.0.0", "--port", "8000"]
