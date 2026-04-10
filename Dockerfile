FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . .

# Expose the mandatory Hugging Face port
EXPOSE 7860

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--access-log"]