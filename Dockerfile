FROM python:3.11-slim

# Install system dependencies (minimal for lightweight deployment)
RUN apt-get update && apt-get install -y \
    curl \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for persistent storage and temp files
RUN mkdir -p /app/chroma_db /app/temp_uploads /app/notebooks

# Expose FastAPI port
EXPOSE 8000

# Health check for FastAPI
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8000}/api/health || exit 1

# Run FastAPI with Uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
