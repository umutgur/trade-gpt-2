FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs

# Set Python path
ENV PYTHONPATH=/app/src

# Expose ports
EXPOSE 8501 8000

# Default command (can be overridden)
CMD ["streamlit", "run", "src/dashboard.py", "--server.address", "0.0.0.0", "--server.port", "8501"]