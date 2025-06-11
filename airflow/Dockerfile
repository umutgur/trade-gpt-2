FROM apache/airflow:2.8.4-python3.11

# Switch to root to install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone to Turkey
ENV TZ=Europe/Istanbul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Create directories with proper permissions
RUN mkdir -p /app/logs /app/models && chown -R airflow /app/logs /app/models

# Switch back to airflow user
USER airflow

# Copy and install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt