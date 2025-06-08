FROM apache/airflow:2.7.3-python3.11

# Switch to root to install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Copy and install requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt