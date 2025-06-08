FROM freqtradeorg/freqtrade:stable

# Switch to root to install additional packages
USER root

# Install additional Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create necessary directories
RUN mkdir -p /freqtrade/user_data/data /freqtrade/user_data/logs

# Switch back to freqtrade user
USER freqtrade

# Set working directory
WORKDIR /freqtrade