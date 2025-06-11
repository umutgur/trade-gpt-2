FROM freqtradeorg/freqtrade:stable

# Install additional Python packages if needed
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create necessary directories
RUN mkdir -p /freqtrade/user_data/data /freqtrade/user_data/logs

# Ensure proper permissions
RUN chown -R ftuser:ftuser /freqtrade/user_data

# Switch back to ftuser
USER ftuser
WORKDIR /freqtrade