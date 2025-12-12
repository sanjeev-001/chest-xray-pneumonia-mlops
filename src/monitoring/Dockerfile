FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy monitoring-specific requirements
COPY monitoring/requirements.txt .

# Install PyTorch from official index, then other requirements from PyPI
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1 torchvision==0.15.2 && \
    pip install --no-cache-dir -r requirements.txt

# Copy monitoring module
COPY monitoring/ ./monitoring/

# Copy other necessary modules
COPY training/ ./training/
COPY deployment/ ./deployment/
COPY data_pipeline/ ./data_pipeline/

# Create directories for monitoring data
RUN mkdir -p /app/monitoring/baseline_stats
RUN mkdir -p /app/metrics_backup

# Set environment variables
ENV PYTHONPATH=/app
ENV PORT=8005
ENV PROMETHEUS_PORT=8080

# Expose ports
EXPOSE 8005 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

# Run the monitoring service
CMD ["python", "-m", "monitoring.main"]