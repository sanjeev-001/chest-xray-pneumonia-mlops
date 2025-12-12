# Dockerfile for Chest X-Ray Pneumonia Detection API
FROM python:3.10-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy deployment-specific requirements (minimal dependencies)
COPY deployment/requirements.txt .

# Install Python dependencies with NumPy version pinning
# Install PyTorch first, then other packages, then force NumPy 1.x
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1 torchvision==0.15.2 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --force-reinstall "numpy>=1.21.0,<2.0.0"

# Copy application code
COPY training/ ./training/
COPY deployment/ ./deployment/
COPY monitoring/ ./monitoring/

# Don't install pyproject.toml to avoid dependency conflicts
# The modules are already in place and Python will find them

# Create models directory (models will be mounted as volume)
RUN mkdir -p ./models

# Create non-root user
RUN useradd -m -u 1000 apiuser && chown -R apiuser:apiuser /app

USER apiuser

# Expose port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Run the application
CMD ["uvicorn", "deployment.api:app", "--host", "0.0.0.0", "--port", "8004"]
