FROM python:3.11-slim-bullseye

WORKDIR /app

# Install system dependencies for scientific libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy data pipeline specific requirements
COPY data_pipeline/requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY data_pipeline/ ./data_pipeline/
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 mlops && chown -R mlops:mlops /app
USER mlops

EXPOSE 8001

CMD ["python", "-m", "data_pipeline.main"]
