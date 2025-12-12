FROM python:3.10-slim-bullseye

WORKDIR /app

# Install system dependencies for PyTorch and OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy training-specific requirements
COPY training/requirements.txt .

# Install PyTorch from official index, then other requirements from PyPI
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1 torchvision==0.15.2 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY training/ ./training/
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 mlops && chown -R mlops:mlops /app
USER mlops

EXPOSE 8002

CMD ["python", "-m", "training.main"]