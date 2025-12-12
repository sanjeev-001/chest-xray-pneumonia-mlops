FROM python:3.10-slim-bullseye

WORKDIR /app

# Copy model registry specific requirements
COPY model_registry/requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY model_registry/ ./model_registry/
COPY pyproject.toml .

# Install the package
RUN pip install -e .

# Create non-root user
RUN useradd -m -u 1000 mlops && chown -R mlops:mlops /app
USER mlops

EXPOSE 8003

CMD ["python", "-m", "model_registry.main"]