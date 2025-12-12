.PHONY: help install install-dev build test lint format clean docker-build docker-up docker-down k8s-deploy k8s-delete

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install production dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev: ## Install development dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

build: ## Build all Docker images
	docker build -f data_pipeline/Dockerfile -t chest-xray-mlops/data-pipeline:latest .
	docker build -f training/Dockerfile -t chest-xray-mlops/training:latest .
	docker build -f model_registry/Dockerfile -t chest-xray-mlops/model-registry:latest .
	docker build -f deployment/Dockerfile -t chest-xray-mlops/deployment:latest .
	docker build -f monitoring/Dockerfile -t chest-xray-mlops/monitoring:latest .

test: ## Run tests
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing

lint: ## Run linting
	flake8 data_pipeline/ training/ model_registry/ deployment/ monitoring/
	mypy data_pipeline/ training/ model_registry/ deployment/ monitoring/

format: ## Format code
	black data_pipeline/ training/ model_registry/ deployment/ monitoring/
	isort data_pipeline/ training/ model_registry/ deployment/ monitoring/

clean: ## Clean up build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf htmlcov/

docker-build: build ## Build Docker images

docker-up: ## Start services with Docker Compose
	docker-compose up -d

docker-down: ## Stop services with Docker Compose
	docker-compose down

docker-logs: ## View Docker Compose logs
	docker-compose logs -f

k8s-deploy: ## Deploy to Kubernetes
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/secrets.yaml
	kubectl apply -f k8s/postgres.yaml
	kubectl apply -f k8s/minio.yaml
	kubectl apply -f k8s/data-pipeline.yaml
	kubectl apply -f k8s/training.yaml
	kubectl apply -f k8s/model-registry.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/monitoring.yaml

k8s-delete: ## Delete Kubernetes resources
	kubectl delete -f k8s/ --ignore-not-found=true

k8s-status: ## Check Kubernetes deployment status
	kubectl get all -n chest-xray-mlops

setup-dev: install-dev ## Set up development environment
	@echo "Development environment setup complete!"
	@echo "Run 'make docker-up' to start local services"