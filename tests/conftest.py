"""
Pytest configuration and fixtures
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def data_pipeline_client():
    """Test client for data pipeline service"""
    from data_pipeline.main import app
    return TestClient(app)


@pytest.fixture
def training_client():
    """Test client for training service"""
    from training.main import app
    return TestClient(app)


@pytest.fixture
def model_registry_client():
    """Test client for model registry service"""
    from model_registry.main import app
    return TestClient(app)


@pytest.fixture
def deployment_client():
    """Test client for deployment service"""
    from deployment.api import app
    return TestClient(app)


@pytest.fixture
def monitoring_client():
    """Test client for monitoring service"""
    from monitoring.main import app
    return TestClient(app)