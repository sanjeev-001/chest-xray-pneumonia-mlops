# Contributing to Chest X-Ray Pneumonia Detection MLOps System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project adheres to a code of conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- Git
- Basic understanding of MLOps concepts

### Setting Up Development Environment

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/chest-xray-pneumonia-mlops.git
   cd chest-xray-pneumonia-mlops
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   make install-dev
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

5. **Start local services:**
   ```bash
   make docker-up
   ```

## Development Workflow

### Branch Naming Convention

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `hotfix/description` - Critical production fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Making Changes

1. **Create a new branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the coding standards

3. **Run tests:**
   ```bash
   make test
   ```

4. **Run linting and formatting:**
   ```bash
   make lint
   make format
   ```

5. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add batch prediction endpoint
fix: resolve memory leak in data pipeline
docs: update API documentation
test: add integration tests for monitoring service
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guide
- Use [Black](https://black.readthedocs.io/) for code formatting (line length: 88)
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use type hints for function signatures
- Write docstrings for all public functions and classes

### Code Quality Tools

The project uses several tools to maintain code quality:

```bash
# Format code
black .
isort .

# Lint code
flake8 .
mypy .

# Or use make commands
make format
make lint
```

### Documentation

- Add docstrings to all public functions, classes, and modules
- Update relevant documentation when making changes
- Include code examples in docstrings where appropriate

Example docstring format:
```python
def predict_pneumonia(image_path: str, model_version: str = "latest") -> dict:
    """
    Predict pneumonia from chest X-ray image.
    
    Args:
        image_path: Path to the chest X-ray image file
        model_version: Version of the model to use (default: "latest")
    
    Returns:
        Dictionary containing prediction results with keys:
        - prediction: "NORMAL" or "PNEUMONIA"
        - confidence: Float between 0 and 1
        - processing_time: Time taken in milliseconds
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is invalid
    
    Example:
        >>> result = predict_pneumonia("xray.jpg")
        >>> print(result["prediction"])
        PNEUMONIA
    """
    pass
```

## Testing Guidelines

### Writing Tests

- Write tests for all new features and bug fixes
- Aim for at least 80% code coverage
- Use descriptive test names that explain what is being tested
- Follow the Arrange-Act-Assert pattern

Example test structure:
```python
def test_predict_pneumonia_with_valid_image():
    """Test pneumonia prediction with a valid X-ray image."""
    # Arrange
    image_path = "tests/fixtures/normal_xray.jpg"
    expected_prediction = "NORMAL"
    
    # Act
    result = predict_pneumonia(image_path)
    
    # Assert
    assert result["prediction"] == expected_prediction
    assert 0 <= result["confidence"] <= 1
    assert result["processing_time"] > 0
```

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_data_pipeline.py -v

# Run with coverage report
pytest --cov=. --cov-report=html

# Run specific test
pytest tests/test_data_pipeline.py::test_ingest_data -v
```

### Test Categories

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test interactions between components
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test system performance and scalability

## Submitting Changes

### Pull Request Process

1. **Update your branch with the latest main:**
   ```bash
   git checkout main
   git pull origin main
   git checkout your-branch
   git rebase main
   ```

2. **Push your changes:**
   ```bash
   git push origin your-branch
   ```

3. **Create a Pull Request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what was changed and why
   - Reference to related issues (e.g., "Fixes #123")
   - Screenshots or examples if applicable

4. **PR Checklist:**
   - [ ] Tests pass locally
   - [ ] Code follows style guidelines
   - [ ] Documentation is updated
   - [ ] Commit messages follow convention
   - [ ] No merge conflicts
   - [ ] Added tests for new features

### Code Review Process

- All PRs require at least one approval
- Address review comments promptly
- Keep PRs focused and reasonably sized
- Be open to feedback and suggestions

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

- Clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and stack traces
- Screenshots if applicable

### Feature Requests

When requesting features, please include:

- Clear description of the feature
- Use case and motivation
- Proposed implementation (if any)
- Potential impact on existing functionality

### Issue Labels

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Documentation improvements
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `question` - Further information requested

## Development Tips

### Useful Commands

```bash
# Start all services
make docker-up

# Stop all services
make docker-down

# View logs
docker-compose logs -f [service-name]

# Run specific service
docker-compose up [service-name]

# Rebuild images
make build

# Clean up
make clean
```

### Debugging

- Use Python debugger (pdb) for debugging
- Check service logs in Docker containers
- Use MLflow UI to track experiments
- Monitor Prometheus metrics for performance issues

### Performance Considerations

- Profile code for performance bottlenecks
- Optimize database queries
- Use batch processing where possible
- Consider memory usage for large datasets

## Questions?

If you have questions or need help:

- Check existing documentation in the `docs/` folder
- Search existing issues on GitHub
- Create a new issue with the `question` label
- Reach out to maintainers

Thank you for contributing to this project!
