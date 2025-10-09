.PHONY: help install install-dev format lint check clean test notebook

help:
	@echo "Available commands:"
	@echo "  make install       - Install project dependencies"
	@echo "  make install-dev   - Install project and development dependencies"
	@echo "  make format        - Format code with black and isort"
	@echo "  make lint          - Run linting checks"
	@echo "  make check         - Run all code quality checks"
	@echo "  make clean         - Clean up cache and temporary files"
	@echo "  make notebook      - Start Jupyter notebook server"
	@echo "  make clear-outputs - Clear all notebook outputs"

install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: install
	pip install black isort flake8 pylint nbqa pre-commit
	pre-commit install

format:
	@echo "Formatting Python files with Black..."
	black .
	@echo "Sorting imports with isort..."
	isort .
	@echo "Formatting Jupyter notebooks..."
	nbqa black . || true
	nbqa isort . || true
	@echo "✅ Formatting complete!"

lint:
	@echo "Running flake8..."
	flake8 . --count --statistics
	@echo "Linting Jupyter notebooks..."
	nbqa flake8 . --exit-zero || true
	@echo "✅ Linting complete!"

check: format lint
	@echo "Running pylint..."
	find . -name "*.py" -type f ! -path "./venv/*" ! -path "./.venv/*" | xargs pylint --exit-zero || true
	@echo "✅ All checks complete!"

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete!"

notebook:
	jupyter notebook

clear-outputs:
	@echo "Clearing notebook outputs..."
	find . -name "*.ipynb" ! -path "*/.*" -exec jupyter nbconvert --clear-output --inplace {} \;
	@echo "✅ Notebook outputs cleared!"
