# Data-Science-Capstone

[![Python Code Quality](https://github.com/OmaryElkady/Data-Science-Capstone/actions/workflows/python-code-quality.yml/badge.svg)](https://github.com/OmaryElkady/Data-Science-Capstone/actions/workflows/python-code-quality.yml)
[![Security Scan](https://github.com/OmaryElkady/Data-Science-Capstone/actions/workflows/security-scan.yml/badge.svg)](https://github.com/OmaryElkady/Data-Science-Capstone/actions/workflows/security-scan.yml)
[![Notebook Checks](https://github.com/OmaryElkady/Data-Science-Capstone/actions/workflows/notebook-checks.yml/badge.svg)](https://github.com/OmaryElkady/Data-Science-Capstone/actions/workflows/notebook-checks.yml)

This is my Data Science senior year capstone!

## Code Quality & CI/CD

This repository uses GitHub Actions to ensure code quality and maintain clean Python code:

### Automated Workflows

1. **Python Code Quality** (`python-code-quality.yml`)
   - Runs on every push and pull request
   - Checks code formatting with Black
   - Validates import sorting with isort
   - Lints Python files with flake8
   - Lints Jupyter notebooks with nbqa
   - Runs pylint for additional code quality checks

2. **Security Scan** (`security-scan.yml`)
   - Runs on push, pull requests, and weekly schedule
   - Scans dependencies for known vulnerabilities using pip-audit
   - Checks for security issues with Safety

3. **Notebook Checks** (`notebook-checks.yml`)
   - Validates Jupyter notebook format
   - Checks if notebooks have cleared outputs (best practice for version control)
   - Optionally executes notebooks to ensure they run without errors

### Local Development Setup

To maintain code quality locally, you can use pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install

# (Optional) Run against all files
pre-commit run --all-files
```

The pre-commit configuration includes:
- Code formatting (Black, isort)
- Linting (flake8)
- Notebook linting (nbqa)
- General file checks (trailing whitespace, large files, etc.)

### Dependencies

Install project dependencies:

```bash
pip install -r requirements.txt
```

### Project Structure

```
.
├── .github/workflows/      # GitHub Actions workflows
├── eda_flights_sample.ipynb # Exploratory Data Analysis notebook
├── eda_outputs/            # EDA visualization outputs
├── requirements.txt        # Python dependencies
└── .pre-commit-config.yaml # Pre-commit hooks configuration
```

## Development Guidelines

1. **Code Formatting**: All Python code should follow Black formatting standards (line length: 127)
2. **Import Sorting**: Use isort with Black profile for consistent import ordering
3. **Linting**: Code should pass flake8 checks with minimal warnings
4. **Notebooks**: Clear outputs before committing (use `jupyter nbconvert --clear-output`)
5. **Security**: Keep dependencies up-to-date and address any security vulnerabilities

## Contributing

When contributing to this repository:
1. Create a new branch for your changes
2. Ensure all CI checks pass
3. Clear notebook outputs before committing
4. Write descriptive commit messages
5. Submit a pull request for review
