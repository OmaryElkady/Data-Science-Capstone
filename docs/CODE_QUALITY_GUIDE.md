# Code Quality Guide

This guide provides quick solutions to common code quality issues detected by our CI/CD pipeline.

## Quick Fix Commands

Run these commands before committing to ensure your code passes all checks:

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Check for linting errors
flake8 .

# Lint Jupyter notebooks
nbqa flake8 .
```

## Common Issues and Solutions

### 1. Black Formatting Issues

**Issue**: `would reformat file.py`

**Solution**:
```bash
black file.py
```

Black will automatically fix formatting issues including:
- Line length (max 127 characters)
- Indentation
- Spacing around operators
- Quote consistency

### 2. Import Sorting Issues

**Issue**: `Imports are incorrectly sorted and/or formatted`

**Solution**:
```bash
isort file.py
```

isort will organize imports in the correct order:
1. Standard library imports
2. Third-party imports
3. Local application imports

### 3. Flake8 Linting Errors

#### E231: Missing whitespace after ','

**Bad:**
```python
my_list = [1,2,3]
```

**Good:**
```python
my_list = [1, 2, 3]
```

#### E225: Missing whitespace around operator

**Bad:**
```python
result = x+y
```

**Good:**
```python
result = x + y
```

#### E402: Module level import not at top of file

**Bad:**
```python
print("Hello")
import os
```

**Good:**
```python
import os

print("Hello")
```

#### F401: Module imported but unused

**Solution**: Remove the unused import or use it in your code.

**Bad:**
```python
import pandas as pd
import numpy as np  # Not used anywhere

data = pd.DataFrame()
```

**Good:**
```python
import pandas as pd

data = pd.DataFrame()
```

#### F811: Redefinition of unused variable

**Solution**: Remove duplicate imports or variable definitions.

### 4. Jupyter Notebook Issues

#### Module imports not at top

In notebooks, imports should be in the first code cell:

**Bad:**
```python
# Cell 1
print("Starting analysis")

# Cell 2
import pandas as pd
```

**Good:**
```python
# Cell 1
import pandas as pd

# Cell 2
print("Starting analysis")
```

#### Clearing notebook outputs

Always clear outputs before committing:

```bash
jupyter nbconvert --clear-output --inplace your_notebook.ipynb
```

Or in Jupyter interface: `Cell > All Output > Clear`

### 5. Line Length Issues

Maximum line length is 127 characters. For long lines:

**Bad:**
```python
result = some_very_long_function_name(parameter1, parameter2, parameter3, parameter4, parameter5, parameter6, parameter7, parameter8)
```

**Good:**
```python
result = some_very_long_function_name(
    parameter1,
    parameter2,
    parameter3,
    parameter4,
    parameter5,
    parameter6,
    parameter7,
    parameter8,
)
```

## Pylint Issues

### Missing docstrings

Add docstrings to functions and classes:

```python
def analyze_data(df):
    """
    Analyze the input dataframe.
    
    Args:
        df: Input dataframe to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    return {"mean": df.mean()}
```

### Invalid name

Use descriptive variable names following Python conventions:
- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_CASE`

## Pre-commit Hooks

To catch these issues before committing:

```bash
# Install pre-commit
pip install pre-commit

# Set up hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## IDE Integration

### VS Code

Install these extensions:
- Python (Microsoft)
- Black Formatter
- isort
- Flake8

Add to `.vscode/settings.json`:
```json
{
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "127"],
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": ["--max-line-length=127"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

### PyCharm

1. Go to `Settings > Tools > Black`
2. Enable "On save"
3. Go to `Settings > Editor > Code Style > Python`
4. Set line length to 127

## Testing Your Changes

Before pushing:

```bash
# 1. Format code
black .
isort .

# 2. Check for issues
flake8 .
pylint $(find . -name "*.py" -type f)

# 3. For notebooks
nbqa black .
nbqa isort .
nbqa flake8 .

# 4. Clear notebook outputs
find . -name "*.ipynb" -not -path "*/.*" -exec jupyter nbconvert --clear-output --inplace {} \;
```

## Getting Help

- Check the [CONTRIBUTING.md](../CONTRIBUTING.md) guide
- Review existing PRs that passed CI checks
- Open an issue if you're stuck

## References

- [Black Documentation](https://black.readthedocs.io/)
- [isort Documentation](https://pycqa.github.io/isort/)
- [Flake8 Documentation](https://flake8.pycqa.org/)
- [PEP 8 Style Guide](https://www.python.org/dev/peps/pep-0008/)
