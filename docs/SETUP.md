# Development Environment Setup

This guide will help you set up your local development environment for contributing to this Data Science project.

## Prerequisites

- Python 3.8 or higher
- Git
- pip (Python package installer)

## Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone https://github.com/OmaryElkady/Data-Science-Capstone.git
cd Data-Science-Capstone
```

### 2. Create a Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt, indicating the virtual environment is active.

### 3. Install Project Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all the data science libraries needed for the project:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter
- scipy

### 4. Install Development Tools (Recommended)

For code quality checks and formatting:

```bash
pip install black isort flake8 pylint nbqa pre-commit
```

### 5. Set Up Pre-commit Hooks (Optional but Recommended)

Pre-commit hooks automatically check your code before each commit:

```bash
pre-commit install
```

To test the setup:
```bash
pre-commit run --all-files
```

### 6. Verify Installation

Test that everything is installed correctly:

```bash
python -c "import pandas; import numpy; import matplotlib; print('✅ All dependencies installed successfully!')"
```

### 7. Start Jupyter Notebook

```bash
jupyter notebook
```

This will open Jupyter in your default web browser.

## IDE Setup

### Visual Studio Code (Recommended)

1. **Install VS Code**: Download from [code.visualstudio.com](https://code.visualstudio.com/)

2. **Install Python Extension**: Open VS Code and install the "Python" extension by Microsoft

3. **Configure Settings**: Create or edit `.vscode/settings.json` in the project root:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "127"],
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.flake8Args": ["--max-line-length=127"],
    "python.linting.pylintEnabled": false,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter"
    }
}
```

4. **Install Recommended Extensions**:
   - Python (ms-python.python)
   - Black Formatter (ms-python.black-formatter)
   - Pylance (ms-python.vscode-pylance)
   - Jupyter (ms-toolsai.jupyter)

### PyCharm

1. **Open Project**: File → Open → Select the project directory

2. **Configure Interpreter**: 
   - File → Settings → Project → Python Interpreter
   - Add Interpreter → Existing → Select `venv/bin/python`

3. **Configure Black**:
   - Settings → Tools → Black
   - Check "On save"
   - Set line length to 127

4. **Configure Flake8**:
   - Settings → Editor → Inspections → Python
   - Enable Flake8 inspections
   - Configure max line length: 127

## Quick Reference Commands

### Code Formatting

```bash
# Format all Python files
black .

# Sort imports
isort .

# Format Jupyter notebooks
nbqa black .
```

### Code Checking

```bash
# Run flake8
flake8 .

# Run pylint
pylint $(find . -name "*.py" -type f)

# Check notebooks
nbqa flake8 .
```

### Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Clear notebook outputs
jupyter nbconvert --clear-output --inplace notebook.ipynb

# Execute notebook
jupyter nbconvert --execute --inplace notebook.ipynb
```

### Git Workflow

```bash
# Create a new branch
git checkout -b feature/my-feature

# Check status
git status

# Stage changes
git add .

# Commit with message
git commit -m "Description of changes"

# Push to remote
git push origin feature/my-feature
```

## Troubleshooting

### Virtual Environment Issues

**Problem**: Can't activate virtual environment

**Solution**:
- Make sure you're in the project directory
- On Windows, you might need to run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**:
1. Ensure virtual environment is activated (you should see `(venv)` in prompt)
2. Reinstall requirements: `pip install -r requirements.txt`

### Jupyter Kernel Issues

**Problem**: Jupyter can't find packages

**Solution**:
```bash
# Install ipykernel
pip install ipykernel

# Create kernel with virtual environment
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```

Then select this kernel in Jupyter: Kernel → Change kernel → Python (venv)

### Pre-commit Hook Failures

**Problem**: Pre-commit hooks are blocking commits

**Solution**:
```bash
# Auto-fix formatting issues
black .
isort .

# Review and stage changes
git add .
git commit -m "Your message"
```

## Next Steps

1. Read the [CONTRIBUTING.md](../CONTRIBUTING.md) guide
2. Check out the [CODE_QUALITY_GUIDE.md](CODE_QUALITY_GUIDE.md) for common issues
3. Review the [example_script.py](../example_script.py) to see code quality standards
4. Start exploring the notebooks in the repository

## Getting Help

- **GitHub Issues**: Open an issue if you encounter problems
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the README and other docs files

## Useful Resources

- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [Jupyter Documentation](https://jupyter.org/documentation)
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [Black Code Style](https://black.readthedocs.io/)

Happy coding! 🎉
