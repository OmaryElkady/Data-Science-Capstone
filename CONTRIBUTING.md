# Contributing to Data Science Capstone

Thank you for contributing to this project! This guide will help you set up your development environment and follow our code quality standards.

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/OmaryElkady/Data-Science-Capstone.git
cd Data-Science-Capstone
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Development Tools (Optional but Recommended)

```bash
pip install black isort flake8 pylint nbqa pre-commit
```

### 5. Set Up Pre-commit Hooks (Optional but Recommended)

Pre-commit hooks automatically check your code before each commit:

```bash
pre-commit install
```

## Code Quality Standards

### Python Code Style

- **Formatter**: Black (line length: 127)
- **Import Sorting**: isort (Black profile)
- **Linter**: flake8 with custom rules
- **Code Quality**: pylint for additional checks

### Running Quality Checks Locally

Before pushing code, you can run these checks locally:

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Lint with flake8
flake8 .

# Lint Jupyter notebooks
nbqa flake8 .

# Run pylint
find . -name "*.py" -type f | xargs pylint
```

### Jupyter Notebook Guidelines

1. **Clear Outputs Before Committing**: Always clear notebook outputs before committing to keep the repository clean:
   ```bash
   jupyter nbconvert --clear-output --inplace your_notebook.ipynb
   ```

2. **Keep Notebooks Executable**: Ensure your notebooks can run from top to bottom without errors

3. **Add Markdown Documentation**: Use markdown cells to explain your analysis and findings

4. **Use Meaningful Variable Names**: Even in notebooks, use descriptive variable names

## Workflow

### Creating a New Feature or Fix

1. **Create a Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**: Implement your feature or fix

3. **Test Locally**: Run the quality checks mentioned above

4. **Commit Changes**:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. **Push to GitHub**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**: Go to GitHub and create a PR from your branch

### Pull Request Guidelines

- Ensure all CI checks pass before requesting review
- Write clear, descriptive PR titles and descriptions
- Reference any related issues in your PR description
- Respond to review comments promptly

## Continuous Integration

Our repository uses GitHub Actions for CI/CD. The following checks run automatically on every push and pull request:

### 1. Python Code Quality
- Black formatting check
- isort import sorting check
- flake8 linting
- pylint code quality analysis
- Jupyter notebook linting with nbqa

### 2. Security Scanning
- Dependency vulnerability scanning with pip-audit
- Security checks with Safety

### 3. Notebook Validation
- Notebook format validation
- Output clearing checks
- Optional notebook execution tests

If any of these checks fail, you'll need to fix the issues before your PR can be merged.

## Troubleshooting

### Pre-commit Hook Failures

If pre-commit hooks are blocking your commit:

1. **Auto-fix Issues**: Many formatters can auto-fix issues:
   ```bash
   black .
   isort .
   ```

2. **Review Changes**: Check what was changed and stage the fixes:
   ```bash
   git add .
   git commit -m "Your message"
   ```

3. **Skip Hooks (Not Recommended)**: Only in emergency:
   ```bash
   git commit --no-verify -m "Your message"
   ```

### CI Check Failures

1. Pull the latest changes from main
2. Run checks locally to reproduce the issue
3. Fix the issues
4. Push the fixes

## Questions or Issues?

If you have questions or run into issues:
- Open an issue on GitHub
- Check existing issues for similar problems
- Review the documentation in the README

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
- Help others learn and grow

Thank you for contributing! 🎉
