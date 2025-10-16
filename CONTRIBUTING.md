# Contributing to Unified AI Analytics Platform

Thank you for your interest in contributing to the Unified AI Analytics Platform! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Basic understanding of machine learning concepts
- Familiarity with scikit-learn, pandas, and numpy

### Areas to Contribute

- **New ML Algorithms**: Implement additional models
- **Feature Engineering**: Add new preprocessing techniques
- **Explainability**: Enhance XAI capabilities
- **Documentation**: Improve docs, add examples
- **Testing**: Increase test coverage
- **Bug Fixes**: Fix reported issues
- **Performance**: Optimize code for speed/memory
- **UI/UX**: Improve dashboard and API

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/unified-ai-platform.git
cd unified-ai-platform

# Add upstream remote
git remote add upstream https://github.com/mohin-io/unified-ai-platform.git
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda
conda create -n unified-ai python=3.9
conda activate unified-ai
```

### 3. Install Dependencies

```bash
# Install package in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install from requirements
pip install -r requirements.txt
```

### 4. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

This will automatically run code formatting and linting before each commit.

### 5. Verify Setup

```bash
# Run tests to verify everything works
pytest tests/

# Check code quality
black --check src/ tests/
flake8 src/ tests/
mypy src/
```

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**Bug Report Template:**

```markdown
**Description**
A clear description of the bug.

**To Reproduce**
Steps to reproduce the behavior:
1. Load data with '...'
2. Train model '...'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- Package version: [e.g., 0.1.0]

**Additional Context**
Any other relevant information.
```

### Suggesting Features

Feature suggestions are welcome! Please provide:

- **Use Case**: Why is this feature needed?
- **Proposed Solution**: How should it work?
- **Alternatives**: Any alternative approaches considered?
- **Examples**: Example usage or similar implementations

### Submitting Changes

1. **Create a Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

   Branch naming conventions:
   - `feature/`: New features
   - `fix/`: Bug fixes
   - `docs/`: Documentation changes
   - `refactor/`: Code refactoring
   - `test/`: Adding tests

2. **Make Changes**
   - Write clean, readable code
   - Follow coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

   Commit message format:
   ```
   <type>(<scope>): <subject>

   <body>

   <footer>
   ```

   Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

4. **Push to Your Fork**
   ```bash
   git push origin feature/amazing-feature
   ```

5. **Create Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill in the PR template
   - Submit for review

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line Length**: 100 characters (not 79)
- **Quotes**: Double quotes for strings
- **Imports**: Organized with isort
- **Type Hints**: Required for all public functions
- **Docstrings**: Google style for all public classes/functions

### Code Formatting

We use **Black** for code formatting:

```bash
# Format code
black src/ tests/

# Check formatting
black --check src/ tests/
```

### Import Organization

We use **isort** for import sorting:

```bash
# Sort imports
isort src/ tests/

# Check import order
isort --check-only src/ tests/
```

### Type Checking

We use **mypy** for static type checking:

```bash
# Type check
mypy src/ --ignore-missing-imports
```

### Example Code Style

```python
"""
Module docstring explaining the purpose.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd

from src.utils import Config


class ExampleClass:
    """
    Brief class description.

    This class demonstrates the coding standards we follow. It includes
    type hints, comprehensive docstrings, and follows PEP 8 guidelines.

    Attributes:
        param1: Description of param1
        param2: Description of param2

    Example:
        >>> obj = ExampleClass(param1="value")
        >>> result = obj.method()
    """

    def __init__(self, param1: str, param2: int = 0):
        """
        Initialize the class.

        Args:
            param1: Description and purpose
            param2: Description with default value explanation

        The initialization sets up the object state and validates parameters.
        """
        self.param1 = param1
        self.param2 = param2

    def method(
        self,
        arg1: Union[pd.DataFrame, np.ndarray],
        arg2: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Brief method description.

        Detailed explanation of what the method does, how it works,
        and why certain design decisions were made.

        Args:
            arg1: Input data description
            arg2: Optional parameter description

        Returns:
            Description of return value

        Raises:
            ValueError: When and why this is raised
            TypeError: When and why this is raised

        Example:
            >>> obj = ExampleClass("test")
            >>> result = obj.method(data, optional_arg)
        """
        if arg2 is None:
            arg2 = []

        # Implementation with clear comments
        # explaining non-obvious logic
        result = self._helper_method(arg1)

        return result

    def _helper_method(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Private helper method.

        Private methods start with underscore and don't need as extensive
        documentation as public methods.

        Args:
            data: Input DataFrame

        Returns:
            Processed DataFrame
        """
        return data.copy()
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/              # Unit tests for individual components
├── integration/       # Integration tests for workflows
└── performance/       # Performance and benchmark tests
```

### Writing Tests

We use **pytest** for testing:

```python
import pytest
import pandas as pd
from src.preprocessing import DataLoader


class TestDataLoader:
    """Test suite for DataLoader class."""

    @pytest.fixture
    def sample_data(self):
        """Fixture providing sample data for tests."""
        return pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })

    def test_load_csv(self, tmp_path):
        """Test CSV loading functionality."""
        # Arrange
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("a,b\n1,2\n3,4")
        loader = DataLoader()

        # Act
        df = loader.load_from_csv(str(csv_file))

        # Assert
        assert len(df) == 2
        assert list(df.columns) == ['a', 'b']

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises error."""
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_from_csv("nonexistent.csv")
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/unit/test_data_loader.py

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/unit/test_data_loader.py::TestDataLoader::test_load_csv

# Run with verbose output
pytest tests/ -v

# Run and stop on first failure
pytest tests/ -x
```

### Test Coverage

- Aim for **80%+ code coverage**
- Focus on critical paths and edge cases
- Test both success and failure scenarios
- Include integration tests for workflows

## Documentation

### Docstring Format

We use **Google style** docstrings:

```python
def function(arg1: int, arg2: str = "default") -> bool:
    """
    Brief description of the function.

    More detailed explanation of what the function does, including any
    important algorithms, assumptions, or design decisions.

    Args:
        arg1: Description of arg1 and its purpose
        arg2: Description of arg2, including default behavior

    Returns:
        Description of return value and its format

    Raises:
        ValueError: When arg1 is negative
        TypeError: When arg2 is not a string

    Example:
        >>> result = function(42, "test")
        >>> print(result)
        True

    Note:
        Any additional notes, warnings, or important information.
    """
    pass
```

### Documentation Files

- **README.md**: Overview, installation, quick start
- **docs/PLAN.md**: Project blueprint and architecture
- **docs/API_REFERENCE.md**: Detailed API documentation
- **docs/TUTORIALS.md**: Step-by-step tutorials
- **docs/FAQ.md**: Frequently asked questions

### Building Documentation

```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme

# Build HTML documentation
cd docs/
make html

# View documentation
open _build/html/index.html
```

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation
- [ ] No merge conflicts with main branch
- [ ] Commit messages are clear and descriptive

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe how you tested your changes.

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented complex code
- [ ] Updated documentation
- [ ] Added tests
- [ ] All tests pass
- [ ] No warnings or errors

## Related Issues
Closes #issue_number
```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs tests and linting
2. **Code Review**: At least one maintainer reviews the code
3. **Discussion**: Address reviewer comments and suggestions
4. **Approval**: Maintainer approves after review
5. **Merge**: Maintainer merges the PR

### After Merge

- Delete your branch
- Update your fork:
  ```bash
  git checkout main
  git pull upstream main
  git push origin main
  ```

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Pull Requests**: Code contributions and reviews

### Getting Help

- Check existing issues and documentation
- Ask questions in GitHub Discussions
- Tag relevant maintainers for specific areas

### Recognition

Contributors are recognized in:
- [README.md](README.md) contributors section
- Release notes for significant contributions
- GitHub contributors page

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with the `question` label
- Start a discussion in GitHub Discussions
- Reach out to maintainers

Thank you for contributing to making machine learning more accessible!
