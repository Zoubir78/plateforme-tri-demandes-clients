# Data Science Boilerplate

Welcome to the **Data Science Boilerplate**: A modern, production-ready template for data science projects with best practices.

This documentation contains the following sections:

- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Package Customization](#package-customization)
- [Development Workflow](#development-workflow)
- [Documentation](#documentation)

# Quick Start

## 1. Create a new repository

### Option A: Use as GitHub Template
1. Click "Use this template" on GitHub
2. Choose "Create a new repository"
3. Name your repository

### Option B: Clone and Customize
```bash
git clone <this-repo-url>
cd <your-project-name>
```

## 2. Customize Package Name (Optional)

You can easily customize the package name to match your project:

```bash
# Customize during setup
make setup PACKAGE_NAME=my-awesome-project

# Or customize separately
make customize-package PACKAGE_NAME=my-awesome-project
```

The package name will automatically be converted to a valid Python module name (hyphens become underscores).

## 3. Setup Development Environment

```bash
# Install dependencies and setup virtual environment
make setup

# Install pre-commit hooks
make install_precommit
```

That's it! Your development environment is ready.

# Project Structure

This boilerplate provides a recommended repository structure following Python best practices:

```
├── src/                           # Source code (src layout)
│   └── your_package_name/         # Your main package
│       ├── __init__.py
│       ├── core/                  # Core functionality
│       │   ├── __init__.py
│       │   ├── data_processing.py
│       │   └── models.py
│       └── utils/                 # Utility functions
│           ├── __init__.py
│           └── helpers.py
├── tests/                         # Test files
│   ├── unit_tests/
│   ├── integration_tests/
│   └── data/
├── notebooks/                     # Jupyter notebooks
├── docs/                          # Documentation
├── config/                        # Configuration files
├── bin/                           # Executable scripts
├── data/                          # Data files (gitignored)
├── secrets/                       # Sensitive files (gitignored)
├── pyproject.toml                 # Project configuration
├── requirements.txt               # Runtime dependencies
├── Makefile                       # Development commands
└── .pre-commit-config.yaml        # Pre-commit hooks
```

## Key Benefits of This Structure

- **Src Layout**: Prevents import confusion and follows Python best practices
- **Clear Separation**: Source code, tests, notebooks, and configuration are clearly separated
- **Scalable**: Easy to add new modules and packages
- **Standard**: Follows industry conventions

# Package Customization

The boilerplate makes it easy to customize the package name for your project:

## How It Works

- **Package Name**: The name you specify (e.g., "my-awesome-project")
- **Python Module**: Automatically converted (e.g., "my_awesome_project")

## Customization Commands

```bash
# Show current configuration
make customize-package

# Customize package name
make customize-package PACKAGE_NAME=my-ds-project

# Customize during setup
make setup PACKAGE_NAME=my-ds-project
```

## What Gets Updated

- `pyproject.toml`: Package name and module configuration
- `src/` directory: Renamed to match your module name
- All imports will work with the new module name

## Examples

| Package Name | Python Module | Import Statement |
|--------------|---------------|------------------|
| `my-project` | `my_project` | `from my_project import ...` |
| `awesome-ds` | `awesome_ds` | `from awesome_ds import ...` |
| `ml-pipeline` | `ml_pipeline` | `from ml_pipeline import ...` |

# Development Workflow

## Available Commands

```bash
# Setup development environment
make setup

# Install pre-commit hooks
make install_precommit

# Format and lint code
make format

# Run tests
make run_tests

# Serve documentation locally
make serve_docs_locally

# Deploy documentation
make deploy_docs
```

## Code Quality Tools

This boilerplate includes several tools to maintain code quality:

- **Pre-commit hooks**: Automatically format and lint code on commit
- **Ruff**: Fast Python linter and formatter
- **Bandit**: Security vulnerability detection
- **Pytest**: Testing framework
- **nbstripout**: Clean Jupyter notebook outputs

## Git Workflow

1. **Pre-commit hooks** automatically format and lint your code
2. **Tests run on push** to ensure code quality
3. **CI pipeline** validates code on GitHub
4. **Pull request template** helps with code reviews

# Documentation

## Local Development

```bash
# Serve documentation locally
make serve_docs_locally
```

This will serve the documentation at `http://localhost:8001`

## Publishing

The documentation is automatically built and deployed to GitHub Pages on each push to the `main` branch.

To manually deploy:
```bash
make deploy_docs
```

## Configuration

- **MkDocs**: Documentation generator
- **Material for MkDocs**: Beautiful theme
- **GitHub Pages**: Automatic deployment

# Adding Dependencies

## Runtime Dependencies

Add to `requirements.txt`:
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

## Development Dependencies

Add to `pyproject.toml` under `[project.optional-dependencies]`:
```toml
dev = [
    "pre-commit>=3.0.0",
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]
```

# Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `make run_tests`
5. Format code: `make format`
6. Submit a pull request

# License

This project is licensed under the MIT License - see the LICENSE file for details.
