SHELL := /bin/bash

PYTHON_VERSION := 3.11

# Package customization - users can override this
PACKAGE_NAME ?= boilerplate-datascience

.DEFAULT_GOAL = help

# help: help					- Display this makefile's help information
.PHONY: help
help:
	@grep "^# help\:" Makefile | grep -v grep | sed 's/\# help\: //' | sed 's/\# help\://'

# help: customize-package			- Customize package name
.PHONY: customize-package
customize-package:
	@echo "🔧 Customizing package name..."
	@echo "Current package name: $(PACKAGE_NAME)"
	@echo "Python module name: $(shell echo $(PACKAGE_NAME) | tr '-' '_')"
	@echo ""
	@echo "To customize, run:"
	@echo "  make customize-package PACKAGE_NAME=my-awesome-project"
	@echo ""
	@if [ "$(PACKAGE_NAME)" != "boilerplate-datascience" ]; then \
		echo "🔄 Updating package configuration..."; \
		sed -i.bak 's/name = "boilerplate-datascience"/name = "$(PACKAGE_NAME)"/' pyproject.toml; \
		sed -i.bak 's/packages = \["boilerplate_datascience"\]/packages = ["$(shell echo $(PACKAGE_NAME) | tr '-' '_')"]/' pyproject.toml; \
		mv src/boilerplate_datascience src/$(shell echo $(PACKAGE_NAME) | tr '-' '_'); \
		echo "✅ Package customized to: $(PACKAGE_NAME)"; \
		echo "✅ Python module: $(shell echo $(PACKAGE_NAME) | tr '-' '_')"; \
	else \
		echo "ℹ️  Using default package name. Run with PACKAGE_NAME to customize."; \
	fi

# help: setup					- Create a virtual environment and install dependencies
.PHONY: setup
setup: customize-package create-venv install-dev-requirements

ENV_NAME := $(shell echo $(notdir $(CURDIR)) | sed 's/^[0-9]*-//' | tr '[:upper:]' '[:lower:]')

.PHONY : create-venv, install-dev-requirements, install-requirements
create-venv:
	@if ! pyenv virtualenvs | grep -q $(ENV_NAME); then pyenv virtualenv $(ENV_NAME); \
	else echo "virtualenv already exists"; fi
	# Set virtualenv as local
	@pyenv local $(ENV_NAME)
	@echo "✅ Virtualenv $(ENV_NAME) created and set as local"

install-dev-requirements:
	@pip install --upgrade pip --quiet
	@pip install -r requirements.txt
	@pip install -e ".[dev]"
	@echo "✅ Requirements installed"

install-requirements:
	@pip install --upgrade pip --quiet
	@pip install -r requirements.txt
	@pip install .
	@echo "✅ Requirements installed"

# help: install_precommit			- Install pre-commit hooks
.PHONY: install_precommit
install_precommit:
	@pre-commit install -t pre-commit
	@pre-commit install -t pre-push

# help: format			- format code using the precommits
.PHONY: format
format:
	@pre-commit run -a

# help: serve_docs_locally			- Serve docs locally on port 8001
.PHONY: serve_docs_locally
serve_docs_locally:
	@mkdocs serve --livereload -a localhost:8001

# help: deploy_docs				- Deploy documentation to GitHub Pages
.PHONY: deploy_docs
deploy_docs:
	@mkdocs build
	@mkdocs gh-deploy

# help: run_tests			- Run repository's tests
.PHONY: run_tests
run_tests:
	@pytest tests/
