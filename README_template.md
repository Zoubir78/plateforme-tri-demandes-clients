<div align="center">

# boilerplate-datascience-artefact

[![CI status](https://github.com/artefactory/boilerplate-datascience-artefact/actions/workflows/ci.yaml/badge.svg)](https://github.com/artefactory/boilerplate-datascience-artefact/actions/workflows/ci.yaml?query=branch%3Amain)
[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)]()

[![Linting , formatting, imports sorting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/artefactory/boilerplate-datascience-artefact/blob/main/.pre-commit-config.yaml)
</div>

TODO: if not done already, check out the [Skaff documentation](https://artefact.roadie.so/catalog/default/component/repo-builder-ds/docs/) for more information about the generated repository.

"Boilerplate for Artefact's Data Science team."

## Table of Contents

- [boilerplate-datascience-artefact](#boilerplate-datascience-artefact)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Documentation](#documentation)
  - [Repository Structure](#repository-structure)

## Installation

To install the required packages in a virtual environment, run the following command:

```bash
make setup
```

A complete list of available commands can be found using the following command:

```bash
make help
```

## Usage

TODO: Add usage instructions here

## Documentation

TODO: Github pages is not enabled by default, you need to enable it in the repository settings: Settings > Pages > Source: "Deploy from a branch" / Branch: "gh-pages" / Folder: "/(root)"

A detailed documentation of this project is available [here](https://artefactory.github.io/boilerplate-datascience-artefact/)

To serve the documentation locally, run the following command:

```bash
mkdocs serve
```

To build it and deploy it to GitHub pages, run the following command:

```bash
make deploy_docs
```

## Repository Structure

```
.
├── .github    <- GitHub Actions workflows and PR template
├── bin        <- Bash files
├── config     <- Configuration files
├── docs       <- Documentation files (mkdocs)
├── lib        <- Python modules
├── notebooks  <- Jupyter notebooks
├── secrets    <- Secret files (ignored by git)
└── tests      <- Unit tests
```
