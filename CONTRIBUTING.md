# Contributing to this Research Repository

Thank you for your interest in contributing to this research repository! This project is
maintained for **educational, academic research, and collaborative open-source
learning** purposes.

______________________________________________________________________

## Code of Conduct

All contributors are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).
Please report any concerns by opening an issue on GitHub or reaching out to the
maintainer on [X (Twitter)](https://x.com/bernardbdas) or
[LinkedIn](https://linkedin.com/in/bernardbdas).

______________________________________________________________________

## Development Environment Setup

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic Python
environment management.

### 1. Prerequisites

- Python 3.12+ (Python 3.13 recommended)
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh` or
  `brew install uv`)
- `just` command runner (`brew install just`)

### 2. Clone and Install Dependencies

```bash
git clone https://github.com/bernardbdas/research.git
cd research

# Install all dependencies and set up pre-commit hooks via justfile
just setup

# Activate the virtual environment
source .venv/bin/activate
```

______________________________________________________________________

## Coding Standards & Tooling

We enforce strict formatting and linting rules configured in `pyproject.toml` and
automated via `justfile`:

- **Code Quality Check**:

  ```bash
  just check
  ```

- **Linting & Auto-Fixing**:

  ```bash
  just lint-fix
  ```

- **Formatting (Code, Notebooks, Markdown)**:

  ```bash
  just format
  ```

- **Pre-Commit Validation**:

  ```bash
  just pre-commit
  ```

______________________________________________________________________

## Guidelines for Models & Datasets

### 1. Data-Agnostic Model Backbones (`models/`)

- All backbones defined in `models/` should remain **data-agnostic** (avoid hardcoding
  dataset-specific dimensions).
- Use **Global Average Pooling** (`jnp.mean(x, axis=(1, 2))`) before classification
  heads to handle variable spatial resolutions.
- Verify new backbones with `just test-models`.

### 2. Data Pipeline (`utils/data_processing.py`)

- **Do not commit dataset files** to git. Raw data caches belong in `data/raw/` and
  processed Parquet files in `data/processed/` (both are git-ignored).
- Use `prepare_and_save_dataset()` or `just prepare-data [dataset] [val_ratio]` to
  prepare datasets into Parquet with distinct `train`, `val`, and `test` splits.
- For Flower partition experiments, use `get_federated_dataset_from_parquet()` for
  zero-copy local loading.

### 3. Notebooks (`notebooks/`)

- Clear heavy cell outputs before committing notebooks.
- Ensure all notebook cells run sequentially from top to bottom.

______________________________________________________________________

## Submitting Pull Requests

1. **Fork** the repository and create a feature branch
   (`git checkout -b feat/your-feature`).

1. Implement your changes following our coding and design guidelines.

1. Ensure all tests and pre-commit checks pass:

   ```bash
   just check
   just pre-commit
   ```

1. Commit your changes with clear, descriptive commit messages.

1. Push to your branch and open a **Pull Request**.

______________________________________________________________________

## License Notice

By contributing to this repository, you agree that your contributions will be licensed
under the project's [PolyForm Noncommercial License 1.0.0](LICENSE) for non-commercial
educational and research use.
