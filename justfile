# Default recipe: display available commands
default:
    @just --list

# Set up environment and install pre-commit hooks
setup:
    uv sync --dev
    uv run pre-commit install
    @echo "[✓] Environment and pre-commit hooks set up successfully."

# Run Ruff linter on all Python scripts and Jupyter notebooks
lint:
    uv run ruff check .

# Automatically fix lint issues across Python scripts and Jupyter notebooks
lint-fix:
    uv run ruff check . --fix

# Format Python files, Jupyter notebooks, and Markdown documentation
format:
    uv run ruff format .
    uv run mdformat README.md CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md
    @echo "[✓] Code, notebooks, and markdown formatted."

# Check formatting without writing changes
format-check:
    uv run ruff format --check .
    uv run mdformat --check README.md CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md

# Run full code quality check (lint + format check)
check: lint format-check

# Run all pre-commit hooks manually on all files
pre-commit:
    uv run pre-commit run --all-files

# Re-render all PlantUML architecture diagrams to high-resolution SVG
diagrams:
    plantuml -tsvg diagrams/**/*.puml
    @echo "[✓] All diagrams re-rendered to SVG."

# Verify all Flax model backbones (forward pass & parameter count)
test-models:
    uv run python -c "\
    import jax, jax.numpy as jnp; \
    from models import get_model, list_models, count_parameters; \
    dummy = jnp.ones((2, 32, 32, 3)); \
    key = jax.random.PRNGKey(0); \
    print('Testing models:'); \
    [print(f'  {m:<12} -> {count_parameters(get_model(m).init(key, dummy, train=False)):,} params') for m in list_models()]; \
    print('[✓] All models initialized successfully.')"

# Download and preprocess a dataset into data/raw and data/processed in Parquet format
prepare-data dataset="cifar10" val_ratio="0.1":
    uv run python -c "\
    from dotenv import load_dotenv, find_dotenv; \
    load_dotenv(find_dotenv('local.env')); \
    from utils import prepare_and_save_dataset; \
    prepare_and_save_dataset('{{dataset}}', val_ratio={{val_ratio}})"

# Clean temporary caches, bytecode, and build artifacts
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name ".ruff_cache" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
    find . -name "*.pyc" -delete
    find . -name ".DS_Store" -delete
    @echo "[✓] Caches cleaned."
