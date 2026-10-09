# Suppress recipe command echoing globally (show only program output)
set quiet

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

# List supported benchmark datasets, optionally filtered by modality (vision, audio, tabular, text, multimodal)
list-datasets modality="":
    uv run python -c "\
    from utils import DATASET_CATALOG; \
    mod = '{{modality}}'.strip().lower(); \
    cats = {mod: DATASET_CATALOG[mod]} if mod in DATASET_CATALOG else DATASET_CATALOG; \
    print('\n'.join(f'[{k.upper()}] ({len(v)} datasets):\n' + '\n'.join(f'  - {ds}' for ds in v) for k, v in cats.items()))"

# Download and preprocess a dataset into data/raw and data/processed in Parquet format
prepare-data dataset="cifar10" val_ratio="0.1" force="false":
    uv run python -c "from utils import prepare_and_save_dataset; prepare_and_save_dataset('{{dataset}}', val_ratio={{val_ratio}}, force='{{force}}')"

# Download and preprocess multiple datasets by space-separated list (e.g. just prepare-datasets "mnist cifar100 femnist")
prepare-datasets datasets val_ratio="0.1":
    @for d in {{datasets}}; do just prepare-data $$d {{val_ratio}}; done

# Download and preprocess all datasets for a given modality (vision, audio, tabular, text, multimodal)
prepare-modality modality="vision" val_ratio="0.1" force="false":
    uv run python -c "from utils import prepare_modality_datasets; prepare_modality_datasets('{{modality}}', val_ratio={{val_ratio}}, force='{{force}}')"

# Download and preprocess core benchmark datasets (cifar10, mnist) into data/
prepare-core-data:
    @just prepare-data cifar10 0.1
    @just prepare-data mnist 0.1

# Download and preprocess all benchmark datasets across all modalities into data/
prepare-all-data val_ratio="0.1":
    @just prepare-modality vision {{val_ratio}}
    @just prepare-modality audio {{val_ratio}}
    @just prepare-modality tabular {{val_ratio}}
    @just prepare-modality text {{val_ratio}}
    @just prepare-modality multimodal {{val_ratio}}

# Clean temporary caches, bytecode, and build artifacts
clean:
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name ".ruff_cache" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +
    find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
    find . -name "*.pyc" -delete
    find . -name ".DS_Store" -delete
    @echo "[✓] Caches cleaned."

# Clean raw download cache from data/raw/
clean-raw-data:
    uv run python -c "from utils import clean_raw_data; clean_raw_data()"

# Clean processed Parquet files from data/processed/
clean-processed-data:
    uv run python -c "from utils import clean_processed_data; clean_processed_data()"

# Delete cached datasets (all or a specific dataset) to reclaim disk space
clean-data dataset="":
    uv run python -c "from utils import clean_dataset_data; clean_dataset_data('{{dataset}}')"

# Clean all data for a specific modality (vision, audio, tabular, text, multimodal)
clean-modality modality="vision":
    uv run python -c "from utils import clean_modality_data; clean_modality_data('{{modality}}')"

# Clean all bytecode, caches, AND all downloaded dataset files
clean-all: clean clean-data
