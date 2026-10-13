# AGENTS Guidelines & Rules

## 1. Mandatory End-to-End Notebook Execution Rule (CRITICAL)

**Always run the entire notebook from start to finish and check for errors after
finishing any work!**

- Before presenting any modified or newly authored Jupyter notebook (`.ipynb`) to the
  user, you MUST execute all cells sequentially (e.g., using
  `scratch/execute_notebook.py` or cell-by-cell execution) to verify that the notebook
  runs cleanly with zero runtime exceptions, zero syntax errors, and zero missing
  imports or variables.
- If any error is encountered during execution, fix it autonomously and re-run until all
  cells pass with 100% success.
- Never present unfinished or unverified notebooks to the user.

## 2. Pluggable Backbones & Truthful Logging

- Notebooks must configure their model architecture through a pluggable backbone
  configuration at the top (e.g., `BACKBONE_NAME = "simple_cnn"` using
  `models.get_model()`).
- Never hardcode misleading architecture names (e.g., "VGG16" or "ResNet50") in print
  statements, table headers, or plot labels when a different backbone is running. Always
  dynamically query `active_model.__class__.__name__` or `BACKBONE_NAME`.

## 3. Approved Datasets

- Strictly use official datasets from **Flower Datasets** (`flwr_datasets`) or **Hugging
  Face Hub** (e.g., `ylecun/mnist`, `zalando-datasets/fashion_mnist`, `flwrlabs/usps`).
- Always persist and cache downloaded data within the repository under `data/raw/` or
  `data/processed/` using `utils.data_processing`. Never allow downloads to default to
  `~/.cache`.

## 4. Model Checkpointing

- All trained models must be checkpointed and serialized using
  `models.checkpointing.save_model_checkpoint` into the `checkpoints/` directory.
- The `checkpoints/` directory must remain ignored by git (in `.gitignore`) with
  `checkpoints/.gitkeep` preserving the directory structure.

## 5. Architectural Cleanliness & Zero Boilerplate

- Follow all guidelines in `.agents/rules/notebook-style.md`.
- No custom class definitions in notebook cells: all architectures, aggregators, and
  utilities must live under `models/`, `federated/`, and `utils/`.
