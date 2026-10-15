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

## 6. Mandatory Literature Citations for Technical Claims (RULE)

- Whenever asserting empirical behaviors, theoretical properties, computational
  complexity bounds, privacy guarantees, or architectural designs, ALWAYS cite
  authoritative literature with formal scholarly citations (Author(s), Year, Title,
  Publication Venue/arXiv).
- Examples:
  - Federated Averaging: McMahan et al. (2017), *Communication-Efficient Learning of
    Deep Networks from Decentralized Data*, AISTATS.
  - Non-IID Drift & Heterogeneity: Li et al. (2020), *Federated Optimization in
    Heterogeneous Networks (FedProx)*, MLSys; Karimireddy et al. (2020), *SCAFFOLD*,
    ICML.
  - Differential Privacy: Dwork et al. (2006), *Calibrating Noise to Sensitivity in
    Private Data Analysis*, TCC; Abadi et al. (2016), *Deep Learning with Differential
    Privacy*, ACM CCS.
  - Homomorphic Encryption: Paillier (1999), *Public-Key Cryptosystems Based on
    Composite Degree Residuosity Classes*, EUROCRYPT.
  - Secure Multi-Party Computation: Shamir (1979), *How to Share a Secret*, CACM;
    Bonawitz et al. (2017), *Practical Secure Aggregation for Privacy-Preserving Machine
    Learning*, ACM CCS.
  - Adaptive Optimization: Reddi et al. (2020), *Adaptive Federated Optimization
    (FedAdam)*, ICLR; Takezawa et al. (2025), *FedMuon: Federated Learning with
    Bias-Corrected Matrix Polar Orthogonalization*, arXiv.
  - Test-Time Training: Sun et al. (2020), *Test-Time Training with Self-Supervision for
    Generalization under Distribution Shifts*, ICML; Sun et al. (2024), *Learning to
    (Learn at Test Time): RNNs with Expressive Hidden States*, arXiv.
  - Neural Long-Term Memory / Titans: Behrouz et al. (2024), *Titans: Learning to
    Memorize at Test Time*, arXiv.
  - Continual & Class-Incremental Learning: Kirkpatrick et al. (2017), *Overcoming
    Catastrophic Forgetting in Neural Networks (EWC)*, PNAS; Rebuffi et al. (2017),
    *iCaRL: Incremental Classifier and Representation Learning*, CVPR.
- Never state claims as speculative or ungrounded opinions; ground all claims firmly in
  published peer-reviewed or established pre-print literature.
