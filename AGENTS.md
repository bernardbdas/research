# AGENTS Guidelines & Rules

## 1. Notebook Execution On Explicit User Permission Only (CRITICAL HARDCODED RULE)

**NEVER execute notebooks automatically! Only start execution when the user tells you
explicitly!**

- Notebook execution drains excessive time and laptop compute resources. Therefore,
  autonomous or background execution of notebooks (`.ipynb`) is strictly prohibited
  unless explicitly requested by the user.
- **Workflow**: Finish all code, documentation, mathematical formulation,
  visual/presentation, and structural changes first.
- Once all modifications are complete, ask the user whether more work needs to be done
  or if you should proceed with execution.
- If the user gives explicit permission to run, proceed with execution. Otherwise,
  inquire about pending tasks and complete them first.
- Never execute notebooks speculatively, preemptively, or prematurely.

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

## 7. Publication-Quality Plot Aesthetics & Palette Standards (HARDCODED RULE)

- **Strictly No Black & White or Hatched Plots**: Never produce grayscale, monochrome,
  or black-and-white hatched plots or graphs. Always use vivid, modern, aesthetically
  pleasing color palettes (e.g., Deep/Dark modern palettes, Viridis, Plasma, Seaborn
  deep, distinct contrasting hex colors like `#2b5c8f`, `#d95f02`, `#7570b3`, `#e7298a`,
  `#1b9e77`, `#2ca02c`, `#3498db`, `#e67e22`).
- **No Dark Backgrounds Under Dark/Bold Typography (Contrasting Colors)**: Never use
  dark colored backgrounds, dark heatmap colormaps (e.g., dark purple baselines in
  `viridis` when plotting sparse matrices), or dark shaded boxes when text/labels are
  bold and dark. Always use light, high-contrast backgrounds (e.g., `YlGnBu`, `Blues`,
  clean white backgrounds) so that dark brutalist typography has sharp, unambiguous
  contrast and readability.
- **Colorbar Placement & Collision Prevention**: Colorbars must never overlap or collide
  with subplot axis titles, ticks, or labels. Always place colorbars vertically on the
  right (`orientation="vertical"`, `fraction=0.02`, `pad=0.03`) or in dedicated axes
  with ample margins.
- **Plot Dimensions & Spacing**: Plots must be large, legible, and spacious — **never
  small or cramped**. Standard dimensions should be `figsize=(14, 6)` or larger for
  single plots, and `figsize=(16, 6)` to `figsize=(18, 6)` for multi-panel subplots
  (never use small sizes like `(7, 3.8)` or `(6.5, 3.2)`).
- **Typography & Labels**: Always provide descriptive axis labels and titles with
  generous spacing and padding (`labelpad=10`, title `pad=12` or more). Use clear,
  legible font weights and sizes (titles $\\ge 12-14$pt bold, axis labels $\\ge 10-12$pt
  bold, tick labels $\\ge 9-11$pt).
- **Legends & Grids**: Include high-contrast, clearly positioned legends with
  appropriate padding, and subtle, clean gridlines (`linestyle="--"`, `alpha=0.3` to
  `0.4`).
- **Mathematical Formulations in Markdown & Plots**: All LaTeX in titles, labels, or
  annotations must use raw Python strings (`r"..."`) with proper double-escaped
  backslashes where appropriate. All markdown math MUST use valid delimiters (`$...$` or
  `$$...$$`) with zero unescaped control characters (`\a`, `\b`, `\t`), balanced dollar
  signs, and properly enclosed variables (e.g., `$K=10$`, `$\alpha=0.5$`,
  `$\approx 3.32$ bits`).
