______________________________________________________________________

## trigger: always_on

# Notebook Authoring Guidelines & Personal Style

When creating or editing Jupyter Notebooks (`.ipynb`) in this repository:

1. **Hierarchy & Headings**:

   - Primary sections: `## **1. Title**`, `## **2. Title**`
   - Subsections: `### **A. Subtitle**`, `### **B. Subtitle**`
   - Use bold titles inside Markdown headers (e.g.
     `## **1. Datasets & Partitioning Regimes**`).
   - **NO Setup Heading**: Never add a `Setup & Environment` or `Prerequisites` markdown
     header at the top. The first two code cells are self-explanatory to human
     programmers. Numbering starts at `## **1. ...**` for the actual subject content.

1. **Clean & Concise Markdown (Zero Boilerplate)**:

   - Keep markdown cells short, punchy, and to the point.
   - Use clean Markdown tables for decision matrices, comparisons, and configs.
   - Avoid generic AI commentary, filler paragraphs, or verbose essay-like text.
   - Use `<font size="2" color="red">**Tip:** ...</font>` for high-value hints where
     applicable.

1. **Code Style & Library Idioms**:

   - Rely directly on library idioms and built-ins (e.g.,
     `flwr_datasets.metrics.compute_counts`, `compute_frequencies`,
     `flwr_datasets.visualization.plot_label_distributions`).
   - Avoid reinventing existing library utilities with bloated manual scripts.
   - Keep code cells focused: one logical operation per cell.
   - Display summaries or inspect outputs directly (`df.head()`, `summary_df`).

1. **No Custom Class Definitions in Notebooks (Strict Rule)**:

   - Never define custom helper classes, algorithms, cryptosystems, or model
     architectures directly inside `.ipynb` notebook cells.
   - All classes and reusable utilities must reside in their dedicated Python modules
     under the source packages (`federated/`, `models/`, `utils/`).
   - Notebooks must import these custom classes and functions individually from their
     implementations under the source tree.
   - Notebook cells should focus strictly on orchestrating workflows, executing
     experiments, inspecting outputs, and visualizing results.

1. **Environment & Imports**:

   - Cell 0 loads environment: `load_dotenv(find_dotenv("local.env"))`.
   - Cell 1 handles imports and enables progress bars: `enable_progress_bar()`.
   - Never write bloated boilerplate when a simple function gets the job done cleanly.

1. **Dataset Persistence**:

   - Whenever requesting or working with any dataset using `flwr_datasets` or Hugging
     Face `datasets`, ALL data must be saved and cached strictly within the local
     `data/` directory (`data/raw/` for raw cache and `data/processed/` for Parquet
     exports).
   - Use `get_federated_dataset()` or `prepare_and_save_dataset()` from
     `utils.data_processing`.
   - Never allow downloads to default to `~/.cache`.

1. **Notebook Execution On Explicit User Permission Only (CRITICAL HARDCODED RULE)**:

   - **DO NOT execute notebooks automatically!** Running notebooks drains excessive time
     and laptop compute resources.
   - Autonomous or speculative execution of notebooks (`.ipynb`) is strictly prohibited.
   - **Required Workflow**: Finish all code, documentation, mathematical formulation,
     visual/presentation, and structural changes first.
   - Once all edits are complete, ask the user whether more work needs to be done or if
     you should proceed with execution.
   - Only execute notebooks if the user gives explicit affirmative permission. If not,
     inquire about pending tasks and complete them first.

1. **Publication-Quality Plot Aesthetics & Palette Standards (HARDCODED RULE)**:

   - **Strictly No Black & White or Hatched Plots**: Never produce grayscale,
     monochrome, or hatched plots/graphs. Always use vivid, modern, aesthetically
     pleasing color palettes (e.g., Deep/Dark modern palettes, Viridis, Plasma, Seaborn
     deep, distinct contrasting hex colors like `#2b5c8f`, `#d95f02`, `#7570b3`,
     `#e7298a`, `#1b9e77`, `#2ca02c`, `#3498db`).
   - **No Dark Backgrounds Under Dark/Bold Typography (Contrasting Colors)**: Never use
     dark colored backgrounds, dark heatmap colormaps (e.g., dark purple in `viridis`
     for zero-valued matrix entries), or dark shaded boxes under dark/bold typography.
     Always use light, high-contrast backgrounds (e.g., `YlGnBu`, `Blues`, white
     backgrounds) so that dark brutalist typography has sharp, unambiguous contrast and
     readability.
   - **Colorbar Placement & Spacing**: Colorbars must never collide or overlap with axis
     titles, ticks, or labels. Always place colorbars vertically on the right side
     (`orientation="vertical"`, `fraction=0.02`, `pad=0.03`) or in dedicated sub-axes
     with ample margins.
   - **Plot Dimensions & Spacing**: Make plots bigger, never small or cramped! Minimum
     width $\\ge 12-14$, height $\\ge 5-6$ (e.g. `figsize=(14, 6)` or larger for single
     plots, `figsize=(16, 6)` to `figsize=(18, 6)` for multi-panel subplots). Never use
     cramped dimensions like `(6.5, 3.2)` or `(7, 3.8)`.
   - **Typography & Labels**: Always give proper, highly descriptive labels to all axes
     and plots with generous padding (`labelpad=10`, title `pad=12`). Use clear legible
     fonts (title $\\ge 12-14$pt bold, labels $\\ge 10-12$pt bold, ticks $\\ge 9-11$pt).
   - **Legends & Grids**: Ensure high contrast, clear legends with proper spacing, and
     subtle clean gridlines (`alpha=0.3` to `0.4`).
   - **Mathematical Expressions in Markdown & Plots**: Ensure all LaTeX mathematical
     expressions are properly formatted and escaped (e.g. `$\text{Dir}(\alpha)$`,
     `$\approx$`, `$K=10$`, `$\alpha=0.5$`, `$2$ classes`) so KaTeX and Matplotlib
     render them cleanly without syntax or escaping errors.
