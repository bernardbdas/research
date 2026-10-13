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

1. **Mandatory End-to-End Notebook Execution & Verification (Strict Rule)**:

   - Always execute the whole notebook from start to finish (e.g. using a headless test
     run or cell-by-cell execution) to verify that every cell runs with zero errors.
   - Never present modified notebooks to the user without validating that the code
     executes cleanly end-to-end.
   - If any exception, syntax error, missing argument, or runtime bug is detected, fix
     it autonomously and re-verify before finalizing the turn.
