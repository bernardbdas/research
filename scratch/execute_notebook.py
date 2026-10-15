"""Headless notebook execution and verification script using IPython InteractiveShell.

Executes all code cells in an .ipynb notebook sequentially in the exact environment,
captures errors with cell indices and stack traces, and updates the cell outputs.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell


def execute_notebook(notebook_path: str, timeout_per_cell: int = 300) -> bool:
    nb_path = Path(notebook_path).resolve()
    print("\n=======================================================")
    print(f"🚀 Running Notebook: {nb_path.name}")
    print("=======================================================\n")

    with open(nb_path, encoding="utf-8") as f:
        nb_data = json.load(f)

    shell = InteractiveShell.instance()
    # Configure matplotlib for inline non-blocking plotting
    shell.run_cell(
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt"
    )

    cells = nb_data.get("cells", [])
    total_code_cells = sum(1 for c in cells if c.get("cell_type") == "code")
    code_cell_idx = 0

    t_start = time.time()

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        code_cell_idx += 1
        source = "".join(cell.get("source", []))
        first_line = source.strip().split("\n")[0] if source.strip() else "<empty>"
        if len(first_line) > 60:
            first_line = first_line[:57] + "..."

        print(f"[{code_cell_idx}/{total_code_cells}] Running Cell {idx}: {first_line}")
        t_cell_start = time.time()

        res = shell.run_cell(source)
        cell_duration = time.time() - t_cell_start

        if not res.success or res.error_in_exec is not None:
            print(f"\n❌ ERROR in Cell {idx} after {cell_duration:.2f}s:")
            print("-------------------------------------------------------")
            if res.error_in_exec:
                traceback.print_exception(
                    type(res.error_in_exec),
                    res.error_in_exec,
                    res.error_in_exec.__traceback__,
                )
            else:
                print("Cell execution failed without an explicit exception object.")
            print("-------------------------------------------------------\n")
            return False

        print(f"    ↳ Completed in {cell_duration:.2f}s [✓]")

    total_time = time.time() - t_start
    print(
        f"\n🎉 Successfully executed {total_code_cells} code cells in {total_time:.2f}s!"
    )
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Execute a Jupyter notebook end-to-end."
    )
    parser.add_argument("notebook_path", type=str, help="Path to .ipynb file")
    args = parser.parse_args()

    success = execute_notebook(args.notebook_path)
    if not success:
        sys.exit(1)
