import glob
import json
from pathlib import Path


def purge_meta_terms():
    # 1. Clean all notebooks
    for nb_path in sorted(glob.glob("notebooks/*.ipynb")):
        with open(nb_path, encoding="utf-8") as f:
            nb = json.load(f)

        modified = False
        for cell in nb.get("cells", []):
            new_source = []
            for line in cell.get("source", []):
                old_line = line
                # Replace meta labels in markdown headers
                line = line.replace(
                    "Brutalist Convergence & Drift Penalty Plots",
                    "Convergence Trajectories & Drift Penalty Dynamics",
                )
                line = line.replace(
                    "Brutalist Convergence & Complexity Frontiers",
                    "Convergence Trajectories & Complexity Frontiers",
                )
                line = line.replace(
                    "Brutalist Privacy vs. Utility Pareto Frontier",
                    "Privacy vs. Utility Pareto Frontier",
                )
                line = line.replace(
                    "Brutalist Trilemma Pareto & Overhead Visualizations",
                    "Privacy Trilemma Pareto & Latency Visualizations",
                )
                line = line.replace(
                    "Brutalist Convergence & Singular Value Dynamics",
                    "Convergence Dynamics & Singular Value Spectra",
                )
                line = line.replace(
                    "Brutalist Convergence Visualizations (Slide 39 Reproduction)",
                    "Convergence Dynamics & Optimizer Comparison (Slide 39 Reproduction)",
                )
                line = line.replace(
                    "Brutalist Retention Matrix & Forgetting Dynamics",
                    "Task Retention Matrix & Catastrophic Forgetting Dynamics",
                )
                line = line.replace(
                    "Brutalist Retention Heatmaps & Forgetting Comparison",
                    "Task Retention Heatmaps & Forgetting Comparison",
                )
                line = line.replace(
                    "Publication-Grade Brutalist Plots",
                    "Statistical Heterogeneity Visual Analytics",
                )

                # Replace comments in code
                line = line.replace(
                    "# Brutalist Matplotlib Theme", "# Plotting Configuration"
                )
                line = line.replace(
                    "# Brutalist plotting defaults", "# Plotting theme defaults"
                )

                # Catch generic occurrences
                line = line.replace("Brutalist ", "")
                line = line.replace("brutalist ", "")
                line = line.replace(" Brutalist", "")
                line = line.replace(" brutalist", "")

                if line != old_line:
                    modified = True
                new_source.append(line)
            cell["source"] = new_source

        if modified:
            with open(nb_path, "w", encoding="utf-8") as f:
                json.dump(nb, f, indent=1)
            print(f"[✓] Purged meta styling terms from {nb_path}")

    # 2. Clean README.md
    readme_path = Path("notebooks/README.md")
    if readme_path.exists():
        text = readme_path.read_text(encoding="utf-8")
        text = text.replace(
            "Executed in a brutalist, bold publication graphic style.",
            "Executed in a publication-grade, high-contrast visual and mathematical style.",
        )
        text = text.replace(
            "Each notebook in this curriculum adheres to a strict, high-contrast, publication-grade brutalist aesthetic:",
            "Each notebook in this curriculum adheres to strict, publication-grade aesthetic standards:",
        )
        readme_path.write_text(text, encoding="utf-8")
        print("[✓] Updated notebooks/README.md")


if __name__ == "__main__":
    purge_meta_terms()
