"""Polish 09-fedmuon-matrix-optimization.ipynb with brutalist layout, Mermaid flowchart, and publication styling."""

import json

nb_path = "notebooks/09-fedmuon-matrix-optimization.ipynb"
with open(nb_path) as f:
    nb = json.load(f)

# Find cell with load_dotenv
env_cell = None
for c in nb["cells"]:
    if c["cell_type"] == "code" and "load_dotenv" in "".join(c.get("source", [])):
        env_cell = c
        break

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
prog_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from datasets.utils.logging import enable_progress_bar\n",
        "enable_progress_bar()\n",
    ],
}

# Update matplotlib styling in code
for c in code_cells:
    src = "".join(c.get("source", []))
    if "plt.subplots" in src or "plt.figure" in src:
        c["source"] = [
            "plt.rcParams.update({\n",
            '    "font.family": "sans-serif",\n',
            '    "font.weight": "bold",\n',
            '    "axes.edgecolor": "#111111",\n',
            '    "axes.linewidth": 1.3,\n',
            '    "grid.color": "#555555",\n',
            '    "grid.linestyle": "--",\n',
            '    "grid.alpha": 0.35,\n',
            '    "figure.autolayout": True,\n',
            "})\n",
        ] + c["source"]
        break

# Target markdown cell
target_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## **1. [TARGET] Matrix Polar Orthogonalization & Bias-Corrected FedMuon**\n",
        "\n",
        "> [!IMPORTANT]\n",
        "> **CENTRAL THESIS**: Coordinate-wise Euclidean averaging (FedAvg) treats matrix weights as independent scalars, ignoring the non-convex geometry of matrix manifolds. The **Muon** optimizer uses a Linear Minimization Oracle (LMO) over the spectral norm via quintic Newton-Schulz iterations: $X_{k+1} = X_k (a I + b X_k^T X_k + c (X_k^T X_k)^2)$. However, because the LMO is non-linear $\\text{lmo}(G_1 + G_2) \\neq \\text{lmo}(G_1) + \\text{lmo}(G_2)$, running Muon naively on clients (**LocalMuon**) causes severe operator bias and divergence. **FedMuon** (Takezawa et al., 2025) introduces control variates to provably cancel drift while maintaining matrix orthogonalization.\n",
        ">\n",
        "> **TARGET METRIC**: Prove singular value collapse to $\\sigma_i \\equiv 1.0$ under 5 polynomial iterations and benchmark convergence of FedMuon vs LocalMuon vs FedAvg on non-IID CIFAR-10.\n",
        ">\n",
        "> **SCHOLARLY CITATIONS**:\n",
        "> - Takezawa, Y., Bao, H., Niwa, K., & Sato, I. (2025). *FedMuon: Federated Learning with Bias-Corrected Matrix Polar Orthogonalization*. arXiv:2502.14815.\n",
        "> - Higham, N. J. (2008). *Functions of Matrices: Theory and Computation*. Society for Industrial and Applied Mathematics (SIAM).\n",
        "> - Bernstein, J., et al. (2018). *signSGD: Compressed Distributed SGD with Majority Vote*. ICML.\n",
    ],
}

mermaid_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## **2. [MECHANICS] Quintic Newton-Schulz LMO Pipeline**\n",
        "\n",
        "```mermaid\n",
        "flowchart TD\n",
        '    A["Client Local Gradient Matrix: G_i in R^{d x m}"] --> B["Spectral Norm Normalization: X_0 = G_i / (||G_i||_F + eps)"]\n',
        '    B --> C["Quintic Newton-Schulz Iteration: X_{k+1} = X_k(a*I + b*X_k^T*X_k + c*(X_k^T*X_k)^2)"]\n',
        '    C --> D["Polar Orthogonal Projection: All Singular Values Collapse to 1.0"]\n',
        '    D --> E{"Bias Correction Engine"}\n',
        '    E -->|"LocalMuon (Uncorrected)"| F["Operator Drift: Error Proportional to Non-IID Skew"]\n',
        '    E -->|"FedMuon (Algorithm 1)"| G["Control Variates (M_i - C_i + C) Provably Eliminate Drift"]\n',
        "```\n",
    ],
}

# Clean existing markdown headers to follow brutalist numbering
clean_cells = [env_cell, prog_cell, target_cell, mermaid_cell]
for c in nb["cells"]:
    if c == env_cell:
        continue
    if c["cell_type"] == "markdown":
        src = "".join(c.get("source", []))
        if "## **1. Environment Setup" in src or "# **FedMuon" in src:
            continue
        elif "## **2. Mathematical Verification" in src:
            c["source"] = [
                "## **3. [VERIFICATION] Mathematical Verification of the Newton-Schulz LMO**\n",
                "\n",
                "Verifying the quintic polynomial singular value collapse without computing full SVD.\n",
            ]
        elif "## **3. Non-IID Dataset Partitioning" in src:
            c["source"] = [
                "## **4. [BENCHMARK] Non-IID Dirichlet Dataset Partitioning (Heterogeneity Setup)**\n",
                "\n",
                "Partitioning CIFAR-10 with $\\alpha=0.5$ across 5 clients.\n",
            ]
        elif "## **4. Algorithmic Formulation" in src:
            c["source"] = [
                "## **5. [ALGORITHM] Algorithmic Formulation: Client-Side FedMuon (Algorithm 1)**\n",
                "\n",
                "Implementing bias-corrected client momentum updates.\n",
            ]
        elif "## **5. Comparative Benchmark Simulation" in src:
            c["source"] = [
                "## **6. [EXPERIMENT] Multi-Round Comparative Benchmark Simulation**\n",
                "\n",
                "Simulating 5 communication rounds across FedAvg, LocalMuon, FedMuon, and Server-Side FedMuon.\n",
            ]
        elif "## **6. Comparative Evaluation" in src:
            c["source"] = [
                "## **7. [VISUAL ANALYTICS] Brutalist Convergence & Singular Value Dynamics**\n",
                "\n",
                "Comparing convergence trajectories and final-round test accuracy.\n",
            ]
        elif "## **7. Quantitative Performance Matrix" in src:
            c["source"] = [
                "## **8. [VERDICT] FedMuon Empirical Performance Matrix & Failure Modes**\n",
                "\n",
                "Structured performance synthesis.\n",
            ]
        elif "## **8. Academic Conclusion" in src:
            c["source"] = [
                "### **Academic Findings & PhD Research Takeaways**:\n",
                "\n",
                src.split("---")[-1].strip() + "\n",
            ]
    clean_cells.append(c)

nb["cells"] = clean_cells
with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

print("✓ Successfully polished 09-fedmuon-matrix-optimization.ipynb")
