"""Polish 04-dynamic-aggregation.ipynb with brutalist styling and Mermaid flowchart."""

import json

nb_path = "notebooks/04-dynamic-aggregation.ipynb"
with open(nb_path) as f:
    nb = json.load(f)

# Update Cell 2 (Title & Problem Statement)
nb["cells"][2]["source"] = [
    "## **1. [TARGET] Dynamic & Adaptive Aggregation under Heterogeneity (Gap 2)**\n",
    "\n",
    "> [!IMPORTANT]\n",
    "> **CENTRAL THESIS**: Static coordinate-wise federated averaging (FedAvg) fails under statistical heterogeneity, suffering severe client drift and objective inconsistency. Adaptive server optimization (FedAdam), proximal drift regularizers (FedProx), robust non-linear estimators (FedMedian, FedTrimmedMean), and matrix polar orthogonalization (FedMuon) mitigate drift while trading off computational complexity and cryptographic compatibility.\n",
    ">\n",
    "> **TARGET METRIC**: Quantify empirical convergence, Pareto complexity, and real-time inference accuracy on target sequence `[2, 4, 9, 0, 0, 3, 1]` across Dirichlet Non-IID partitions ($\\alpha=0.5$).\n",
    ">\n",
    "> **SCHOLARLY CITATIONS**:\n",
    "> - Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). *Federated Optimization in Heterogeneous Networks (FedProx)*. MLSys.\n",
    "> - Yin, D., Chen, Y., Kannan, R., & Bartlett, P. (2018). *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates*. ICML.\n",
    "> - Reddi, S., et al. (2020). *Adaptive Federated Optimization (FedAdam)*. ICLR.\n",
    "> - Takezawa, Y., et al. (2025). *FedMuon: Federated Learning with Bias-Corrected Matrix Polar Orthogonalization*. arXiv.\n",
]

# Insert Mermaid diagram in markdown cell after Cell 3
mermaid_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## **2. [MECHANICS] Aggregation Taxonomy & Server Update Dynamics**\n",
        "\n",
        "```mermaid\n",
        "flowchart TD\n",
        '    A["Decentralized Client Updates: Δw_1 .. Δw_K"] --> B{"Server Aggregation Engine"}\n',
        '    B -->|"Static Linear Baseline"| C["FedAvg: w <- w + sum(p_k * Δw_k)"]\n',
        '    B -->|"Proximal Drift Constraint"| D["FedProx: Loss + mu/2 ||w - w_t||^2"]\n',
        '    B -->|"Nonlinear Order Statistics"| E["FedMedian / FedTrimmedMean"]\n',
        '    B -->|"Second-Moment Adaptive Scaling"| F["FedAdam: m_t / (sqrt(v_t) + eps)"]\n',
        '    B -->|"Matrix Polar Orthogonalization"| G["FedMuon: Quintic Newton-Schulz"]\n',
        '    C --> H["Empirical Validation: Slide 14 Dashboard & Sequence [2,4,9,0,0,3,1]"]\n',
        "    D --> H\n",
        "    E --> H\n",
        "    F --> H\n",
        "    G --> H\n",
        "```\n",
    ],
}

# Insert after Cell 3
nb["cells"].insert(4, mermaid_cell)

# Update headings
for c in nb["cells"]:
    if c["cell_type"] == "markdown":
        src = "".join(c["source"])
        if "## **2. Statistical Heterogeneity" in src:
            c["source"] = [
                "## **3. [BENCHMARK] Statistical Heterogeneity & Dirichlet Partitioning**\n",
                "\n",
                "Simulating non-IID client partitions with $\\alpha=0.5$ across MNIST and USPS.\n",
            ]
        elif "## **3. The Federated Aggregation" in src:
            c["source"] = [
                "## **4. [TAXONOMY] Aggregator Complexity & Cryptographic Compatibility**\n",
                "\n",
                src.split("---")[-1].strip() + "\n",
            ]
        elif "## **4. Federated Simulation" in src:
            c["source"] = [
                "## **5. [EXPERIMENT] Multi-Round Federated Simulation Engine**\n",
                "\n",
                "Executing multi-round federated training with sub-second checkpoint caching.\n",
            ]
        elif "## **5. Performance Comparison" in src:
            c["source"] = [
                "## **6. [VISUAL ANALYTICS] Brutalist Convergence & Complexity Frontiers**\n",
                "\n",
                "Tracking convergence trajectories, loss stability, and computational complexity.\n",
            ]
        elif "## **6. Real-Time Sample" in src:
            c["source"] = [
                "## **7. [VALIDATION] Real-Time Inference: Target Sequence `[2, 4, 9, 0, 0, 3, 1]`**\n",
                "\n",
                "Evaluating class predictions and probability calibrations on target sequence.\n",
            ]
        elif "## **7. Slide-Ready Comprehensive Dashboard" in src:
            c["source"] = [
                "## **8. [PRESENTATION] Slide 14 Presentation Dashboard (Gap 2 Solution)**\n",
                "\n",
                "Comprehensive high-impact figure for presentation slide.\n",
            ]
        elif "### **Key Takeaways & Resolution" in src:
            c["source"] = [
                "## **9. [VERDICT] Empirical Performance Matrix & Failure Modes**\n",
                "\n",
                src,
            ]

with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

print("✓ Successfully polished 04-dynamic-aggregation.ipynb")
