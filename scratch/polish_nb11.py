"""Polish 11-class-incremental-continual-learning.ipynb with brutalist layout and Mermaid flowchart."""

import json

nb_path = "notebooks/11-class-incremental-continual-learning.ipynb"
with open(nb_path) as f:
    nb = json.load(f)

env_cell = None
for c in nb["cells"]:
    if c["cell_type"] == "code" and "load_dotenv" in "".join(c.get("source", [])):
        env_cell = c
        break

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

target_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## **1. [TARGET] Continual Vision Transfer & The Stability-Plasticity Dilemma**\n",
        "\n",
        "> [!IMPORTANT]\n",
        "> **CENTRAL THESIS**: When deep models are adapted sequentially across non-stationary task streams without access to past data, gradient updates on new tasks overwrite orthogonal weight subspaces, causing catastrophic forgetting of previously acquired knowledge ($R_{T, i} \\ll R_{i, i}$). While exemplar memory replay mitigates forgetting, storing raw historical exemplars on edge clients violates data privacy. We quantify the **Stability-Plasticity Dilemma** via Backward Transfer ($BWT$) and Forward Transfer ($FWT$).\n",
        ">\n",
        "> **TARGET METRIC**: Quantify empirical forgetting $\\Delta_{\\text{forget}} = R_{1,1} - R_{3,1}$ across 3 sequential vision tasks under Naive Fine-Tuning vs. Exemplar Memory Replay.\n",
        ">\n",
        "> **SCHOLARLY CITATIONS**:\n",
        "> - Kirkpatrick, J., Pascanu, R., et al. (2017). *Overcoming Catastrophic Forgetting in Neural Networks (EWC)*. PNAS.\n",
        "> - Rebuffi, S. A., Kolesnikov, A., et al. (2017). *iCaRL: Incremental Classifier and Representation Learning*. CVPR.\n",
        "> - De Lange, M., Aljundi, R., et al. (2021). *A Continual Learning Survey: Defying Forgetting in Classification Tasks*. IEEE TPAMI.\n",
    ],
}

mermaid_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## **2. [MECHANICS] Class-Incremental Learning & Retention Matrix Pipeline**\n",
        "\n",
        "```mermaid\n",
        "flowchart TD\n",
        '    A["Task Stream Arrival: Task 1 (Classes 0-3)"] --> B["Task 2 (Classes 4-6)"]\n',
        '    B --> C["Task 3 (Classes 7-9)"]\n',
        '    A --> D["Head Expansion: Expand W_out from C_1 to C_1+C_2+C_3"]\n',
        "    B --> D\n",
        "    C --> D\n",
        '    D --> E{"Continual Paradigm"}\n',
        '    E -->|"Naive Fine-Tuning"| F["Unconstrained SGD: Overwrites Past Task Subspaces (Severe Forgetting)"]\n',
        '    E -->|"Exemplar Replay"| G["Buffer Storage: Interleaves Past Samples (Privacy & Memory Cost)"]\n',
        '    F --> H["Task Evaluation Matrix R_{i,j}: Measure BWT and Final Accuracy"]\n',
        "    G --> H\n",
        "```\n",
    ],
}

clean_cells = [env_cell, prog_cell, target_cell, mermaid_cell]
for c in nb["cells"]:
    if c == env_cell:
        continue
    if c["cell_type"] == "markdown":
        src = "".join(c.get("source", []))
        if "## **1. Environment" in src or "# **ImageNet Transfer" in src:
            continue
        elif "## **2. Pure JAX/Flax Visual" in src:
            c["source"] = [
                "## **3. [ARCHITECTURE] Pure JAX/Flax Visual Feature Extractor**\n",
                "\n",
                "Extracting deep representations with frozen convolutional trunks.\n",
            ]
        elif "## **3. Dataset Ingestion" in src:
            c["source"] = [
                "## **4. [BENCHMARK] Sequential Class-Incremental Task Protocol**\n",
                "\n",
                "Partitioning CIFAR-10 into 3 disjoint incremental tasks.\n",
            ]
        elif "## **4. Feature Embedding" in src:
            c["source"] = [
                "## **5. [EXECUTION] Precomputing Task Feature Embeddings**\n",
                "\n",
                "Precomputing invariant representations for fast execution.\n",
            ]
        elif "## **5. Dynamic Class-Incremental" in src:
            c["source"] = [
                "## **6. [ARCHITECTURE] Dynamic Classification Head Expansion**\n",
                "\n",
                "Dynamically expanding output logits as new classes arrive.\n",
            ]
        elif "## **6. Optimization & Evaluation" in src:
            c["source"] = [
                "## **7. [OPTIMIZATION] JIT-Compiled Training & Multi-Task Evaluation Routines**\n",
                "\n",
                "Constructing pure JAX step functions.\n",
            ]
        elif "## **7. Continual Learning Paradigm 1" in src:
            c["source"] = [
                "## **8. [BENCHMARK] Paradigm 1: Naive Sequential Fine-Tuning**\n",
                "\n",
                "Simulating sequential adaptation without memory replay.\n",
            ]
        elif "## **8. Continual Learning Paradigm 2" in src:
            c["source"] = [
                "## **9. [BENCHMARK] Paradigm 2: Exemplar Memory Replay**\n",
                "\n",
                "Simulating replay buffer preservation.\n",
            ]
        elif "## **9. Continual Learning Quantitative" in src:
            c["source"] = [
                "## **10. [VISUAL ANALYTICS] Brutalist Retention Matrix & Forgetting Dynamics**\n",
                "\n",
                "Heatmaps of retention matrices $R_{i,j}$ and accuracy progression.\n",
            ]
        elif "## **10. Final Summary" in src:
            c["source"] = [
                "## **11. [VERDICT] Stability-Plasticity Empirical Performance Matrix**\n",
                "\n",
                src.split("---")[-1].strip() + "\n",
            ]
    clean_cells.append(c)

nb["cells"] = clean_cells
with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

print("✓ Successfully polished 11-class-incremental-continual-learning.ipynb")
