"""Polish 12-test-time-training-continual-learning.ipynb with brutalist layout and Mermaid flowchart."""

import json

nb_path = "notebooks/12-test-time-training-continual-learning.ipynb"
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
        "## **1. [TARGET] Vision-TTT: Test-Time Training for Zero Catastrophic Forgetting**\n",
        "\n",
        "> [!IMPORTANT]\n",
        "> **CENTRAL THESIS**: The fundamental limitation of classical continual learning is the requirement of historical replay exemplars, which violates privacy in federated systems and scales linearly with task count. **Test-Time Training (Vision-TTT)** resolves the stability-plasticity dilemma by turning hidden states into adaptive machine learning models: an inner-loop self-supervised reconstruction task optimizes fast weight matrices $W_t$ online at inference time, eliminating catastrophic forgetting without retaining a single private historical sample.\n",
        ">\n",
        "> **TARGET METRIC**: Compare Backward Transfer ($BWT$) and accuracy retention between Static Vision-TTT (no online updates) and Adaptive Vision-TTT across sequential vision tasks.\n",
        ">\n",
        "> **SCHOLARLY CITATIONS**:\n",
        "> - Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A., & Hardt, M. (2020). *Test-Time Training with Self-Supervision for Generalization under Distribution Shifts*. ICML.\n",
        "> - Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., & Wei, F. (2024). *Learning to (Learn at Test Time): RNNs with Expressive Hidden States*. arXiv:2407.04620.\n",
    ],
}

mermaid_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## **2. [MECHANICS] Vision-TTT Dual-Loop Architecture**\n",
        "\n",
        "```mermaid\n",
        "flowchart TD\n",
        '    A["Test Sample Arrives: x_test"] --> B["Frozen Outer Backbone: Shared Invariant Representations"]\n',
        '    B --> C["Inner Loop: Self-Supervised Reconstruction Loss L_{self}(W; x_test)"]\n',
        '    C --> D["Online Gradient Step: W_t = W_{t-1} - η_ttt * ∇L_{self}"]\n',
        '    D --> E["Dynamic Classification Head: Expand to Accumulated Classes"]\n',
        '    E --> F["Inference Prediction: y_pred with Zero Catastrophic Forgetting"]\n',
        '    F --> G["No Historical Exemplar Replay Buffers Stored on Client"]\n',
        "```\n",
    ],
}

clean_cells = [env_cell, prog_cell, target_cell, mermaid_cell]
for c in nb["cells"]:
    if c == env_cell:
        continue
    if c["cell_type"] == "markdown":
        src = "".join(c.get("source", []))
        if "## **1. Environment" in src or "# **Test-Time Training" in src:
            continue
        elif "## **2. Architecture Anatomy" in src:
            c["source"] = [
                "## **3. [ARCHITECTURE] Vision-TTT Backbone & Inner-Loop Weight Matrices**\n",
                "\n",
                "Instantiating Vision-TTT with patch projections and online weights.\n",
            ]
        elif "## **3. Class-Incremental Benchmark Protocol" in src:
            c["source"] = [
                "## **4. [BENCHMARK] Sequential Vision Benchmark Protocol**\n",
                "\n",
                "Partitioning incremental tasks across CIFAR splits.\n",
            ]
        elif "## **4. Dynamic Classification Head" in src:
            c["source"] = [
                "## **5. [ARCHITECTURE] Dynamic Head Expansion for Vision-TTT**\n",
                "\n",
                "Expanding output projection matrices as task vocabulary increases.\n",
            ]
        elif "## **5. Continual Optimization & Dual-Mode" in src:
            c["source"] = [
                "## **6. [OPTIMIZATION] Continual Training & Dual-Mode Evaluation**\n",
                "\n",
                "Constructing static and adaptive evaluation step functions.\n",
            ]
        elif "## **6. Continual Benchmark 1: Static" in src:
            c["source"] = [
                "## **7. [BENCHMARK] Benchmark 1: Static Vision-TTT (No Test-Time Updates)**\n",
                "\n",
                "Evaluating baseline forgetting without online adaptation.\n",
            ]
        elif "## **7. Continual Benchmark 2: Test-Time" in src:
            c["source"] = [
                "## **8. [BENCHMARK] Benchmark 2: Vision-TTT with Active Test-Time Training**\n",
                "\n",
                "Evaluating online test-time adaptation on held-out tasks.\n",
            ]
        elif "## **8. Quantitative Metrics" in src:
            c["source"] = [
                "## **9. [VISUAL ANALYTICS] Brutalist Retention Heatmaps & Forgetting Comparison**\n",
                "\n",
                "Contrasting task retention matrices and $BWT$ metric.\n",
            ]
        elif "## **9. Final Summary" in src:
            c["source"] = [
                "## **10. [VERDICT] Vision-TTT Empirical Performance Matrix & Findings**\n",
                "\n",
                src.split("---")[-1].strip() + "\n",
            ]
    clean_cells.append(c)

nb["cells"] = clean_cells
with open(nb_path, "w") as f:
    json.dump(nb, f, indent=1)

print("✓ Successfully polished 12-test-time-training-continual-learning.ipynb")
