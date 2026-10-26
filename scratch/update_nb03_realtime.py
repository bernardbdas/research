import json
from pathlib import Path


def update_nb03():
    p = Path("notebooks/03-client-drift.ipynb")
    with open(p, encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Cell 4 imports
    c4 = "".join(nb["cells"][4]["source"])
    if "plot_sample_classifications" not in c4:
        c4 = c4.replace(
            "    predict_with_confidence,\n",
            "    predict_with_confidence,\n    plot_sample_classifications,\n",
        )
        nb["cells"][4]["source"] = [line + "\n" for line in c4.splitlines()]

    # 2. Update Cell 9: Remove "Brutalist" and "Plots"
    nb["cells"][9]["source"] = [
        "## **5. [VISUAL ANALYTICS] Convergence Dynamics & Drift Penalty**\n",
        "\n",
        "### **A. Convergence Dynamics Across Partition Regimes (MNIST & Fashion-MNIST)**\n",
    ]

    # 3. Update Cell 13: Clean section 6 header matching image 1
    nb["cells"][13]["source"] = [
        "## **6. [VALIDATION] Real-Time Sample Classifications: Sequence [2, 4, 9, 0, 0, 3, 1]**\n",
        "\n",
        "We test all 9 models on the specified target digit sequence `[2, 4, 9, 0, 0, 3, 1]` extracted from the test split.\n",
    ]

    # 4. Insert or update the in-situ visual classification cells after Cell 14
    # Cell 14 is the code that generates seq_df
    # We add Cell 14b for the in-situ visual plots
    visual_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-visual-seq-mnist",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Real-time inference on MNIST for Target Sequence [2, 4, 9, 0, 0, 3, 1] across all 9 models\n",
            "mnist_model_preds = {}\n",
            "regimes = [('iid', 'IID'), ('dirichlet_alpha_0.5', 'Dirichlet'), ('pathological_2_classes', 'Pathological')]\n",
            "aggregators = [('fedavg', 'FedAvg'), ('fedprox', 'FedProx'), ('fedmedian', 'FedMedian')]\n",
            "\n",
            "for part_tag, reg_label in regimes:\n",
            "    for strat_tag, agg_label in aggregators:\n",
            '        _, p = unpack_history_and_params(results["mnist"][part_tag][strat_tag])\n',
            "        preds, confs, _ = predict_with_confidence(model_inst, p, seq_x)\n",
            '        mnist_model_preds[f"{reg_label} [{agg_label}]"] = (preds, confs)\n',
            "\n",
            "fig = plot_sample_classifications(\n",
            "    images=seq_x,\n",
            "    true_labels=seq_y,\n",
            "    model_predictions=mnist_model_preds,\n",
            '    class_names=get_class_names("mnist"),\n',
            "    num_samples=7,\n",
            "    is_cifar=False,\n",
            '    title="MNIST: Real-Time Sample Classifications Across Regimes & Aggregation Strategies",\n',
            "    figsize=(18, 6.2),\n",
            "    fontsize=7.5,\n",
            ")\n",
            "plt.show()\n",
        ],
    }

    visual_cell_fashion = {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-visual-seq-fashion",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Real-time inference on Fashion-MNIST across aggregators & regimes\n",
            'fashion_test = load_processed_dataset("zalando-datasets/fashion_mnist", split="test")\n',
            "fash_x, fash_y = extract_sequence_samples(fashion_test, TARGET_SEQ)\n",
            "\n",
            "fashion_model_preds = {}\n",
            "for part_tag, reg_label in regimes:\n",
            "    for strat_tag, agg_label in aggregators:\n",
            '        _, p = unpack_history_and_params(results["fashion-mnist"][part_tag][strat_tag])\n',
            "        preds, confs, _ = predict_with_confidence(model_inst, p, fash_x)\n",
            '        fashion_model_preds[f"{reg_label} [{agg_label}]"] = (preds, confs)\n',
            "\n",
            "fig = plot_sample_classifications(\n",
            "    images=fash_x,\n",
            "    true_labels=fash_y,\n",
            "    model_predictions=fashion_model_preds,\n",
            '    class_names=get_class_names("fashion_mnist"),\n',
            "    num_samples=7,\n",
            "    is_cifar=False,\n",
            '    title="Fashion-MNIST: Real-Time Sample Classifications Across Regimes & Aggregation Strategies",\n',
            "    figsize=(18, 6.2),\n",
            "    fontsize=7.5,\n",
            ")\n",
            "plt.show()\n",
        ],
    }

    visual_cell_usps = {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-visual-seq-usps",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Real-time inference on USPS postal digits across aggregators & regimes\n",
            'usps_test = load_processed_dataset("flwrlabs/usps", split="test")\n',
            "usps_x, usps_y = extract_sequence_samples(usps_test, TARGET_SEQ)\n",
            "\n",
            "usps_model_preds = {}\n",
            "for part_tag, reg_label in regimes:\n",
            "    for strat_tag, agg_label in aggregators:\n",
            '        _, p = unpack_history_and_params(results["usps"][part_tag][strat_tag])\n',
            "        preds, confs, _ = predict_with_confidence(model_inst, p, usps_x)\n",
            '        usps_model_preds[f"{reg_label} [{agg_label}]"] = (preds, confs)\n',
            "\n",
            "fig = plot_sample_classifications(\n",
            "    images=usps_x,\n",
            "    true_labels=usps_y,\n",
            "    model_predictions=usps_model_preds,\n",
            '    class_names=get_class_names("usps"),\n',
            "    num_samples=7,\n",
            "    is_cifar=False,\n",
            '    title="USPS: Real-Time Sample Classifications Across Regimes & Aggregation Strategies",\n',
            "    figsize=(18, 6.2),\n",
            "    fontsize=7.5,\n",
            ")\n",
            "plt.show()\n",
        ],
    }

    # Reconstruct cells
    # Check if visual cells already exist
    existing_ids = [c.get("id") for c in nb["cells"]]
    if "cell-visual-seq-mnist" not in existing_ids:
        # Insert right after Cell 14
        nb["cells"].insert(15, visual_cell)
        nb["cells"].insert(16, visual_cell_fashion)
        nb["cells"].insert(17, visual_cell_usps)

    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(
        "[✓] Successfully updated notebooks/03-client-drift.ipynb with in-situ real-time visualizations!"
    )


if __name__ == "__main__":
    update_nb03()
