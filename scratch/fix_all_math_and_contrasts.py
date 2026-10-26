import json
from pathlib import Path


def run_fixes():
    # ==========================================
    # 01-data-partitioning.ipynb
    # ==========================================
    p1 = Path("notebooks/01-data-partitioning.ipynb")
    with open(p1, encoding="utf-8") as f:
        nb1 = json.load(f)

    # Cell 5
    c5 = "".join(nb1["cells"][5]["source"])
    c5 = c5.replace(
        "We instantiate =10$ client partitions under IID, Dirichlet ($\alpha=0.5$), and Pathological ($ classes/client) allocations.",
        r"We instantiate $K=10$ client partitions under IID, Dirichlet ($\alpha=0.5$), and Pathological ($2$ classes/client) allocations.",
    )
    c5 = c5.replace(
        "We instantiate =10$ client partitions under IID, Dirichlet ($alpha=0.5$), and Pathological ($ classes/client) allocations.",
        r"We instantiate $K=10$ client partitions under IID, Dirichlet ($\alpha=0.5$), and Pathological ($2$ classes/client) allocations.",
    )
    nb1["cells"][5]["source"] = [line + "\n" for line in c5.splitlines()]

    # Cell 8: Heatmap contrasting palette & vertical colorbar
    nb1["cells"][8]["source"] = [
        "fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)\n",
        "\n",
        "datasets = [\n",
        '    ("IID (Uniform Baseline)", counts_iid),\n',
        '    ("Dirichlet Skew (alpha=0.5)", counts_dir),\n',
        '    ("Pathological (2 Classes/Node)", counts_path),\n',
        "]\n",
        "\n",
        "for ax, (title, df) in zip(axes, datasets):\n",
        "    data = df.values\n",
        '    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", interpolation="nearest")\n',
        '    ax.set_title(title.upper(), fontsize=12, fontweight="bold", pad=14)\n',
        '    ax.set_xlabel("CLASS ID (0-9)", fontsize=11, fontweight="bold", labelpad=10)\n',
        "    ax.set_xticks(range(10))\n",
        "    ax.set_yticks(range(NUM_CLIENTS))\n",
        '    ax.set_yticklabels([f"C{i:02d}" for i in range(NUM_CLIENTS)], fontsize=10, fontweight="bold")\n',
        "    ax.grid(False)\n",
        "    for spine in ax.spines.values():\n",
        "        spine.set_linewidth(1.3)\n",
        '        spine.set_edgecolor("#333333")\n',
        "\n",
        'axes[0].set_ylabel("CLIENT SILO ID", fontsize=11, fontweight="bold", labelpad=10)\n',
        'cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical", fraction=0.02, pad=0.03)\n',
        'cbar.set_label("SAMPLE COUNT PER CLASS", fontweight="bold", fontsize=11, labelpad=10)\n',
        "cbar.ax.tick_params(labelsize=10)\n",
        'cbar.outline.set_edgecolor("#333333")\n',
        "cbar.outline.set_linewidth(1.2)\n",
        "plt.show()\n",
    ]

    # Cell 9: Shannon entropy formula fix
    nb1["cells"][9]["source"] = [
        "### **B. Statistical Heterogeneity & Shannon Entropy Metric**\n",
        "\n",
        "Shannon entropy $H_i = -\\sum_{c=0}^9 p_{ic} \\log_2(p_{ic})$ measures partition diversity. Maximum theoretical entropy is $\\log_2(10) \\approx 3.322$ bits.\n",
    ]

    # Cell 10: Shannon entropy plot labels
    c10 = "".join(nb1["cells"][10]["source"])
    c10 = c10.replace(
        'label="Theoretical Max Entropy (3.32 bits)"',
        'label=r"Theoretical Max Entropy ($\\approx 3.32$ bits)"',
    )
    nb1["cells"][10]["source"] = [line + "\n" for line in c10.splitlines()]

    # Cell 13: Takeaways LaTeX
    c13 = "".join(nb1["cells"][13]["source"])
    c13 = c13.replace("( approx $3.32$ bits)", r"($\approx 3.32$ bits)")
    c13 = c13.replace("($alpha=0.5$)", r"($\alpha=0.5$)")
    nb1["cells"][13]["source"] = [line + "\n" for line in c13.splitlines()]

    with open(p1, "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=1)
    print("[✓] Fixed notebooks/01-data-partitioning.ipynb")

    # ==========================================
    # 04-dynamic-aggregation.ipynb
    # ==========================================
    p4 = Path("notebooks/04-dynamic-aggregation.ipynb")
    with open(p4, encoding="utf-8") as f:
        nb4 = json.load(f)

    c22 = "".join(nb4["cells"][22]["source"])
    c22 = c22.replace('cmap="viridis"', 'cmap="Blues"')
    c22 = c22.replace('cmap="gray"', 'cmap="Blues"')
    nb4["cells"][22]["source"] = [line + "\n" for line in c22.splitlines()]

    with open(p4, "w", encoding="utf-8") as f:
        json.dump(nb4, f, indent=1)
    print("[✓] Fixed notebooks/04-dynamic-aggregation.ipynb")


if __name__ == "__main__":
    run_fixes()
