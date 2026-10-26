import json
from pathlib import Path


def fix_all():
    # ----------------------------------------------------
    # Notebook 01: Data Partitioning
    # ----------------------------------------------------
    p1 = Path("notebooks/01-data-partitioning.ipynb")
    with open(p1, encoding="utf-8") as f:
        nb1 = json.load(f)

    # Cell 2: LaTeX fix
    c2 = "".join(nb1["cells"][2]["source"])
    c2 = c2.replace("\x07lpha", "\\alpha")
    c2 = c2.replace("\text{Dir}(", "\\text{Dir}(")
    c2 = c2.replace("\t", "\\t")  # if any
    c2 = c2.replace("(=2$ classes per client)", "($2$ classes per client)")
    # ensure exact clean formula
    c2 = c2.replace(
        "Dirichlet Label Skew ($	ext{Dir}( lpha)$ for $ lpha \\in \\{10.0, 0.5, 0.1\\}$)",
        r"Dirichlet Label Skew ($\text{Dir}(\alpha)$ for $\alpha \in \{10.0, 0.5, 0.1\}$)",
    )
    c2 = c2.replace(
        "Dirichlet Label Skew ($\text{Dir}(\alpha)$ for $\alpha \\in \\{10.0, 0.5, 0.1\\}$)",
        r"Dirichlet Label Skew ($\text{Dir}(\alpha)$ for $\alpha \in \{10.0, 0.5, 0.1\}$)",
    )
    nb1["cells"][2]["source"] = [line + "\n" for line in c2.splitlines()]

    # Cell 4: Theme
    c4 = "".join(nb1["cells"][4]["source"])
    c4 = c4.replace(
        "# Brutalist plotting defaults: sharp, high-contrast monochrome",
        "# Publication-grade plotting theme: spacious, high-contrast, bold typography",
    )
    c4 = c4.replace('"axes.edgecolor": "#111111"', '"axes.edgecolor": "#333333"')
    c4 = c4.replace('"grid.color": "#555555"', '"grid.color": "#cccccc"')
    nb1["cells"][4]["source"] = [line + "\n" for line in c4.splitlines()]

    # Cell 5: LaTeX fix
    c5 = "".join(nb1["cells"][5]["source"])
    c5 = c5.replace("\x07lpha", "\\alpha")
    c5 = c5.replace(
        "We instantiate =10$ client partitions under IID, Dirichlet ($ lpha=0.5$), and Pathological ($ classes/client) allocations.",
        r"We instantiate $K=10$ client partitions under IID, Dirichlet ($\alpha=0.5$), and Pathological ($2$ classes/client) allocations.",
    )
    nb1["cells"][5]["source"] = [line + "\n" for line in c5.splitlines()]

    # Cell 8: Heatmap colors & size
    c8 = "".join(nb1["cells"][8]["source"])
    c8 = c8.replace("figsize=(18, 5)", "figsize=(18, 6)")
    c8 = c8.replace('cmap="Greys"', 'cmap="viridis"')
    c8 = c8.replace('"#111111"', '"#333333"')
    nb1["cells"][8]["source"] = [line + "\n" for line in c8.splitlines()]

    # Cell 9: LaTeX fix
    c9 = "".join(nb1["cells"][9]["source"])
    c9 = c9.replace("\x07pprox", "\\approx")
    c9 = c9.replace(
        "Shannon entropy  = -\\sum_{c=0}^9 p_{ic} \\log_2(p_{ic})$ measures partition diversity. Maximum theoretical entropy is $\\log_2(10)  pprox 3.322$ bits.",
        r"Shannon entropy $H_i = -\sum_{c=0}^9 p_{ic} \log_2(p_{ic})$ measures partition diversity. Maximum theoretical entropy is $\log_2(10) \approx 3.322$ bits.",
    )
    nb1["cells"][9]["source"] = [line + "\n" for line in c9.splitlines()]

    # Cell 10: Bar chart colors, remove hatches, enlarge
    c10 = "".join(nb1["cells"][10]["source"])
    c10 = c10.replace("figsize=(10, 4.5)", "figsize=(16, 6)")
    c10 = c10.replace('color="#111111"', 'color="#2b5c8f"')
    c10 = c10.replace(
        'color="#777777", edgecolor="black", linewidth=1.2, hatch="//"',
        'color="#d95f02", edgecolor="black", linewidth=1.2',
    )
    c10 = c10.replace(
        'color="#E0E0E0", edgecolor="black", linewidth=1.2, hatch="xx"',
        'color="#7570b3", edgecolor="black", linewidth=1.2',
    )
    c10 = c10.replace(
        'color="black", linestyle=":", linewidth=1.5, label="Theoretical Max Entropy (3.32 bits)"',
        'color="#e7298a", linestyle=":", linewidth=2.0, label="Theoretical Max Entropy (3.32 bits)"',
    )
    c10 = c10.replace(
        'color="black", linestyle="--", linewidth=1.5, label="2-Class Ceiling (1.00 bit)"',
        'color="#1b9e77", linestyle="--", linewidth=2.0, label="2-Class Ceiling (1.00 bit)"',
    )
    nb1["cells"][10]["source"] = [line + "\n" for line in c10.splitlines()]

    # Cell 13: LaTeX fix
    c13 = "".join(nb1["cells"][13]["source"])
    c13 = c13.replace("\x07pprox", "\\approx")
    c13 = c13.replace("\x07lpha", "\\alpha")
    c13 = c13.replace("(  pprox 3.32$ bits)", r"($\approx 3.32$ bits)")
    c13 = c13.replace("($ lpha=0.5$)", r"($\alpha=0.5$)")
    c13 = c13.replace("($ classes)", r"($2$ classes)")
    c13 = c13.replace("exactly .00$ bit", r"exactly $1.00$ bit")
    nb1["cells"][13]["source"] = [line + "\n" for line in c13.splitlines()]

    with open(p1, "w", encoding="utf-8") as f:
        json.dump(nb1, f, indent=1)
    print("[✓] Cleaned notebooks/01-data-partitioning.ipynb")

    # ----------------------------------------------------
    # Notebook 02: Federation Penalty
    # ----------------------------------------------------
    p2 = Path("notebooks/02-federation-penalty.ipynb")
    with open(p2, encoding="utf-8") as f:
        nb2 = json.load(f)

    # Cell 10: Size and colors
    c10 = "".join(nb2["cells"][10]["source"])
    c10 = c10.replace("figsize=(15, 5)", "figsize=(16, 6)")
    c10 = c10.replace('color="#111111"', 'color="#2b5c8f"')
    c10 = c10.replace('color="#777777"', 'color="#d95f02"')
    nb2["cells"][10]["source"] = [line + "\n" for line in c10.splitlines()]

    # Cell 12: Bar chart size and colors, remove hatch
    c12 = "".join(nb2["cells"][12]["source"])
    c12 = c12.replace("figsize=(7, 4.2)", "figsize=(14, 6)")
    c12 = c12.replace(
        'color=["#111111", "#666666"], edgecolor="black", linewidth=1.3, width=0.45, hatch="//"',
        'color=["#2b5c8f", "#d95f02"], edgecolor="black", linewidth=1.3, width=0.45',
    )
    nb2["cells"][12]["source"] = [line + "\n" for line in c12.splitlines()]

    # Cell 16: Aggregator trajectory colors & size
    c16 = "".join(nb2["cells"][16]["source"])
    c16 = c16.replace("figsize=(15, 5)", "figsize=(16, 6)")
    old_styles = """styles = {
    "FedAvg": ("#111111", "-", "o"),
    "FedProx (mu=0.01)": ("#444444", "--", "s"),
    "FedMedian": ("#777777", ":", "^"),
    "FedTrimmedMean (beta=0.1)": ("#999999", "-.", "v"),
    "FedAdam (eta=0.05)": ("#111111", "-", "*"),
}"""
    new_styles = """styles = {
    "FedAvg": ("#2b5c8f", "-", "o"),
    "FedProx (mu=0.01)": ("#d95f02", "--", "s"),
    "FedMedian": ("#7570b3", ":", "^"),
    "FedTrimmedMean (beta=0.1)": ("#1b9e77", "-.", "v"),
    "FedAdam (eta=0.05)": ("#e7298a", "-", "*"),
}"""
    c16 = c16.replace(old_styles, new_styles)
    nb2["cells"][16]["source"] = [line + "\n" for line in c16.splitlines()]

    with open(p2, "w", encoding="utf-8") as f:
        json.dump(nb2, f, indent=1)
    print("[✓] Cleaned notebooks/02-federation-penalty.ipynb")

    # ----------------------------------------------------
    # Notebook 03: Client Drift
    # ----------------------------------------------------
    p3 = Path("notebooks/03-client-drift.ipynb")
    with open(p3, encoding="utf-8") as f:
        nb3 = json.load(f)

    # Cell 10: Size and colors
    c10 = "".join(nb3["cells"][10]["source"])
    c10 = c10.replace("figsize=(18, 4.8)", "figsize=(18, 6)")
    c10 = c10.replace(
        '("fedavg", ("#111111", "-")), ("fedprox", ("#555555", "--")), ("fedmedian", ("#888888", ":"))',
        '("fedavg", ("#2b5c8f", "-")), ("fedprox", ("#d95f02", "--")), ("fedmedian", ("#7570b3", ":"))',
    )
    nb3["cells"][10]["source"] = [line + "\n" for line in c10.splitlines()]

    # Cell 12: Bar chart size and colors
    c12 = "".join(nb3["cells"][12]["source"])
    c12 = c12.replace("figsize=(9, 4.5)", "figsize=(14, 6)")
    c12 = c12.replace('color="#111111"', 'color="#d95f02"')
    c12 = c12.replace(
        'color="#E0E0E0", edgecolor="black", linewidth=1.2, hatch="//"',
        'color="#e7298a", edgecolor="black", linewidth=1.2',
    )
    nb3["cells"][12]["source"] = [line + "\n" for line in c12.splitlines()]

    with open(p3, "w", encoding="utf-8") as f:
        json.dump(nb3, f, indent=1)
    print("[✓] Cleaned notebooks/03-client-drift.ipynb")

    # ----------------------------------------------------
    # Notebook 04: Dynamic Aggregation
    # ----------------------------------------------------
    p4 = Path("notebooks/04-dynamic-aggregation.ipynb")
    with open(p4, encoding="utf-8") as f:
        nb4 = json.load(f)

    c22 = "".join(nb4["cells"][22]["source"])
    c22 = c22.replace('cmap="gray"', 'cmap="viridis"')
    nb4["cells"][22]["source"] = [line + "\n" for line in c22.splitlines()]

    with open(p4, "w", encoding="utf-8") as f:
        json.dump(nb4, f, indent=1)
    print("[✓] Cleaned notebooks/04-dynamic-aggregation.ipynb")

    # ----------------------------------------------------
    # Notebook 05: Differential Privacy
    # ----------------------------------------------------
    p5 = Path("notebooks/05-differential-privacy.ipynb")
    with open(p5, encoding="utf-8") as f:
        nb5 = json.load(f)

    c8 = "".join(nb5["cells"][8]["source"])
    c8 = c8.replace("figsize=(15, 5)", "figsize=(16, 6)")
    old_dp_styles = """styles = {
    "Baseline (No Privacy)": ("#111111", "-", "s"),
    "Moderate DP (sigma=0.8)": ("#555555", "--", "o"),
    "Strict DP (sigma=1.5)": ("#888888", ":", "^"),
}"""
    new_dp_styles = """styles = {
    "Baseline (No Privacy)": ("#2b5c8f", "-", "s"),
    "Moderate DP (sigma=0.8)": ("#d95f02", "--", "o"),
    "Strict DP (sigma=1.5)": ("#7570b3", ":", "^"),
}"""
    c8 = c8.replace(old_dp_styles, new_dp_styles)
    nb5["cells"][8]["source"] = [line + "\n" for line in c8.splitlines()]

    with open(p5, "w", encoding="utf-8") as f:
        json.dump(nb5, f, indent=1)
    print("[✓] Cleaned notebooks/05-differential-privacy.ipynb")

    # ----------------------------------------------------
    # Notebook 06: Homomorphic Encryption
    # ----------------------------------------------------
    p6 = Path("notebooks/06-homomorphic-encryption.ipynb")
    with open(p6, encoding="utf-8") as f:
        nb6 = json.load(f)

    # Cell 8: Size and colors
    c8 = "".join(nb6["cells"][8]["source"])
    c8 = c8.replace("figsize=(15, 4.8)", "figsize=(16, 6)")
    c8 = c8.replace('color="#111111"', 'color="#2b5c8f"')
    c8 = c8.replace('color="#888888"', 'color="#d95f02"')
    c8 = c8.replace('color="#222222"', 'color="#7570b3"')
    nb6["cells"][8]["source"] = [line + "\n" for line in c8.splitlines()]

    # Cell 11: LaTeX fix
    c11 = "".join(nb6["cells"][11]["source"])
    c11 = c11.replace("\x07pprox", "\\approx")
    c11 = c11.replace(
        "($ pprox 32\\times$ communication byte overhead)",
        r"($\approx 32\times$ communication byte overhead)",
    )
    nb6["cells"][11]["source"] = [line + "\n" for line in c11.splitlines()]

    with open(p6, "w", encoding="utf-8") as f:
        json.dump(nb6, f, indent=1)
    print("[✓] Cleaned notebooks/06-homomorphic-encryption.ipynb")

    # ----------------------------------------------------
    # Notebook 07: SMPC
    # ----------------------------------------------------
    p7 = Path("notebooks/07-secure-multi-party-computation.ipynb")
    with open(p7, encoding="utf-8") as f:
        nb7 = json.load(f)

    c8 = "".join(nb7["cells"][8]["source"])
    c8 = c8.replace("figsize=(15, 4.8)", "figsize=(16, 6)")
    c8 = c8.replace(
        'color="#111111", linewidth=2.2, linestyle="-"',
        'color="#2b5c8f", linewidth=2.2, linestyle="-"',
    )
    c8 = c8.replace(
        'color="#888888", linewidth=1.5, linestyle=":", marker="x"',
        'color="#d95f02", linewidth=1.8, linestyle=":", marker="x"',
    )
    c8 = c8.replace(
        'color="#111111", linewidth=2.5, linestyle="-"',
        'color="#2b5c8f", linewidth=2.5, linestyle="-"',
    )
    c8 = c8.replace(
        'color="#666666", linewidth=2.0, linestyle="--", marker="o"',
        'color="#1b9e77", linewidth=2.0, linestyle="--", marker="o"',
    )
    nb7["cells"][8]["source"] = [line + "\n" for line in c8.splitlines()]

    with open(p7, "w", encoding="utf-8") as f:
        json.dump(nb7, f, indent=1)
    print("[✓] Cleaned notebooks/07-secure-multi-party-computation.ipynb")

    # ----------------------------------------------------
    # Notebook 08: Hybrid Privacy Benchmark
    # ----------------------------------------------------
    p8 = Path("notebooks/08-hybrid-privacy-benchmark.ipynb")
    with open(p8, encoding="utf-8") as f:
        nb8 = json.load(f)

    c8 = "".join(nb8["cells"][8]["source"])
    c8 = c8.replace("figsize=(15, 4.8)", "figsize=(16, 6)")
    old_hybrid_styles = """styles = {
    "Baseline FedAvg (No Privacy)": ("#111111", "-", "s"),
    "Pure DP-SGD (σ=0.8)": ("#777777", ":", "v"),
    "Pure SecAgg (Zero-Sum)": ("#333333", "--", "o"),
    "Hybrid (DP + SecAgg)": ("#111111", "-.", "*"),
}"""
    new_hybrid_styles = """styles = {
    "Baseline FedAvg (No Privacy)": ("#2b5c8f", "-", "s"),
    "Pure DP-SGD (σ=0.8)": ("#d95f02", ":", "v"),
    "Pure SecAgg (Zero-Sum)": ("#7570b3", "--", "o"),
    "Hybrid (DP + SecAgg)": ("#1b9e77", "-.", "*"),
}"""
    c8 = c8.replace(old_hybrid_styles, new_hybrid_styles)
    c8 = c8.replace(
        'color=["#111111", "#555555", "#888888", "#E0E0E0"], edgecolor="black", linewidth=1.2, width=0.5, hatch="//"',
        'color=["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"], edgecolor="black", linewidth=1.2, width=0.5',
    )
    nb8["cells"][8]["source"] = [line + "\n" for line in c8.splitlines()]

    with open(p8, "w", encoding="utf-8") as f:
        json.dump(nb8, f, indent=1)
    print("[✓] Cleaned notebooks/08-hybrid-privacy-benchmark.ipynb")

    # ----------------------------------------------------
    # Notebook 09: FedMuon Matrix Optimization
    # ----------------------------------------------------
    p9 = Path("notebooks/09-fedmuon-matrix-optimization.ipynb")
    with open(p9, encoding="utf-8") as f:
        nb9 = json.load(f)

    # Cell 7
    c7 = "".join(nb9["cells"][7]["source"]).replace(
        "figsize=(12, 4)", "figsize=(14, 6)"
    )
    nb9["cells"][7]["source"] = [line + "\n" for line in c7.splitlines()]

    # Cell 11
    c11 = "".join(nb9["cells"][11]["source"]).replace(
        "figsize=(9, 4.5)", "figsize=(14, 6)"
    )
    nb9["cells"][11]["source"] = [line + "\n" for line in c11.splitlines()]

    # Cell 24
    c24 = "".join(nb9["cells"][24]["source"]).replace(
        "figsize=(14, 4.8)", "figsize=(16, 6)"
    )
    nb9["cells"][24]["source"] = [line + "\n" for line in c24.splitlines()]

    # Cell 25
    c25 = "".join(nb9["cells"][25]["source"]).replace(
        "figsize=(8, 4)", "figsize=(14, 6)"
    )
    nb9["cells"][25]["source"] = [line + "\n" for line in c25.splitlines()]

    with open(p9, "w", encoding="utf-8") as f:
        json.dump(nb9, f, indent=1)
    print("[✓] Cleaned notebooks/09-fedmuon-matrix-optimization.ipynb")

    # ----------------------------------------------------
    # Notebook 10: FedM3 Multimomentum Benchmark
    # ----------------------------------------------------
    p10 = Path("notebooks/10-fedm3-multimomentum-benchmark.ipynb")
    with open(p10, encoding="utf-8") as f:
        nb10 = json.load(f)

    # Cell 12
    c12 = "".join(nb10["cells"][12]["source"]).replace(
        "figsize=(9, 4)", "figsize=(14, 6)"
    )
    nb10["cells"][12]["source"] = [line + "\n" for line in c12.splitlines()]

    # Cell 16
    c16 = "".join(nb10["cells"][16]["source"]).replace(
        "figsize=(13, 5)", "figsize=(16, 6)"
    )
    c16 = c16.replace('"FedAvg": "#4A5568"', '"FedAvg": "#2b5c8f"')
    nb10["cells"][16]["source"] = [line + "\n" for line in c16.splitlines()]

    with open(p10, "w", encoding="utf-8") as f:
        json.dump(nb10, f, indent=1)
    print("[✓] Cleaned notebooks/10-fedm3-multimomentum-benchmark.ipynb")

    # ----------------------------------------------------
    # Notebook 11: Class Incremental Continual Learning
    # ----------------------------------------------------
    p11 = Path("notebooks/11-class-incremental-continual-learning.ipynb")
    with open(p11, encoding="utf-8") as f:
        nb11 = json.load(f)

    c19 = "".join(nb11["cells"][19]["source"]).replace(
        "figsize=(18, 5)", "figsize=(18, 6)"
    )
    nb11["cells"][19]["source"] = [line + "\n" for line in c19.splitlines()]

    with open(p11, "w", encoding="utf-8") as f:
        json.dump(nb11, f, indent=1)
    print("[✓] Cleaned notebooks/11-class-incremental-continual-learning.ipynb")

    # ----------------------------------------------------
    # Notebook 12: Test-Time Training Continual Learning
    # ----------------------------------------------------
    p12 = Path("notebooks/12-test-time-training-continual-learning.ipynb")
    with open(p12, encoding="utf-8") as f:
        nb12 = json.load(f)

    c17 = "".join(nb12["cells"][17]["source"]).replace(
        "figsize=(18, 5)", "figsize=(18, 6)"
    )
    nb12["cells"][17]["source"] = [line + "\n" for line in c17.splitlines()]

    with open(p12, "w", encoding="utf-8") as f:
        json.dump(nb12, f, indent=1)
    print("[✓] Cleaned notebooks/12-test-time-training-continual-learning.ipynb")

    # ----------------------------------------------------
    # Notebook 13: Titans Neural Memory Benchmark
    # ----------------------------------------------------
    p13 = Path("notebooks/13-titans-neural-memory-benchmark.ipynb")
    with open(p13, encoding="utf-8") as f:
        nb13 = json.load(f)

    # Cell 9
    c9 = "".join(nb13["cells"][9]["source"]).replace(
        "figsize=(7, 3.8)", "figsize=(14, 6)"
    )
    nb13["cells"][9]["source"] = [line + "\n" for line in c9.splitlines()]

    # Cell 14
    c14 = "".join(nb13["cells"][14]["source"]).replace(
        "figsize=(12, 5)", "figsize=(15, 6)"
    )
    nb13["cells"][14]["source"] = [line + "\n" for line in c14.splitlines()]

    # Cell 21
    c21 = "".join(nb13["cells"][21]["source"]).replace(
        "figsize=(13, 4.5)", "figsize=(16, 6)"
    )
    nb13["cells"][21]["source"] = [line + "\n" for line in c21.splitlines()]

    # Cell 22
    c22 = "".join(nb13["cells"][22]["source"]).replace(
        "figsize=(7, 3.8)", "figsize=(14, 6)"
    )
    nb13["cells"][22]["source"] = [line + "\n" for line in c22.splitlines()]

    # Cell 26
    c26 = "".join(nb13["cells"][26]["source"]).replace(
        "figsize=(6.5, 3.2)", "figsize=(14, 6)"
    )
    nb13["cells"][26]["source"] = [line + "\n" for line in c26.splitlines()]

    # Cell 31
    c31 = "".join(nb13["cells"][31]["source"]).replace(
        "figsize=(14, 5)", "figsize=(16, 6)"
    )
    nb13["cells"][31]["source"] = [line + "\n" for line in c31.splitlines()]

    # Cell 32
    c32 = "".join(nb13["cells"][32]["source"]).replace(
        "figsize=(7, 3.8)", "figsize=(14, 6)"
    )
    nb13["cells"][32]["source"] = [line + "\n" for line in c32.splitlines()]

    # Cell 35
    c35 = "".join(nb13["cells"][35]["source"]).replace(
        "figsize=(17, 4.8)", "figsize=(18, 6)"
    )
    nb13["cells"][35]["source"] = [line + "\n" for line in c35.splitlines()]

    with open(p13, "w", encoding="utf-8") as f:
        json.dump(nb13, f, indent=1)
    print("[✓] Cleaned notebooks/13-titans-neural-memory-benchmark.ipynb")


if __name__ == "__main__":
    fix_all()
