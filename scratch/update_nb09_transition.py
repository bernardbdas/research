import json

with open(
    "notebooks/03-federated-aggregation/09-fedmuon-matrix-optimization.ipynb",
    encoding="utf-8",
) as f:
    nb9 = json.load(f)

cell29_text = "".join(nb9["cells"][29]["source"])

transition_qa = """
- **Q: Where does FedMuon reach its theoretical limit, and how does FedM3 (Ours) advance beyond it?**
  **A:** FedMuon uses a single momentum track and applies polar orthogonalization strictly to static weights on the server. Under severe statistical skew ($\\alpha \\le 0.1$), this single momentum buffer accumulates conflicting client updates, causing **orthogonalization overshoot** (since polar orthogonalization forces all singular values to $\\sigma_i \\equiv 1.0$, boosting noise-dominated directions equally with signal directions). Furthermore, FedMuon cannot adapt to temporal domain drift or catastrophic forgetting at the edge. **FedM3 (Ours)** advances beyond FedMuon through three modular innovations:
  1. **FedM3 Multi-Scale Engine**: Decouples fast local acceleration ($\\beta_{\\text{fast}}=0.80$) from slow global consensus ($\\beta_{\\text{slow}}=0.98$) using convex blending ($\\gamma=0.60$) and Nesterov lookahead before polar orthogonalization.
  2. **Continuous Memory System (CMS)**: Equips edge clients with a dynamic fast-weight working memory ($M_t \\in \\mathbb{R}^{D \\times D}$) that updates during the forward pass via associative surprise ($S_t = v_t - k_t M_{t-1}$) and adaptive retention ($\\alpha_t$).
  3. **HOPE-FL Architecture**: Couples client-side CMS with server-side FedM3, eliminating both spatial client drift and temporal catastrophic forgetting.
"""

if "- **Q: Where does FedMuon reach its theoretical limit" not in cell29_text:
    # Insert right before Data Analysis Key Findings
    marker = "### **Data Analysis Key Findings**"
    assert marker in cell29_text, "Marker not found"
    parts = cell29_text.split(marker)
    new_cell29_text = parts[0] + transition_qa + "\n" + marker + parts[1]
    nb9["cells"][29]["source"] = [
        line + "\n" for line in new_cell29_text.split("\n")[:-1]
    ] + [new_cell29_text.split("\n")[-1]]

    with open(
        "notebooks/03-federated-aggregation/09-fedmuon-matrix-optimization.ipynb",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(nb9, f, indent=1)
    print("Successfully updated notebook 09 with FedMuon-to-FedM3 transition!")
else:
    print("Already present in notebook 09.")
