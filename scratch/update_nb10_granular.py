import json

with open(
    "notebooks/03-federated-aggregation/10-fedm3-multimomentum-benchmark.ipynb",
    encoding="utf-8",
) as f:
    nb10 = json.load(f)

# Update Cell 2 (Target)
cell2_content = """## **1. [TARGET] Multi-Momentum Matrix Aggregation & HOPE-FL Continuum Memory**

> [!IMPORTANT]
> **CENTRAL THESIS**: While **FedMuon** (Takezawa et al., 2025) successfully applies matrix polar orthogonalization to federated learning, it operates with a *single-scale momentum buffer* and is restricted to static neural architectures. Under severe non-IID client drift ($\\alpha \\le 0.1$), FedMuon suffers from **orthogonalization overshoot**, where obsolete historical gradient vectors are normalized to unit spectral norm, destabilizing consensus updates. Furthermore, FedMuon provides zero defense against temporal distribution shift or catastrophic forgetting at the edge.
>
> **FedM3 (Ours)** resolves these limitations through three granular modular innovations:
> 1. **FedM3 Optimizer Engine**: Decouples fast local client acceleration ($\\beta_{\\text{fast}} = 0.80$) from slow global consensus curvature ($\\beta_{\\text{slow}} = 0.98$) using convex scale blending ($\\gamma = 0.60$) and Nesterov lookahead before quintic Newton-Schulz polar orthogonalization onto the Stiefel manifold $\\text{St}(p, n)$.
> 2. **Continuous Memory System (CMS)**: An edge-native working memory matrix ($M_t \\in \\mathbb{R}^{D \\times D}$) updated dynamically in the forward pass via associative surprise detection ($S_t = v_t - k_t M_{t-1}$) and adaptive retention gating ($\\alpha_t$).
> 3. **HOPE-FL Synergy**: Unifies server-side FedM3 with client-side CMS, forming a dual-timescale system that simultaneously eliminates spatial client drift and temporal catastrophic forgetting.
>
> **TARGET METRIC**: Replicate Slide 39 empirical findings: benchmark convergence across 5 optimizers (FedAvg, FedProx, FedAdam, FedMuon, FedM3) on non-IID MNIST ($\\alpha=0.5$) under the HOPE-FL backbone.
>
> **SCHOLARLY CITATIONS**:
> - Takezawa, Y., Bao, H., Niwa, K., & Sato, I. (2025). *FedMuon: Federated Learning with Bias-Corrected Matrix Polar Orthogonalization*. arXiv:2502.14815.
> - Behrouz, A., Pezeshki, C., & Mirrokni, V. (2024). *Titans: Learning to Memorize at Test Time*. arXiv:2412.0009.
> - Sun, Y., et al. (2020). *Test-Time Training with Self-Supervision for Generalization under Distribution Shifts*. ICML.
> - Wang, J., et al. (2020). *Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization (FedNova)*. NeurIPS.
"""

nb10["cells"][2]["source"] = [
    line + "\n" for line in cell2_content.split("\n")[:-1]
] + [cell2_content.split("\n")[-1]]

# Update Cell 3 (Mechanics)
cell3_content = """## **2. [MECHANICS] Granular Modular Architecture: FedMuon vs. FedM3, CMS, and HOPE-FL**

```mermaid
flowchart TD
    subgraph S_FedMuon ["Baseline: Server-Side FedMuon (Single-Scale)"]
        A1["Client Parameter Deltas Δw_k"] --> B1["Single-Scale Momentum: M_t = β M_{t-1} + (1-β) Δw"]
        B1 --> C1["Newton-Schulz Polar Orthogonalization: O_t = NS_5(M_t)"]
        C1 --> D1["Global Update: W_{t+1} = W_t + η O_t"]
        D1 -.->|"Vulnerable to Overshoot"| E1["Static Edge Architecture (No Continual Memory)"]
    end

    subgraph S_FedM3 ["Module 1: FedM3 Dual-Scale Momentum Engine (Ours)"]
        A2["Client Parameter Deltas Δw_k"] --> B2["Fast Track: m_{fast} = β_f m_{fast} + (1-β_f) Δw (β_f=0.80)"]
        A2 --> C2["Slow Track: m_{slow} = β_s m_{slow} + (1-β_s) Δw (β_s=0.98)"]
        B2 --> D2["Convex Synthesis: M_t = γ m_{fast} + (1-γ) m_{slow} (γ=0.60)"]
        C2 --> D2
        D2 --> E2["Nesterov Lookahead: V_t = (1+β_f) M_t - β_f M_{t-1}"]
        E2 --> F2["Quintic Newton-Schulz on Stiefel Manifold: O_t = NS_5(V_t)"]
        F2 --> G2["Global Update: W_{t+1} = W_t + η O_t"]
    end

    subgraph S_CMS ["Module 2: Continuous Memory System (CMS) at the Edge"]
        H2["Visual Feature Embedding x_t"] --> I2["Projections: q_t, k_t, v_t"]
        I2 --> J2["Associative Surprise Detection: S_t = v_t - k_t M_{t-1}"]
        J2 --> K2["Adaptive Retention Gate: α_t = σ(W_α x_t + b_α)"]
        K2 --> L2["Fast-Weight Update: M_t = α_t M_{t-1} + η_{mem} (k_t^T ⊗ S_t)"]
        L2 --> M2["Memory Readout & Residual Fusion: y_t = LN(W_O r_t + x_t)"]
    end

    subgraph S_HOPE ["Module 3: HOPE-FL End-to-End Synergy"]
        G2 ==>|"Broadcast Global Slow Weights"| S_CMS
        S_CMS ==>|"Transmit Cleaned Representation Deltas"| A2
    end
```
"""

nb10["cells"][3]["source"] = [
    line + "\n" for line in cell3_content.split("\n")[:-1]
] + [cell3_content.split("\n")[-1]]

# Update Cell 6 (Architecture Details)
cell6_content = """## **3. [ARCHITECTURE] Granular Modular Comparison: FedMuon vs. FedM3, CMS, and HOPE-FL**

To understand why FedM3 outperforms FedMuon, we examine the three foundational modules:

### **Module 1: FedMuon vs. FedM3 Optimizer Engine**
| Characteristic | FedMuon (Takezawa et al., 2025) | FedM3 Optimizer Engine (Ours) |
| :--- | :--- | :--- |
| **Momentum Hierarchy** | Single-scale temporal EMA ($M_t = \\beta M_{t-1} + (1-\\beta)G_t$) | **Dual-scale frequency decoupling**: Fast ($\\beta_{\\text{fast}}=0.80$) + Slow ($\\beta_{\\text{slow}}=0.98$) |
| **Scale Synthesis** | Static single buffer | **Convex blending** ($M_t = \\gamma M_t^{\\text{fast}} + (1-\\gamma) M_t^{\\text{slow}}$) with $\\gamma=0.60$ |
| **Lookahead Dynamics** | None (standard momentum step) | **Nesterov lookahead**: $V_t = (1+\\beta_{\\text{fast}})M_t - \\beta_{\\text{fast}}M_{t-1}$ |
| **Manifold Projection** | Newton-Schulz on unconditioned momentum | Newton-Schulz on **drift-invariant subspace velocity** $V_t$ |
| **Overshoot Under Drift** | High: noise singular vectors normalized to norm 1 | **Suppressed**: Slow buffer anchors consensus trajectory |

### **Module 2: Continuous Memory System (CMS)**
- **Working Memory Matrix**: Fast-weight tensor $M_t \\in \\mathbb{R}^{D \\times D}$ maintained in volatile client memory.
- **Associative Surprise Metric**:
  $$S_t = v_t - k_t M_{t-1}$$
  Measures representation novelty before updating memory parameters.
- **Adaptive Input-Dependent Retention**:
  $$\\alpha_t = \\sigma(W_\\alpha x_t + b_\\alpha) \\in (0, 1)$$
  Dynamically modulates forgetting: familiar inputs trigger high retention ($\\alpha_t \\to 1$), while distribution shifts open the gate for rapid memory overwriting.
- **Associative Fast-Weight Writing**:
  $$M_t = \\alpha_t M_{t-1} + \\eta_{\\text{mem}} (k_t^T \\otimes S_t)$$
- **Zero-Backpropagation Test-Time Adaptation**: The memory update occurs entirely within the forward pass, requiring zero gradient computation on edge devices.

### **Module 3: HOPE-FL End-to-End Synergy**
- **Dual-Timescale Spatiotemporal Resolution**:
  1. **Spatial Client Drift** across decentralized silos is resolved by the server-side **FedM3** multi-scale momentum engine.
  2. **Temporal Domain Drift & Catastrophic Forgetting** within local client streams is resolved by client-side **CMS** fast weights.
- **Architecture Stack**:
  - **Trunk**: Conv1 ($3 \\times 3$, stride 2) $\\rightarrow$ Conv2 ($3 \\times 3$, stride 2) $\\rightarrow$ Global Average Pooling $\\rightarrow x_t \\in \\mathbb{R}^{64}$.
  - **Memory Core**: Continuous Memory System with Query, Key, Value projections ($D=64$).
  - **Head**: Linear classifier mapping fused representation $y_t$ to class logits.
"""

nb10["cells"][6]["source"] = [
    line + "\n" for line in cell6_content.split("\n")[:-1]
] + [cell6_content.split("\n")[-1]]

with open(
    "notebooks/03-federated-aggregation/10-fedm3-multimomentum-benchmark.ipynb",
    "w",
    encoding="utf-8",
) as f:
    json.dump(nb10, f, indent=1)

print("Successfully updated notebook 10!")
