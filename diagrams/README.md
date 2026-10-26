# System Architecture & UML Design Specifications

This directory contains publication-grade **Unified Modeling Language (UML)** and
**System Architecture** diagrams documenting the mathematical foundations, structural
composition, algorithmic flows, and communication protocols of the federated research
framework.

All diagrams are written in declarative [PlantUML](https://plantuml.com/) (`.puml`) and
compiled to vector-quality Scalable Vector Graphics (`.svg`).

______________________________________________________________________

## Directory Structure

```text
diagrams/
├── 01-class-diagrams/          # Structural design of models & data pipeline
│   ├── model_architectures.puml / .svg       # Backbones, HOPEFLNet & ContinuousMemorySystem
│   └── data_pipeline_classes.puml / .svg     # Partitioners, HuggingFace Hub & Multimodal Catalog
├── 02-sequence-diagrams/       # Temporal protocols & step-by-step lifecycles
│   ├── fl_training_round.puml / .svg         # Round coordinator, FedM3/FedAvg aggregation
│   └── differential_privacy_dp_sgd.puml / .svg # Parallel vmap clipping & RDP accountant
├── 03-system-architecture/     # Component topology & core innovation architectures
│   ├── fedm3_architecture.puml / .svg        # Dual-scale momentum + Newton-Schulz Stiefel projection
│   ├── cms_architecture.puml / .svg          # Associative surprise gate & fast-weight memory
│   ├── hope_fl_architecture.puml / .svg      # End-to-end synergy: Vision Trunk + CMS + FedM3
│   ├── fl_system_overview.puml / .svg        # High-level server-client distributed topology
│   └── data_flow_pipeline.puml / .svg        # Raw cache, Parquet serialization & FDS partitioners
└── 04-activity-diagrams/       # Control flow, decision branches & execution pipelines
    ├── client_execution_lifecycle.puml / .svg # Client epoch loop with CMS memory updates
    └── communication_efficiency.puml / .svg  # Top-K + INT8 compression & FedM3 server update
```

______________________________________________________________________

## 1. Core Innovation System Architectures (`03-system-architecture/`)

### 1.1 FedM3: Federated Multi-scale Momentum Muon Optimizer (`fedm3_architecture.puml`)

- **Design Objective:** Resolves severe client drift across non-IID client cohorts by
  decoupling high-frequency local gradient noise from low-frequency global trajectory
  drift, followed by unconstrained matrix polar orthogonalization onto the Stiefel
  manifold $\\text{St}(p, n) = {X \\in \\mathbb{R}^{n \\times p} : X^T X = I}$.
- **Architectural Flow:**
  1. **Client Cohort Tier:** $K$ selected clients compute local mini-batch updates
     $\\Delta W_t^k = W\_{t, E}^k - W_t$ and transmit parameter deltas to the central
     coordinator.
  1. **Gradient Delta Gathering:** Weighted aggregation over sample counts: $$\\Delta
     W_t = \\sum\_{k \\in S_t} \\frac{|D_k|}{|D|} \\Delta W_t^k$$
  1. **Dual-Scale Momentum Engine:** Computes fast and slow temporal momentum tracks and
     blends them convexly with Nesterov lookahead: $$M_t^{\\text{fast}} =
     \\beta\_{\\text{fast}} M\_{t-1}^{\\text{fast}} + (1 - \\beta\_{\\text{fast}})
     \\Delta W_t \\quad (\\beta\_{\\text{fast}} = 0.80)$$ $$M_t^{\\text{slow}} =
     \\beta\_{\\text{slow}} M\_{t-1}^{\\text{slow}} + (1 - \\beta\_{\\text{slow}})
     \\Delta W_t \\quad (\\beta\_{\\text{slow}} = 0.98)$$ $$M_t = \\gamma
     M_t^{\\text{fast}} + (1 - \\gamma) M_t^{\\text{slow}} \\quad (\\gamma = 0.60)$$
     $$V_t = (1 + \\beta\_{\\text{fast}}) M_t - \\beta\_{\\text{fast}} M\_{t-1}$$
  1. **Quintic Newton-Schulz Polar Orthogonalization:** Iterative polynomial
     approximation of $V_t (V_t^T V_t)^{-1/2}$ for 2D matrix weights: $$X_0 =
     \\frac{V_t}{|V_t|_F + \\epsilon}$$ $$X_{k+1} = a X_k + b (X_k X_k^T) X_k + c (X_k
     X_k^T)^2 X_k \\quad (a = 3.4445, ; b = -4.7750, ; c = 2.0315)$$ $$O_t = X_5
     \\approx \\mathcal{P}\_{\\text{Stiefel}}(V_t)$$
  1. **Global Consensus Update:** $W\_{t+1} = W_t + \\eta O_t$, serialized and broadcast
     to the next cohort.

______________________________________________________________________

### 1.2 Continuous Memory System (CMS) (`cms_architecture.puml`)

- **Design Objective:** Forward-pass test-time dynamic adaptation mechanism enabling
  edge devices to adapt to local covariate shifts without overwriting slow frozen trunk
  parameters (mitigating catastrophic forgetting).
- **Architectural Flow:**
  1. **Input Representation:** Feature vector $x_t \\in \\mathbb{R}^D$ projected to
     query, key, and value subspaces: $$q_t = x_t W_q, \\quad k_t = x_t W_k, \\quad v_t
     = x_t W_v$$
  1. **Associative Surprise Detection:** Compares incoming value against current memory
     retrieval: $$S_t = v_t - k_t M\_{t-1}$$
  1. **Adaptive Retention & Fast-Weight Writing:** Dynamic input-dependent forgetting
     gate updates associative fast-weight matrix $M_t \\in \\mathbb{R}^{D \\times D}$:
     $$\\alpha_t = \\sigma(W\_\\alpha x_t + b\_\\alpha) \\in (0, 1)$$ $$M_t = \\alpha_t
     M\_{t-1} + \\eta\_{\\text{mem}} (k_t^T \\otimes S_t)$$
  1. **Readout & Residual Fusion:** Associative readout $r_t = q_t M_t$ followed by
     projection and residual connection: $$y_t = \\text{LayerNorm}(W_O r_t + x_t)$$

______________________________________________________________________

### 1.3 HOPE-FL: Federated Higher-Order Predictive Memory Architecture (`hope_fl_architecture.puml`)

- **Design Objective:** Unifies client-side neural test-time memory adaptation (CMS)
  with server-side dual-momentum polar orthogonalization (FedM3), creating a
  dual-timescale system resilient to both spatial client drift and temporal domain
  drift.
- **Architectural Flow:**
  1. **Edge Client Trunk:** Conv1 ($3 \\times 3$, stride 2) $\\rightarrow$ Conv2 ($3
     \\times 3$, stride 2) $\\rightarrow$ Global Average Pooling $\\rightarrow$ $x_t
     \\in \\mathbb{R}^{64}$.
  1. **Client CMS Fast-Weight Engine:** Forward-pass associative memory matrix $M_t$
     learns sequence-level and batch-level continuous contexts in real time.
  1. **Classification Head:** Linear projection from fused embedding $y_t \\in
     \\mathbb{R}^{64}$ to class logits $\\hat{y} \\in \\mathbb{R}^{C}$.
  1. **Client-to-Server Uplink:** Compressed slow-weight deltas $\\Delta W_t^k$
     transmitted via gRPC.
  1. **FedM3 Aggregation Engine:** Dual-scale temporal momentum blending and quintic
     Newton-Schulz polar orthogonalization.

______________________________________________________________________

### 1.4 Architectural Granularity: FedMuon vs. FedM3, CMS, and HOPE-FL

| Dimension / Capability       | FedMuon (Takezawa et al., 2025)                                             | FedM3 + CMS + HOPE-FL (Ours)                                                      |
| :--------------------------- | :-------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- |
| **Optimization Focus**       | Spatial matrix polar orthogonalization on server                            | **Spatiotemporal optimization & test-time neural memory**                         |
| **Momentum Tracking**        | Single-scale temporal EMA ($M_t = \\beta M\_{t-1} + (1-\\beta)\\Delta W_t$) | **Dual-scale frequency decoupling**: Fast ($\\beta=0.80$) + Slow ($\\beta=0.98$)  |
| **Momentum Synthesis**       | Static single track                                                         | **Convex blending** ($\\gamma=0.60$) + **Nesterov lookahead** $V_t$               |
| **Manifold Projection**      | Newton-Schulz on unconditioned momentum                                     | Newton-Schulz on **drift-invariant subspace velocity** $V_t$                      |
| **Overshoot Resistance**     | Low: obsolete client drift vectors amplified to norm 1                      | **High**: Slow buffer filters high-frequency client noise                         |
| **On-Device Memory Module**  | **None** (Static feedforward weights only)                                  | **Continuous Memory System (CMS)** fast-weight matrix $M_t$                       |
| **Memory Writing Mechanism** | N/A (Standard SGD backward pass only)                                       | **Associative surprise** ($S_t = v_t - k_t M\_{t-1}$) + adaptive gate $\\alpha_t$ |
| **Catastrophic Forgetting**  | Severe on non-stationary client streams                                     | **Zero forgetting** via dynamic fast weights ($M_t$) in forward pass              |
| **System-Level Synergy**     | Isolated server algorithm                                                   | **HOPE-FL**: Client CMS + Server FedM3 unified co-design                          |

______________________________________________________________________

## 2. Structural Class Diagrams (`01-class-diagrams/`)

### 2.1 Model Architectures (`model_architectures.puml`)

- **Base Module:** `flax.linen.Module` interface.
- **Core Backbones:** `SimpleCNN`, `ResNet18` (with `BasicBlock`), `MobileNetV2` (with
  `InvertedResidual`), `VGG16`.
- **Advanced Memory Architectures:**
  - `ContinuousMemorySystem`: Dynamic $M_t$ matrix state, surprise detection, and
    adaptive retention.
  - `HOPEFLNet`: Hybrid vision trunk coupled with CMS fast-weight working memory and
    dynamic classifier.
- **Model Registry:** `ModelRegistry` providing centralized `get_model()`,
  `list_models()`, and parameter counting.

### 2.2 Data Pipeline Classes (`data_pipeline_classes.puml`)

- **Partitioner Hierarchy:** `Partitioner` base class with `IidPartitioner`,
  `DirichletPartitioner`, `PathologicalPartitioner`, `ShardPartitioner`, and
  `ExponentialPartitioner`.
- **Multi-Modal Dataset Catalog:** Centralized taxonomy supporting 15 Vision, 4 Audio, 5
  Tabular, 9 Text, and 1 Multimodal dataset from Hugging Face Hub and Flower Datasets
  (`flwr_datasets`).
- **Data Engine:** `DataProcessingPipeline` bridging Parquet files into Flower's
  `FederatedDataset`.

______________________________________________________________________

## 3. Sequence & Activity Diagrams (`02-sequence-diagrams/` & `04-activity-diagrams/`)

### 3.1 Federated Training Round (`fl_training_round.puml`)

- Step-by-step communication between `ServerApp`, `ClientManager`, `AggregatorStrategy`
  (FedM3 / FedAvg), and edge `ClientApp`.
- Covers client sampling, parameter broadcast, JAX local execution, delta collection,
  and Stiefel polar orthogonalization.

### 3.2 Differential Privacy DP-SGD (`differential_privacy_dp_sgd.puml`)

- Vectorized per-sample gradients via `jax.vmap(jax.grad(loss_fn))`.
- Gradient norm clipping to threshold $C$ and calibrated Gaussian noise injection
  $\\mathcal{N}(0, \\sigma^2 C^2 I)$.
- Real-time privacy tracking via Rényi Differential Privacy (RDP) moments accountant.

### 3.3 Client Execution Lifecycle (`client_execution_lifecycle.puml`)

- Detailed client decision tree: Invitation $\\rightarrow$ Sampling check $\\rightarrow$
  Weight unpacking $\\rightarrow$ CMS fast-weight initialization $\\rightarrow$
  Epoch/mini-batch forward-backward passes $\\rightarrow$ Top-K / INT8 compression
  $\\rightarrow$ Uplink transmission.

### 3.4 Communication Efficiency Pipeline (`communication_efficiency.puml`)

- **Client Uplink:** Absolute coordinate sorting $\\rightarrow$ Top-$K%$ sparsification
  $\\rightarrow$ Stochastic uniform quantization (FP32 $\\rightarrow$ INT8)
  $\\rightarrow$ Coordinate bit-packing ($\\sim 40\\times$ bandwidth reduction).
- **Server Downlink:** Dequantization $\\rightarrow$ Weighted gathering $\\rightarrow$
  FedM3 dual-scale momentum $\\rightarrow$ Quintic Newton-Schulz polar orthogonalization
  $\\rightarrow$ Broadcast.

______________________________________________________________________

## 4. Literature Citations

All theoretical principles and architectural patterns depicted in these diagrams are
grounded in published peer-reviewed and established pre-print literature:

1. **Federated Optimization & Client Drift:**
   - McMahan et al. (2017), *Communication-Efficient Learning of Deep Networks from
     Decentralized Data*, AISTATS.
   - Li et al. (2020), *Federated Optimization in Heterogeneous Networks (FedProx)*,
     MLSys.
   - Karimireddy et al. (2020), *SCAFFOLD: Stochastic Controlled Averaging for Federated
     Learning*, ICML.
1. **Matrix Polar Orthogonalization & Muon Optimization:**
   - Takezawa et al. (2025), *FedMuon: Federated Learning with Bias-Corrected Matrix
     Polar Orthogonalization*, arXiv.
   - Higham (1986), *Computing the Polar Decomposition—with Applications*, SIAM J. Sci.
     Stat. Comput.
1. **Test-Time Training & Dynamic Fast-Weight Memory:**
   - Sun et al. (2020), *Test-Time Training with Self-Supervision for Generalization
     under Distribution Shifts*, ICML.
   - Sun et al. (2024), *Learning to (Learn at Test Time): RNNs with Expressive Hidden
     States*, arXiv.
   - Behrouz et al. (2024), *Titans: Learning to Memorize at Test Time*, arXiv.
1. **Differential Privacy & Compression:**
   - Abadi et al. (2016), *Deep Learning with Differential Privacy*, ACM CCS.
   - Alistarh et al. (2017), *QSGD: Communication-Efficient Learning with Quantized
     Gradients*, NeurIPS.

______________________________________________________________________

## 🛠️ How to Re-render or Modify Diagrams

All source diagrams are written in standard [PlantUML](https://plantuml.com/) (`.puml`).

### Render All Diagrams to SVG:

```bash
plantuml -tsvg diagrams/**/*.puml
```

### Render All Diagrams to PNG:

```bash
plantuml -tpng diagrams/**/*.puml
```

### Render to PDF (for LaTeX publication):

```bash
plantuml -tpdf diagrams/**/*.puml
```
