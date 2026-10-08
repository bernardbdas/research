# System Architecture & UML Design Specifications

This directory contains publication-grade **Unified Modeling Language (UML)** and
**System Architecture** diagrams documenting the design, structural composition,
protocols, and workflows of this research framework.

______________________________________________________________________

## 📁 Directory Structure

```text
diagrams/
├── 01-class-diagrams/          # Structural design of models & data pipeline
│   ├── model_architectures.puml / .svg
│   └── data_pipeline_classes.puml / .svg
├── 02-sequence-diagrams/       # Temporal protocols & step-by-step lifecycles
│   ├── fl_training_round.puml / .svg
│   └── differential_privacy_dp_sgd.puml / .svg
├── 03-system-architecture/     # Component topology & end-to-end data flow
│   ├── fl_system_overview.puml / .svg
│   └── data_flow_pipeline.puml / .svg
└── 04-activity-diagrams/       # Control flow, decision branches & algorithms
    ├── client_execution_lifecycle.puml / .svg
    └── communication_efficiency.puml / .svg
```

______________________________________________________________________

## 1. Structural Class Diagrams (`01-class-diagrams/`)

### 🔹 Model Architectures (`model_architectures.puml`)

- **Purpose:** Explains the Flax / JAX neural network backbone hierarchy.
- **Key Components:**
  - Base class `flax.linen.Module`.
  - Submodules: `BasicBlock` (ResNet skip connections) and `InvertedResidual`
    (MobileNetV2 linear bottlenecks).
  - Concrete backbones: `SimpleCNN`, `ResNet18`, `MobileNetV2`, `VGG16`.
  - Factory & Registry: `ModelRegistry` exposing `get_model()`, `list_models()`, and
    `count_parameters()`.
  - All models leverage **Global Average Pooling** across spatial dimensions for
    data-agnostic resolution handling.

### 🔹 Data Pipeline Classes (`data_pipeline_classes.puml`)

- **Purpose:** Documents data management, partitioners, and local storage serialization.
- **Key Components:**
  - Flower `Partitioner` abstraction: `IidPartitioner`, `DirichletPartitioner`,
    `PathologicalPartitioner`.
  - Hugging Face `Dataset` & `DatasetDict` integration.
  - Apache Parquet storage layer (`train.parquet`, `val.parquet`, `test.parquet`).
  - `DataProcessingPipeline` factory bridging Parquet files directly into Flower's
    `FederatedDataset`.

______________________________________________________________________

## 2. Interaction Sequence Diagrams (`02-sequence-diagrams/`)

### 🔹 Federated Training Round (`fl_training_round.puml`)

- **Purpose:** Demonstrates the end-to-end communication protocol between the Central
  Server and Edge Clients.
- **Protocol Steps:**
  1. Round configuration & client sampling ($S_t$).
  1. Broadcast of global model weights $\\mathbf{w}\_t$.
  1. On-device local training loop with JAX/Flax and Optax optimizer.
  1. Client result submission ($\\mathbf{w}\_{t+1}^k, |\\mathcal{D}\_k|$).
  1. FedAvg weighted aggregation: $\\mathbf{w}\_{t+1} = \\sum
     \\frac{|\\mathcal{D}_k|}{|\\mathcal{D}|} \\mathbf{w}_{t+1}^k$.
  1. Centralized and federated evaluation logging.

### 🔹 Differential Privacy DP-SGD (`differential_privacy_dp_sgd.puml`)

- **Purpose:** Details the localized per-sample gradient perturbation loop using JAX.
- **Steps:**
  1. Vectorized per-sample gradients via `jax.vmap(jax.grad(loss_fn))`.
  1. Per-sample gradient norm calculation and clipping to threshold $C$.
  1. Calibrated Gaussian noise addition $\\mathcal{N}(0, \\sigma^2 C^2 \\mathbf{I})$.
  1. Privacy budget accumulation ($\\epsilon, \\delta$) via Rényi Differential Privacy
     (RDP).

______________________________________________________________________

## 3. System Architecture & Topology (`03-system-architecture/`)

### 🔹 Federated Learning System Overview (`fl_system_overview.puml`)

- **Purpose:** High-level architectural topology of the Federated Learning deployment.
- **Highlights:**
  - Separation of concerns between Central Server (Orchestration, Strategy, Evaluation)
    and Edge Clients.
  - Zero raw-data transmission guarantee over the secure parameter transport channel
    (gRPC / TLS).
  - Client-side computation stack: Model Backbone $\\rightarrow$ Optimizer
    $\\rightarrow$ Privacy Engine $\\rightarrow$ Compression.

### 🔹 Data Flow & Partitioning Pipeline (`data_flow_pipeline.puml`)

- **Purpose:** Traces the lifecycle of datasets from remote acquisition to
  client-partition consumption.
- **Highlights:**
  - Remote Hugging Face Hub download to local raw disk cache (`data/raw/`).
  - Deterministic splitting into 90% train, 10% validation, and global test Parquet
    files (`data/processed/`).
  - High-throughput ingestion into Flower `FederatedDataset` with IID and Dirichlet
    label skew partitioners.

______________________________________________________________________

## 4. Behavioral Activity Diagrams (`04-activity-diagrams/`)

### 🔹 Client Execution Lifecycle (`client_execution_lifecycle.puml`)

- **Purpose:** Flowchart of operations and decision branches performed by an edge client
  upon receiving a server round invitation.
- **Branches:**
  - Sampling selection check (Selected vs Idle).
  - Epoch & mini-batch iterations.
  - Conditional branch: Standard SGD vs DP-SGD (vectorized clipping & noise injection).
  - Conditional branch: Full precision FP32 vs Sparsified & Quantized uplink payload.

### 🔹 Communication Efficiency & Uplink Compression (`communication_efficiency.puml`)

- **Purpose:** Details the uplink payload compression and server-side reconstruction
  algorithm.
- **Pipeline:**
  - Raw delta calculation ($\\Delta \\mathbf{w}\_t^k$).
  - Top-$K%$ magnitude sparsification.
  - Stochastic uniform quantization (FP32 $\\rightarrow$ INT8/INT4).
  - Payload bit-packing ($\\sim 40\\times$ uplink bandwidth reduction).
  - Server-side dequantization, vector reconstruction, and FedAvg accumulation.

______________________________________________________________________

## 🛠️ How to Re-render or Modify Diagrams

All source diagrams are written in standard [PlantUML](https://plantuml.com/) (`.puml`).

### Render All Diagrams to SVG:

```bash
just diagrams
```

*(Or run `plantuml -tsvg diagrams/**/*.puml` directly)*

### Render All Diagrams to PNG:

```bash
plantuml -tpng diagrams/**/*.puml
```

### Render to PDF (for LaTeX publication):

```bash
plantuml -tpdf diagrams/**/*.puml
```
