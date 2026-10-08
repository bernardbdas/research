# Federated Learning & Privacy-Preserving Machine Learning Research

An empirical research and benchmarking platform for **Federated Learning (FL)**,
**Privacy Preservation (Differential Privacy & Secure Aggregation)**, and
**Communication Efficiency** using **Flower (`flwr`)** and **JAX / Flax**.

______________________________________________________________________

## 🎯 Research Objectives

1. **Fair & Justified Baseline Comparison:** Evaluate identical data-agnostic neural
   network backbones across **Centralized Training**, **Federated IID**, and **Federated
   Non-IID (Dirichlet Skew)** regimes.
1. **Privacy-Utility Tradeoffs:** Investigate Local Differential Privacy (LDP via JAX
   vectorization and DP-SGD) and Central Differential Privacy (CDP) under varying
   privacy budgets ($\\epsilon, \\delta$).
1. **Communication Efficiency:** Benchmark model compression, weight quantization, and
   client selection strategies to minimize uplink communication bottlenecks.

______________________________________________________________________

## 🏗️ Repository Architecture

```text
research/
├── .github/workflows/       # GitHub Actions CI for automated linting & formatting
├── diagrams/                # System architectures and communication diagrams
├── models/                  # Shared data-agnostic neural network backbones in Flax
│   ├── simple_cnn.py        # Canonical FedAvg benchmark
│   ├── resnet.py            # ResNet-18 with Global Average Pooling
│   ├── mobilenet.py         # MobileNetV2 with inverted residuals
│   ├── vgg.py               # VGG-16 adapted for efficient FL aggregation
│   └── __init__.py          # Unified model registry & factory
├── notebooks/               # Research experiments & empirical benchmarks
│   ├── 00-preliminaries/    # Partitioning analysis (IID vs Dirichlet Non-IID)
│   ├── 01-privacy-preservation/     # Differential Privacy & cryptographic protocols
│   ├── 02-communication-efficiency/ # Model compression & sparsification
│   └── 03-federated-aggregation/    # Baseline comparisons (Centralized vs FedAvg)
├── utils/                   # Reusable data pipelines & metrics
│   ├── data_processing.py   # Raw caching, train/val/test split, Parquet export
│   └── __init__.py
├── data/                    # Local data directories (git-ignored, preserved with .gitkeep)
│   ├── raw/                 # Raw dataset download caches
│   └── processed/           # Processed Apache Parquet partitions
├── .pre-commit-config.yaml  # Pre-commit hooks for Ruff and mdformat
├── pyproject.toml           # Project dependencies managed via uv
└── LICENSE                  # PolyForm Noncommercial License 1.0.0
```

______________________________________________________________________

## 🧠 Shared Model Backbones

All backbones in [`models/`](models/) use **Global Average Pooling** across spatial
dimensions, making them completely data-agnostic (supporting arbitrary image resolutions
and channels):

```python
import jax
import jax.numpy as jnp
from models import count_parameters, get_model

# Instantiate any backbone dynamically
model = get_model("simple_cnn", num_classes=10)
# Options: "simple_cnn", "resnet18", "mobilenetv2", "vgg16"

# Initialize with dummy input (batch_size, H, W, C)
variables = model.init(jax.random.PRNGKey(0), jnp.ones((1, 32, 32, 3)), train=False)
print(f"Trainable Parameters: {count_parameters(variables):,}")
```

______________________________________________________________________

## 📊 Data Pipeline (Raw Caching & Parquet Export)

To ensure high-throughput local training and clean separation of concerns:

- Datasets are downloaded to `data/raw/<dataset>/`.
- Processed into deterministic `train`, `val`, and `test` splits and exported to
  `data/processed/<dataset>/*.parquet`.
- Loaded directly into Flower's `FederatedDataset` without network overhead.

```python
from flwr_datasets.partitioner import DirichletPartitioner
from utils import get_federated_dataset_from_parquet, prepare_and_save_dataset

# 1. Download and preprocess
prepare_and_save_dataset("cifar10", val_ratio=0.1)

# 2. Partition for Federated Learning via Flower
fds = get_federated_dataset_from_parquet(
    dataset_name="cifar10",
    partitioners={
        "train": DirichletPartitioner(
            num_partitions=10, partition_by="label", alpha=0.3
        )
    },
)

# 3. Load client partition
client_0_data = fds.load_partition(0, split="train")
```

______________________________________________________________________

## 🚀 Quickstart & Setup

### 1. Prerequisites

- Python 3.12+ (Python 3.13 recommended)
- [`uv`](https://github.com/astral-sh/uv) package manager

### 2. Environment Setup

```bash
# Clone the repository
git clone https://github.com/bernardbdas/research.git
cd research

# Install all dependencies with uv
uv sync --dev

# Activate virtual environment
source .venv/bin/activate

# Install pre-commit hooks
uv run pre-commit install
```

### 3. Linting & Code Formatting

```bash
# Check code & notebooks with Ruff
uv run ruff check . --fix

# Format code & notebooks
uv run ruff format .

# Format markdown documentation
uv run mdformat README.md CONTRIBUTING.md
```

______________________________________________________________________

## 📜 License

This project is licensed under the **PolyForm Noncommercial License 1.0.0**. It is
open-sourced exclusively for **educational, academic research, and learning purposes**.
Commercial use of any kind is strictly prohibited. See [`LICENSE`](LICENSE) for full
details.

For contribution guidelines, see [`CONTRIBUTING.md`](CONTRIBUTING.md).
