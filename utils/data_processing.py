"""Data pipeline utilities for raw caching, preprocessing, and Parquet export.

Downloads datasets into data/raw/, creates train/val/test splits,
and exports them to data/processed/ in Apache Parquet format.
"""

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import Partitioner

# Common dataset shortcuts mapped to canonical Hugging Face Hub repositories
DATASET_ALIASES: dict[str, str] = {
    "cifar10": "uoft-cs/cifar10",
    "cifar100": "uoft-cs/cifar100",
    "mnist": "ylecun/mnist",
    "fashion_mnist": "zalando-datasets/fashion_mnist",
}


def get_project_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent.parent


def resolve_dataset_name(dataset_name: str) -> str:
    """Resolve shorthand dataset names to canonical Hugging Face repository IDs."""
    key = dataset_name.strip().lower()
    return DATASET_ALIASES.get(key, dataset_name)


def prepare_and_save_dataset(
    dataset_name: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
    subset: str | None = None,
    **load_dataset_kwargs: Any,
) -> dict[str, Path]:
    """Download dataset to data/raw, create train/val/test splits, and save as Parquet.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g., 'cifar10', 'mnist', 'uoft-cs/cifar10').
    val_ratio : float
        Proportion of training set to reserve for validation if none exists. Default is 0.1 (10%).
    seed : int
        Random seed for reproducible train/val splitting. Default is 42.
    raw_dir : Optional[Path | str]
        Directory for raw downloaded cache. Defaults to '<root>/data/raw/<dataset_name>'.
    processed_dir : Optional[Path | str]
        Directory for processed Parquet files. Defaults to '<root>/data/processed/<dataset_name>'.
    subset : Optional[str]
        Optional dataset configuration/subset name if required.
    **load_dataset_kwargs : Any
        Additional keyword arguments forwarded to datasets.load_dataset.

    Returns
    -------
    dict[str, Path]
        Dictionary mapping split names ('train', 'val', 'test') to their Parquet file paths.
    """
    root = get_project_root()
    canonical_repo = resolve_dataset_name(dataset_name)
    clean_folder_name = dataset_name.replace("/", "_").lower()

    raw_path = Path(raw_dir) if raw_dir else root / "data" / "raw" / clean_folder_name
    proc_path = (
        Path(processed_dir)
        if processed_dir
        else root / "data" / "processed" / clean_folder_name
    )

    raw_path.mkdir(parents=True, exist_ok=True)
    proc_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Downloading/caching raw '{canonical_repo}' into {raw_path}...")
    ds_dict = load_dataset(
        canonical_repo,
        name=subset,
        cache_dir=str(raw_path),
        **load_dataset_kwargs,
    )

    train_split: Dataset | None = None
    val_split: Dataset | None = None
    test_split: Dataset | None = None

    if "test" in ds_dict:
        test_split = ds_dict["test"]

    if "validation" in ds_dict:
        val_split = ds_dict["validation"]
    elif "val" in ds_dict:
        val_split = ds_dict["val"]

    if "train" in ds_dict:
        train_data = ds_dict["train"]
        if val_split is None and val_ratio > 0.0:
            print(
                f"[*] Creating validation split ({int(val_ratio * 100)}%) from train split..."
            )
            split_dict = train_data.train_test_split(test_size=val_ratio, seed=seed)
            train_split = split_dict["train"]
            val_split = split_dict["test"]
        else:
            train_split = train_data
    elif test_split is not None:
        train_split = test_split

    saved_paths: dict[str, Path] = {}
    splits_to_save = {
        "train": train_split,
        "val": val_split,
        "test": test_split,
    }

    for split_name, dataset in splits_to_save.items():
        if dataset is not None:
            file_path = proc_path / f"{split_name}.parquet"
            print(
                f"[*] Exporting {split_name} split ({len(dataset)} samples) to {file_path}..."
            )
            dataset.to_parquet(str(file_path))
            saved_paths[split_name] = file_path

    print(f"[✓] Successfully processed and saved '{dataset_name}' into {proc_path}")
    return saved_paths


def load_processed_dataset(
    dataset_name: str,
    split: str | None = None,
    processed_dir: Path | str | None = None,
) -> Dataset | DatasetDict:
    """Load preprocessed Parquet dataset splits from data/processed/.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. 'cifar10').
    split : Optional[str]
        Split name to load ('train', 'val', or 'test'). If None, loads all available splits.
    processed_dir : Optional[Path | str]
        Path to processed data directory. Defaults to '<root>/data/processed/<dataset_name>'.

    Returns
    -------
    Dataset | DatasetDict
        Hugging Face Dataset for the split, or DatasetDict containing all splits.
    """
    root = get_project_root()
    clean_folder_name = dataset_name.replace("/", "_").lower()
    proc_path = (
        Path(processed_dir)
        if processed_dir
        else root / "data" / "processed" / clean_folder_name
    )

    if not proc_path.exists():
        raise FileNotFoundError(
            f"Processed dataset directory not found at {proc_path}. "
            f"Run prepare_and_save_dataset('{dataset_name}') first."
        )

    if split:
        file_path = proc_path / f"{split}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Split file not found: {file_path}")
        return Dataset.from_parquet(str(file_path))

    files = {}
    for candidate in ["train", "val", "test"]:
        p = proc_path / f"{candidate}.parquet"
        if p.exists():
            files[candidate] = str(p)

    return DatasetDict({s: Dataset.from_parquet(f) for s, f in files.items()})


def get_federated_dataset_from_parquet(
    dataset_name: str,
    partitioners: dict[str, Partitioner | int],
    processed_dir: Path | str | None = None,
) -> FederatedDataset:
    """Create a Flower FederatedDataset directly from the preprocessed Parquet files.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (e.g. 'cifar10').
    partitioners : dict[str, Partitioner | int]
        Flower partitioners dictionary (e.g. {'train': IidPartitioner(num_partitions=10)}).
    processed_dir : Optional[Path | str]
        Path to processed data directory. Defaults to '<root>/data/processed/<dataset_name>'.

    Returns
    -------
    FederatedDataset
        Configured FederatedDataset ready for client partition loading.
    """
    root = get_project_root()
    clean_folder_name = dataset_name.replace("/", "_").lower()
    proc_path = (
        Path(processed_dir)
        if processed_dir
        else root / "data" / "processed" / clean_folder_name
    )

    data_files = {}
    for split in ["train", "val", "test"]:
        file_path = proc_path / f"{split}.parquet"
        if file_path.exists():
            data_files[split] = str(file_path)

    if not data_files:
        raise FileNotFoundError(f"No Parquet split files found at {proc_path}")

    return FederatedDataset(
        dataset="parquet",
        data_files=data_files,
        partitioners=partitioners,
    )
