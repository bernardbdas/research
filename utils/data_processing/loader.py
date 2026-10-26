"""Dataset loading and Flower FederatedDataset integration."""

from pathlib import Path
from typing import Any

from utils.data_processing.config import configure_hf_cache_dir
from utils.data_processing.paths import DatasetPaths
from utils.data_processing.pipeline import prepare_and_save_dataset

# Ensure Hugging Face cache and environment are configured
configure_hf_cache_dir()

from datasets import Dataset, DatasetDict  # noqa: E402
from flwr_datasets import FederatedDataset  # noqa: E402
from flwr_datasets.partitioner import Partitioner  # noqa: E402


def load_processed_dataset(
    dataset_name: str,
    split: str | None = None,
    modality: str | None = None,
    processed_dir: Path | str | None = None,
    auto_prepare: bool = True,
    **prepare_kwargs: Any,
) -> Dataset | DatasetDict:
    """Load preprocessed Parquet dataset splits from data/processed/<modality>/.

    Parameters
    ----------
    dataset_name : str
        Dataset name or alias (e.g. 'cifar10', 'mnist').
    split : Optional[str]
        Split name to load ('train', 'val', or 'test'). If None, returns DatasetDict.
    modality : Optional[str]
        Modality category override ('vision', 'audio', etc.). Automatically resolved if None.
    processed_dir : Optional[Path | str]
        Custom directory for processed data. Defaults to '<root>/data/processed/<modality>/<dataset>'.
    auto_prepare : bool
        If True and processed Parquet files do not exist, automatically downloads, caches,
        and preprocesses the dataset into data/raw/ and data/processed/. Defaults to True.
    **prepare_kwargs : Any
        Forwarded to prepare_and_save_dataset if auto_prepare is triggered.

    Returns
    -------
    Dataset | DatasetDict
        Loaded Hugging Face Dataset or DatasetDict.
    """
    paths = DatasetPaths.from_name(
        dataset_name, modality=modality, processed_dir=processed_dir
    )

    if not paths.has_processed_parquets():
        if auto_prepare:
            prepare_and_save_dataset(
                dataset_name=dataset_name,
                modality=modality,
                raw_dir=paths.raw_dir,
                processed_dir=paths.processed_dir,
                **prepare_kwargs,
            )
        else:
            raise FileNotFoundError(
                f"Processed dataset directory not found at {paths.processed_dir}. "
                f"Run prepare_and_save_dataset('{dataset_name}') first."
            )

    if split:
        file_path = paths.parquet_file(split)
        if not file_path.exists():
            raise FileNotFoundError(f"Split file not found: {file_path}")
        return Dataset.from_parquet(str(file_path))

    files = {
        candidate: str(paths.parquet_file(candidate))
        for candidate in ["train", "val", "test"]
        if paths.parquet_file(candidate).exists()
    }
    return DatasetDict({s: Dataset.from_parquet(f) for s, f in files.items()})


def get_federated_dataset_from_parquet(
    dataset_name: str,
    partitioners: dict[str, Partitioner | int],
    modality: str | None = None,
    processed_dir: Path | str | None = None,
) -> FederatedDataset:
    """Create a Flower FederatedDataset directly from preprocessed Parquet files in data/processed/.

    Parameters
    ----------
    dataset_name : str
        Name or shortcut of the dataset (e.g. 'cifar10', 'mnist').
    partitioners : dict[str, Partitioner | int]
        Flower partitioners dictionary (e.g. {'train': IidPartitioner(num_partitions=10)}).
    modality : Optional[str]
        Modality category override ('vision', 'audio', etc.). Automatically resolved if None.
    processed_dir : Optional[Path | str]
        Custom processed directory. Defaults to '<root>/data/processed/<modality>/<dataset>'.

    Returns
    -------
    FederatedDataset
        Configured FederatedDataset ready for client partition loading.
    """
    paths = DatasetPaths.from_name(
        dataset_name, modality=modality, processed_dir=processed_dir
    )

    data_files = {
        split: str(paths.parquet_file(split))
        for split in ["train", "val", "test"]
        if paths.parquet_file(split).exists()
    }

    if not data_files:
        raise FileNotFoundError(
            f"No Parquet split files found at {paths.processed_dir}"
        )

    return FederatedDataset(
        dataset="parquet",
        data_files=data_files,
        partitioners=partitioners,
    )


def get_federated_dataset(
    dataset_name: str,
    partitioners: dict[str, Partitioner | int],
    modality: str | None = None,
    auto_prepare: bool = True,
    val_ratio: float = 0.1,
    **load_dataset_kwargs: Any,
) -> FederatedDataset:
    """Load or create a Flower FederatedDataset guaranteed to persist in data/<modality>/.

    Parameters
    ----------
    dataset_name : str
        Name or shortcut of the dataset (e.g. 'cifar10', 'mnist', 'femnist', 'speech_commands').
    partitioners : dict[str, Partitioner | int]
        Flower partitioner dictionary mapping split to Partitioner instance.
    modality : Optional[str]
        Modality category override ('vision', 'audio', etc.). Automatically resolved if None.
    auto_prepare : bool
        If True and processed Parquet does not exist, automatically runs
        prepare_and_save_dataset() into data/raw/<modality>/ and data/processed/<modality>/.
    val_ratio : float
        Ratio of training data to reserve for validation if preparing. Default is 0.1.
    **load_dataset_kwargs : Any
        Additional keyword arguments forwarded to dataset loader.

    Returns
    -------
    FederatedDataset
        Configured Flower FederatedDataset backed by data/.
    """
    paths = DatasetPaths.from_name(dataset_name, modality=modality)

    if paths.has_processed_parquets():
        return get_federated_dataset_from_parquet(
            dataset_name=dataset_name,
            partitioners=partitioners,
            modality=modality,
            processed_dir=paths.processed_dir,
        )

    if auto_prepare:
        prepare_and_save_dataset(
            dataset_name=dataset_name,
            val_ratio=val_ratio,
            modality=modality,
            raw_dir=paths.raw_dir,
            processed_dir=paths.processed_dir,
            **load_dataset_kwargs,
        )
        return get_federated_dataset_from_parquet(
            dataset_name=dataset_name,
            partitioners=partitioners,
            modality=modality,
            processed_dir=paths.processed_dir,
        )

    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    load_kwargs = dict(load_dataset_kwargs)
    load_kwargs.setdefault("cache_dir", str(paths.raw_dir))
    return FederatedDataset(
        dataset=paths.canonical_id,
        partitioners=partitioners,
        **load_kwargs,
    )
