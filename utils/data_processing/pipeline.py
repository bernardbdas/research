"""Data pipeline execution: download, split extraction, and Parquet export."""

import os
from pathlib import Path
from typing import Any

from utils.data_processing.catalog import (
    DATASET_DEFAULT_CONFIGS,
    HF_MODERN_MIRRORS,
    list_supported_datasets,
)
from utils.data_processing.config import configure_hf_cache_dir
from utils.data_processing.paths import DatasetPaths

# Ensure Hugging Face cache and environment are configured
configure_hf_cache_dir()

from datasets import Dataset, DatasetDict, load_dataset  # noqa: E402


def _extract_dataset_splits(
    ds_dict: DatasetDict, val_ratio: float, seed: int
) -> dict[str, Dataset]:
    """Cleanly extract train, validation, and test splits with automatic splitting."""
    train: Dataset | None = ds_dict.get("train")
    test: Dataset | None = ds_dict.get("test")
    val: Dataset | None = ds_dict.get("validation") or ds_dict.get("val")

    # If validation split is absent, create one from the training set
    if train and val is None and val_ratio > 0.0:
        print(
            f"[*] Creating validation split ({int(val_ratio * 100)}%) from train split..."
        )
        split_dict = train.train_test_split(test_size=val_ratio, seed=seed)
        train, val = split_dict["train"], split_dict["test"]
    elif train is None and test is not None:
        train = test

    splits = {"train": train, "val": val, "test": test}
    return {k: v for k, v in splits.items() if v is not None}


def prepare_and_save_dataset(
    dataset_name: str,
    val_ratio: float = 0.1,
    seed: int = 42,
    modality: str | None = None,
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
    subset: str | None = None,
    force: bool | str = False,
    **load_dataset_kwargs: Any,
) -> dict[str, Path]:
    """Download dataset to data/raw/<modality>/, create splits, and save to data/processed/<modality>/.

    Parameters
    ----------
    dataset_name : str
        Dataset alias or canonical repository ID (e.g. 'cifar10', 'mnist', 'ylecun/mnist').
    val_ratio : float
        Proportion of training set to reserve for validation if none exists. Default is 0.1.
    seed : int
        Random seed for reproducible train/val splitting. Default is 42.
    modality : Optional[str]
        Modality category override ('vision', 'audio', etc.). Automatically resolved if None.
    raw_dir : Optional[Path | str]
        Custom directory for raw download cache. Defaults to '<root>/data/raw/<modality>/<dataset>'.
    processed_dir : Optional[Path | str]
        Custom directory for processed Parquet files. Defaults to '<root>/data/processed/<modality>/<dataset>'.
    subset : Optional[str]
        Optional dataset configuration/subset name if required.
    force : bool | str
        If True, forces re-download and re-export even if data exists locally. Default is False.
    **load_dataset_kwargs : Any
        Additional keyword arguments forwarded to datasets.load_dataset.

    Returns
    -------
    dict[str, Path]
        Dictionary mapping split names ('train', 'val', 'test') to their Parquet file paths.
    """
    should_force = force is True or str(force).strip().lower() in (
        "true",
        "1",
        "yes",
    )
    paths = DatasetPaths.from_name(
        dataset_name,
        modality=modality,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

    # Quick scan: verify if dataset is already cached and processed locally
    if not should_force and paths.has_processed_parquets():
        search_dirs = [
            paths.processed_dir,
            paths.processed_dir.parent.parent / paths.clean_name,
        ]
        existing_files: dict[str, Path] = {}
        for d in search_dirs:
            if d.exists():
                for p in d.glob("*.parquet"):
                    existing_files.setdefault(p.stem, p)
        if existing_files:
            summary = ", ".join(f"{s}.parquet" for s in sorted(existing_files.keys()))
            print(
                f"[⚡] Dataset '{dataset_name}' already exists locally at {paths.processed_dir}. "
                f"Skipping download ({summary})."
            )
            return existing_files

    paths.ensure_directories()

    load_kwargs = dict(load_dataset_kwargs)
    if "token" not in load_kwargs and "HF_TOKEN" in os.environ:
        load_kwargs["token"] = os.environ["HF_TOKEN"]

    target_repo = paths.canonical_id
    target_subset = subset
    if target_repo in HF_MODERN_MIRRORS:
        fallback_repo, fallback_subset = HF_MODERN_MIRRORS[target_repo]
        target_repo = fallback_repo
        target_subset = target_subset or fallback_subset

    if not target_subset:
        target_subset = DATASET_DEFAULT_CONFIGS.get(
            target_repo
        ) or DATASET_DEFAULT_CONFIGS.get(paths.canonical_id)

    print(f"[*] Downloading/caching raw '{target_repo}' into {paths.raw_dir}...")
    ds_dict = load_dataset(
        target_repo,
        name=target_subset,
        cache_dir=str(paths.raw_dir),
        **load_kwargs,
    )

    splits = _extract_dataset_splits(ds_dict, val_ratio=val_ratio, seed=seed)
    saved_paths: dict[str, Path] = {}

    for split_name, dataset in splits.items():
        file_path = paths.parquet_file(split_name)
        print(
            f"[*] Exporting {split_name} split ({len(dataset)} samples) to {file_path}..."
        )
        dataset.to_parquet(str(file_path))
        saved_paths[split_name] = file_path

    print(
        f"[✓] Successfully processed and saved '{dataset_name}' into {paths.processed_dir}"
    )
    return saved_paths


def prepare_modality_datasets(
    modality: str = "vision",
    val_ratio: float = 0.1,
    seed: int = 42,
    force: bool | str = False,
    **load_dataset_kwargs: Any,
) -> dict[str, dict[str, Path]]:
    """Download and prepare all benchmark datasets for a specific modality.

    Parameters
    ----------
    modality : str
        Modality category ('vision', 'audio', 'tabular', 'text', 'multimodal').
    val_ratio : float
        Proportion of training set to reserve for validation if none exists.
    seed : int
        Random seed for reproducible train/val splitting.
    force : bool | str
        If True, forces re-download and re-export even if data exists locally.
    **load_dataset_kwargs : Any
        Additional keyword arguments forwarded to datasets.load_dataset.

    Returns
    -------
    dict[str, dict[str, Path]]
        Dictionary mapping canonical dataset ID to split Parquet paths.
    """
    datasets = list_supported_datasets(modality)
    if not isinstance(datasets, list):
        datasets = [ds for ds_list in datasets.values() for ds in ds_list]

    print(f"[*] Preparing {len(datasets)} datasets for modality: {modality}")
    results: dict[str, dict[str, Path]] = {}
    for ds in datasets:
        try:
            results[ds] = prepare_and_save_dataset(
                ds,
                val_ratio=val_ratio,
                seed=seed,
                modality=modality,
                force=force,
                **load_dataset_kwargs,
            )
        except Exception as e:
            print(f"[✗] Skipping dataset '{ds}' due to error: {e}")
    return results
