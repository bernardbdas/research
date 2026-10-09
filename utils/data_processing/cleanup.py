"""Disk and cache cleanup utilities for raw and processed datasets."""

import shutil

from utils.data_processing.catalog import resolve_dataset_name
from utils.data_processing.config import get_project_root


def clean_raw_data() -> None:
    """Clean all cached raw downloads from data/raw/ while preserving .gitkeep."""
    raw_dir = get_project_root() / "data" / "raw"
    if not raw_dir.exists():
        print("[✓] Raw directory does not exist. Nothing to clean.")
        return

    count = 0
    for p in raw_dir.iterdir():
        if p.name != ".gitkeep":
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
            count += 1
    print(f"[✓] Raw download cache cleaned from data/raw/ ({count} entries removed).")


def clean_processed_data() -> None:
    """Clean all preprocessed Parquet files from data/processed/ while preserving .gitkeep."""
    proc_dir = get_project_root() / "data" / "processed"
    if not proc_dir.exists():
        print("[✓] Processed directory does not exist. Nothing to clean.")
        return

    count = 0
    for p in proc_dir.iterdir():
        if p.name != ".gitkeep":
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
            count += 1
    print(
        f"[✓] Processed Parquet files cleaned from data/processed/ ({count} entries removed)."
    )


def clean_modality_data(modality: str) -> None:
    """Clean all data for a specific modality from data/raw/ and data/processed/."""
    root = get_project_root()
    mod = modality.strip().lower()
    cleaned = 0
    for parent in [root / "data" / "raw", root / "data" / "processed"]:
        target = parent / mod
        if target.exists() and target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
            cleaned += 1
    print(f"[✓] Cleaned data for modality '{mod}' ({cleaned} directories removed).")


def clean_dataset_data(dataset_name: str | None = None) -> None:
    """Delete raw downloads and processed Parquets for a dataset, or all datasets if none specified."""
    target = (dataset_name or "").strip()
    root = get_project_root()
    raw_dir = root / "data" / "raw"
    proc_dir = root / "data" / "processed"

    if not target:
        # Nuclear clean of all datasets
        clean_raw_data()
        clean_processed_data()
        print("[✓] All cached raw and processed datasets deleted from data/.")
        return

    canonical = resolve_dataset_name(target)
    slugs = {
        target,
        canonical.replace("/", "_").lower(),
        target.replace("/", "_").lower(),
        target.split("/")[-1].lower(),
    }

    deleted_count = 0
    for parent in [raw_dir, proc_dir]:
        if not parent.exists():
            continue
        for s in slugs:
            for p in list(parent.rglob(s)):
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                    deleted_count += 1

    print(
        f"[✓] Deleted data for dataset '{target}' ({deleted_count} directories removed)."
    )
