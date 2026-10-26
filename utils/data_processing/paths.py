"""Path resolution and data folder organization for raw and processed datasets."""

from dataclasses import dataclass
from pathlib import Path

from utils.data_processing.catalog import get_dataset_modality, resolve_dataset_name
from utils.data_processing.config import get_project_root


@dataclass(frozen=True)
class DatasetPaths:
    """Encapsulates local storage paths and file resolution for a dataset organized by modality."""

    canonical_id: str
    clean_name: str
    modality: str
    raw_dir: Path
    processed_dir: Path

    @classmethod
    def from_name(
        cls,
        dataset_name: str,
        modality: str | None = None,
        raw_dir: Path | str | None = None,
        processed_dir: Path | str | None = None,
    ) -> "DatasetPaths":
        """Resolve paths for a given dataset name organized by modality folder."""
        canonical = resolve_dataset_name(dataset_name)
        clean = canonical.replace("/", "_").lower()
        mod = (modality or get_dataset_modality(canonical)).lower()
        root = get_project_root()

        raw = Path(raw_dir) if raw_dir else root / "data" / "raw" / mod / clean
        proc = (
            Path(processed_dir)
            if processed_dir
            else root / "data" / "processed" / mod / clean
        )
        return cls(
            canonical_id=canonical,
            clean_name=clean,
            modality=mod,
            raw_dir=raw,
            processed_dir=proc,
        )

    def parquet_file(self, split: str) -> Path:
        """Return the expected path for a specific Parquet split file."""
        mod_file = self.processed_dir / f"{split}.parquet"
        if mod_file.exists():
            return mod_file
        # Backward compatibility fallback for legacy flat path
        legacy_file = (
            self.processed_dir.parent.parent / self.clean_name / f"{split}.parquet"
        )
        if legacy_file.exists():
            return legacy_file
        return mod_file

    def has_processed_parquets(self) -> bool:
        """Check if preprocessed Parquet files already exist (modality-aware or legacy)."""
        if self.processed_dir.exists() and any(self.processed_dir.glob("*.parquet")):
            return True
        legacy = self.processed_dir.parent.parent / self.clean_name
        return legacy.exists() and any(legacy.glob("*.parquet"))

    def ensure_directories(self) -> None:
        """Create raw and processed directories if they do not exist."""
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
