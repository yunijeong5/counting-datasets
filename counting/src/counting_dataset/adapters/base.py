from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Protocol

from counting_dataset.core.schema import (
    ClassRecord,
    ImageRecord,
    InstanceAnnotationRecord,
    SplitType,
)


@dataclass(frozen=True)
class AdapterContext:
    """
    Common config passed to all adapters.
    raw_root: path to your repository's raw/ directory (dataset as they were released by the original authors).
    """

    raw_root: Path


class DatasetAdapter(Protocol):
    """
    Dataset adapter interface: raw dataset -> canonical records.
    Each adapter's output should be deterministic (stable order) to keep IDs stable.
    """

    # Short unique dataset key used in class_key and provenance.dataset
    dataset: str

    def iter_classes(self, ctx: AdapterContext) -> Iterable[ClassRecord]: ...

    def iter_images(self, ctx: AdapterContext) -> Iterable[ImageRecord]: ...

    def iter_annotations(
        self, ctx: AdapterContext
    ) -> Iterable[InstanceAnnotationRecord]: ...


def stable_sort_paths(paths: Iterable[Path]) -> List[Path]:
    """
    Deterministic ordering across platforms.
    """
    return sorted(paths, key=lambda p: str(p).replace("\\", "/"))


def infer_split_from_string(s: str) -> SplitType:
    s2 = (s or "").strip().lower()
    if s2 in ("train", "training", "trn"):
        return SplitType.TRAIN
    if s2 in ("valid", "validation", "val"):
        return SplitType.VAL
    if s2 in ("test", "testing", "tst"):
        return SplitType.TEST
    return SplitType.UNSPECIFIED
