"""
Custom Dataset Adapter Template

Copy this file into:
  counting/src/counting_dataset/adapters/<your_dataset>.py

Then:
  1) Replace <YOUR_DATASET_KEY> (e.g., "dota", "my_dataset")
  2) Implement iter_classes(), iter_images(), iter_annotations()
  3) Add your adapter to IndexBuilder.build([...]) to index it.

Design notes:
  - image_id is a stable hash of (dataset + original_relpath).
  - ann_id is a stable hash of (image_id + class_key + geometry + source [+ salt]).
  - role="instance" annotations are the canonical counted objects.
    Alternative geometry goes under role != "instance" and appears in target["aux"][role].
  - Only keep metadata in ImageRecord.meta / AnnotationRecord.meta that enables
    API-level filtering (meta_filter) or is needed in the target dict downstream.
    Prefer the raw dataset files as the authoritative source for everything else.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image

from counting_dataset.adapters.base import AdapterContext
from counting_dataset.core.ids import make_ann_id, make_image_id, normalize_relpath
from counting_dataset.core.schema import (
    AnnType,
    ClassRecord,
    ImageRecord,
    InstanceAnnotationRecord,
    SourceType,
    SplitType,
    Point,
    HBB,
    OBB,
    Polygon,
)


def _slugify(name: str) -> str:
    """Normalize class/category names into stable class_key suffixes."""
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"


class CustomDatasetAdapter:
    """
    Adapter for raw/<YOUR_RAW_DIR>/ ... (describe the dataset here).

    Raw layout:
      raw/<YOUR_RAW_DIR>/
        - ... (list files/directories that matter)
    """

    dataset = "<YOUR_DATASET_KEY>"

    def _dataset_root(self, ctx: AdapterContext) -> Path:
        return ctx.raw_root / "<YOUR_RAW_DIR>"

    @staticmethod
    def _split_dirs() -> Sequence[Tuple[str, SplitType]]:
        return [
            ("train", SplitType.TRAIN),
            ("val", SplitType.VAL),
            ("test", SplitType.TEST),
        ]

    def iter_classes(self, ctx: AdapterContext) -> Iterable[ClassRecord]:
        # TODO: Replace with real discovery logic.
        # Example (single-class):
        # yield ClassRecord(class_key=f"{self.dataset}/object", dataset=self.dataset, name="object")
        return
        yield  # pragma: no cover

    def iter_images(self, ctx: AdapterContext) -> Iterable[ImageRecord]:
        root = self._dataset_root(ctx)

        for split_dir, split in self._split_dirs():
            img_dir = root / split_dir / "images"
            if not img_dir.exists():
                continue

            for img_path in sorted(img_dir.glob("*.jpg")):
                rel = normalize_relpath(f"{split_dir}/images/{img_path.name}")
                abs_path = root / rel

                with Image.open(abs_path) as im:
                    width, height = im.size

                image_id = make_image_id(self.dataset, rel)

                yield ImageRecord(
                    image_id=image_id,
                    path=str(abs_path),
                    width=int(width),
                    height=int(height),
                    split=split,
                    dataset=self.dataset,
                    original_relpath=rel,
                    original_filename=img_path.name,
                    original_id=img_path.stem,
                    # Only include meta fields that enable meta_filter() filtering:
                    # meta={"scene": ..., "sensor": ...},
                )

    def iter_annotations(
        self, ctx: AdapterContext
    ) -> Iterable[InstanceAnnotationRecord]:
        root = self._dataset_root(ctx)

        for split_dir, _split in self._split_dirs():
            img_dir = root / split_dir / "images"
            ann_dir = root / split_dir / "annotations"
            if not img_dir.exists() or not ann_dir.exists():
                continue

            for img_path in sorted(img_dir.glob("*.jpg")):
                rel = normalize_relpath(f"{split_dir}/images/{img_path.name}")
                image_id = make_image_id(self.dataset, rel)

                # TODO: parse your annotation source for this image
                parsed: List[dict] = []

                for instance_index, obj in enumerate(parsed):
                    cat = obj.get("category", "object")
                    class_key = f"{self.dataset}/{_slugify(cat)}"

                    x, y, w, h = obj["bbox_xywh"]
                    geom = HBB(x=float(x), y=float(y), w=float(w), h=float(h))

                    yield InstanceAnnotationRecord(
                        ann_id=make_ann_id(
                            image_id=image_id,
                            class_key=class_key,
                            ann_type=AnnType.HBB,
                            geometry=geom,
                            source=SourceType.ORIGINAL,
                            instance_index=instance_index,
                        ),
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=AnnType.HBB,
                        geometry=geom,
                        role="instance",
                        instance_index=instance_index,
                        source=SourceType.ORIGINAL,
                        # Only include annotation meta that is useful downstream:
                        # meta={"difficult": obj.get("difficulty", 0)},
                    )
