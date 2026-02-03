"""
Custom Dataset Adapter Template

Copy this file into:
  counting/src/counting_dataset/adapters/<your_dataset>.py

Then:
  1) Replace <YOUR_DATASET_KEY> (e.g., "dota", "my_dataset")
  2) Implement:
       - iter_classes()
       - iter_images()
       - iter_annotations()
  3) Add your adapter to IndexBuilder.build([...]) to index it.

Design notes (matches this repo's schema):
  - image_id is a stable hash of (dataset + original_relpath).
  - ann_id is a stable hash of (image_id + class_key + geometry + source [+ salt]).
  - role="instance" annotations are the canonical counted objects.
    Any alternative or auxiliary geometry should use role != "instance" and will show up under target["aux"][role].
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from counting_dataset.adapters.base import AdapterContext
from counting_dataset.core.ids import make_ann_id, make_image_id, normalize_relpath
from counting_dataset.core.schema import (
    AnnType,
    ClassRecord,
    ImageRecord,
    InstanceAnnotationRecord,
    Provenance,
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

    Strategy:
      - Define class keys:
          class_key = "<YOUR_DATASET_KEY>/<slugified_class_name>"
      - Define split mapping:
          train -> SplitType.TRAIN, val -> SplitType.VAL, test -> SplitType.TEST
        (or SplitType.UNSPECIFIED if no split info is available)
      - Emit canonical counted objects with:
          role="instance"
      - Emit auxiliary / alternative annotations with:
          role="<something_descriptive>"  (e.g., "exemplar", "hbb", "prompt", "mask", ...)
      - Store dataset-specific per-image metadata in ImageRecord.meta (sensor, gsd, etc.)
      - Store dataset-specific per-annotation metadata in InstanceAnnotationRecord.meta
        (difficulty flags, annotator IDs, original IDs, etc.)
    """

    # Dataset key used in class_key and image_id hashing.
    dataset = "<YOUR_DATASET_KEY>"

    # -----------------------
    # Paths / discovery helpers
    # -----------------------

    def _dataset_root(self, ctx: AdapterContext) -> Path:
        """
        Return the dataset root under ctx.raw_root.

        Example:
          return ctx.raw_root / "my_dataset_v1"
        """
        return ctx.raw_root / "<YOUR_RAW_DIR>"

    # Optional: define split dirs if the dataset is laid out by split on disk.
    @staticmethod
    def _split_dirs() -> Sequence[Tuple[str, SplitType]]:
        """
        If your dataset is organized as raw/<dir>/{train,val,test}/..., define it here.
        Otherwise, you can ignore this and yield SplitType.UNSPECIFIED for all images.
        """
        return [
            ("train", SplitType.TRAIN),
            ("val", SplitType.VAL),
            ("test", SplitType.TEST),
        ]

    # -----------------------
    # Required adapter interface
    # -----------------------

    def iter_classes(self, ctx: AdapterContext) -> Iterable[ClassRecord]:
        """
        Yield all classes in this dataset.

        Tip:
          - If your dataset has a taxonomy/categories file, parse it here.
          - If not, you can discover classes by scanning annotation files.
        """
        # TODO: Replace this with real discovery logic.
        # Example: single-class dataset:
        # yield ClassRecord(class_key=f"{self.dataset}/object", dataset=self.dataset, name="object", meta={})
        return
        yield  # pragma: no cover

    def iter_images(self, ctx: AdapterContext) -> Iterable[ImageRecord]:
        """
        Yield all images with provenance + split + width/height.

        Requirements:
          - ImageRecord.path must be the *absolute* path to the image on disk.
          - Provenance.original_relpath must be relative to the dataset root (under raw/).
          - image_id must be derived from (dataset, original_relpath) using make_image_id().
        """
        root = self._dataset_root(ctx)

        # TODO: Replace with your dataset’s image enumeration logic.
        # Example (split-based directory):
        for split_dir, split in self._split_dirs():
            img_dir = root / split_dir / "images"
            if not img_dir.exists():
                continue

            for img_path in sorted(img_dir.glob("*.jpg")):
                rel = normalize_relpath(f"{split_dir}/images/{img_path.name}")
                abs_path = root / rel

                # Prefer reading true dimensions from the file.
                with Image.open(abs_path) as im:
                    width, height = im.size

                image_id = make_image_id(self.dataset, rel)

                meta = {
                    # TODO: dataset-specific metadata fields
                    # "sensor": ...,
                }

                yield ImageRecord(
                    image_id=image_id,
                    path=str(abs_path),
                    width=int(width),
                    height=int(height),
                    split=split,
                    provenance=Provenance(
                        dataset=self.dataset,
                        original_relpath=rel,
                        original_filename=img_path.name,
                        original_id=img_path.stem,  # or another stable image identifier
                        sha1=None,
                        size_bytes=None,
                    ),
                    counts={},  # filled by IndexBuilder aggregates
                    meta=meta,
                )

    def iter_annotations(
        self, ctx: AdapterContext
    ) -> Iterable[InstanceAnnotationRecord]:
        """
        Yield all annotations.

        Canonical counting annotations must use:
          role="instance"

        Examples:
          - Points:  AnnType.POINT + Point(x,y)
          - HBB:     AnnType.HBB   + HBB(x,y,w,h)
          - OBB:     AnnType.OBB   + OBB(corners=(x1,y1,...,x4,y4))
          - Polygon: AnnType.POLYGON + Polygon(points=(x1,y1,...))
        """
        root = self._dataset_root(ctx)

        # TODO: Replace with your dataset’s annotation parsing.
        # The skeleton below assumes you can iterate images and find an annotation file per image.
        for split_dir, split in self._split_dirs():
            img_dir = root / split_dir / "images"
            ann_dir = root / split_dir / "annotations"
            if not img_dir.exists() or not ann_dir.exists():
                continue

            for img_path in sorted(img_dir.glob("*.jpg")):
                rel = normalize_relpath(f"{split_dir}/images/{img_path.name}")
                image_id = make_image_id(self.dataset, rel)

                # TODO: parse your annotation source for this image
                # Example placeholders:
                # parsed = [{"category": "object", "bbox_xywh": (x,y,w,h), "difficulty": 0}, ...]

                parsed: List[dict] = []

                instance_index = 0
                for obj in parsed:
                    cat = obj.get("category", "object")
                    class_key = f"{self.dataset}/{_slugify(cat)}"

                    # TODO: choose geometry type based on dataset
                    # Example HBB:
                    x, y, w, h = obj["bbox_xywh"]
                    geom = HBB(x=float(x), y=float(y), w=float(w), h=float(h))
                    ann_type = AnnType.HBB

                    ann_id = make_ann_id(
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=ann_type,
                        geometry=geom,
                        source=SourceType.ORIGINAL,
                        instance_index=instance_index,
                        # Use salt if you need extra disambiguation when geometry may repeat
                        salt=f"{split_dir}:{img_path.name}:{instance_index}",
                    )

                    yield InstanceAnnotationRecord(
                        ann_id=ann_id,
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=ann_type,
                        geometry=geom,
                        role="instance",  # canonical counted objects
                        instance_index=instance_index,
                        source=SourceType.ORIGINAL,  # TODO: use best fitting source type
                        score=None,
                        meta={
                            "split": split.value,
                            "image_name": img_path.name,
                            # TODO: add dataset-specific fields
                            # "difficult": obj.get("difficulty", 0),
                            # "original_ann_id": obj.get("ann_id"),
                        },
                    )

                    instance_index += 1
