from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from counting_dataset.adapters.base import AdapterContext
from counting_dataset.core.ids import make_ann_id, make_image_id, normalize_relpath
from counting_dataset.core.schema import (
    AnnType,
    ClassRecord,
    ImageRecord,
    InstanceAnnotationRecord,
    Point,
    Provenance,
    SourceType,
    SplitType,
)


class AerialElephantAdapter:
    """
    Adapter for raw/aerial-elephant-dataset/ with CSV metadata and point annotations.

    Raw layout:
      raw/aerial-elephant-dataset/
        - training_images/               (JPEG images)
        - test_images/                   (JPEG images)
        - training_images.csv            (image-level metadata)
        - test_images.csv
        - training_elephants.csv         (point annotations)
        - test_elephants.csv

    Strategy:
      - Define a single class: "aerial_elephant/elephant".
      - Map splits by filename prefix:
          training_* -> SplitType.TRAIN
          test_*     -> SplitType.TEST
      - Emit one POINT instance annotation per row in *_elephants.csv
        with source=SourceType.ORIGINAL.
      - Read image width/height from *_images.csv and store remaining
        metadata fields (e.g., GSD, altitude) in ImageRecord.meta.
    """

    dataset = "aerial_elephant"

    def _dataset_root(self, ctx: AdapterContext) -> Path:
        return ctx.raw_root / "aerial-elephant-dataset"

    def iter_classes(self, ctx: AdapterContext) -> Iterable[ClassRecord]:
        yield ClassRecord(
            class_key=f"{self.dataset}/elephant",
            dataset=self.dataset,
            name="elephant",
            meta={},
        )

    @staticmethod
    def _maybe_float(x):
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        try:
            return float(s)
        except Exception:
            return None

    @staticmethod
    def _maybe_int(x):
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        try:
            return int(float(s))
        except Exception:
            return None

    def _dataset_root_from_csv_parent(self, parent: Path) -> Path:
        # parent will be raw/aerial-elephant-dataset
        # (helper to keep the code safe even if called differently)
        if parent.name == "aerial-elephant-dataset":
            return parent
        # fallback: assume directory exists at expected location
        # (not strictly needed, but avoids accidental misuse)
        return parent / "aerial-elephant-dataset"

    def _read_image_csv(
        self, csv_path: Path, split_dir: str, split: SplitType
    ) -> Dict[str, ImageRecord]:
        """
        Returns mapping image_name -> ImageRecord for that split.
        """
        root = csv_path.parent
        ds_root = self._dataset_root_from_csv_parent(root)

        out: Dict[str, ImageRecord] = {}
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = str(row["image_name"]).strip()
                if not image_name:
                    continue

                # image_name is a hash-like stem; images are .jpg
                rel = normalize_relpath(f"{split_dir}/{image_name}.jpg")
                abs_path = ds_root / rel

                width = int(float(row.get("image_width", 0) or 0))
                height = int(float(row.get("image_height", 0) or 0))

                image_id = make_image_id(self.dataset, rel)

                meta = {
                    "sortie_id": self._maybe_int(row.get("sortie_id")),
                    "gsd": self._maybe_float(row.get("gsd")),
                    "measured_altitude": self._maybe_float(
                        row.get("measured_altitude")
                    ),
                    "terrain_altitude": self._maybe_float(row.get("terrain_altitude")),
                    "gps_altitude": self._maybe_float(row.get("gps_altitude")),
                }

                out[image_name] = ImageRecord(
                    image_id=image_id,
                    path=str(abs_path),
                    width=width,
                    height=height,
                    split=split,
                    provenance=Provenance(
                        dataset=self.dataset,
                        original_relpath=rel,
                        original_filename=f"{image_name}.jpg",
                        original_id=image_name,  # stable identifier used by CSVs
                        sha1=None,
                        size_bytes=None,
                    ),
                    counts={},  # builder fills from annotations
                    meta=meta,
                )
        return out

    def iter_images(self, ctx: AdapterContext) -> Iterable[ImageRecord]:
        root = self._dataset_root(ctx)

        train_images_csv = root / "training_images.csv"
        test_images_csv = root / "test_images.csv"

        train_map = self._read_image_csv(
            train_images_csv, "training_images", SplitType.TRAIN
        )
        test_map = self._read_image_csv(test_images_csv, "test_images", SplitType.TEST)

        # deterministic yield order
        for name in sorted(train_map.keys()):
            yield train_map[name]
        for name in sorted(test_map.keys()):
            yield test_map[name]

    def _read_elephant_csv(self, csv_path: Path) -> List[Tuple[str, float, float]]:
        """
        Returns list of (image_name, x, y) rows.
        """
        rows: List[Tuple[str, float, float]] = []
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = str(row["image_name"]).strip()
                if not image_name:
                    continue
                try:
                    x = float(row["x"])
                    y = float(row["y"])
                except Exception:
                    continue
                rows.append((image_name, x, y))
        # deterministic
        rows.sort(key=lambda t: (t[0], t[1], t[2]))
        return rows

    def iter_annotations(
        self, ctx: AdapterContext
    ) -> Iterable[InstanceAnnotationRecord]:
        root = self._dataset_root(ctx)

        # Build image_name -> (image_id, split_dir) map from image CSVs
        train_map = self._read_image_csv(
            root / "training_images.csv", "training_images", SplitType.TRAIN
        )
        test_map = self._read_image_csv(
            root / "test_images.csv", "test_images", SplitType.TEST
        )

        image_name_to_image_id: Dict[str, str] = {}
        image_name_to_split: Dict[str, str] = {}
        for name, rec in train_map.items():
            image_name_to_image_id[name] = rec.image_id
            image_name_to_split[name] = rec.split.value
        for name, rec in test_map.items():
            image_name_to_image_id[name] = rec.image_id
            image_name_to_split[name] = rec.split.value

        class_key = f"{self.dataset}/elephant"

        # instance_index should be unique per image across all instances
        per_image_counter: Dict[str, int] = {}

        # Iterate train then test (both deterministic)
        for split_label, csv_file in [
            ("train", root / "training_elephants.csv"),
            ("test", root / "test_elephants.csv"),
        ]:
            rows = self._read_elephant_csv(csv_file)
            for image_name, x, y in rows:
                image_id = image_name_to_image_id.get(image_name)
                if image_id is None:
                    # CSV inconsistency: point references unknown image
                    continue

                instance_index = per_image_counter.get(image_id, 0)
                per_image_counter[image_id] = instance_index + 1

                geom = Point(x=float(x), y=float(y))

                ann_id = make_ann_id(
                    image_id=image_id,
                    class_key=class_key,
                    ann_type=AnnType.POINT,
                    geometry=geom,
                    source=SourceType.ORIGINAL,
                    instance_index=instance_index,
                    salt=None,
                )

                yield InstanceAnnotationRecord(
                    ann_id=ann_id,
                    image_id=image_id,
                    class_key=class_key,
                    ann_type=AnnType.POINT,
                    geometry=geom,
                    source=SourceType.ORIGINAL,
                    score=None,
                    instance_index=instance_index,
                    meta={
                        "image_name": image_name,
                        "split": image_name_to_split.get(image_name, split_label),
                    },
                )
