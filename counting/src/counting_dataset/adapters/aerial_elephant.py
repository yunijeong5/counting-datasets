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
    """

    dataset = "aerial_elephant"

    def _dataset_root(self, ctx: AdapterContext) -> Path:
        return ctx.raw_root / "aerial-elephant-dataset"

    def iter_classes(self, ctx: AdapterContext) -> Iterable[ClassRecord]:
        yield ClassRecord(
            class_key=f"{self.dataset}/elephant",
            dataset=self.dataset,
            name="elephant",
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

    def _read_image_csv(
        self, csv_path: Path, split_dir: str, split: SplitType
    ) -> Dict[str, ImageRecord]:
        root = csv_path.parent

        out: Dict[str, ImageRecord] = {}
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_name = str(row["image_name"]).strip()
                if not image_name:
                    continue

                rel = normalize_relpath(f"{split_dir}/{image_name}.jpg")
                abs_path = root / rel

                width = int(float(row.get("image_width", 0) or 0))
                height = int(float(row.get("image_height", 0) or 0))

                image_id = make_image_id(self.dataset, rel)

                out[image_name] = ImageRecord(
                    image_id=image_id,
                    path=str(abs_path),
                    width=width,
                    height=height,
                    split=split,
                    dataset=self.dataset,
                    original_relpath=rel,
                    original_filename=f"{image_name}.jpg",
                    original_id=image_name,
                )
        return out

    def iter_images(self, ctx: AdapterContext) -> Iterable[ImageRecord]:
        root = self._dataset_root(ctx)

        train_map = self._read_image_csv(
            root / "training_images.csv", "training_images", SplitType.TRAIN
        )
        test_map = self._read_image_csv(
            root / "test_images.csv", "test_images", SplitType.TEST
        )

        for name in sorted(train_map.keys()):
            yield train_map[name]
        for name in sorted(test_map.keys()):
            yield test_map[name]

    def _read_elephant_csv(self, csv_path: Path) -> List[Tuple[str, float, float]]:
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
        rows.sort(key=lambda t: (t[0], t[1], t[2]))
        return rows

    def iter_annotations(
        self, ctx: AdapterContext
    ) -> Iterable[InstanceAnnotationRecord]:
        root = self._dataset_root(ctx)

        train_map = self._read_image_csv(
            root / "training_images.csv", "training_images", SplitType.TRAIN
        )
        test_map = self._read_image_csv(
            root / "test_images.csv", "test_images", SplitType.TEST
        )

        image_name_to_image_id: Dict[str, str] = {}
        for name, rec in {**train_map, **test_map}.items():
            image_name_to_image_id[name] = rec.image_id

        class_key = f"{self.dataset}/elephant"
        per_image_counter: Dict[str, int] = {}

        for _split_label, csv_file in [
            ("train", root / "training_elephants.csv"),
            ("test", root / "test_elephants.csv"),
        ]:
            rows = self._read_elephant_csv(csv_file)
            for image_name, x, y in rows:
                image_id = image_name_to_image_id.get(image_name)
                if image_id is None:
                    continue

                instance_index = per_image_counter.get(image_id, 0)
                per_image_counter[image_id] = instance_index + 1

                geom = Point(x=float(x), y=float(y))

                yield InstanceAnnotationRecord(
                    ann_id=make_ann_id(
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=AnnType.POINT,
                        geometry=geom,
                        source=SourceType.ORIGINAL,
                        instance_index=instance_index,
                        salt=None,
                    ),
                    image_id=image_id,
                    class_key=class_key,
                    ann_type=AnnType.POINT,
                    geometry=geom,
                    source=SourceType.ORIGINAL,
                    instance_index=instance_index,
                )
