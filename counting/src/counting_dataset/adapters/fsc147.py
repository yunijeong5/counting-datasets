from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image

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
    HBB,
)


def _slugify(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"


class FSC147Adapter:
    """
    Adapter for raw/FSC147_384/ with single-class counting targets and exemplar boxes.

    Raw layout:
      raw/FSC147_384/
        - images_384_VarV2/
        - ImageClasses_FSC147.txt
        - Train_Test_Val_FSC_147.json
        - annotation_FSC147_384.json
        - gt_density_map_adaptive_384_VarV2/   (optional)

    Annotations:
      - POINT instances per image (ground-truth count annotations).
      - HBB exemplar boxes (role="exemplar") — the 3 user-provided reference boxes.
        exemplar_index in meta identifies which of the 3 this is.
    Image meta:
      - density_map_relpath: path to the .npy density map, if it exists.
    """

    dataset = "fsc147"

    def _dataset_root(self, ctx: AdapterContext) -> Path:
        return ctx.raw_root / "FSC147_384"

    def _load_image_to_class(self, ctx: AdapterContext) -> Dict[str, str]:
        root = self._dataset_root(ctx)
        path = root / "ImageClasses_FSC147.txt"

        mapping: Dict[str, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = re.split(r"\t+", line)
                if len(parts) < 2:
                    parts = line.split(maxsplit=1)
                    if len(parts) < 2:
                        continue
                img = parts[0].strip()
                cls = parts[1].strip()
                if not img or not cls:
                    continue
                mapping[img] = f"{self.dataset}/{_slugify(cls)}"
        return mapping

    def _load_splits(self, ctx: AdapterContext) -> Dict[str, SplitType]:
        root = self._dataset_root(ctx)
        path = root / "Train_Test_Val_FSC_147.json"
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        train = set(obj.get("train", []) or [])
        val = set(obj.get("val", []) or []) | set(obj.get("val_coco", []) or [])
        test = set(obj.get("test", []) or []) | set(obj.get("test_coco", []) or [])

        split_map: Dict[str, SplitType] = {}
        for im in sorted(test):
            split_map[im] = SplitType.TEST
        for im in sorted(val):
            split_map[im] = SplitType.VAL
        for im in sorted(train):
            split_map[im] = SplitType.TRAIN
        return split_map

    def _load_annotations(self, ctx: AdapterContext) -> Dict[str, dict]:
        root = self._dataset_root(ctx)
        path = root / "annotation_FSC147_384.json"
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _corners_to_hbb(corners: List[List[float]]) -> Optional[HBB]:
        if not isinstance(corners, list) or len(corners) == 0:
            return None
        xs, ys = [], []
        for p in corners:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                continue
            try:
                xs.append(float(p[0]))
                ys.append(float(p[1]))
            except Exception:
                continue
        if not xs or not ys:
            return None
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return HBB(x=x0, y=y0, w=max(0.0, x1 - x0), h=max(0.0, y1 - y0))

    def iter_classes(self, ctx: AdapterContext) -> Iterable[ClassRecord]:
        image_to_ck = self._load_image_to_class(ctx)
        ck_to_name: Dict[str, str] = {}
        for _img, ck in image_to_ck.items():
            name = ck.split("/", 1)[1] if "/" in ck else ck
            ck_to_name.setdefault(ck, name)

        for ck in sorted(ck_to_name.keys()):
            yield ClassRecord(
                class_key=ck,
                dataset=self.dataset,
                name=ck_to_name[ck],
            )

    def iter_images(self, ctx: AdapterContext) -> Iterable[ImageRecord]:
        root = self._dataset_root(ctx)

        image_to_ck = self._load_image_to_class(ctx)
        split_map = self._load_splits(ctx)
        ann = self._load_annotations(ctx)

        for img_name in sorted(set(split_map.keys())):
            if img_name not in image_to_ck:
                continue

            rel = normalize_relpath(f"images_384_VarV2/{img_name}")
            abs_path = root / rel
            split = split_map[img_name]

            with Image.open(abs_path) as im:
                width, height = im.size

            image_id = make_image_id(self.dataset, rel)

            # Density map path (optional auxiliary file)
            stem = Path(img_name).stem
            density_rel = normalize_relpath(f"gt_density_map_adaptive_384_VarV2/{stem}.npy")
            density_abs = root / density_rel
            density_rel_out = density_rel if density_abs.exists() else None

            yield ImageRecord(
                image_id=image_id,
                path=str(abs_path),
                width=int(width),
                height=int(height),
                split=split,
                dataset=self.dataset,
                original_relpath=rel,
                original_filename=img_name,
                original_id=img_name,
                meta={"density_map_relpath": density_rel_out},
            )

    def iter_annotations(
        self, ctx: AdapterContext
    ) -> Iterable[InstanceAnnotationRecord]:
        image_to_ck = self._load_image_to_class(ctx)
        split_map = self._load_splits(ctx)
        ann = self._load_annotations(ctx)

        per_image_counter: Dict[str, int] = {}

        for img_name in sorted(set(split_map.keys())):
            if img_name not in image_to_ck:
                continue

            a = ann.get(img_name, None) if isinstance(ann, dict) else None
            if not isinstance(a, dict):
                continue

            pts = a.get("points", None)
            if not isinstance(pts, list):
                continue

            rel = normalize_relpath(f"images_384_VarV2/{img_name}")
            image_id = make_image_id(self.dataset, rel)
            class_key = image_to_ck[img_name]

            per_image_counter.setdefault(image_id, 0)

            for p_i, p in enumerate(pts):
                if not isinstance(p, (list, tuple)) or len(p) != 2:
                    continue
                try:
                    x = float(p[0])
                    y = float(p[1])
                except Exception:
                    continue

                geom = Point(x=x, y=y)
                instance_index = per_image_counter[image_id]
                per_image_counter[image_id] += 1

                yield InstanceAnnotationRecord(
                    ann_id=make_ann_id(
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=AnnType.POINT,
                        geometry=geom,
                        source=SourceType.ORIGINAL,
                        instance_index=instance_index,
                        salt=str(p_i),
                    ),
                    image_id=image_id,
                    class_key=class_key,
                    ann_type=AnnType.POINT,
                    geometry=geom,
                    source=SourceType.ORIGINAL,
                    instance_index=instance_index,
                )

            boxes = a.get("box_examples_coordinates", None)
            if isinstance(boxes, list):
                for ex_i, corners in enumerate(boxes):
                    hbb = self._corners_to_hbb(corners)
                    if hbb is None:
                        continue

                    instance_index = per_image_counter.get(image_id, 0)
                    per_image_counter[image_id] = instance_index + 1

                    yield InstanceAnnotationRecord(
                        ann_id=make_ann_id(
                            image_id=image_id,
                            class_key=class_key,
                            ann_type=AnnType.HBB,
                            geometry=hbb,
                            source=SourceType.ORIGINAL,
                            instance_index=instance_index,
                            salt=f"exemplar:{ex_i}",
                        ),
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=AnnType.HBB,
                        geometry=hbb,
                        source=SourceType.ORIGINAL,
                        instance_index=instance_index,
                        role="exemplar",
                        meta={"exemplar_index": ex_i},
                    )
