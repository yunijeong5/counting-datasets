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
    Provenance,
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
        - images_384_VarV2/                  (resized JPEG images)
        - ImageClasses_FSC147.txt            (image-to-class mapping)
        - Train_Test_Val_FSC_147.json        (train/val/test splits)
        - annotation_FSC147_384.json         (points and exemplar boxes)
        - gt_density_map_adaptive_384_VarV2/ (optional density maps)

    Strategy:
      - Create one class per category listed in ImageClasses_FSC147.txt:
          class_key = "fsc147/<slugified_class_name>"
      - Assign splits using Train_Test_Val_FSC_147.json
        (unioning *_coco variants where present).
      - Emit one POINT instance annotation per annotated point.
      - Emit exemplar boxes as auxiliary annotations:
          ann_type=HBB, role="exemplar"
      - Store density map paths and other per-image metadata in ImageRecord.meta.
    """

    dataset = "fsc147"

    def _dataset_root(self, ctx: AdapterContext) -> Path:
        return ctx.raw_root / "FSC147_384"

    def _load_image_to_class(self, ctx: AdapterContext) -> Dict[str, str]:
        """
        Returns mapping: image_name (e.g., "2.jpg") -> class_key (e.g., "fsc147/sea_shells")
        """
        root = self._dataset_root(ctx)
        path = root / "ImageClasses_FSC147.txt"

        mapping: Dict[str, str] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # lines look like: "2.jpg\tsea shells"
                parts = re.split(r"\t+", line)
                if len(parts) < 2:
                    # sometimes space-separated; fall back
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
        """
        Returns mapping: image_name -> split type, based on union of {val,val_coco} and {test,test_coco}.
        """
        root = self._dataset_root(ctx)
        path = root / "Train_Test_Val_FSC_147.json"
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        train = set(obj.get("train", []) or [])
        val = set(obj.get("val", []) or []) | set(obj.get("val_coco", []) or [])
        test = set(obj.get("test", []) or []) | set(obj.get("test_coco", []) or [])

        # If overlaps exist, prefer train > val > test deterministically
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
        """
        corners: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        Converts to axis-aligned bbox (HBB) in xywh.
        """
        if not isinstance(corners, list) or len(corners) == 0:
            return None
        xs = []
        ys = []
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
        # collect unique class_keys
        ck_to_name: Dict[str, str] = {}
        for img, ck in image_to_ck.items():
            # recover display name from slug (best-effort). If you prefer the raw name, we could store it while parsing.
            name = ck.split("/", 1)[1] if "/" in ck else ck
            ck_to_name.setdefault(ck, name)

        for ck in sorted(ck_to_name.keys()):
            yield ClassRecord(
                class_key=ck,
                dataset=self.dataset,
                name=ck_to_name[ck],
                meta={},
            )

    def iter_images(self, ctx: AdapterContext) -> Iterable[ImageRecord]:
        root = self._dataset_root(ctx)

        image_to_ck = self._load_image_to_class(ctx)
        split_map = self._load_splits(ctx)
        ann = self._load_annotations(ctx)

        # Use union of known images from split_map (more authoritative for availability)
        image_names = sorted(set(split_map.keys()))

        for img_name in image_names:
            if img_name not in image_to_ck:
                # If class mapping missing, skip (should be rare)
                continue

            rel = normalize_relpath(f"images_384_VarV2/{img_name}")
            abs_path = root / rel
            split = split_map[img_name]

            # Prefer actual on-disk resized dimensions (matches coord space)
            with Image.open(abs_path) as im:
                width, height = im.size

            image_id = make_image_id(self.dataset, rel)

            # Pull per-image annotation entry if present
            a = ann.get(img_name, {}) if isinstance(ann, dict) else {}

            # Density map relpath (optional)
            stem = Path(img_name).stem
            density_rel = normalize_relpath(
                f"gt_density_map_adaptive_384_VarV2/{stem}.npy"
            )
            density_abs = root / density_rel
            density_rel_out = density_rel if density_abs.exists() else None

            meta = {
                "class_key": image_to_ck[
                    img_name
                ],  # redundant but convenient; single class per image.
                "density_map_relpath": density_rel_out,
                # keep original H/W + ratios if available
                "orig_H": a.get("H"),
                "orig_W": a.get("W"),
                "ratio_h": a.get("ratio_h"),
                "ratio_w": a.get("ratio_w"),
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
                    original_filename=img_name,
                    original_id=img_name,
                    sha1=None,
                    size_bytes=None,
                ),
                counts={},  # builder fills from annotations
                meta=meta,
            )

    def iter_annotations(
        self, ctx: AdapterContext
    ) -> Iterable[InstanceAnnotationRecord]:
        image_to_ck = self._load_image_to_class(ctx)
        split_map = self._load_splits(ctx)
        ann = self._load_annotations(ctx)

        # instance_index per image
        per_image_counter: Dict[str, int] = {}

        # deterministic iteration over image names
        for img_name in sorted(set(split_map.keys())):
            if img_name not in image_to_ck:
                continue

            # Ensure annotation exists
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

                ann_id = make_ann_id(
                    image_id=image_id,
                    class_key=class_key,
                    ann_type=AnnType.POINT,
                    geometry=geom,
                    source=SourceType.ORIGINAL,
                    instance_index=instance_index,
                    salt=str(p_i),
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
                        "image_name": img_name,
                        "split": split_map[img_name].value,
                    },
                )

            boxes = a.get("box_examples_coordinates", None)
            if isinstance(boxes, list):
                for ex_i, corners in enumerate(boxes):
                    hbb = self._corners_to_hbb(corners)
                    if hbb is None:
                        continue

                    instance_index = per_image_counter.get(image_id, 0)
                    per_image_counter[image_id] = instance_index + 1

                    ann_id = make_ann_id(
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=AnnType.HBB,
                        geometry=hbb,
                        source=SourceType.ORIGINAL,
                        instance_index=instance_index,
                        salt=f"exemplar:{ex_i}",
                    )

                    yield InstanceAnnotationRecord(
                        ann_id=ann_id,
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=AnnType.HBB,
                        geometry=hbb,
                        source=SourceType.ORIGINAL,
                        score=None,
                        instance_index=instance_index,
                        meta={
                            "image_name": img_name,
                            "split": split_map[img_name].value,
                            "is_exemplar": True,
                            "exemplar_index": ex_i,
                        },
                        role="exemplar",
                    )
