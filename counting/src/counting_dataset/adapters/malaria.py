from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from counting_dataset.adapters.base import AdapterContext, DatasetAdapter
from counting_dataset.core.ids import make_ann_id, make_image_id, normalize_relpath
from counting_dataset.core.schema import (
    SplitType,
    AnnType,
    SourceType,
    HBB,
    ClassRecord,
    ImageRecord,
    InstanceAnnotationRecord,
)


def _slugify_class_name(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "unknown"


class MalariaAdapter(DatasetAdapter):
    """
    Adapter for raw/malaria/ with JSON annotations and bounding boxes.

    Raw layout:
      raw/malaria/
        - images/            (*.png)
        - training.json
        - test.json
    """

    dataset = "malaria"

    def _dataset_root(self, ctx: AdapterContext) -> Path:
        return ctx.raw_root / "malaria"

    def _load_list_json(self, path: Path) -> List[dict]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list JSON at {path}, got {type(data)}")
        return data

    def _pathname_to_relpath(self, pathname: str) -> str:
        p = (pathname or "").strip()
        if p.startswith("/"):
            p = p[1:]
        return normalize_relpath(p)

    def _parse_bbox_rc(self, obj: dict) -> tuple[float, float, float, float]:
        bb = obj.get("bounding_box", {}) or {}
        mn = bb.get("minimum", {}) or {}
        mx = bb.get("maximum", {}) or {}
        rmin = float(mn.get("r", 0))
        cmin = float(mn.get("c", 0))
        rmax = float(mx.get("r", 0))
        cmax = float(mx.get("c", 0))
        return rmin, cmin, rmax, cmax

    def _rc_to_xywh(self, rmin: float, cmin: float, rmax: float, cmax: float) -> HBB:
        return HBB(x=cmin, y=rmin, w=max(0.0, cmax - cmin), h=max(0.0, rmax - rmin))

    def _iter_entries(self, ctx: AdapterContext) -> Iterable[Tuple[SplitType, dict]]:
        root = self._dataset_root(ctx)
        train_entries = self._load_list_json(root / "training.json")
        test_entries = self._load_list_json(root / "test.json")

        def _key(e: dict) -> Tuple[str, str]:
            img = e.get("image", {}) or {}
            return (str(img.get("pathname", "")), str(img.get("checksum", "")))

        for e in sorted(train_entries, key=_key):
            yield (SplitType.TRAIN, e)
        for e in sorted(test_entries, key=_key):
            yield (SplitType.TEST, e)

    def iter_classes(self, ctx: AdapterContext) -> Iterable[ClassRecord]:
        seen: Dict[str, str] = {}
        for _, entry in self._iter_entries(ctx):
            for obj in entry.get("objects", []) or []:
                cat = str(obj.get("category", "")).strip()
                slug = _slugify_class_name(cat)
                if slug not in seen:
                    seen[slug] = cat or slug

        for slug in sorted(seen.keys()):
            yield ClassRecord(
                class_key=f"{self.dataset}/{slug}",
                dataset=self.dataset,
                name=seen[slug],
            )

    def iter_images(self, ctx: AdapterContext) -> Iterable[ImageRecord]:
        root = self._dataset_root(ctx)
        seen_relpaths: Set[str] = set()

        for split, entry in self._iter_entries(ctx):
            img = entry.get("image", {}) or {}

            checksum = str(img.get("checksum", "")).strip() or None
            pathname = str(img.get("pathname", "")).strip()
            shape = img.get("shape", {}) or {}

            height = int(shape.get("r", 0))
            width = int(shape.get("c", 0))

            relpath = self._pathname_to_relpath(pathname)
            if relpath in seen_relpaths:
                continue
            seen_relpaths.add(relpath)

            image_id = make_image_id(self.dataset, relpath)
            abs_path = root / relpath

            yield ImageRecord(
                image_id=image_id,
                path=str(abs_path),
                width=width,
                height=height,
                split=split,
                dataset=self.dataset,
                original_relpath=relpath,
                original_filename=Path(relpath).name,
                original_id=checksum,
            )

    def iter_annotations(
        self, ctx: AdapterContext
    ) -> Iterable[InstanceAnnotationRecord]:
        # Build slug set for class_key consistency
        slug_set: Set[str] = set()
        for cr in self.iter_classes(ctx):
            slug_set.add(cr.class_key.split("/", 1)[1])

        for split, entry in self._iter_entries(ctx):
            img = entry.get("image", {}) or {}
            pathname = str(img.get("pathname", "")).strip()
            checksum = str(img.get("checksum", "")).strip() or None

            relpath = self._pathname_to_relpath(pathname)
            image_id = make_image_id(self.dataset, relpath)

            objects = entry.get("objects", []) or []

            def _obj_key(t):
                idx, obj = t
                cat = str(obj.get("category", "")).strip()
                rmin, cmin, rmax, cmax = self._parse_bbox_rc(obj)
                return (_slugify_class_name(cat), rmin, cmin, rmax, cmax, idx)

            indexed_sorted = sorted(enumerate(objects), key=_obj_key)

            for instance_index, (_orig_idx, obj) in enumerate(indexed_sorted):
                cat = str(obj.get("category", "")).strip()
                slug = _slugify_class_name(cat)
                class_key = f"{self.dataset}/{slug}"
                rmin, cmin, rmax, cmax = self._parse_bbox_rc(obj)
                geom = self._rc_to_xywh(rmin, cmin, rmax, cmax)

                yield InstanceAnnotationRecord(
                    ann_id=make_ann_id(
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=AnnType.HBB,
                        geometry=geom,
                        source=SourceType.ORIGINAL,
                        instance_index=instance_index,
                        salt=checksum or "",
                    ),
                    image_id=image_id,
                    class_key=class_key,
                    ann_type=AnnType.HBB,
                    geometry=geom,
                    source=SourceType.ORIGINAL,
                    instance_index=instance_index,
                )
