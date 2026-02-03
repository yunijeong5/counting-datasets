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
    Provenance,
    ClassRecord,
    ImageRecord,
    InstanceAnnotationRecord,
)


def _slugify_class_name(name: str) -> str:
    """
    Convert a category string like "red blood cell" into a stable identifier
    used in class_key. Keep it simple and deterministic.
    """
    s = (name or "").strip().lower()
    # Replace any run of non-alnum with underscores
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

    Strategy:
      - Treat each distinct object category as a class:
          class_key = "malaria/<slugified_category>"
      - Map JSON files to splits:
          training.json -> SplitType.TRAIN
          test.json     -> SplitType.TEST
      - Emit one HBB instance annotation per object using
        bounding boxes derived from row/column min/max coordinates.
      - Store original checksums and image paths in provenance/metadata
        for traceability.
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
        """
        Input like "/images/foo.png" or "images/foo.png" -> "images/foo.png"
        """
        p = (pathname or "").strip()
        if p.startswith("/"):
            p = p[1:]
        # Now p should be like "images/....png"
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
        # Convert row/col bounds into xywh in pixel coords:
        # x = cmin, y = rmin
        # w = cmax - cmin, h = rmax - rmin
        # (Assuming "maximum" is an exclusive or inclusive boundary; either way,
        # this is consistent. If you discover off-by-ones later, you can adjust.)
        x = cmin
        y = rmin
        w = max(0.0, cmax - cmin)
        h = max(0.0, rmax - rmin)
        return HBB(x=x, y=y, w=w, h=h)

    def _iter_entries(self, ctx: AdapterContext) -> Iterable[Tuple[SplitType, dict]]:
        root = self._dataset_root(ctx)
        train_path = root / "training.json"
        test_path = root / "test.json"

        train_entries = self._load_list_json(train_path)
        test_entries = self._load_list_json(test_path)

        # Deterministic ordering: sort by pathname, then checksum (if present)
        def _key(e: dict) -> Tuple[str, str]:
            img = e.get("image", {}) or {}
            pathname = str(img.get("pathname", ""))
            checksum = str(img.get("checksum", ""))
            return (pathname, checksum)

        for e in sorted(train_entries, key=_key):
            yield (SplitType.TRAIN, e)
        for e in sorted(test_entries, key=_key):
            yield (SplitType.TEST, e)

    def iter_classes(self, ctx: AdapterContext) -> Iterable[ClassRecord]:
        seen: Dict[str, str] = {}  # slug -> original display name (first seen)
        for _, entry in self._iter_entries(ctx):
            for obj in entry.get("objects", []) or []:
                cat = str(obj.get("category", "")).strip()
                slug = _slugify_class_name(cat)
                if slug not in seen:
                    seen[slug] = cat or slug

        # Deterministic output order
        for slug in sorted(seen.keys()):
            name = seen[slug]
            yield ClassRecord(
                class_key=f"{self.dataset}/{slug}",
                dataset=self.dataset,
                name=name,  # original display name
                meta={
                    "slug": slug,
                },
            )

    def iter_images(self, ctx: AdapterContext) -> Iterable[ImageRecord]:
        root = self._dataset_root(ctx)

        # If an image appears in both JSONs (unlikely), keep first split encountered
        # but deterministic since we iterate train then test.
        seen_relpaths: Set[str] = set()

        for split, entry in self._iter_entries(ctx):
            img = entry.get("image", {}) or {}

            checksum = str(img.get("checksum", "")).strip() or None
            pathname = str(img.get("pathname", "")).strip()
            shape = img.get("shape", {}) or {}

            # shape has {"r": height, "c": width, "channels": 3}
            height = int(shape.get("r", 0))
            width = int(shape.get("c", 0))

            relpath = self._pathname_to_relpath(pathname)
            if relpath in seen_relpaths:
                # skip duplicate
                continue
            seen_relpaths.add(relpath)

            image_id = make_image_id(self.dataset, relpath)
            abs_path = root / relpath  # raw/malaria/images/...

            yield ImageRecord(
                image_id=image_id,
                path=str(abs_path),
                width=width,
                height=height,
                split=split,
                provenance=Provenance(
                    dataset=self.dataset,
                    original_relpath=relpath,
                    original_filename=Path(relpath).name,
                    original_id=checksum,  # native checksum acts as a stable original id
                    sha1=None,
                    size_bytes=None,
                ),
                counts={},  # filled later by index builder or derived on the fly for efficiency
                meta={
                    "checksum": checksum,
                    "channels": (
                        int(shape.get("channels", 0)) if "channels" in shape else None
                    ),
                },
            )

    def iter_annotations(
        self, ctx: AdapterContext
    ) -> Iterable[InstanceAnnotationRecord]:
        # Build a mapping from slug -> display name (optional)
        # (not strictly needed, but helps keep class_key consistent with iter_classes)
        slug_to_display: Dict[str, str] = {}
        for cr in self.iter_classes(ctx):
            slug = str(cr.meta.get("slug", cr.name))
            slug_to_display[slug] = cr.name

        for split, entry in self._iter_entries(ctx):
            img = entry.get("image", {}) or {}
            pathname = str(img.get("pathname", "")).strip()
            checksum = str(img.get("checksum", "")).strip() or None

            relpath = self._pathname_to_relpath(pathname)
            image_id = make_image_id(self.dataset, relpath)

            objects = entry.get("objects", []) or []

            # Deterministic ordering within each image:
            # sort by category + bbox coords + original index (stable tiebreak)
            def _obj_key(t):
                idx, obj = t
                cat = str(obj.get("category", "")).strip()
                rmin, cmin, rmax, cmax = self._parse_bbox_rc(obj)
                return (_slugify_class_name(cat), rmin, cmin, rmax, cmax, idx)

            indexed = list(enumerate(objects))
            indexed_sorted = sorted(indexed, key=_obj_key)

            # instance_index is per image (not per class).
            for instance_index, (orig_idx, obj) in enumerate(indexed_sorted):
                cat = str(obj.get("category", "")).strip()
                slug = _slugify_class_name(cat)
                class_key = f"{self.dataset}/{slug}"
                rmin, cmin, rmax, cmax = self._parse_bbox_rc(obj)
                geom = self._rc_to_xywh(rmin, cmin, rmax, cmax)

                ann_id = make_ann_id(
                    image_id=image_id,
                    class_key=class_key,
                    ann_type=AnnType.HBB,
                    geometry=geom,
                    source=SourceType.ORIGINAL,
                    instance_index=instance_index,
                    # checksum helps if you ever change path normalization, but not required
                    salt=checksum or "",
                )

                yield InstanceAnnotationRecord(
                    ann_id=ann_id,
                    image_id=image_id,
                    class_key=class_key,
                    ann_type=AnnType.HBB,
                    geometry=geom,
                    source=SourceType.ORIGINAL,
                    score=None,
                    instance_index=instance_index,
                    meta={
                        "split": split.value,
                        "category_raw": cat,
                        "checksum": checksum,
                        "original_object_index": orig_idx,
                        "bbox_rc_minmax": {
                            "rmin": rmin,
                            "cmin": cmin,
                            "rmax": rmax,
                            "cmax": cmax,
                        },
                    },
                )
