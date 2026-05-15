from __future__ import annotations

import json
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
)


class PenguinAdapter:
    """
    Adapter for raw/penguin/ with crowd-sourced point annotations.

    Raw layout:
      raw/penguin/
        - images/
        - split.json
        - annotation.json

    Annotation semantics (annotation.json):
      - Each entry in obj["dots"] has: {"imName": <stem>, "xy": ...}
      - "xy" == null        -> UNLABELED
      - "xy" == list        -> reviewed; each list item is one annotator's output
                               (may be "_NaN_" or [] to indicate an empty vote)

    Indexing scope (include_unlabeled):
      - False (default): skip images with no annotation entry or xy=null.
        (78,078 of 81,941 images are unlabeled — avoids opening them all.)
      - True: index all images in split.json.

    Image meta stores crowd statistics (review_status, annotator counts) as a
    build-time data carrier for the index builder's image_review_stats table.
    These stats drive downstream crowd-quality filtering and are not surfaced
    directly in __getitem__.

    Annotation meta stores annotator_index per point to support inter-annotator
    agreement analysis.
    """

    dataset = "penguin"

    def __init__(self, *, include_unlabeled: bool = False):
        self.include_unlabeled = bool(include_unlabeled)

    def _dataset_root(self, ctx: AdapterContext) -> Path:
        return ctx.raw_root / "penguin"

    def _load_split_map(self, ctx: AdapterContext) -> Dict[str, SplitType]:
        root = self._dataset_root(ctx)
        with (root / "split.json").open("r", encoding="utf-8") as f:
            obj = json.load(f)

        imdb = obj.get("imdb", {}) or {}
        out: Dict[str, SplitType] = {}

        def add_many(items: List[str], split: SplitType) -> None:
            for p in items:
                out[normalize_relpath(f"images/{p}")] = split

        add_many(imdb.get("train", []), SplitType.TRAIN)
        add_many(imdb.get("val", []), SplitType.VAL)
        add_many(imdb.get("test", []), SplitType.TEST)
        return out

    def _index_relpaths_by_stem(self, ctx: AdapterContext) -> Dict[str, List[str]]:
        root = self._dataset_root(ctx)
        images_dir = root / "images"
        mapping: Dict[str, List[str]] = {}

        for sub in sorted([p for p in images_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
            for img_path in sorted(sub.glob("*.JPG")):
                rel = normalize_relpath(str(img_path.relative_to(root)))
                mapping.setdefault(img_path.stem, []).append(rel)
            for img_path in sorted(sub.glob("*.jpg")):
                rel = normalize_relpath(str(img_path.relative_to(root)))
                mapping.setdefault(img_path.stem, []).append(rel)

        return mapping

    def _load_annotation_stats(self, ctx: AdapterContext) -> Dict[str, dict]:
        """Returns mapping: imName(stem) -> crowd stats dict."""
        root = self._dataset_root(ctx)
        with (root / "annotation.json").open("r", encoding="utf-8") as f:
            obj = json.load(f)

        stats: Dict[str, dict] = {}
        for d in obj.get("dots", []) or []:
            im_name = str(d.get("imName", "")).strip()
            if not im_name:
                continue

            xy = d.get("xy", None)

            if xy is None:
                stats[im_name] = {
                    "review_status": "unreviewed",
                    "num_annotator_entries": 0,
                    "num_empty_votes": 0,
                    "num_point_votes": 0,
                    "num_points_total": 0,
                }
                continue

            if not isinstance(xy, list):
                stats[im_name] = {
                    "review_status": "malformed",
                    "num_annotator_entries": 0,
                    "num_empty_votes": 0,
                    "num_point_votes": 0,
                    "num_points_total": 0,
                }
                continue

            num_empty = 0
            num_point_votes = 0
            total_points = 0

            for entry in xy:
                if entry == "_NaN_":
                    num_empty += 1
                    continue
                if isinstance(entry, list):
                    if len(entry) == 0:
                        num_empty += 1
                        continue
                    cnt = sum(
                        1
                        for pt in entry
                        if isinstance(pt, (list, tuple))
                        and len(pt) == 2
                        and _safe_float(pt[0]) is not None
                        and _safe_float(pt[1]) is not None
                    )
                    if cnt == 0:
                        num_empty += 1
                    else:
                        num_point_votes += 1
                    total_points += cnt
                else:
                    num_empty += 1

            stats[im_name] = {
                "review_status": "reviewed",
                "num_annotator_entries": len(xy),
                "num_empty_votes": num_empty,
                "num_point_votes": num_point_votes,
                "num_points_total": total_points,
            }

        return stats

    def iter_classes(self, ctx: AdapterContext) -> Iterable[ClassRecord]:
        yield ClassRecord(
            class_key=f"{self.dataset}/penguin",
            dataset=self.dataset,
            name="penguin",
        )

    def iter_images(self, ctx: AdapterContext) -> Iterable[ImageRecord]:
        root = self._dataset_root(ctx)
        split_map = self._load_split_map(ctx)
        ann_stats = self._load_annotation_stats(ctx)

        for rel in sorted(split_map.keys()):
            abs_path = root / rel
            split = split_map[rel]
            stem = Path(rel).stem

            crowd = ann_stats.get(stem, None)

            if not self.include_unlabeled:
                if crowd is None:
                    continue
                if crowd.get("review_status") == "unreviewed":
                    continue

            if crowd is None:
                crowd = {"review_status": "missing_in_annotation_json"}

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
                original_filename=Path(rel).name,
                # crowd dict is a build-time data carrier for image_review_stats;
                # not surfaced in __getitem__
                meta={"crowd": crowd},
            )

    def iter_annotations(
        self, ctx: AdapterContext
    ) -> Iterable[InstanceAnnotationRecord]:
        root = self._dataset_root(ctx)
        split_map = self._load_split_map(ctx)
        stem_to_relpaths = self._index_relpaths_by_stem(ctx)

        class_key = f"{self.dataset}/penguin"

        with (root / "annotation.json").open("r", encoding="utf-8") as f:
            obj = json.load(f)

        dots_sorted = sorted(obj.get("dots", []) or [], key=lambda d: str(d.get("imName", "")))
        per_image_counter: Dict[str, int] = {}

        for d in dots_sorted:
            im_name = str(d.get("imName", "")).strip()
            xy = d.get("xy", None)

            if not im_name or xy is None or not isinstance(xy, list):
                continue

            relpaths = stem_to_relpaths.get(im_name, [])
            if not relpaths:
                continue

            chosen_rel: Optional[str] = None
            for r in relpaths:
                if r in split_map:
                    chosen_rel = r
                    break
            if chosen_rel is None:
                chosen_rel = sorted(relpaths)[0]

            if chosen_rel not in split_map:
                continue

            image_id = make_image_id(self.dataset, chosen_rel)
            per_image_counter.setdefault(image_id, 0)

            for annotator_index, entry in enumerate(xy):
                if entry == "_NaN_":
                    continue
                if not isinstance(entry, list) or len(entry) == 0:
                    continue

                for pt_i, pt in enumerate(entry):
                    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                        continue
                    x = _safe_float(pt[0])
                    y = _safe_float(pt[1])
                    if x is None or y is None:
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
                            source=SourceType.CROWDSOURCE,
                            instance_index=instance_index,
                            salt=f"{annotator_index}:{pt_i}",
                        ),
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=AnnType.POINT,
                        geometry=geom,
                        source=SourceType.CROWDSOURCE,
                        instance_index=instance_index,
                        meta={"annotator_index": annotator_index},
                    )


def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None
