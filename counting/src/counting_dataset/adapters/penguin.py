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
    Provenance,
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
      - Each entry in obj["dots"] has:
          {"imName": <stem>, "xy": ...}
      - "xy" meanings:
          * xy == null        -> UNLABELED (nobody reviewed / no annotator outputs)
          * xy == list        -> reviewed; each list item is an annotator output
                                (may be "_NaN_" or [] to indicate empty votes)
      - Not all images in the dataset has an entry in annotation.json.

    Indexing scope (include_unlabeled):
      - include_unlabeled=False (default):
          Skip images whose annotation entry is missing in annotation.json or has {"xy": null}. Both are treated as unlabled.
          This avoids opening tens of thousands of unlabeled images just to read size. For context, 78,078 of 81,941 total images are unlabeled.
      - include_unlabeled=True:
          Index all images from split.json, regardless of whether it has no entry in annotation.json or xy is null.

    Strategy:
      - Define a single class: "penguin/penguin".
      - Assign splits using split.json.
      - Emit one POINT annotation per point per annotator with source=CROWDSOURCE.
      - Treat "_NaN_" and [] as empty votes (no points emitted).
      - Store crowd stats under ImageRecord.meta["crowd"] for downstream filtering.
    """

    dataset = "penguin"

    def __init__(self, *, include_unlabeled: bool = False):
        if not include_unlabeled:
            self.include_unlabeled = bool(include_unlabeled)

    def _dataset_root(self, ctx: AdapterContext) -> Path:
        return ctx.raw_root / "penguin"

    def _load_split_map(self, ctx: AdapterContext) -> Dict[str, SplitType]:
        """
        Returns mapping from image relpath (e.g., "images/DAMOa/DAMOa2014a_000123.JPG")
        to SplitType.
        """
        root = self._dataset_root(ctx)
        split_path = root / "split.json"
        with split_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        imdb = obj.get("imdb", {}) or {}
        out: Dict[str, SplitType] = {}

        def add_many(items: List[str], split: SplitType) -> None:
            for p in items:
                # split.json paths are relative to penguin/images/, so we prefix "images/"
                rel = normalize_relpath(f"images/{p}")
                out[rel] = split

        add_many(imdb.get("train", []), SplitType.TRAIN)
        add_many(imdb.get("val", []), SplitType.VAL)
        add_many(imdb.get("test", []), SplitType.TEST)

        return out

    def _index_relpaths_by_stem(self, ctx: AdapterContext) -> Dict[str, List[str]]:
        """
        Map `imName` style stems (no ext, no subdir) to one or more relpaths.
        Example key: "BAILa2014a_000003"
        Value: ["images/BAILa/BAILa2014a_000003.JPG"]
        """
        root = self._dataset_root(ctx)
        images_dir = root / "images"

        mapping: Dict[str, List[str]] = {}

        for sub in sorted(
            [p for p in images_dir.iterdir() if p.is_dir()], key=lambda p: p.name
        ):
            for img_path in sorted(sub.glob("*.JPG")):
                stem = img_path.stem
                rel = normalize_relpath(str(img_path.relative_to(root)))
                mapping.setdefault(stem, []).append(rel)

            for img_path in sorted(sub.glob("*.jpg")):
                stem = img_path.stem
                rel = normalize_relpath(str(img_path.relative_to(root)))
                mapping.setdefault(stem, []).append(rel)

        return mapping

    def _load_annotation_stats(self, ctx: AdapterContext) -> Dict[str, dict]:
        """
        Returns mapping: imName(stem) -> crowd stats dict.
        """
        root = self._dataset_root(ctx)
        ann_path = root / "annotation.json"
        with ann_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        stats: Dict[str, dict] = {}
        dots = obj.get("dots", []) or []
        for d in dots:
            im_name = str(d.get("imName", "")).strip()
            if not im_name:
                continue

            xy = d.get("xy", None)  # null on JSON

            if xy is None:
                # unlabeled/unreviewed: nobody has checked yet
                stats[im_name] = {
                    "review_status": "unreviewed",
                    "num_annotator_entries": 0,
                    "num_empty_votes": 0,
                    "num_point_votes": 0,
                    "num_points_total": 0,
                    "num_points_per_annotator": [],
                }
                continue

            if not isinstance(xy, list):
                stats[im_name] = {
                    "review_status": "malformed",
                    "num_annotator_entries": 0,
                    "num_empty_votes": 0,
                    "num_point_votes": 0,
                    "num_points_total": 0,
                    "num_points_per_annotator": [],
                }
                continue

            num_empty = 0
            num_point_votes = 0
            points_per_annotator: List[int] = []
            total_points = 0

            for entry in xy:
                if entry == "_NaN_":
                    num_empty += 1
                    points_per_annotator.append(0)
                    continue

                if isinstance(entry, list):
                    if len(entry) == 0:
                        num_empty += 1
                        points_per_annotator.append(0)
                        continue

                    cnt = 0
                    for pt in entry:
                        if isinstance(pt, (list, tuple)) and len(pt) == 2:
                            try:
                                float(pt[0])
                                float(pt[1])
                                cnt += 1
                            except Exception:
                                pass

                    if cnt == 0:
                        num_empty += 1
                    else:
                        num_point_votes += 1

                    points_per_annotator.append(cnt)
                    total_points += cnt
                    continue

                num_empty += 1
                points_per_annotator.append(0)

            stats[im_name] = {
                "review_status": "reviewed",
                "num_annotator_entries": len(xy),
                "num_empty_votes": num_empty,
                "num_point_votes": num_point_votes,
                "num_points_total": total_points,
                "num_points_per_annotator": points_per_annotator,
            }

        return stats

    def iter_classes(self, ctx: AdapterContext) -> Iterable[ClassRecord]:
        yield ClassRecord(
            class_key=f"{self.dataset}/penguin",  # no penguin species information provided.
            dataset=self.dataset,
            name="penguin",
            meta={},
        )

    def iter_images(self, ctx: AdapterContext) -> Iterable[ImageRecord]:
        root = self._dataset_root(ctx)
        split_map = self._load_split_map(ctx)
        ann_stats = self._load_annotation_stats(ctx)

        # Only include images that appear in split.json (source of truth)
        relpaths = sorted(split_map.keys())

        for rel in relpaths:
            abs_path = root / rel
            split = split_map[rel]
            stem = Path(rel).stem  # e.g., "BAILa2014a_000003"

            crowd = ann_stats.get(stem, None)

            # If include_unlabeled=False (default), we only index images that have an
            # entry in annotation.json *AND* are not {"xy": null}.
            if not self.include_unlabeled:
                if crowd is None:
                    # image does not have an entry in annotation.json.
                    # missing_in_annotation_json -> treat as unlabeled for indexing scope
                    continue
                if crowd.get("review_status") == "unreviewed":
                    # {"xy": null} -> unlabeled per your definition
                    continue

            if crowd is None:
                crowd = {"review_status": "missing_in_annotation_json"}

            meta = {"crowd": crowd}

            # Only reliable size source is the actual image.
            with Image.open(abs_path) as im:
                width, height = im.size

            image_id = make_image_id(self.dataset, rel)

            yield ImageRecord(
                image_id=image_id,
                path=str(abs_path),
                width=int(width),
                height=int(height),
                split=split,
                provenance=Provenance(
                    dataset=self.dataset,
                    original_relpath=rel,
                    original_filename=Path(rel).name,
                    original_id=None,
                    sha1=None,
                    size_bytes=None,
                ),
                counts={},
                meta=meta,
            )

    def iter_annotations(
        self, ctx: AdapterContext
    ) -> Iterable[InstanceAnnotationRecord]:
        root = self._dataset_root(ctx)
        split_map = self._load_split_map(ctx)
        stem_to_relpaths = self._index_relpaths_by_stem(ctx)

        class_key = f"{self.dataset}/penguin"

        ann_path = root / "annotation.json"
        with ann_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        dots = obj.get("dots", []) or []
        dots_sorted = sorted(dots, key=lambda d: str(d.get("imName", "")))

        per_image_counter: Dict[str, int] = {}

        for d in dots_sorted:
            im_name = str(d.get("imName", "")).strip()
            xy = d.get("xy", None)

            if not im_name:
                continue

            # unlabeled images have xy == None; no points to emit
            if xy is None:
                continue

            if not isinstance(xy, list):
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
                if not isinstance(entry, list):
                    continue
                if len(entry) == 0:
                    continue

                for pt_i, pt in enumerate(entry):
                    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                        continue
                    try:
                        x = float(pt[0])
                        y = float(pt[1])
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
                        source=SourceType.CROWDSOURCE,
                        instance_index=instance_index,
                        salt=f"{annotator_index}:{pt_i}",
                    )

                    yield InstanceAnnotationRecord(
                        ann_id=ann_id,
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=AnnType.POINT,
                        geometry=geom,
                        source=SourceType.CROWDSOURCE,
                        score=None,
                        instance_index=instance_index,
                        meta={
                            "image_name": im_name,
                            "chosen_relpath": chosen_rel,
                            "split": split_map[chosen_rel].value,
                            "annotator_index": annotator_index,
                            "annotator_marked_empty": False,
                            "raw_entry_type": "points",
                        },
                    )
