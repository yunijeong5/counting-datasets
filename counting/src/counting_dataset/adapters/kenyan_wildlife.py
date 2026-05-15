from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from counting_dataset.adapters.base import AdapterContext, DatasetAdapter
from counting_dataset.core.ids import make_ann_id, make_image_id, normalize_relpath
from counting_dataset.core.schema import (
    AnnType,
    ClassRecord,
    HBB,
    ImageRecord,
    InstanceAnnotationRecord,
    SourceType,
    SplitType,
)


def _slugify(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"


class KenyanWildlifeAdapter(DatasetAdapter):
    """
    Adapter for raw/kenyan-wildlife-aerial-survey/ in COCO format.

    Raw layout:
      raw/kenyan-wildlife-aerial-survey/
        coco/
          train/
            _annotations.coco.json
            *.jpg
          valid/
            _annotations.coco.json
            *.jpg
          test/
            _annotations.coco.json
            *.jpg
    """

    dataset = "kenyan_wildlife"

    def _dataset_root(self, ctx: AdapterContext) -> Path:
        return ctx.raw_root / "kenyan-wildlife-aerial-survey"

    def _split_dirs(self) -> List[Tuple[str, SplitType]]:
        return [
            ("train", SplitType.TRAIN),
            ("test", SplitType.TEST),
            ("valid", SplitType.VAL),
        ]

    def _load_coco(self, path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Expected COCO dict at {path}, got {type(data)}")
        return data

    def _build_category_maps(self, coco: dict) -> Tuple[Dict[int, str], Dict[int, str]]:
        id_to_key: Dict[int, str] = {}
        id_to_name: Dict[int, str] = {}

        cats = coco.get("categories", []) or []
        for c in sorted(cats, key=lambda x: int(x.get("id", -1))):
            cid = int(c["id"])
            name = str(c.get("name", "")).strip()
            supercat = str(c.get("supercategory", "")).strip().lower()

            if name.strip().lower() == "objects" and supercat in ("none", "", "objects"):
                continue

            slug = _slugify(name)
            class_key = f"{self.dataset}/{slug}"
            id_to_key[cid] = class_key
            id_to_name[cid] = name or slug

        return id_to_key, id_to_name

    def iter_classes(self, ctx: AdapterContext) -> Iterable[ClassRecord]:
        root = self._dataset_root(ctx)
        seen: Dict[str, ClassRecord] = {}

        for split_dir, _ in self._split_dirs():
            ann_path = root / "coco" / split_dir / "_annotations.coco.json"
            coco = self._load_coco(ann_path)
            id_to_key, id_to_name = self._build_category_maps(coco)

            for cid in sorted(id_to_key.keys()):
                ck = id_to_key[cid]
                if ck in seen:
                    continue
                seen[ck] = ClassRecord(
                    class_key=ck,
                    dataset=self.dataset,
                    name=id_to_name[cid],
                )

        for ck in sorted(seen.keys()):
            yield seen[ck]

    def iter_images(self, ctx: AdapterContext) -> Iterable[ImageRecord]:
        root = self._dataset_root(ctx)
        seen_relpaths: Set[str] = set()

        for split_dir, split_type in self._split_dirs():
            ann_path = root / "coco" / split_dir / "_annotations.coco.json"
            coco = self._load_coco(ann_path)

            images = coco.get("images", []) or []
            images_sorted = sorted(
                images,
                key=lambda im: (str(im.get("file_name", "")), int(im.get("id", -1))),
            )

            for im in images_sorted:
                file_name = str(im["file_name"])
                width = int(im.get("width", 0))
                height = int(im.get("height", 0))

                relpath = normalize_relpath(f"coco/{split_dir}/{file_name}")
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
                    split=split_type,
                    dataset=self.dataset,
                    original_relpath=relpath,
                    original_filename=Path(file_name).name,
                    original_id=str(im.get("id")) if "id" in im else None,
                )

    def iter_annotations(
        self, ctx: AdapterContext
    ) -> Iterable[InstanceAnnotationRecord]:
        root = self._dataset_root(ctx)

        for split_dir, _split_type in self._split_dirs():
            ann_path = root / "coco" / split_dir / "_annotations.coco.json"
            coco = self._load_coco(ann_path)

            id_to_key, _ = self._build_category_maps(coco)

            coco_imgid_to_relpath: Dict[int, str] = {}
            for im in coco.get("images", []) or []:
                coco_imgid = int(im["id"])
                file_name = str(im["file_name"])
                relpath = normalize_relpath(f"coco/{split_dir}/{file_name}")
                coco_imgid_to_relpath[coco_imgid] = relpath

            anns = coco.get("annotations", []) or []
            anns_sorted = sorted(
                anns,
                key=lambda a: (
                    int(a.get("image_id", -1)),
                    int(a.get("category_id", -1)),
                    int(a.get("id", -1)),
                ),
            )

            per_image_counter: Dict[int, int] = {}

            for a in anns_sorted:
                coco_imgid = int(a["image_id"])
                coco_catid = int(a["category_id"])

                if coco_catid not in id_to_key:
                    continue

                relpath = coco_imgid_to_relpath[coco_imgid]
                image_id = make_image_id(self.dataset, relpath)

                bbox = a.get("bbox", None)
                if bbox is None or len(bbox) != 4:
                    print(
                        f"Invalid bounding box annotation format in image {relpath}. "
                        f"Expected four [x, y, w, h] values; got {len(bbox) if bbox else 0} values."
                    )
                    continue

                x, y, w, h = map(float, bbox)
                geom = HBB(x=x, y=y, w=max(0.0, w), h=max(0.0, h))

                i = per_image_counter.get(coco_imgid, 0)
                per_image_counter[coco_imgid] = i + 1

                class_key = id_to_key[coco_catid]

                yield InstanceAnnotationRecord(
                    ann_id=make_ann_id(
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=AnnType.HBB,
                        geometry=geom,
                        source=SourceType.ORIGINAL,
                        instance_index=i,
                        salt=str(a.get("id", "")),
                    ),
                    image_id=image_id,
                    class_key=class_key,
                    ann_type=AnnType.HBB,
                    geometry=geom,
                    source=SourceType.ORIGINAL,
                    instance_index=i,
                )
