from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from counting_dataset.adapters.base import AdapterContext, DatasetAdapter
from counting_dataset.core.ids import make_ann_id, make_image_id, normalize_relpath
from counting_dataset.core.schema import (
    AnnType,
    ClassRecord,
    HBB,
    ImageRecord,
    InstanceAnnotationRecord,
    Point,
    SourceType,
    SplitType,
)

try:
    import numpy as np
    import skimage as ski

    _HAVE_SKIMAGE = True
except ImportError:
    _HAVE_SKIMAGE = False


def _tile_number(tile_name: str) -> int:
    """'tile_33.jpg' -> 33"""
    stem = tile_name.rsplit(".", 1)[0]
    return int(stem.split("_", 1)[1])


def _compute_pseudo_bboxes(
    img_path: Path, sorted_regions: List[dict]
) -> Optional[List[Tuple[float, float, float, float]]]:
    """
    Estimate per-bird bounding boxes using Otsu thresholding.
    Returns None when scikit-image is absent, the image is unreadable, or
    there are no regions.
    """
    if not _HAVE_SKIMAGE or not sorted_regions:
        return None

    try:
        img = ski.io.imread(str(img_path))
    except Exception:
        return None

    gray = ski.color.rgb2gray(img)
    H, W = gray.shape

    try:
        thresh = ski.filters.threshold_otsu(gray)
    except Exception:
        return None

    fg = gray < thresh
    total_area = float(np.sum(fg))
    n = len(sorted_regions)
    if total_area == 0 or n == 0:
        return None

    r = np.sqrt(total_area / n) / 2 * 1.5
    r_buf = r * 2

    mask = np.zeros((H, W), dtype=np.float32)
    for reg in sorted_regions:
        cx = int(reg["shape_attributes"]["cx"])
        cy = int(reg["shape_attributes"]["cy"])
        x0, x1 = max(0, cx - int(r_buf)), min(W, cx + int(r_buf))
        y0, y1 = max(0, cy - int(r_buf)), min(H, cy + int(r_buf))
        mask[y0:y1, x0:x1] = 1.0

    masked_area = float(np.sum(mask * fg))
    if masked_area > 0:
        r = np.sqrt(masked_area / n) / 2 * 1.5

    boxes: List[Tuple[float, float, float, float]] = []
    for reg in sorted_regions:
        cx = float(reg["shape_attributes"]["cx"])
        cy = float(reg["shape_attributes"]["cy"])
        boxes.append((cx - r, cy - r, cx + r, cy + r))
    return boxes


class BirdsAdapter(DatasetAdapter):
    """
    Adapter for raw/birds/ with per-tile VGG Image Annotator point labels.

    Raw layout::

        raw/birds/
          tiles/
            tiles_DSC5214/
              <batch_id>/
                tile_<N>.jpg
              labels/
                <batch_id>.json
            tiles_DSC5295/
              ...

    * Unit of analysis is the tile (200×200).
    * One class: birds/bird.
    * scene ("sky" or "reeds") stored in image meta for meta_filter use.
    * Two annotation types per labeled point:
      - POINT / role="instance" / source=ORIGINAL — ground-truth count.
      - HBB  / role="hbb"      / source=GENERATED — Otsu pseudo-bbox
        (only emitted when scikit-image is installed; does not count).
    * Tiles with zero annotations are still indexed as ImageRecords.
    """

    dataset = "birds"

    _SOURCE_NAMES: Dict[str, str] = {
        "tiles_DSC5214": "sky",
        "tiles_DSC5295": "reeds",
    }

    def _dataset_root(self, ctx: AdapterContext) -> Path:
        return ctx.raw_root / "birds"

    def _tiles_root(self, ctx: AdapterContext) -> Path:
        return self._dataset_root(ctx) / "tiles"

    def _iter_label_files(self, ctx: AdapterContext) -> Iterable[Tuple[str, int, Path]]:
        tiles_root = self._tiles_root(ctx)
        src_dirs = sorted(
            p
            for p in tiles_root.iterdir()
            if p.is_dir() and p.name.startswith("tiles_")
        )
        for src_dir in src_dirs:
            labels_dir = src_dir / "labels"
            if not labels_dir.is_dir():
                continue
            for lf in sorted(labels_dir.glob("*.json"), key=lambda p: int(p.stem)):
                yield src_dir.name, int(lf.stem), lf

    def _tile_relpath(self, src_img: str, batch: int, tile_name: str) -> str:
        return normalize_relpath(f"birds/tiles/{src_img}/{batch}/{tile_name}")

    @staticmethod
    def _tile_dims(path: Path) -> Tuple[int, int]:
        try:
            from PIL import Image as _PIL

            with _PIL.open(path) as im:
                return im.size
        except Exception:
            return 200, 200

    @staticmethod
    def _sorted_entries(data: Dict) -> List[Tuple[str, dict]]:
        return sorted(
            data.items(),
            key=lambda kv: _tile_number(kv[1]["filename"].split("/")[-1]),
        )

    def iter_classes(self, ctx: AdapterContext) -> Iterable[ClassRecord]:
        yield ClassRecord(
            class_key=f"{self.dataset}/bird",
            dataset=self.dataset,
            name="bird",
        )

    def iter_images(self, ctx: AdapterContext) -> Iterable[ImageRecord]:
        dataset_root = self._dataset_root(ctx)
        seen: Set[str] = set()

        for src_img, batch, label_path in self._iter_label_files(ctx):
            with label_path.open("r", encoding="utf-8") as f:
                data: Dict = json.load(f)

            for _key, value in self._sorted_entries(data):
                filename = value["filename"]
                tile_name = filename.split("/")[-1]
                relpath = self._tile_relpath(src_img, batch, tile_name)

                if relpath in seen:
                    continue
                seen.add(relpath)

                abs_path = dataset_root / "tiles" / src_img / str(batch) / tile_name
                image_id = make_image_id(self.dataset, relpath)
                w, h = self._tile_dims(abs_path)

                yield ImageRecord(
                    image_id=image_id,
                    path=str(abs_path),
                    width=w,
                    height=h,
                    split=SplitType.UNSPECIFIED,
                    dataset=self.dataset,
                    original_relpath=relpath,
                    original_filename=tile_name,
                    meta={"scene": self._SOURCE_NAMES.get(src_img, src_img)},
                )

    def iter_annotations(
        self, ctx: AdapterContext
    ) -> Iterable[InstanceAnnotationRecord]:
        dataset_root = self._dataset_root(ctx)
        class_key = f"{self.dataset}/bird"

        for src_img, batch, label_path in self._iter_label_files(ctx):
            with label_path.open("r", encoding="utf-8") as f:
                data: Dict = json.load(f)

            for _key, value in self._sorted_entries(data):
                filename = value["filename"]
                tile_name = filename.split("/")[-1]
                regions = value.get("regions") or []
                if not regions:
                    continue

                relpath = self._tile_relpath(src_img, batch, tile_name)
                image_id = make_image_id(self.dataset, relpath)

                sorted_regions = sorted(
                    regions,
                    key=lambda r: (
                        r["shape_attributes"]["cx"],
                        r["shape_attributes"]["cy"],
                    ),
                )

                abs_path = dataset_root / "tiles" / src_img / str(batch) / tile_name
                pseudo_boxes = _compute_pseudo_bboxes(abs_path, sorted_regions)

                for inst_idx, reg in enumerate(sorted_regions):
                    cx = float(reg["shape_attributes"]["cx"])
                    cy = float(reg["shape_attributes"]["cy"])
                    pt_geom = Point(x=cx, y=cy)

                    yield InstanceAnnotationRecord(
                        ann_id=make_ann_id(
                            image_id=image_id,
                            class_key=class_key,
                            ann_type=AnnType.POINT,
                            geometry=pt_geom,
                            source=SourceType.ORIGINAL,
                            instance_index=inst_idx,
                        ),
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=AnnType.POINT,
                        geometry=pt_geom,
                        role="instance",
                        source=SourceType.ORIGINAL,
                        instance_index=inst_idx,
                    )

                    if pseudo_boxes is not None:
                        x1, y1, x2, y2 = pseudo_boxes[inst_idx]
                        hbb_geom = HBB(
                            x=float(x1),
                            y=float(y1),
                            w=float(x2 - x1),
                            h=float(y2 - y1),
                        )
                        yield InstanceAnnotationRecord(
                            ann_id=make_ann_id(
                                image_id=image_id,
                                class_key=class_key,
                                ann_type=AnnType.HBB,
                                geometry=hbb_geom,
                                source=SourceType.GENERATED,
                                instance_index=inst_idx,
                            ),
                            image_id=image_id,
                            class_key=class_key,
                            ann_type=AnnType.HBB,
                            geometry=hbb_geom,
                            role="hbb",
                            source=SourceType.GENERATED,
                            instance_index=inst_idx,
                        )
