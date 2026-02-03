from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Literal

from PIL import Image

from counting_dataset.adapters.base import AdapterContext
from counting_dataset.core.ids import make_ann_id, make_image_id, normalize_relpath
from counting_dataset.core.schema import (
    AnnType,
    ClassRecord,
    ImageRecord,
    InstanceAnnotationRecord,
    Provenance,
    SourceType,
    SplitType,
    HBB,
    OBB,
)


def _slugify(name: str) -> str:
    s = (name or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"


class DOTAAdapter:
    """
    Adapter for raw/dota_v1-5/ with oriented and horizontal bounding box annotations.

    Raw layout:
      raw/dota_v1-5/
        train/
          images/                       (*.png)
          annotation/
            obb/                        (<image>.txt)
            hbb/                        (<image>.txt)
        val/
          images/
          annotation/
            obb/
            hbb/

    Annotation format:
      Each <image>.txt contains optional metadata lines followed by instance rows:
        x1 y1 x2 y2 x3 y3 x4 y4 category difficult

      Vertices are ordered clockwise. `difficult` is 1 (difficult) or 0.

    Strategy:
      - Create one class per category:
          class_key = "dota/<slugified_category>"
      - Map split directories:
          train -> SplitType.TRAIN
          val   -> SplitType.VAL
      - Emit OBB annotations as canonical instances:
          ann_type=OBB, role="instance"
      - Emit HBB annotations as alternative geometry for the same objects:
          ann_type=HBB, role="hbb"
      - Pair OBB and HBB rows by annotation row index when possible and
        store pairing metadata for traceability.
    """

    dataset = "dota"

    def _dataset_root(self, ctx: AdapterContext) -> Path:
        return ctx.raw_root / "dota_v1-5"

    @staticmethod
    def _split_dirs() -> Sequence[Tuple[str, SplitType]]:
        return [("train", SplitType.TRAIN), ("val", SplitType.VAL)]

    def _images_dir(self, ctx: AdapterContext, split_dir: str) -> Path:
        return self._dataset_root(ctx) / split_dir / "images"

    def _ann_dir(self, ctx: AdapterContext, split_dir: str, kind: str) -> Path:
        # kind in {"obb", "hbb"}
        return self._dataset_root(ctx) / split_dir / "annotation" / kind

    @staticmethod
    def _parse_ann_file(
        path: Path,
        *,
        mode: Literal["both", "header", "rows"] = "both",
    ) -> Tuple[Dict[str, str], List[dict]]:
        """
        mode:
          - "both": parse header + rows
          - "header": parse header only (stop when first instance row is encountered)
          - "rows": parse rows only (ignore header)
        """
        header: Dict[str, str] = {}
        rows: List[dict] = []

        if not path.exists():
            return header, rows

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue

                # detect header line
                is_header = ":" in raw and not re.match(r"^\s*-?\d+(\.\d+)?\b", raw)

                if is_header:
                    if mode != "rows":
                        k, v = raw.split(":", 1)
                        k = (k or "").strip().lower()
                        v = (v or "").strip()
                        if k:
                            header[k] = v
                    continue

                # from here: not a header line -> likely an instance row
                if mode == "header":
                    # we got everything we need
                    break

                parts = raw.split()
                if len(parts) < 10:
                    continue

                try:
                    coords = [float(x) for x in parts[:8]]
                except Exception:
                    continue

                cat = parts[8]
                try:
                    difficult = int(parts[9])
                except Exception:
                    difficult = 0

                corners = [
                    (coords[0], coords[1]),
                    (coords[2], coords[3]),
                    (coords[4], coords[5]),
                    (coords[6], coords[7]),
                ]
                rows.append(
                    {
                        "corners": corners,
                        "category": cat,
                        "difficult": difficult,
                    }
                )

        return header, rows

    @staticmethod
    def _corners_to_hbb(corners: Sequence[Tuple[float, float]]) -> HBB:
        xs = [float(x) for x, _ in corners]
        ys = [float(y) for _, y in corners]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        return HBB(x=x0, y=y0, w=max(0.0, x1 - x0), h=max(0.0, y1 - y0))

    @staticmethod
    def _make_obb_geometry(corners: Sequence[Tuple[float, float]]) -> OBB:
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = corners
        return OBB(corners=(x1, y1, x2, y2, x3, y3, x4, y4))

    def iter_classes(self, ctx: AdapterContext) -> Iterable[ClassRecord]:
        # Discover classes by scanning OBB annotations (preferred), across splits.
        seen: Dict[str, str] = {}

        for split_dir, _split in self._split_dirs():
            obb_dir = self._ann_dir(ctx, split_dir, "obb")
            if not obb_dir.exists():
                continue

            for txt in sorted(obb_dir.glob("*.txt")):
                _, rows = self._parse_ann_file(txt, mode="rows")
                for r in rows:
                    cat = r.get("category", "")
                    ck = f"{self.dataset}/{_slugify(cat)}"
                    seen.setdefault(ck, _slugify(cat))

        for ck in sorted(seen.keys()):
            yield ClassRecord(
                class_key=ck,
                dataset=self.dataset,
                name=seen[ck],
                meta={},
            )

    def iter_images(self, ctx: AdapterContext) -> Iterable[ImageRecord]:
        root = self._dataset_root(ctx)

        for split_dir, split in self._split_dirs():
            img_dir = self._images_dir(ctx, split_dir)
            obb_dir = self._ann_dir(ctx, split_dir, "obb")
            hbb_dir = self._ann_dir(ctx, split_dir, "hbb")

            if not img_dir.exists():
                continue

            for img_path in sorted(img_dir.glob("*.png")):
                img_name = img_path.stem  # e.g. "P0003"
                rel = normalize_relpath(f"{split_dir}/images/{img_path.name}")
                abs_path = root / rel

                with Image.open(abs_path) as im:
                    width, height = im.size

                image_id = make_image_id(self.dataset, rel)

                obb_txt = obb_dir / f"{img_name}.txt"
                hbb_txt = hbb_dir / f"{img_name}.txt"
                obb_header, _ = self._parse_ann_file(obb_txt, mode="header")
                hbb_header, _ = self._parse_ann_file(hbb_txt, mode="header")

                # union: obb overrides hbb if both present
                header = dict(hbb_header)
                header.update(obb_header)

                meta = {
                    # common normalized keys (best-effort)
                    "imagesource": header.get("imagesource"),
                    "gsd": header.get("gsd"),
                    # preserve full header for completeness
                    "header": header,
                    "has_obb": obb_txt.exists(),
                    "has_hbb": hbb_txt.exists(),
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
                        original_filename=img_path.name,
                        original_id=img_name,
                        sha1=None,
                        size_bytes=None,
                    ),
                    counts={},  # builder fills from role="instance"
                    meta=meta,
                )

    def iter_annotations(
        self, ctx: AdapterContext
    ) -> Iterable[InstanceAnnotationRecord]:
        """
        Emits:
          - OBB as role="instance" (canonical)
          - HBB as role="hbb" (paired alternative geometry)
        """

        for split_dir, split in self._split_dirs():
            img_dir = self._images_dir(ctx, split_dir)
            obb_dir = self._ann_dir(ctx, split_dir, "obb")
            hbb_dir = self._ann_dir(ctx, split_dir, "hbb")

            if not img_dir.exists():
                continue

            for img_path in sorted(img_dir.glob("*.png")):
                img_name = img_path.stem
                rel = normalize_relpath(f"{split_dir}/images/{img_path.name}")
                image_id = make_image_id(self.dataset, rel)

                obb_txt = obb_dir / f"{img_name}.txt"
                hbb_txt = hbb_dir / f"{img_name}.txt"

                _, obb_rows = self._parse_ann_file(obb_txt, mode="rows")
                _, hbb_rows = self._parse_ann_file(hbb_txt, mode="rows")

                # 1) Canonical OBB instances
                for row_i, r in enumerate(obb_rows):
                    corners = r["corners"]
                    cat = r["category"]
                    difficult = int(r.get("difficult", 0) or 0)

                    class_key = f"{self.dataset}/{_slugify(cat)}"
                    geom = self._make_obb_geometry(corners)

                    ann_id = make_ann_id(
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=AnnType.OBB,
                        geometry=geom,
                        source=SourceType.ORIGINAL,
                        instance_index=row_i,
                        salt=f"obb:{row_i}",
                    )

                    yield InstanceAnnotationRecord(
                        ann_id=ann_id,
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=AnnType.OBB,
                        geometry=geom,
                        role="instance",
                        instance_index=row_i,
                        source=SourceType.ORIGINAL,
                        score=None,
                        meta={
                            "split": split.value,
                            "image_name": img_path.name,
                            "difficult": difficult,
                            "annotation_row_index": row_i,
                            "paired_with_ann_type": AnnType.HBB.value,
                        },
                    )

                # 2) Auxiliary HBB geometry (paired by row index)
                for row_i, r in enumerate(hbb_rows):
                    corners = r["corners"]
                    cat = r["category"]
                    difficult = int(r.get("difficult", 0) or 0)

                    class_key = f"{self.dataset}/{_slugify(cat)}"
                    geom_hbb = self._corners_to_hbb(corners)

                    ann_id = make_ann_id(
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=AnnType.HBB,
                        geometry=geom_hbb,
                        source=SourceType.ORIGINAL,
                        instance_index=row_i,
                        salt=f"hbb:{row_i}",
                    )

                    yield InstanceAnnotationRecord(
                        ann_id=ann_id,
                        image_id=image_id,
                        class_key=class_key,
                        ann_type=AnnType.HBB,
                        geometry=geom_hbb,
                        role="hbb",
                        instance_index=row_i,
                        source=SourceType.ORIGINAL,
                        score=None,
                        meta={
                            "split": split.value,
                            "image_name": img_path.name,
                            "difficult": difficult,
                            "annotation_row_index": row_i,
                            "paired_with_ann_type": AnnType.OBB.value,
                        },
                    )
