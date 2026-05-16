"""
Generate and cache per-tile Otsu pseudo-bounding boxes for the birds dataset.

Run once on the machine that holds the raw data (before building the index):
    python raw/birds/generate_bird_bbox_cache.py   # from project root
    python generate_bird_bbox_cache.py              # from raw/birds/

Output: raw/birds/pseudo_bboxes_cache.json

Format:
    {
      "<tiles_dir>/<batch>/<tile_name>": [[x1, y1, x2, y2], ...],
      ...
    }

Each entry maps a tile (identified by its path relative to raw/birds/tiles/)
to the list of Otsu pseudo-bboxes, one per annotated point, in the same order
as the annotations sorted by (cx, cy).  Tiles where Otsu fails (skimage absent,
unreadable image, uniform image) have no entry; the image-level adapter treats
a missing entry as "no HBB for this tile."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent          # raw/birds/
PROJECT_ROOT = HERE.parent.parent     # project root
RAW_BIRDS = HERE
TILES_ROOT = RAW_BIRDS / "tiles"
CACHE_PATH = RAW_BIRDS / "pseudo_bboxes_cache.json"

sys.path.insert(0, str(PROJECT_ROOT / "counting" / "src"))
from counting_dataset.adapters.birds_tiles import _compute_pseudo_bboxes, _tile_number


def _sorted_entries(data: dict) -> list:
    return sorted(
        data.items(),
        key=lambda kv: _tile_number(kv[1]["filename"].split("/")[-1]),
    )


def main() -> None:
    if not TILES_ROOT.exists():
        print(f"ERROR: tiles root not found: {TILES_ROOT}", file=sys.stderr)
        sys.exit(1)

    src_dirs = sorted(
        p for p in TILES_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("tiles_")
    )

    cache: dict = {}
    annotated_tiles = 0
    cached_tiles = 0

    for src_dir in src_dirs:
        labels_dir = src_dir / "labels"
        if not labels_dir.is_dir():
            continue

        print(f"Processing {src_dir.name} ...")
        for lf in sorted(labels_dir.glob("*.json"), key=lambda p: int(p.stem)):
            batch = int(lf.stem)
            with lf.open("r", encoding="utf-8") as f:
                data = json.load(f)

            for _key, value in _sorted_entries(data):
                filename = value["filename"]
                tile_name = filename.split("/")[-1]
                regions = value.get("regions") or []
                if not regions:
                    continue

                annotated_tiles += 1

                sorted_regions = sorted(
                    regions,
                    key=lambda r: (
                        r["shape_attributes"]["cx"],
                        r["shape_attributes"]["cy"],
                    ),
                )

                abs_path = src_dir / str(batch) / tile_name
                boxes = _compute_pseudo_bboxes(abs_path, sorted_regions)

                if boxes is not None:
                    cache_key = f"{src_dir.name}/{batch}/{tile_name}"
                    cache[cache_key] = [[x1, y1, x2, y2] for x1, y1, x2, y2 in boxes]
                    cached_tiles += 1

                if annotated_tiles % 200 == 0:
                    print(f"  {annotated_tiles} annotated tiles processed, {cached_tiles} cached ...")

    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(cache, f)

    print(f"\nDone.")
    print(f"  annotated tiles : {annotated_tiles}")
    print(f"  cached (Otsu OK): {cached_tiles}")
    print(f"  output          : {CACHE_PATH}")


if __name__ == "__main__":
    main()
