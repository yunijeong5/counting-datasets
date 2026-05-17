"""
Build the counting dataset index (counting/data/index.sqlite).

Usage:
    python build_index.py

Control which datasets to include by toggling the True/False flags in the
DATASETS section below.  The script resolves all paths relative to its own
location, so it works regardless of your working directory.
"""

from __future__ import annotations

import time
from pathlib import Path

from counting_dataset.api.counting_dataset_index import CountingDatasetIndex
from counting_dataset.index.builder import IndexBuilder

# ---------------------------------------------------------------------------
# DATASETS — toggle True/False to include/exclude each dataset
# ---------------------------------------------------------------------------

from counting_dataset.adapters.birds import BirdsAdapter
from counting_dataset.adapters.dota import DOTAAdapter
from counting_dataset.adapters.kenyan_wildlife import KenyanWildlifeAdapter
from counting_dataset.adapters.malaria import MalariaAdapter
from counting_dataset.adapters.aerial_elephant import AerialElephantAdapter
from counting_dataset.adapters.fsc147 import FSC147Adapter
from counting_dataset.adapters.penguin import PenguinAdapter

DATASETS = {
    "birds": (True, BirdsAdapter()),
    "dota": (True, DOTAAdapter()),
    "kenyan_wildlife": (True, KenyanWildlifeAdapter()),
    "malaria": (True, MalariaAdapter()),
    "aerial_elephant": (False, AerialElephantAdapter()),
    "fsc147": (False, FSC147Adapter()),
    "penguin": (False, PenguinAdapter()),
}

# ---------------------------------------------------------------------------
# Paths — relative to this script
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent
RAW_ROOT = HERE / "raw"
OUT_ROOT = HERE / "counting" / "data"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    adapters = [adapter for enabled, adapter in DATASETS.values() if enabled]
    enabled_names = [name for name, (enabled, _) in DATASETS.items() if enabled]

    print(f"Building index with {len(adapters)} dataset(s): {', '.join(enabled_names)}")
    print(f"  raw root : {RAW_ROOT}")
    print(f"  out root : {OUT_ROOT}")
    print()

    builder = IndexBuilder(raw_root=RAW_ROOT, out_root=OUT_ROOT)

    t0 = time.perf_counter()
    db_path = builder.build(adapters, overwrite=True, show_progress=True)
    elapsed = time.perf_counter() - t0

    print(f"\nDone in {elapsed:.1f}s  →  {db_path}")

    # ---------------------------------------------------------------------------
    # Sanity check — basic statistics via CountingDatasetIndex
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INDEX SUMMARY")
    print("=" * 60)

    index = CountingDatasetIndex(OUT_ROOT)
    all_classes = index.list_classes(apply_policy=False)
    datasets = index.available_datasets()

    total_classes = len(all_classes)
    total_instances = sum(c["num_instances"] for c in all_classes)
    total_images = sum(
        len(index.load_dataset(ds, load_images=False, preload_annotations=False))
        for ds in datasets
    )

    print(f"  datasets : {len(datasets)}  ({', '.join(datasets)})")
    print(f"  images   : {total_images:,}")
    print(f"  classes  : {total_classes}")
    print(f"  instances: {total_instances:,}")

    # Per-dataset breakdown
    print()
    print(f"  {'dataset':<20} {'images':>8} {'classes':>8} {'instances':>12}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*12}")
    for ds in sorted(datasets):
        ds_classes = [c for c in all_classes if c["dataset"] == ds]
        n_images = len(
            index.load_dataset(ds, load_images=False, preload_annotations=False)
        )
        n_classes = len(ds_classes)
        n_instances = sum(c["num_instances"] for c in ds_classes)
        print(f"  {ds:<20} {n_images:>8,} {n_classes:>8} {n_instances:>12,}")

    # Per-class breakdown
    print()
    print(f"  {'class_key':<40} {'images':>8} {'instances':>12}")
    print(f"  {'-'*40} {'-'*8} {'-'*12}")
    for c in sorted(all_classes, key=lambda c: c["class_key"]):
        print(f"  {c['class_key']:<40} {c['num_images']:>8,} {c['num_instances']:>12,}")
