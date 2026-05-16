"""
Build the counting dataset index (counting/data/index.sqlite).

Usage:
    python build_index.py

Control which datasets to include by toggling the True/False flags in the
DATASETS section below.  The script resolves all paths relative to its own
location, so it works regardless of your working directory.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

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
    # "dota":             (True,  DOTAAdapter()),
    # "kenyan_wildlife":  (True,  KenyanWildlifeAdapter()),
    # "malaria":          (True,  MalariaAdapter()),
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
    # Sanity check — basic statistics from the built index
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INDEX SUMMARY")
    print("=" * 60)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Totals
    total_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    total_classes = conn.execute("SELECT COUNT(*) FROM classes").fetchone()[0]
    total_instances = conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE role = 'instance'"
    ).fetchone()[0]

    print(f"  datasets : {len(enabled_names)}  ({', '.join(enabled_names)})")
    print(f"  images   : {total_images:,}")
    print(f"  classes  : {total_classes}")
    print(f"  instances: {total_instances:,}")

    # Per-dataset breakdown
    print()
    print(f"  {'dataset':<20} {'images':>8} {'classes':>8} {'instances':>12}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*12}")
    rows = conn.execute("""
        SELECT
            i.dataset,
            COUNT(DISTINCT icc.image_id) AS num_images,
            COUNT(DISTINCT icc.class_key) AS num_classes,
            SUM(icc.count)               AS num_instances
        FROM image_class_counts icc
        JOIN images i ON i.image_id = icc.image_id
        GROUP BY i.dataset
        ORDER BY i.dataset
    """).fetchall()
    for r in rows:
        print(
            f"  {r['dataset']:<20} {r['num_images']:>8,} {r['num_classes']:>8} {r['num_instances']:>12,}"
        )

    # Per-class breakdown
    print()
    print(f"  {'class_key':<40} {'images':>8} {'instances':>12}")
    print(f"  {'-'*40} {'-'*8} {'-'*12}")
    class_rows = conn.execute("""
        SELECT
            icc.class_key,
            COUNT(DISTINCT icc.image_id) AS num_images,
            SUM(icc.count)               AS num_instances
        FROM image_class_counts icc
        GROUP BY icc.class_key
        ORDER BY icc.class_key
    """).fetchall()
    for r in class_rows:
        print(f"  {r['class_key']:<40} {r['num_images']:>8,} {r['num_instances']:>12,}")

    conn.close()
