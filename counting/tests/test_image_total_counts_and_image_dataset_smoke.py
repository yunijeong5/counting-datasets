from pathlib import Path
import sqlite3
import pytest

from counting_dataset.adapters.malaria import MalariaAdapter
from counting_dataset.index.builder import IndexBuilder
from counting_dataset.api.counting_dataset_index import CountingDatasetIndex


@pytest.mark.smoke
def test_image_total_counts_table_and_load_dataset_pruning(tmp_path: Path):
    """
    End-to-end smoke test:
      - build index from MalariaAdapter
      - verify image_total_counts exists and is populated
      - verify load_dataset() respects min_total_count/max_total_count pruning
    """
    repo_root = Path.cwd()
    raw_root = repo_root / "raw"
    if not (raw_root / "malaria").exists():
        pytest.skip("raw/malaria not found; skipping smoke test")

    out_root = tmp_path / "counting_data"
    builder = IndexBuilder(raw_root=raw_root, out_root=out_root)
    db_path = builder.build([MalariaAdapter()], overwrite=True)

    assert db_path.exists()

    # Verify image_total_counts exists and has rows
    conn = sqlite3.connect(str(db_path))
    try:
        # table existence
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "image_total_counts" in tables

        n_itc = conn.execute("SELECT COUNT(*) FROM image_total_counts").fetchone()[0]
        n_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        assert n_itc > 0
        # should typically match number of images, but allow missing if there are images with 0 annotations
        assert n_itc <= n_images

        # Get global min/max total_count to test pruning deterministically
        mn, mx = conn.execute(
            "SELECT MIN(total_count), MAX(total_count) FROM image_total_counts"
        ).fetchone()
        assert mn is not None and mx is not None
        mn = int(mn)
        mx = int(mx)
        assert mn >= 0
        assert mx >= mn
    finally:
        conn.close()

    # API tests
    index = CountingDatasetIndex(root=out_root)

    # No pruning (all images in malaria, all splits)
    ds_all = index.load_dataset("malaria", load_images=False)
    assert len(ds_all) == n_images

    # Strong pruning: min_total_count > max => should yield empty
    ds_empty = index.load_dataset("malaria", load_images=False, min_total_count=mx + 1)
    assert len(ds_empty) == 0

    # Pruning at exact min and max should keep at least one image (unless degenerate)
    ds_min = index.load_dataset("malaria", load_images=False, min_total_count=mn)
    assert len(ds_min) == n_images  # since all totals are >= mn

    ds_max = index.load_dataset("malaria", load_images=False, max_total_count=mx)
    assert len(ds_max) == n_images  # since all totals are <= mx

    # If range is tight, dataset should shrink (usually). We don't hard assert shrink,
    # but we can sanity-check that returned targets satisfy the constraint.
    tight_min = mn + (mx - mn) // 2
    ds_tight = index.load_dataset(
        "malaria", load_images=False, min_total_count=tight_min
    )

    for _, tgt in ds_tight:
        assert tgt["total_count"] >= tight_min
