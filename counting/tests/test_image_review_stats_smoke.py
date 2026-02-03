from pathlib import Path
import sqlite3
import pytest

from counting_dataset.adapters.penguin import PenguinAdapter
from counting_dataset.index.builder import IndexBuilder
from counting_dataset.api.counting_dataset_index import CountingDatasetIndex


@pytest.mark.smoke
def test_image_review_stats_populated_and_filters_work(tmp_path: Path):
    repo_root = Path.cwd()
    raw_root = repo_root / "raw"
    if not (raw_root / "penguin").exists():
        pytest.skip("raw/penguin not found; skipping")

    out_root = tmp_path / "counting_data"
    db_path = IndexBuilder(raw_root=raw_root, out_root=out_root).build(
        [PenguinAdapter()], overwrite=True
    )
    assert db_path.exists()

    conn = sqlite3.connect(str(db_path))
    try:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "image_review_stats" in tables

        n_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        n_stats = conn.execute("SELECT COUNT(*) FROM image_review_stats").fetchone()[0]
        assert n_images > 0
        assert n_stats == n_images

        # There should be some distribution of statuses (depends on dataset; at least 'na' or reviewed/unreviewed)
        statuses = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT review_status FROM image_review_stats"
            ).fetchall()
        }
        assert len(statuses) >= 1
    finally:
        conn.close()

    index = CountingDatasetIndex(root=out_root)

    ds_all = index.load_dataset("penguin", load_images=False)
    assert len(ds_all) == n_images

    # crowd_reviewed_only should never increase dataset size
    ds_reviewed = index.load_dataset(
        "penguin", load_images=False, crowd_reviewed_only=True
    )
    assert len(ds_reviewed) <= len(ds_all)

    # min_annotators filter should never increase dataset size
    ds_annot2 = index.load_dataset("penguin", load_images=False, min_annotators=2)
    assert len(ds_annot2) <= len(ds_all)

    # Spot check returned fields exist
    if len(ds_all) > 0:
        _, tgt = ds_all[0]
        assert "review_status" in tgt
        assert "num_annotators" in tgt
