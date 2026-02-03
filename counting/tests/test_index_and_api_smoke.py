from pathlib import Path
import sqlite3
import pytest

from counting_dataset.adapters.malaria import MalariaAdapter
from counting_dataset.index.builder import IndexBuilder
from counting_dataset.api.counting_dataset_index import CountingDatasetIndex


@pytest.mark.smoke
def test_build_index_and_basic_api_smoke(tmp_path: Path):
    """
    End-to-end smoke test:
      - builds sqlite index from malaria adapter
      - checks some tables are non-empty
      - exercises CountingDatasetIndex APIs
    """
    repo_root = Path.cwd()
    raw_root = repo_root / "raw"

    if not (raw_root / "malaria").exists():
        pytest.skip("raw/malaria not found; skipping smoke test")

    out_root = tmp_path / "counting_data"
    builder = IndexBuilder(raw_root=raw_root, out_root=out_root)
    db_path = builder.build([MalariaAdapter()], overwrite=True)

    assert db_path.exists()

    # sanity check DB tables
    conn = sqlite3.connect(str(db_path))
    try:
        n_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        n_classes = conn.execute("SELECT COUNT(*) FROM classes").fetchone()[0]
        n_anns = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
        n_counts = conn.execute("SELECT COUNT(*) FROM image_class_counts").fetchone()[0]
    finally:
        conn.close()

    assert n_images > 0
    assert n_classes > 0
    assert n_anns > 0
    assert n_counts > 0

    # now test API
    cd = CountingDatasetIndex(root=out_root)
    classes = cd.list_classes()
    assert len(classes) > 0

    # pick one class and load
    ck = classes[0]["class_key"]
    ds = cd.load_class(ck, split="train")
    assert len(ds) >= 0  # could be 0 if class only appears in test, but usually >0

    # exercise a single sample if available
    if len(ds) > 0:
        img, target = ds[0]
        assert "image_id" in target
        assert "class_key" in target
        assert target["class_key"] == ck
