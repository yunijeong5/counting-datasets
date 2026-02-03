from pathlib import Path
import pytest
import sqlite3

from counting_dataset.adapters.malaria import MalariaAdapter
from counting_dataset.adapters.kenyan_wildlife import KenyanWildlifeAdapter
from counting_dataset.index.builder import IndexBuilder
from counting_dataset.api.counting_dataset_index import CountingDatasetIndex


@pytest.mark.smoke
def test_build_index_with_kenyan_wildlife(tmp_path: Path):
    repo_root = Path.cwd()
    raw_root = repo_root / "raw"
    if not (raw_root / "kenyan-wildlife-aerial-survey").exists():
        pytest.skip("raw/kenyan-wildlife-aerial-survey not found; skipping")

    out_root = tmp_path / "counting_data"
    db_path = IndexBuilder(raw_root=raw_root, out_root=out_root).build(
        [MalariaAdapter(), KenyanWildlifeAdapter()],
        overwrite=True,
    )
    assert db_path.exists()

    conn = sqlite3.connect(str(db_path))
    try:
        n_images = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        n_classes = conn.execute("SELECT COUNT(*) FROM classes").fetchone()[0]
        n_anns = conn.execute("SELECT COUNT(*) FROM annotations").fetchone()[0]
        assert n_images > 0 and n_classes > 0 and n_anns > 0
    finally:
        conn.close()

    index = CountingDatasetIndex(root=out_root)
    # Kenyan wildlife should have multiple classes (Elephant/Giraffe/Zebra...)
    kw_classes = index.get_classes(datasets=["kenyan_wildlife"], apply_policy=False)
    assert len(kw_classes) >= 2

    # Image-centric dataset should return multi-class targets
    ds = index.load_dataset(
        "kenyan_wildlife", split="train", load_images=False, on_missing_split="empty"
    )
    if len(ds) > 0:
        _, tgt = ds[0]
        assert isinstance(tgt["counts"], dict)
