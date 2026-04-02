from pathlib import Path
import pytest

from counting_dataset.adapters.malaria import MalariaAdapter
from counting_dataset.index.builder import IndexBuilder
from counting_dataset.api.counting_dataset_index import CountingDatasetIndex


@pytest.fixture(scope="module")
def index(tmp_path_factory: pytest.TempPathFactory) -> CountingDatasetIndex:
    repo_root = Path.cwd()
    raw_root = repo_root / "raw"
    if not (raw_root / "malaria").exists():
        pytest.skip("raw/malaria not found; skipping smoke test")

    out_root = tmp_path_factory.mktemp("counting_data")
    builder = IndexBuilder(raw_root=raw_root, out_root=out_root)
    builder.build([MalariaAdapter()], overwrite=True)
    return CountingDatasetIndex(root=out_root)


def test_image_dataset_exposes_width_and_height(index: CountingDatasetIndex):
    ds = index.load_dataset("malaria", load_images=False)
    assert len(ds) > 0

    for i in range(len(ds)):
        _, target = ds[i]
        assert "width" in target, f"sample {i}: 'width' missing from target"
        assert "height" in target, f"sample {i}: 'height' missing from target"
        if target["width"] is not None:
            assert isinstance(target["width"], int) and target["width"] > 0
        if target["height"] is not None:
            assert isinstance(target["height"], int) and target["height"] > 0


def test_class_dataset_exposes_width_and_height(index: CountingDatasetIndex):
    classes = index.list_classes()
    assert len(classes) > 0

    ck = classes[0]["class_key"]
    ds = index.load_class(ck, load_images=False)
    assert len(ds) > 0

    for i in range(len(ds)):
        _, target = ds[i]
        assert "width" in target, f"sample {i}: 'width' missing from target"
        assert "height" in target, f"sample {i}: 'height' missing from target"
        if target["width"] is not None:
            assert isinstance(target["width"], int) and target["width"] > 0
        if target["height"] is not None:
            assert isinstance(target["height"], int) and target["height"] > 0


def test_image_dataset_dimensions_match_db(index: CountingDatasetIndex):
    """width/height in the target must match what is stored in the images table."""
    import sqlite3

    ds = index.load_dataset("malaria", load_images=False)
    assert len(ds) > 0

    conn = sqlite3.connect(str(index.index_path))
    conn.row_factory = sqlite3.Row
    try:
        db = {
            r["image_id"]: (r["width"], r["height"])
            for r in conn.execute("SELECT image_id, width, height FROM images").fetchall()
        }
    finally:
        conn.close()

    for i in range(len(ds)):
        _, target = ds[i]
        iid = target["image_id"]
        assert iid in db, f"image_id {iid!r} not found in DB"
        expected_w, expected_h = db[iid]
        assert target["width"] == expected_w, (
            f"image_id {iid!r}: width mismatch {target['width']} != {expected_w}"
        )
        assert target["height"] == expected_h, (
            f"image_id {iid!r}: height mismatch {target['height']} != {expected_h}"
        )


def test_class_dataset_dimensions_match_db(index: CountingDatasetIndex):
    """width/height in the target must match what is stored in the images table."""
    import sqlite3

    classes = index.list_classes()
    ck = classes[0]["class_key"]
    ds = index.load_class(ck, load_images=False)
    assert len(ds) > 0

    conn = sqlite3.connect(str(index.index_path))
    conn.row_factory = sqlite3.Row
    try:
        db = {
            r["image_id"]: (r["width"], r["height"])
            for r in conn.execute("SELECT image_id, width, height FROM images").fetchall()
        }
    finally:
        conn.close()

    for i in range(len(ds)):
        _, target = ds[i]
        iid = target["image_id"]
        assert iid in db, f"image_id {iid!r} not found in DB"
        expected_w, expected_h = db[iid]
        assert target["width"] == expected_w, (
            f"image_id {iid!r}: width mismatch {target['width']} != {expected_w}"
        )
        assert target["height"] == expected_h, (
            f"image_id {iid!r}: height mismatch {target['height']} != {expected_h}"
        )
