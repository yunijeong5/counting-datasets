from pathlib import Path
import pytest

from counting_dataset.adapters.fsc147 import FSC147Adapter
from counting_dataset.index.builder import IndexBuilder
from counting_dataset.api.counting_dataset_index import CountingDatasetIndex


@pytest.mark.smoke
def test_build_index_with_fsc147(tmp_path: Path):
    repo_root = Path.cwd()
    raw_root = repo_root / "raw"
    if not (raw_root / "FSC147_384").exists():
        pytest.skip("raw/FSC147_384 not found; skipping")

    out_root = tmp_path / "counting_data"
    db_path = IndexBuilder(raw_root=raw_root, out_root=out_root).build(
        [FSC147Adapter()], overwrite=True
    )
    assert db_path.exists()

    index = CountingDatasetIndex(root=out_root)
    assert "fsc147" in index.available_datasets()

    # should have many classes
    classes = index.get_classes(datasets=["fsc147"], apply_policy=False)
    assert len(classes) > 0

    # image-centric load should work
    ds = index.load_dataset(
        "fsc147", split="train", load_images=False, on_missing_split="warn"
    )
    assert len(ds) > 0

    img, tgt = ds[0]
    assert "counts" in tgt
    assert "aux" in tgt

    found = False
    for _, tgt in ds:
        if "exemplar" in (tgt.get("aux") or {}):
            found = True
            break
    assert found
