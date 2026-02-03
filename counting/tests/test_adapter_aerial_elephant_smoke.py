from pathlib import Path
import pytest

from counting_dataset.adapters.aerial_elephant import AerialElephantAdapter
from counting_dataset.index.builder import IndexBuilder
from counting_dataset.api.counting_dataset_index import CountingDatasetIndex


@pytest.mark.smoke
def test_build_index_with_aerial_elephant(tmp_path: Path):
    repo_root = Path.cwd()
    raw_root = repo_root / "raw"
    if not (raw_root / "aerial-elephant-dataset").exists():
        pytest.skip("raw/aerial-elephant-dataset not found; skipping")

    out_root = tmp_path / "counting_data"
    db_path = IndexBuilder(raw_root=raw_root, out_root=out_root).build(
        [AerialElephantAdapter()],
        overwrite=True,
    )
    assert db_path.exists()

    index = CountingDatasetIndex(root=out_root)
    assert "aerial_elephant" in index.available_datasets()

    classes = index.get_classes(datasets=["aerial_elephant"], apply_policy=False)
    assert len(classes) == 1
    assert classes[0]["class_key"] == "aerial_elephant/elephant"

    ds = index.load_dataset("aerial_elephant", split="train", load_images=False, on_missing_split="warn")
    assert len(ds) > 0
