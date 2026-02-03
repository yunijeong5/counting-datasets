from pathlib import Path
import pytest

from counting_dataset.adapters.malaria import MalariaAdapter
from counting_dataset.index.builder import IndexBuilder
from counting_dataset.api.counting_dataset_index import CountingDatasetIndex


@pytest.mark.smoke
def test_split_normalization_train_variants(tmp_path: Path):
    """
    split inputs should be case-insensitive and accept common synonyms.
    E.g., "Train" / "training" should normalize to "train".
    """
    repo_root = Path.cwd()
    raw_root = repo_root / "raw"
    if not (raw_root / "malaria").exists():
        pytest.skip("raw/malaria not found; skipping")

    out_root = tmp_path / "counting_data"
    builder = IndexBuilder(raw_root=raw_root, out_root=out_root)
    builder.build([MalariaAdapter()], overwrite=True)

    index = CountingDatasetIndex(root=out_root)

    classes = index.list_classes(apply_policy=False)
    assert len(classes) > 0
    class_key = classes[0]["class_key"]

    ds1 = index.load_class(class_key, split="train", load_images=False)
    ds2 = index.load_class(class_key, split="Train", load_images=False)
    ds3 = index.load_class(class_key, split="training", load_images=False)

    assert len(ds1) == len(ds2) == len(ds3)


@pytest.mark.smoke
def test_missing_split_handling_load_class(tmp_path: Path):
    """
    If a split doesn't exist for the class, load_class should handle it gracefully:
      - on_missing_split="empty": returns empty dataset
      - on_missing_split="raise": raises ValueError
    """
    repo_root = Path.cwd()
    raw_root = repo_root / "raw"
    if not (raw_root / "malaria").exists():
        pytest.skip("raw/malaria not found; skipping")

    out_root = tmp_path / "counting_data"
    builder = IndexBuilder(raw_root=raw_root, out_root=out_root)
    builder.build([MalariaAdapter()], overwrite=True)

    index = CountingDatasetIndex(root=out_root)

    classes = index.list_classes(apply_policy=False)
    assert len(classes) > 0
    class_key = classes[0]["class_key"]

    # Pick a split that definitely should not exist in malaria (commonly no val).
    ds_empty = index.load_class(
        class_key,
        split="val",
        load_images=False,
        on_missing_split="empty",
    )
    assert len(ds_empty) == 0

    with pytest.raises(ValueError):
        index.load_class(
            class_key,
            split="val",
            load_images=False,
            on_missing_split="raise",
        )


@pytest.mark.smoke
def test_missing_split_handling_load_dataset(tmp_path: Path):
    """
    If a requested split doesn't exist for the dataset, load_dataset should:
      - in empty mode: drop missing splits and still return data for available ones
      - in raise mode: raise ValueError
    """
    repo_root = Path.cwd()
    raw_root = repo_root / "raw"
    if not (raw_root / "malaria").exists():
        pytest.skip("raw/malaria not found; skipping")

    out_root = tmp_path / "counting_data"
    builder = IndexBuilder(raw_root=raw_root, out_root=out_root)
    builder.build([MalariaAdapter()], overwrite=True)

    index = CountingDatasetIndex(root=out_root)

    # "train" should exist; "val" likely doesn't. Empty mode should keep train.
    ds_mixed = index.load_dataset(
        "malaria",
        splits={"train", "val"},
        load_images=False,
        on_missing_split="empty",
    )
    ds_train = index.load_dataset(
        "malaria",
        split="train",
        load_images=False,
        on_missing_split="raise",
    )
    assert len(ds_mixed) == len(ds_train)

    # Raise mode should error if any split is missing
    with pytest.raises(ValueError):
        index.load_dataset(
            "malaria",
            splits={"train", "val"},
            load_images=False,
            on_missing_split="raise",
        )
