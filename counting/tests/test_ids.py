import pytest
from pathlib import Path

from counting_dataset.core.ids import (
    make_ann_id,
    make_image_id,
    normalize_relpath,
    safe_join,
)
from counting_dataset.core.schema import (
    AnnType,
    Point,
    SourceType,
)


def test_normalize_relpath_basic():
    assert normalize_relpath("images/a.jpg") == "images/a.jpg"
    assert normalize_relpath("./images/a.jpg") == "images/a.jpg"
    assert normalize_relpath("images//a.jpg") == "images/a.jpg"
    assert normalize_relpath(r"images\a.jpg") == "images/a.jpg"


def test_normalize_relpath_rejects_absolute():
    with pytest.raises(ValueError):
        normalize_relpath("/abs/path.jpg")
    with pytest.raises(ValueError):
        normalize_relpath(r"C:\abs\path.jpg")


def test_safe_join_blocks_traversal(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()

    # valid join
    p = safe_join(root, "a/b/c.txt")
    assert str(p).endswith("a/b/c.txt")

    # traversal attempts
    with pytest.raises(ValueError):
        safe_join(root, "../evil.txt")
    with pytest.raises(ValueError):
        safe_join(root, "a/../../evil.txt")


def test_make_image_id_deterministic_and_normalizes_paths():
    img1 = make_image_id("FSC147_384", "images_384_VarV2/img_0001.jpg")
    img2 = make_image_id("FSC147_384", r"images_384_VarV2\img_0001.jpg")
    img3 = make_image_id("FSC147_384", "./images_384_VarV2/img_0001.jpg")

    assert img1 == img2 == img3
    assert img1.startswith("img_")


def test_make_ann_id_deterministic_same_inputs_same_id():
    image_id = make_image_id("penguin", "images/BAILa/foo.png")
    class_key = "penguin/penguin"
    geom = Point(x=10.0, y=20.0)

    a1 = make_ann_id(
        image_id=image_id,
        class_key=class_key,
        ann_type=AnnType.POINT,
        geometry=geom,
        source=SourceType.CROWDSOURCE,
        instance_index=0,
    )
    a2 = make_ann_id(
        image_id=image_id,
        class_key=class_key,
        ann_type=AnnType.POINT,
        geometry=geom,
        source=SourceType.CROWDSOURCE,
        instance_index=0,
    )

    assert a1 == a2
    assert a1.startswith("ann_")


def test_make_ann_id_instance_index_disambiguates_duplicates():
    """
    Two annotations with identical geometry can occur (e.g., duplicate crowd points).
    instance_index must disambiguate them deterministically.
    """
    image_id = make_image_id("penguin", "images/BAILa/foo.png")
    class_key = "penguin/penguin"
    geom = Point(x=10.0, y=20.0)

    a0 = make_ann_id(
        image_id=image_id,
        class_key=class_key,
        ann_type=AnnType.POINT,
        geometry=geom,
        source=SourceType.CROWDSOURCE,
        instance_index=0,
    )
    a1 = make_ann_id(
        image_id=image_id,
        class_key=class_key,
        ann_type=AnnType.POINT,
        geometry=geom,
        source=SourceType.CROWDSOURCE,
        instance_index=1,
    )

    assert a0 != a1


def test_make_ann_id_salt_disambiguates_duplicates_without_index():
    image_id = make_image_id("penguin", "images/BAILa/foo.png")
    class_key = "penguin/penguin"
    geom = Point(x=10.0, y=20.0)

    a1 = make_ann_id(
        image_id=image_id,
        class_key=class_key,
        ann_type=AnnType.POINT,
        geometry=geom,
        source=SourceType.CROWDSOURCE,
        salt="worker_001_vote_0003",
    )
    a2 = make_ann_id(
        image_id=image_id,
        class_key=class_key,
        ann_type=AnnType.POINT,
        geometry=geom,
        source=SourceType.CROWDSOURCE,
        salt="worker_007_vote_0101",
    )

    assert a1 != a2


def test_make_ann_id_rejects_negative_instance_index():
    image_id = make_image_id("penguin", "images/BAILa/foo.png")
    with pytest.raises(ValueError):
        make_ann_id(
            image_id=image_id,
            class_key="penguin/penguin",
            ann_type=AnnType.POINT,
            geometry=Point(x=1.0, y=2.0),
            source=SourceType.CROWDSOURCE,
            instance_index=-1,
        )
