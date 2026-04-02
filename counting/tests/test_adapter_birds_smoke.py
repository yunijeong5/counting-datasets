"""
Smoke tests for BirdsAdapter + index build + API.

Requires raw/birds/tiles/ to be present.  All tests are marked @pytest.mark.smoke
and skip gracefully when the raw data is absent.

What is checked
---------------
Adapter layer (no index):
  - exactly one class emitted: birds/bird
  - image records are well-formed (stable ids, positive dims, correct meta keys)
  - both annotation types (point, hbb) appear; points are role="instance",
    hbb is role="hbb"
  - every annotation's image_id resolves to an image record

Index layer (built into tmp_path):
  - images / classes / annotations / image_class_counts tables are populated
  - image_total_counts matches bird point counts (hbb rows excluded)
  - all image paths in the DB exist on disk

API layer (CountingDatasetIndex):
  - load_dataset("birds") returns all tiles
  - meta_filter={"scene": "sky"} and meta_filter={"scene": "reeds"} partition
    the full set (sky + reeds == all)
  - target dict contains "counts", "instances", "aux" keys
  - aux["hbb"] contains hbb entries (when skimage available); CountingClassDataset
    uses {role: [ann, ...]} — no class_key nesting unlike CountingImageDataset
  - load_class("birds/bird") returns only annotated tiles (count >= 1)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Set

import pytest

from counting_dataset.adapters.base import AdapterContext
from counting_dataset.adapters.birds import BirdsAdapter
from counting_dataset.api.counting_dataset_index import CountingDatasetIndex
from counting_dataset.index.builder import IndexBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RAW_BIRDS = Path("raw/birds")


def _skip_if_missing():
    if not (RAW_BIRDS / "tiles").exists():
        pytest.skip("raw/birds/tiles not found; skipping birds smoke test")


def _build_index(tmp_path: Path) -> Path:
    builder = IndexBuilder(raw_root=Path("raw"), out_root=tmp_path / "data")
    return builder.build([BirdsAdapter()], overwrite=True)


# ---------------------------------------------------------------------------
# Adapter-layer tests (no index required)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_birds_adapter_one_class():
    _skip_if_missing()
    ctx = AdapterContext(raw_root=Path("raw"))
    classes = list(BirdsAdapter().iter_classes(ctx))

    assert len(classes) == 1
    assert classes[0].class_key == "birds/bird"
    assert classes[0].dataset == "birds"
    assert classes[0].name == "bird"


@pytest.mark.smoke
def test_birds_adapter_image_records():
    _skip_if_missing()
    ctx = AdapterContext(raw_root=Path("raw"))
    images = list(BirdsAdapter().iter_images(ctx))

    assert len(images) > 0, "No image records emitted"

    ids: Set[str] = set()
    for img in images:
        assert img.image_id.startswith("img_"), f"Bad image_id: {img.image_id}"
        assert img.image_id not in ids, f"Duplicate image_id: {img.image_id}"
        ids.add(img.image_id)

        assert img.width > 0 and img.height > 0, (
            f"Non-positive dims for {img.provenance.original_filename}"
        )
        assert img.provenance.dataset == "birds"
        assert img.split.value == "unspecified"

        assert "scene" in img.meta, "Missing 'scene' key in image.meta"
        assert "source_image" in img.meta, "Missing 'source_image' key in image.meta"
        assert "batch" in img.meta
        assert "tile_number" in img.meta
        assert img.meta["scene"] in ("sky", "reeds"), (
            f"Unexpected scene value: {img.meta['scene']}"
        )


@pytest.mark.smoke
def test_birds_adapter_both_scenes_present():
    _skip_if_missing()
    ctx = AdapterContext(raw_root=Path("raw"))
    images = list(BirdsAdapter().iter_images(ctx))

    scenes = {img.meta["scene"] for img in images}
    assert "sky" in scenes, "No tiles from the 'sky' scene found"
    assert "reeds" in scenes, "No tiles from the 'reeds' scene found"


@pytest.mark.smoke
def test_birds_adapter_annotations():
    _skip_if_missing()
    ctx = AdapterContext(raw_root=Path("raw"))
    ad = BirdsAdapter()

    image_ids = {img.image_id for img in ad.iter_images(ctx)}
    anns = list(ad.iter_annotations(ctx))

    assert len(anns) > 0, "No annotations emitted"

    point_anns = [a for a in anns if a.ann_type.value == "point"]
    hbb_anns   = [a for a in anns if a.ann_type.value == "hbb"]

    assert len(point_anns) > 0, "No point annotations"

    for ann in point_anns:
        assert ann.role == "instance", (
            f"Point annotation should be role='instance', got {ann.role!r}"
        )
        assert ann.source.value == "original"
        assert ann.class_key == "birds/bird"
        assert ann.image_id in image_ids, "Point ann references unknown image_id"
        geom = ann.geometry
        assert hasattr(geom, "x") and hasattr(geom, "y")
        assert geom.x >= 0 and geom.y >= 0

    for ann in hbb_anns:
        assert ann.role == "hbb", (
            f"HBB annotation should be role='hbb', got {ann.role!r}"
        )
        assert ann.source.value == "generated"
        assert ann.class_key == "birds/bird"
        assert ann.image_id in image_ids, "HBB ann references unknown image_id"
        geom = ann.geometry
        assert geom.w > 0 and geom.h > 0, "HBB has non-positive dimensions"


@pytest.mark.smoke
def test_birds_adapter_point_hbb_counts_match():
    """Every annotated tile should have the same number of points and hbb boxes."""
    _skip_if_missing()
    ctx = AdapterContext(raw_root=Path("raw"))
    ad = BirdsAdapter()
    anns = list(ad.iter_annotations(ctx))

    hbb_anns = [a for a in anns if a.ann_type.value == "hbb"]
    if not hbb_anns:
        pytest.skip("No HBB annotations (scikit-image probably not installed)")

    from collections import Counter
    pt_per_image  = Counter(a.image_id for a in anns if a.ann_type.value == "point")
    hbb_per_image = Counter(a.image_id for a in hbb_anns)

    # Only check tiles that received HBBs — tiles whose image file couldn't be
    # read (e.g. not present on this machine) will have points but no HBBs,
    # which is expected behaviour from _compute_pseudo_bboxes returning None.
    for iid, hbb_count in hbb_per_image.items():
        pt_count = pt_per_image.get(iid, 0)
        assert hbb_count == pt_count, (
            f"image {iid}: {hbb_count} hbb boxes but {pt_count} points"
        )


# ---------------------------------------------------------------------------
# Index-layer tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_birds_index_tables_populated(tmp_path: Path):
    _skip_if_missing()
    db_path = _build_index(tmp_path)
    assert db_path.exists()

    conn = sqlite3.connect(str(db_path))
    try:
        n_images  = conn.execute("SELECT COUNT(*) FROM images WHERE dataset='birds'").fetchone()[0]
        n_classes = conn.execute("SELECT COUNT(*) FROM classes WHERE dataset='birds'").fetchone()[0]
        n_pt_anns = conn.execute(
            "SELECT COUNT(*) FROM annotations WHERE class_key='birds/bird' AND ann_type='point'"
        ).fetchone()[0]
        n_hbb_anns = conn.execute(
            "SELECT COUNT(*) FROM annotations WHERE class_key='birds/bird' AND ann_type='hbb'"
        ).fetchone()[0]
        n_counts  = conn.execute(
            "SELECT COUNT(*) FROM image_class_counts WHERE class_key='birds/bird'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert n_images > 0
    assert n_classes == 1
    assert n_pt_anns > 0
    # hbb count >= 0 (depends on scikit-image availability)
    assert n_hbb_anns >= 0
    assert n_counts > 0


@pytest.mark.smoke
def test_birds_index_total_counts_reflect_points_only(tmp_path: Path):
    """image_total_counts should equal per-image point count (hbb role excluded)."""
    _skip_if_missing()
    db_path = _build_index(tmp_path)

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("""
            SELECT itc.image_id, itc.total_count,
                   COUNT(a.ann_id) AS pt_count
            FROM image_total_counts itc
            JOIN images i ON i.image_id = itc.image_id AND i.dataset = 'birds'
            LEFT JOIN annotations a
                   ON a.image_id = itc.image_id
                  AND a.ann_type = 'point'
                  AND a.role = 'instance'
            GROUP BY itc.image_id, itc.total_count
        """).fetchall()
    finally:
        conn.close()

    assert len(rows) > 0
    for image_id, total_count, pt_count in rows:
        assert total_count == pt_count, (
            f"image {image_id}: total_count={total_count} but point count={pt_count}"
        )


@pytest.mark.smoke
def test_birds_index_image_paths_exist(tmp_path: Path):
    _skip_if_missing()

    # The label JSON files live in raw/birds/tiles/…/labels/, but the actual
    # tile .jpg images may not be present on every machine (e.g. a cluster
    # where only labels were synced).  Skip rather than fail in that case.
    tile_images = list((RAW_BIRDS / "tiles").rglob("*.jpg"))
    if not tile_images:
        pytest.skip("No tile .jpg files found; skipping path existence check")

    db_path = _build_index(tmp_path)

    conn = sqlite3.connect(str(db_path))
    try:
        paths = [r[0] for r in conn.execute(
            "SELECT path FROM images WHERE dataset='birds'"
        ).fetchall()]
    finally:
        conn.close()

    missing = [p for p in paths if not Path(p).exists()]
    assert not missing, f"{len(missing)} image path(s) missing, e.g. {missing[:3]}"


# ---------------------------------------------------------------------------
# API-layer tests
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_birds_api_load_dataset(tmp_path: Path):
    _skip_if_missing()
    db_path = _build_index(tmp_path)
    index = CountingDatasetIndex(root=tmp_path / "data")

    ds_all = index.load_dataset("birds", load_images=False)
    assert len(ds_all) > 0

    # spot-check one sample
    _, target = ds_all[0]
    assert "counts" in target
    assert "instances" in target
    assert "aux" in target
    assert "total_count" in target


@pytest.mark.smoke
def test_birds_api_scene_filter_partitions(tmp_path: Path):
    """sky tiles + reeds tiles should equal all tiles (no overlap, full coverage)."""
    _skip_if_missing()
    db_path = _build_index(tmp_path)
    index = CountingDatasetIndex(root=tmp_path / "data")

    ds_all   = index.load_dataset("birds", load_images=False)
    ds_sky   = index.load_dataset("birds", load_images=False, meta_filter={"scene": "sky"})
    ds_reeds = index.load_dataset("birds", load_images=False, meta_filter={"scene": "reeds"})

    assert len(ds_sky) > 0,   "No tiles returned for scene='sky'"
    assert len(ds_reeds) > 0, "No tiles returned for scene='reeds'"
    assert len(ds_sky) + len(ds_reeds) == len(ds_all), (
        f"sky ({len(ds_sky)}) + reeds ({len(ds_reeds)}) != all ({len(ds_all)})"
    )

    # no overlap between scenes
    sky_ids   = {ds_sky[i][1]["image_id"]   for i in range(len(ds_sky))}
    reeds_ids = {ds_reeds[i][1]["image_id"] for i in range(len(ds_reeds))}
    assert sky_ids.isdisjoint(reeds_ids), "Sky and reeds share image_ids"


@pytest.mark.smoke
def test_birds_api_load_class(tmp_path: Path):
    """load_class should return only tiles that have at least one bird."""
    _skip_if_missing()
    db_path = _build_index(tmp_path)
    index = CountingDatasetIndex(root=tmp_path / "data")

    ds = index.load_class("birds/bird", load_images=False)
    assert len(ds) > 0

    for i in range(len(ds)):
        _, target = ds[i]
        assert target["count"] >= 1, (
            f"load_class returned a tile with count={target['count']}"
        )
        assert len(target["instances"]) >= 1


@pytest.mark.smoke
def test_birds_api_aux_hbb_present(tmp_path: Path):
    """When skimage is available, aux should carry hbb boxes for annotated tiles."""
    _skip_if_missing()
    pytest.importorskip("skimage", reason="scikit-image not installed")

    db_path = _build_index(tmp_path)
    index = CountingDatasetIndex(root=tmp_path / "data")

    ds = index.load_class("birds/bird", load_images=False)
    # CountingClassDataset.aux is {role: [ann, ...]} — no class_key nesting.
    found_hbb = False
    for i in range(min(len(ds), 20)):
        _, target = ds[i]
        hbb_list = target.get("aux", {}).get("hbb", [])
        if hbb_list:
            found_hbb = True
            for ann in hbb_list:
                assert ann["ann_type"] == "hbb"
                g = ann["geometry"]
                assert g["w"] > 0 and g["h"] > 0
            break

    assert found_hbb, "No hbb annotations found in first 20 annotated tiles"


@pytest.mark.smoke
def test_birds_scene_tile_and_bird_counts(tmp_path: Path):
    """Exact tile and total bird counts per scene, verified against known ground truth."""
    _skip_if_missing()
    db_path = _build_index(tmp_path)
    index = CountingDatasetIndex(root=tmp_path / "data")

    EXPECTED = {
        "sky":   {"tiles": 925,  "birds": 5847},
        "reeds": {"tiles": 1426, "birds": 12849},
    }

    for scene, expected in EXPECTED.items():
        ds = index.load_dataset("birds", load_images=False, meta_filter={"scene": scene})

        n_tiles = len(ds)
        n_birds = sum(ds[i][1]["total_count"] for i in range(n_tiles))

        assert n_tiles == expected["tiles"], (
            f"scene='{scene}': expected {expected['tiles']} tiles, got {n_tiles}"
        )
        assert n_birds == expected["birds"], (
            f"scene='{scene}': expected {expected['birds']} birds, got {n_birds}"
        )
