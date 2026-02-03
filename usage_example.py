from pathlib import Path

from counting_dataset.index.builder import IndexBuilder
from counting_dataset.adapters.malaria import MalariaAdapter
from counting_dataset.adapters.kenyan_wildlife import KenyanWildlifeAdapter
from counting_dataset.adapters.penguin import PenguinAdapter
from counting_dataset.adapters.aerial_elephant import AerialElephantAdapter
from counting_dataset.adapters.fsc147 import FSC147Adapter
from counting_dataset.adapters.dota import DOTAAdapter

from counting_dataset import CountingDatasetIndex
from counting_dataset.index.policy import FilterPolicy


def _print_class_summary(
    index: CountingDatasetIndex,
    dataset: str,
    apply_policy: bool = False,
    limit: int = 15,
):
    classes = index.get_classes(datasets=[dataset], apply_policy=apply_policy)
    print(f"\n--- Classes ({dataset}) [showing up to {limit}] ---")
    if not classes:
        print("  (none)")
        return classes
    for c in classes[:limit]:
        print(
            f"  {c['class_key']}: images={c['num_images']} instances={c['num_instances']} ann_types={c.get('ann_types')}"
        )
    if len(classes) > limit:
        print(f"  ... (+{len(classes) - limit} more)")
    return classes


def _peek_image_dataset(ds, title: str, n: int = 2):
    print(f"\n--- {title} ---")
    print("  num_images:", len(ds))
    if len(ds) == 0:
        return
    for i in range(min(n, len(ds))):
        img, tgt = ds[i]
        print(f"  sample[{i}] image:", img)
        print(f"    total_count={tgt.get('total_count')}")
        if "review_status" in tgt:
            print(
                f"    review_status={tgt.get('review_status')} "
                f"num_annotators={tgt.get('num_annotators')} "
                f"num_point_votes={tgt.get('num_point_votes')}"
            )
        counts = tgt.get("counts", {})
        if isinstance(counts, dict):
            some = list(sorted(counts.items()))[:8]
            print(f"    classes_present={len(counts)} (showing up to 8): {some}")
        print("tgt keys: ", tgt.keys())


def main():
    index_dest = Path("counting/data")
    rebuild_index = True

    # ------------------------------------------------------------------
    # Build Index
    # ------------------------------------------------------------------
    if rebuild_index:
        builder = IndexBuilder(raw_root=Path("raw"), out_root=index_dest)
        db_path = builder.build(
            [
                MalariaAdapter(),
                KenyanWildlifeAdapter(),
                PenguinAdapter(),
                AerialElephantAdapter(),
                FSC147Adapter(),
                DOTAAdapter(),
            ],
            overwrite=True,
            show_progress=True,
        )
        print("Built:", db_path)

    # ------------------------------------------------------------------
    # Index + discovery
    # ------------------------------------------------------------------
    policy = FilterPolicy(
        min_images_per_class=10,
        # allowed_splits={"train", "test"},
        allowed_ann_types={"hbb", "point", "obb"},
    )

    index = CountingDatasetIndex(root=index_dest, policy=policy)

    print("\n=== Available datasets ===")
    ds_list = index.available_datasets()
    print(ds_list)

    print("\n=== Available splits (global) ===")
    print(index.available_splits())

    for ds_name in ds_list:
        print(f"\n--- Available splits ({ds_name}) ---")
        print(index.available_splits(dataset=ds_name))

    # ------------------------------------------------------------------
    # DOTA (v1.5)
    # ------------------------------------------------------------------
    _print_class_summary(index, "dota", apply_policy=False, limit=10)

    print("\n=== DOTA: load_dataset('dota') (image-centric) ===")
    ds_dota = index.load_dataset(
        "dota",
        split="train",
        load_images=False,
        min_total_count=1,  # ensure at least one canonical instance
        on_missing_split="warn",
        natural_sort=True,
    )
    _peek_image_dataset(ds_dota, "DOTA image-centric peek")

    # Sanity checks:
    # - instances should be OBB (role="instance") and contribute to counts
    # - HBB should appear under aux["hbb"]
    if len(ds_dota) > 0:
        img, tgt = ds_dota[0]
        print("\nDOTA sanity checks on first sample:")

        # show one class bucket’s instance ann_types
        inst = tgt.get("instances", {}) or {}
        if inst:
            ck0 = sorted(inst.keys())[0]
            anns0 = inst[ck0]
            ann_types0 = sorted({a.get("ann_type") for a in anns0})
            print(
                f"  instances bucket example: {ck0} -> {len(anns0)} anns, ann_types={ann_types0}"
            )
            print(anns0[0])
        else:
            print(
                "  [WARN] instances is empty on first sample (unexpected if min_total_count=1)."
            )

        aux = tgt.get("aux", {}) or {}
        hbb_aux = aux.get("hbb", {}) or {}
        if hbb_aux:
            ck0 = sorted(hbb_aux.keys())[0]
            boxes = hbb_aux[ck0]
            print(
                f"  aux['hbb'] present for class {ck0}: {len(boxes)} (showing up to 2)"
            )
            for b in boxes[:2]:
                print("    hbb:", b["geometry"])
        else:
            print("  aux['hbb']: (none in this sample)")

    # Class-centric load: pick one DOTA class
    dota_classes = index.get_classes(datasets=["dota"], apply_policy=False)
    if dota_classes:
        dota_ck = dota_classes[0]["class_key"]
        print(f"\n=== DOTA: load_class({dota_ck}) ===")
        ds_dota_class = index.load_class(
            dota_ck,
            split="train",
            load_images=False,
            min_count=1,
            on_missing_split="warn",
            natural_sort=True,
        )
        print("  class-centric length:", len(ds_dota_class))
        if len(ds_dota_class) > 0:
            img, tgt = ds_dota_class[0]
            print("tgt keys:", tgt.keys())
            print("  image id: ", tgt["image_id"])
            print("  sample image:", img)
            print("  class count in image:", tgt["count"])
            print("  instances: ", tgt["instances"][:2])

    # ------------------------------------------------------------------
    # FSC147
    # ------------------------------------------------------------------
    _print_class_summary(index, "fsc147", apply_policy=False, limit=10)

    print("\n=== FSC147: load_dataset('fsc147') (image-centric) ===")
    ds_fsc = index.load_dataset(
        "fsc147",
        # split="train",
        load_images=False,
        min_total_count=1,  # should have points; safe guard
        on_missing_split="warn",
        natural_sort=True,
    )
    _peek_image_dataset(ds_fsc, "FSC147 image-centric peek")

    # Sanity checks: one "active" class per image; exemplars are aux (role="exemplar")
    if len(ds_fsc) > 0:
        img, tgt = ds_fsc[0]
        nonzero_classes = [ck for ck, cnt in tgt.get("counts", {}).items() if cnt > 0]
        print("\nFSC147 sanity checks on first sample:")
        print("  nonzero_classes:", nonzero_classes)
        if len(nonzero_classes) != 1:
            print("  [WARN] Expected exactly 1 nonzero class for FSC147 image.")

        aux = tgt.get("aux", {}) or {}
        exemplar = aux.get("exemplar", {}) or {}
        # exemplar boxes are stored as HBB records under the same class_key
        if len(exemplar) > 0:
            # show up to 1 class bucket + 2 boxes
            ck0 = sorted(exemplar.keys())[0]
            boxes = exemplar[ck0]
            print(
                f"  exemplar boxes present for class {ck0}: {len(boxes)} (showing up to 2)"
            )
            for b in boxes[:2]:
                print("    exemplar:", b["geometry"])
        else:
            print("  exemplar boxes: (none in this sample)")

    # Class-centric load: pick the first FSC class
    fsc_classes = index.get_classes(datasets=["fsc147"], apply_policy=False)
    if fsc_classes:
        fsc_ck = fsc_classes[0]["class_key"]
        print(f"\n=== FSC147: load_class({fsc_ck}) ===")
        ds_fsc_class = index.load_class(
            fsc_ck,
            # split="train",
            load_images=False,
            min_count=1,
            on_missing_split="warn",
        )
        print("  class-centric length:", len(ds_fsc_class))
        if len(ds_fsc_class) > 0:
            img, tgt = ds_fsc_class[0]
            print("  sample image:", img)
            print("  class count in image:", tgt["count"])

    # ------------------------------------------------------------------
    # AERIAL ELEPHANT
    # ------------------------------------------------------------------
    _print_class_summary(index, "aerial_elephant", apply_policy=False, limit=5)
    print("\n=== Aerial elephant: load_dataset('aerial_elephant') ===")
    ds_ae = index.load_dataset(
        "aerial_elephant",
        split="train",
        load_images=False,
        min_total_count=1,
        on_missing_split="warn",
    )
    _peek_image_dataset(ds_ae, "Aerial elephant image-centric peek")

    # Also test class-centric view (should be exactly one class)
    ae_classes = index.get_classes(datasets=["aerial_elephant"], apply_policy=False)
    if ae_classes:
        ae_ck = ae_classes[0]["class_key"]
        print(f"\n=== Aerial elephant: load_class({ae_ck}) ===")
        ds_ae_class = index.load_class(
            ae_ck,
            split="train",
            load_images=False,
            min_count=1,
            on_missing_split="warn",
        )
        print("  class-centric length:", len(ds_ae_class))
        if len(ds_ae_class) > 0:
            img, tgt = ds_ae_class[0]
            print("  sample image:", img)
            print("  class count in image:", tgt["count"])
            print("  target instances: ", tgt["instances"])

    # ------------------------------------------------------------------
    # MALARIA
    # ------------------------------------------------------------------
    malaria_classes = _print_class_summary(
        index, "malaria", apply_policy=False, limit=10
    )
    if malaria_classes:
        class_key = malaria_classes[0]["class_key"]
        print(f"\n=== Malaria: load_class({class_key}) ===")
        ds_class = index.load_class(
            class_key,
            # split="train",
            load_images=False,
            min_count=5,
            max_count=500,
            on_missing_split="warn",
        )
        print("  class-centric length:", len(ds_class))
        if len(ds_class) > 0:
            img, target = ds_class[0]
            print("  sample image:", img)
            print("  class count in image:", target["count"])

        print("\n=== Malaria: load_dataset('malaria') (image-centric) ===")
        ds_mal = index.load_dataset(
            "malaria",
            # split="train",
            load_images=False,
            min_total_count=1,
            on_missing_split="warn",
        )
        _peek_image_dataset(ds_mal, "Malaria image-centric peek")

    # ------------------------------------------------------------------
    # KENYAN WILDLIFE
    # ------------------------------------------------------------------
    _print_class_summary(index, "kenyan_wildlife", apply_policy=False, limit=10)
    print("\n=== Kenyan wildlife: load_dataset('kenyan_wildlife') ===")
    ds_kw = index.load_dataset(
        "kenyan_wildlife",
        # splits={"train", "test"},
        load_images=False,
        min_total_count=1,  # drop empty images
        max_total_count=500,  # sanity cap
        on_missing_split="warn",
    )
    _peek_image_dataset(ds_kw, "Kenyan wildlife image-centric peek")

    print("\n=== Kenyan wildlife: per-class totals (image-centric) ===")
    totals_kw = {}
    for _, tgt in ds_kw:
        for ck, cnt in tgt.get("counts", {}).items():
            totals_kw[ck] = totals_kw.get(ck, 0) + cnt
    for ck in sorted(totals_kw.keys()):
        print(f"  {ck}: total_instances={totals_kw[ck]}")

    # ------------------------------------------------------------------
    # PENGUIN
    # ------------------------------------------------------------------
    _print_class_summary(index, "penguin", apply_policy=False, limit=5)
    print("\n=== Penguin: load_dataset('penguin') with crowd filters ===")
    ds_peng_all = index.load_dataset(
        "penguin",
        # split="train",
        load_images=False,
        on_missing_split="warn",
    )
    _peek_image_dataset(ds_peng_all, "Penguin (no crowd pruning) peek")

    # If you renamed reviewed_only -> crowd_reviewed_only, use that here.
    # If not renamed yet, change arg name accordingly.
    ds_peng_reviewed = index.load_dataset(
        "penguin",
        # split="train",
        load_images=False,
        crowd_reviewed_only=True,  # keep only images marked as reviewed by crowd
        min_annotators=2,  # at least 2 annotator entries
        on_missing_split="warn",
    )
    _peek_image_dataset(
        ds_peng_reviewed, "Penguin (crowd_reviewed_only + min_annotators=2) peek"
    )


if __name__ == "__main__":
    main()
