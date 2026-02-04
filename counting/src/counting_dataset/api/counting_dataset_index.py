from __future__ import annotations

import sqlite3
from pathlib import Path
import warnings
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from counting_dataset.api.counting_class_dataset import CountingClassDataset
from counting_dataset.api.counting_image_dataset import CountingImageDataset

from counting_dataset.index.policy import FilterPolicy

from counting_dataset.core.splits import normalize_split, normalize_splits


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _available_splits_for_dataset(index_path: Path, dataset: str) -> Set[str]:
    sql = "SELECT DISTINCT split FROM images WHERE dataset = ?"
    with _connect(index_path) as conn:
        rows = conn.execute(sql, [dataset]).fetchall()
    return {r["split"] for r in rows}


def _available_splits_for_class(index_path: Path, class_key: str) -> Set[str]:
    sql = """
    SELECT DISTINCT i.split AS split
    FROM images i
    JOIN image_class_counts icc ON icc.image_id = i.image_id
    WHERE icc.class_key = ?
    """
    with _connect(index_path) as conn:
        rows = conn.execute(sql, [class_key]).fetchall()
    return {r["split"] for r in rows}


class CountingDatasetIndex:
    """
    User-facing registry/query object backed by index.sqlite.

    root is a directory that contains index.sqlite (recommended name),
    e.g. root="counting/data" or root="/data/counting_v1".

    Filtering rules are applied here (not in adapters).

    Answers questions like:
    - What classes exist?
    - Which classes have between 300 and 1000 images?
    - Load me a dataset for FSC147_384/bird.
    """

    def __init__(
        self,
        root: Union[str, Path],
        *,
        index_name: str = "index.sqlite",
        policy: Optional[FilterPolicy] = None,
    ):
        self.root = Path(root)
        self.index_path = self.root / index_name
        if not self.index_path.exists():
            raise FileNotFoundError(f"index not found: {self.index_path}")

        self.policy = policy or FilterPolicy()

    # -------------------------
    # Dataset discovery
    # -------------------------

    def available_datasets(self) -> List[str]:
        """
        Return the list of dataset keys present in the index (e.g., ["malaria", "kenyan_wildlife"]).
        """
        sql = "SELECT DISTINCT dataset FROM images ORDER BY dataset"
        with _connect(self.index_path) as conn:
            rows = conn.execute(sql).fetchall()
        return [r["dataset"] for r in rows]

    def available_splits(self, *, dataset: Optional[str] = None) -> List[str]:
        """
        Return the list of available split strings present in the index.

        If `dataset` is provided, returns splits only for that dataset.
        Otherwise returns splits across the entire index.

        Examples:
            index.available_splits()                 -> ["test","train","val"]
            index.available_splits(dataset="malaria")-> ["test","train"]
        """
        if dataset is None:
            sql = "SELECT DISTINCT split FROM images ORDER BY split"
            params = []
        else:
            sql = "SELECT DISTINCT split FROM images WHERE dataset = ? ORDER BY split"
            params = [dataset]

        with _connect(self.index_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [r["split"] for r in rows]

    # -------------------------
    # Class discovery / stats
    # -------------------------

    def list_classes(self, *, apply_policy: bool = True) -> List[Dict[str, Any]]:
        """
        Returns list of dicts including basic per-class stats:
          - class_key, dataset, name
          - num_images (distinct images containing that class)
          - num_instances (total instances for that class across all images)

        If apply_policy=True, returns only classes allowed by policy.
        """

        sql = """
        SELECT
          c.class_key,
          c.dataset,
          c.name,
          COALESCE(imgs.num_images, 0) AS num_images,
          COALESCE(inst.num_instances, 0) AS num_instances,
          COALESCE(types.ann_types, '') AS ann_types_csv
        FROM classes c
        LEFT JOIN (
          SELECT class_key, COUNT(*) AS num_images
          FROM image_class_counts
          GROUP BY class_key
        ) imgs ON imgs.class_key = c.class_key
        LEFT JOIN (
          SELECT class_key, COUNT(*) AS num_instances
          FROM annotations
          WHERE role = 'instance'
          GROUP BY class_key
        ) inst ON inst.class_key = c.class_key
        LEFT JOIN (
          SELECT class_key, GROUP_CONCAT(DISTINCT ann_type) AS ann_types
          FROM annotations
          WHERE role = 'instance'
          GROUP BY class_key
        ) types ON types.class_key = c.class_key
        ORDER BY c.class_key
        """

        with _connect(self.index_path) as conn:
            rows = conn.execute(sql).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            ann_types = set(filter(None, (d.get("ann_types_csv") or "").split(",")))
            d["ann_types"] = sorted(ann_types)
            d.pop("ann_types_csv", None)

            if apply_policy:
                if not self.policy.class_allowed(
                    dataset=d["dataset"],
                    num_images=int(d["num_images"]),
                    num_instances=int(d["num_instances"]),
                    ann_types=set(d["ann_types"]),
                ):
                    continue

            out.append(d)

        return out

    def _fetch_ann_types_by_role(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Returns:
        {class_key: {role: [ann_type, ...], ...}, ...}
        """
        sql = """
        SELECT class_key, role, ann_type
        FROM annotations
        GROUP BY class_key, role, ann_type
        """
        out: Dict[str, Dict[str, Set[str]]] = {}

        with _connect(self.index_path) as conn:
            rows = conn.execute(sql).fetchall()

        for r in rows:
            ck = r["class_key"]
            role = r["role"] or "instance"
            ann_type = r["ann_type"]
            out.setdefault(ck, {}).setdefault(role, set()).add(ann_type)

        # convert to sorted lists
        out2: Dict[str, Dict[str, List[str]]] = {}
        for ck, role_map in out.items():
            out2[ck] = {role: sorted(list(s)) for role, s in role_map.items()}
        return out2

    def get_classes(
        self,
        *,
        min_images: Optional[int] = None,
        max_images: Optional[int] = None,
        min_instances: Optional[int] = None,
        max_instances: Optional[int] = None,
        datasets: Optional[Sequence[str]] = None,
        ann_types: Optional[Sequence[str]] = None,
        apply_policy: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Ad-hoc filtering on top of (optional) global policy.
        """
        classes = self.list_classes(apply_policy=apply_policy)

        available = set(self.available_datasets())
        if datasets is not None:
            ds_set = {ds.lower() for ds in datasets}
            unknown = ds_set - available
            if unknown:
                raise ValueError(
                    f"Unknown dataset(s) {sorted(unknown)}. "
                    f"Available: {sorted(available)}"
                )
        at_set = {a.lower() for a in ann_types} if ann_types is not None else None

        ann_types_by_role = self._fetch_ann_types_by_role()

        out = []
        for c in classes:
            # filtering
            if min_images is not None and c["num_images"] < min_images:
                continue
            if max_images is not None and c["num_images"] > max_images:
                continue
            if min_instances is not None and c["num_instances"] < min_instances:
                continue
            if max_instances is not None and c["num_instances"] > max_instances:
                continue
            if ds_set is not None and c["dataset"] not in ds_set:
                continue

            ck = c["class_key"]
            role_map = ann_types_by_role.get(ck, {})
            # union of all roles
            all_types = sorted({t for types in role_map.values() for t in types})

            if at_set is not None and len(set(all_types).intersection(at_set)) == 0:
                continue

            instance_types = role_map.get("instance", [])
            aux_types = sorted(
                {
                    t
                    for role, types in role_map.items()
                    if role != "instance"
                    for t in types
                }
            )
            c2 = dict(c)
            c2["ann_types"] = all_types
            c2["instance_ann_types"] = instance_types
            c2["aux_ann_types"] = aux_types
            c2["ann_types_by_role"] = role_map
            out.append(c2)
        return out

    # -------------------------
    # Loading
    # -------------------------

    def load_class(
        self,
        class_key: str,
        *,
        split: Optional[str] = None,
        load_images: bool = True,
        target_format: str = "instances",
        ignore_policy: bool = False,
        min_count: Optional[int] = None,
        max_count: Optional[int] = None,
        on_missing_split: str = "empty",
        natural_sort: bool = False,
    ) -> CountingClassDataset:
        """
        Create a class-centric iterable dataset for a single `class_key`.

        This returns a `CountingClassDataset` where each sample corresponds to an image
        that contains at least one instance of `class_key`. Each sample yields:

            (image, target)

        where `target` contains only the annotations for this class (not other classes).

        Parameters
        ----------
        class_key:
            Fully-qualified class identifier, e.g. "malaria/red_blood_cell" or "FSC147_384/bird".

        split:
            Optional split selector ("train", "val", "test"). If None, includes all splits.

        min_count / max_count:
            Optional *sample-level pruning* based on the number of instances of THIS class
            in a candidate image.

            Concretely, for each image, we look up `image_class_counts.count` for
            (image_id, class_key). The image is included only if:

                min_count <= count <= max_count

            If a bound is None, it is not applied.

            Examples:
            - min_count=5: keep only images where this class appears at least 5 times
            - max_count=200: drop extremely crowded images for this class

        load_images:
            If True, `__getitem__` loads and returns a PIL.Image. If False, returns the image path.

        ignore_policy:
            If False, the global FilterPolicy may prevent loading classes filtered out
            by class-level rules (e.g., too few images). Set True to override.

        target_format:
            Reserved for future extensions. Currently "instances" returns instance-level
            annotations for this class.

        on_missing_split:
            Behavior when no samples satisfy the split query.
            Options: 'empty' (default), 'warn', or 'raise'. If 'empty', returns a zero-length dataset.

        natural_sort:
            If True, apply human-friendly natural sorting (e.g., 2.jpg < 10.jpg) by filename
            after querying the DB. Default False uses lexicographic path order. Intended for
            convenience/debugging only.
        """
        dataset = class_key.lower().split("/")[0]
        if dataset not in set(self.available_datasets()):
            raise ValueError(
                f"Unknown dataset {dataset!r}. Available: {self.available_datasets()}"
            )

        if not ignore_policy:
            allowed = {c["class_key"] for c in self.list_classes(apply_policy=True)}
            if class_key not in allowed:
                raise ValueError(
                    f"class_key {class_key!r} is filtered out by policy. "
                    f"Use ignore_policy=True to override."
                )

        # normalize user input
        split_norm = normalize_split(split) if split is not None else None

        # Apply split restriction from policy
        if not ignore_policy and self.policy.allowed_splits is not None:
            if split_norm is None:
                # If user asked for "all splits", restrict to allowed ones by passing None here
                # and letting CountingClassDataset filter internally. We'll pass allowed_splits.
                allowed_splits = self.policy.allowed_splits
            else:
                if split_norm not in self.policy.allowed_splits:
                    raise ValueError(
                        f"split {split!r} is not allowed by policy.allowed_splits={sorted(self.policy.allowed_splits)}"
                    )
                allowed_splits = {split_norm}
        else:
            allowed_splits = None if split_norm is None else {split_norm}

        # handle missing split gracefully
        if split_norm is not None:
            available = _available_splits_for_class(self.index_path, class_key)
            if split_norm not in available:
                msg = (
                    f"Requested split={split!r} (normalized to {split_norm!r}) is not available "
                    f"for class_key={class_key!r}. Available splits: {sorted(available)}"
                )
                if on_missing_split == "raise":
                    raise ValueError(msg)
                if on_missing_split == "warn":
                    warnings.warn(msg)
                # empty dataset: pass an impossible splits set
                allowed_splits = {split_norm}  # dataset will naturally be empty

        return CountingClassDataset(
            index_path=self.index_path,
            class_key=class_key,
            splits=allowed_splits,
            load_images=load_images,
            target_format=target_format,
            min_count=min_count,
            max_count=max_count,
            natural_sort=natural_sort,
        )

    def load_dataset(
        self,
        dataset: str,
        *,
        split: Optional[str] = None,
        splits: Optional[Set[str]] = None,
        class_keys: Optional[Set[str]] = None,
        min_total_count: Optional[int] = None,
        max_total_count: Optional[int] = None,
        crowd_reviewed_only: bool = False,
        min_annotators: Optional[int] = None,
        max_annotators: Optional[int] = None,
        min_point_votes: Optional[int] = None,
        load_images: bool = True,
        ignore_policy: bool = False,
        on_missing_split: str = "empty",
        natural_sort: bool = False,
    ) -> CountingImageDataset:
        """
        Create an image-centric iterable dataset for an entire dataset (e.g., "malaria").

        This returns a `CountingImageDataset` where each sample corresponds to a single image,
        and the target includes counts and instance annotations for *all classes present*
        in that image (optionally restricted to `class_keys`).

        Each sample yields:

            (image, target)

        where `target` includes:
        - "counts": {class_key: count, ...}
        - "instances": {class_key: [instance, ...], ...}
        - "total_count": total number of annotations in the image across all classes

        Parameters
        ----------
        dataset:
            Dataset key (e.g., "malaria", "penguin", "dota_v15").

        split / splits:
            Optional split selector(s). Use either:
            - split="train" (single split), or
            - splits={"train","test"} (multiple splits)
            If neither is provided, includes all splits (subject to policy restrictions).

        class_keys:
            Optional set of fully-qualified class_keys to restrict the returned targets.
            If provided, counts/instances will only include those classes, but the image
            sampling is still dataset-level.

        min_total_count / max_total_count:
            Optional *sample-level pruning* based on the total number of annotations in an image,
            across ALL classes.

            Concretely, for each image, we look up `image_total_counts.total_count`. The image is
            included only if:

                min_total_count <= total_count <= max_total_count

            If a bound is None, it is not applied.

            Examples:
            - min_total_count=10: drop extremely sparse images (few annotations overall)
            - max_total_count=2000: drop extremely crowded images overall

        crowd_reviewed_only:
            If True, keep only images with crowd review_status == "reviewed"
            according to `image_review_stats`. This is intended for crowd-annotated datasets.
            Non-crowd datasets use review_status="na", so crowd_reviewed_only=True will usually
            return an empty dataset for those datasets.

        min_annotators / max_annotators:
            Optional sample-level pruning based on `image_review_stats.num_annotators`
            (number of annotator entries for the image). Primarily meaningful for crowd datasets.
            Defaults to 0 for non-crowd datasets.

        min_point_votes:
            Optional sample-level pruning based on `image_review_stats.num_point_votes`
            (number of annotator entries that contain at least one valid point set).
            Primarily meaningful for crowd datasets. Defaults to 0 for non-crowd datasets.

        load_images:
            If True, `__getitem__` loads and returns a PIL.Image. If False, returns the image path.

        ignore_policy:
            If False, applies policy.allowed_splits (and any future dataset-level restrictions).
            Set True to override.

        on_missing_split:
            Behavior when no samples satisfy the split query.
            Options: 'empty' (default), 'warn', or 'raise'. If 'empty', returns a zero-length dataset.

        natural_sort:
            If True, apply human-friendly natural sorting (e.g., 2.jpg < 10.jpg) by filename
            after querying the DB. Default False uses lexicographic path order. Intended for
            convenience/debugging only.
        """

        if dataset not in set(self.available_datasets()):
            raise ValueError(
                f"Unknown dataset {dataset!r}. Available: {self.available_datasets()}"
            )

        if splits is not None and split is not None:
            raise ValueError("Provide either split=... or splits={...}, not both.")

        # normalize splits
        if split is not None:
            splits_norm = {normalize_split(split)}
        else:
            splits_norm = normalize_splits(splits)

        # apply policy.allowed_splits (normalize policy too, if it comes from user)
        if not ignore_policy and self.policy.allowed_splits is not None:
            pol = normalize_splits(
                self.policy.allowed_splits
            )  # in case policy has "Train"
            if splits_norm is None:
                splits_norm = set(pol)
            else:
                splits_norm = set(splits_norm).intersection(pol)

        # handle missing splits for the dataset
        if splits_norm is not None:
            available = _available_splits_for_dataset(self.index_path, dataset)
            missing = sorted(set(splits_norm) - set(available))
            if missing:
                msg = (
                    f"Requested splits {sorted(splits_norm)} include missing {missing} for dataset={dataset!r}. "
                    f"Available splits: {sorted(available)}"
                )
                if on_missing_split == "raise":
                    raise ValueError(msg)
                if on_missing_split == "warn":
                    warnings.warn(msg)
                # Drop missing splits.
                if on_missing_split in ("empty", "warn"):
                    splits_norm = set(splits_norm).intersection(available)

                    # If nothing left, dataset will be empty (fine)
                    # (Leave as empty set rather than None to avoid "all splits")
                    if len(splits_norm) == 0:
                        splits_norm = set()

        return CountingImageDataset(
            index_path=self.index_path,
            dataset=dataset,
            splits=splits_norm if splits_norm is not None else None,
            class_keys=class_keys,
            load_images=load_images,
            min_total_count=min_total_count,
            max_total_count=max_total_count,
            crowd_reviewed_only=crowd_reviewed_only,
            min_annotators=min_annotators,
            max_annotators=max_annotators,
            min_point_votes=min_point_votes,
            natural_sort=natural_sort,
        )
