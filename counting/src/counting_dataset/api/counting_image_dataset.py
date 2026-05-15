from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from natsort import natsorted

from PIL import Image


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Per-connection read tuning (safe on NFS and local disks).
    # cache_size: 64 MB in-process page cache.
    # temp_store: keep SQLite's internal temp tables in RAM.
    conn.execute("PRAGMA cache_size=-65536;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


class CountingImageDataset:
    """
    Iterable dataset where each item is an image, and targets include annotations
    for ALL classes present (or optionally restricted to a subset of class_keys).

    Useful for multi-class experiments and per-image statistics.

    preload_annotations (default True): fetch ALL annotation rows for the
    selected images in two bulk queries at construction time (one for counts,
    one for annotations), then serve them from an in-memory dict on every
    __getitem__.  This eliminates two SQLite round-trips per sample.

    Both bulk queries use subqueries to identify the relevant images — no
    IN(list) of IDs, so there is no SQLite variable-count limit, and
    concurrent readers on any filesystem are unaffected.
    """

    def __init__(
        self,
        *,
        index_path: Path,
        dataset: str,
        splits: Optional[Set[str]] = None,
        class_keys: Optional[Set[str]] = None,
        load_images: bool = True,
        # sample-level pruning:
        min_total_count: Optional[int] = None,
        max_total_count: Optional[int] = None,
        # crowd filters:
        crowd_reviewed_only: bool = False,
        min_annotators: Optional[int] = None,
        max_annotators: Optional[int] = None,
        min_point_votes: Optional[int] = None,
        # metadata filters (JSON_EXTRACT on images.meta_json):
        meta_filter: Optional[Dict[str, Any]] = None,
        # yield order:
        natural_sort: Optional[bool] = False,
        # performance:
        preload_annotations: bool = True,
    ):
        self.index_path = Path(index_path)
        self.dataset = dataset
        self.splits = splits
        self.class_keys = class_keys
        self.load_images = load_images
        self.min_total_count = min_total_count
        self.max_total_count = max_total_count
        self.crowd_reviewed_only = crowd_reviewed_only
        self.min_annotators = min_annotators
        self.max_annotators = max_annotators
        self.min_point_votes = min_point_votes
        self.meta_filter = meta_filter
        self.natural_sort = natural_sort
        self.preload_annotations = preload_annotations

        self._image_rows = self._fetch_image_rows()
        if self.natural_sort:
            self._image_rows = natsorted(
                self._image_rows, key=lambda r: Path(r["path"]).name
            )

        # _ann_cache: image_id -> (counts, instances, aux)
        self._ann_cache: Optional[Dict[str, Tuple[
            Dict[str, int],
            Dict[str, List[Dict[str, Any]]],
            Dict[str, Dict[str, List[Dict[str, Any]]]],
        ]]] = None
        if self.preload_annotations and self._image_rows:
            self._ann_cache = self._preload_all_annotations()

    # ------------------------------------------------------------------
    # Image filter helpers shared by _fetch_image_rows and
    # _preload_all_annotations (keeps the two in sync).
    # ------------------------------------------------------------------

    def _image_filter_sql_and_params(self) -> Tuple[str, List[Any]]:
        """
        Return (WHERE-clause fragment, params) that selects the images
        belonging to this dataset object.

        The fragment assumes the query already has:
          FROM images i
          LEFT JOIN image_total_counts itc ON itc.image_id = i.image_id
          LEFT JOIN image_review_stats irs ON irs.image_id = i.image_id
        and starts with "i.dataset = ?".
        """
        parts = ["i.dataset = ?"]
        params: List[Any] = [self.dataset]

        if self.splits is not None:
            ph = ",".join(["?"] * len(self.splits))
            parts.append(f"i.split IN ({ph})")
            params.extend(sorted(self.splits))

        if self.min_total_count is not None:
            parts.append("COALESCE(itc.total_count, 0) >= ?")
            params.append(int(self.min_total_count))

        if self.max_total_count is not None:
            parts.append("COALESCE(itc.total_count, 0) <= ?")
            params.append(int(self.max_total_count))

        if self.crowd_reviewed_only:
            parts.append("COALESCE(irs.review_status, 'na') = 'reviewed'")

        if self.min_annotators is not None:
            parts.append("COALESCE(irs.num_annotators, 0) >= ?")
            params.append(int(self.min_annotators))

        if self.max_annotators is not None:
            parts.append("COALESCE(irs.num_annotators, 0) <= ?")
            params.append(int(self.max_annotators))

        if self.min_point_votes is not None:
            parts.append("COALESCE(irs.num_point_votes, 0) >= ?")
            params.append(int(self.min_point_votes))

        if self.meta_filter:
            for key in sorted(self.meta_filter):
                parts.append(f"JSON_EXTRACT(i.meta_json, '$.{key}') = ?")
                params.append(str(self.meta_filter[key]))

        return " AND ".join(parts), params

    def _fetch_image_rows(self) -> List[sqlite3.Row]:
        """
        Select images from a dataset (+ optional split restriction),
        optionally prune by total annotation count (image_total_counts).
        """
        where, params = self._image_filter_sql_and_params()
        sql = f"""
        SELECT i.image_id, i.path, i.width, i.height, i.split,
            COALESCE(itc.total_count, 0) AS total_count,
            COALESCE(irs.review_status, 'na') AS review_status,
            COALESCE(irs.num_annotators, 0) AS num_annotators,
            COALESCE(irs.num_point_votes, 0) AS num_point_votes
        FROM images i
        LEFT JOIN image_total_counts itc ON itc.image_id = i.image_id
        LEFT JOIN image_review_stats irs ON irs.image_id = i.image_id
        WHERE {where}
        ORDER BY i.path
        """
        with _connect(self.index_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return rows

    def _preload_all_annotations(self) -> Dict[str, Tuple[
        Dict[str, int],
        Dict[str, List[Dict[str, Any]]],
        Dict[str, Dict[str, List[Dict[str, Any]]]],
    ]]:
        """
        Fetch all counts and annotations for the selected images in two bulk
        queries (using subqueries, not IN(list)), then assemble per-image caches.

        Returns: {image_id: (counts, instances, aux)}
          counts:    {class_key: count}
          instances: {class_key: [ann, ...]}   for role == "instance"
          aux:       {role: {class_key: [ann, ...]}} for role != "instance"
        """
        where, params = self._image_filter_sql_and_params()
        image_subquery = f"""
            SELECT i.image_id
            FROM images i
            LEFT JOIN image_total_counts itc ON itc.image_id = i.image_id
            LEFT JOIN image_review_stats irs ON irs.image_id = i.image_id
            WHERE {where}
        """

        # --- counts ---
        counts_sql = f"""
        SELECT image_id, class_key, count
        FROM image_class_counts
        WHERE image_id IN ({image_subquery})
        """
        with _connect(self.index_path) as conn:
            count_rows = conn.execute(counts_sql, params).fetchall()

        image_ids = {r["image_id"] for r in self._image_rows}
        counts_map: Dict[str, Dict[str, int]] = {iid: {} for iid in image_ids}
        for r in count_rows:
            ck = r["class_key"]
            if self.class_keys is not None and ck not in self.class_keys:
                continue
            counts_map[r["image_id"]][ck] = int(r["count"])

        # --- annotations ---
        ann_sql = f"""
        SELECT a.ann_id, a.image_id, a.class_key, a.ann_type, a.source,
               a.instance_index, a.geometry_json, a.meta_json, a.role
        FROM annotations a
        WHERE a.image_id IN ({image_subquery})
        ORDER BY a.image_id ASC, a.role ASC, a.class_key ASC, a.instance_index ASC, a.ann_id ASC
        """
        with _connect(self.index_path) as conn:
            ann_rows = conn.execute(ann_sql, params).fetchall()

        instances_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {iid: {} for iid in image_ids}
        aux_map: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = {iid: {} for iid in image_ids}

        for r in ann_rows:
            ck = r["class_key"]
            if self.class_keys is not None and ck not in self.class_keys:
                continue
            iid = r["image_id"]
            role = r["role"] or "instance"
            ann = {
                "ann_id": r["ann_id"],
                "ann_type": r["ann_type"],
                "source": r["source"],
                "instance_index": r["instance_index"],
                "geometry": json.loads(r["geometry_json"]),
                "meta": json.loads(r["meta_json"]),
                "role": role,
            }
            if role == "instance":
                instances_map[iid].setdefault(ck, []).append(ann)
            else:
                aux_map[iid].setdefault(role, {}).setdefault(ck, []).append(ann)

        return {
            iid: (counts_map[iid], instances_map[iid], aux_map[iid])
            for iid in image_ids
        }

    def __len__(self) -> int:
        return len(self._image_rows)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        row = self._image_rows[idx]
        image_id = row["image_id"]
        path = row["path"]

        if self.load_images:
            img = Image.open(path).convert("RGB")
        else:
            img = path

        target = self._build_target(
            image_id=image_id,
            total_count=int(row["total_count"]),
            width=row["width"],
            height=row["height"],
        )
        target["review_status"] = row["review_status"]
        target["num_annotators"] = int(row["num_annotators"])
        target["num_point_votes"] = int(row["num_point_votes"])
        return img, target

    def _build_target(
        self, *, image_id: str, total_count: int, width: Optional[int], height: Optional[int]
    ) -> Dict[str, Any]:
        if self._ann_cache is not None:
            counts, instances, aux = self._ann_cache[image_id]
        else:
            counts = self._fetch_counts(image_id)
            instances, aux = self._fetch_instances_and_aux(image_id)

        return {
            "image_id": image_id,
            "dataset": self.dataset,
            "width": width,
            "height": height,
            "total_count": total_count,
            "counts": counts,
            "instances": instances,  # role == "instance"
            "aux": aux,  # role != "instance" grouped by role
        }

    def _fetch_counts(self, image_id: str) -> Dict[str, int]:
        sql = """
        SELECT class_key, count
        FROM image_class_counts
        WHERE image_id = ?
        """
        with _connect(self.index_path) as conn:
            rows = conn.execute(sql, [image_id]).fetchall()

        out: Dict[str, int] = {}
        for r in rows:
            ck = r["class_key"]
            if self.class_keys is not None and ck not in self.class_keys:
                continue
            out[ck] = int(r["count"])
        return out

    def _fetch_instances_and_aux(
        self, image_id: str
    ) -> Tuple[
        Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, List[Dict[str, Any]]]]
    ]:
        """
        Returns:
        instances: {class_key: [ann, ...]} for role == "instance"
        aux: {role: {class_key: [ann, ...]}} for role != "instance"
        """
        sql = """
        SELECT ann_id, class_key, ann_type, source, instance_index,
            geometry_json, meta_json, role
        FROM annotations
        WHERE image_id = ?
        ORDER BY role ASC, class_key ASC, instance_index ASC, ann_id ASC
        """
        with _connect(self.index_path) as conn:
            rows = conn.execute(sql, [image_id]).fetchall()

        instances: Dict[str, List[Dict[str, Any]]] = {}
        aux: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for r in rows:
            ck = r["class_key"]
            if self.class_keys is not None and ck not in self.class_keys:
                continue

            role = r["role"] or "instance"
            ann = {
                "ann_id": r["ann_id"],
                "ann_type": r["ann_type"],
                "source": r["source"],
                "instance_index": r["instance_index"],
                "geometry": json.loads(r["geometry_json"]),
                "meta": json.loads(r["meta_json"]),
                "role": role,
            }

            if role == "instance":
                instances.setdefault(ck, []).append(ann)
            else:
                aux.setdefault(role, {}).setdefault(ck, []).append(ann)

        return instances, aux
