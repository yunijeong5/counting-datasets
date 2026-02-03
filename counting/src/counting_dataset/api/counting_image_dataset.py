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
    return conn


class CountingImageDataset:
    """
    Iterable dataset where each item is an image, and targets include annotations
    for ALL classes present (or optionally restricted to a subset of class_keys).

    Useful for multi-class experiments and per-image statistics.
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
        # yield order:
        natural_sort: Optional[bool] = False,
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
        self.natural_sort = natural_sort

        self._image_rows = self._fetch_image_rows()
        if self.natural_sort:
            self._image_rows = natsorted(
                self._image_rows, key=lambda r: Path(r["path"]).name
            )

    def _fetch_image_rows(self) -> List[sqlite3.Row]:
        """
        Select images from a dataset (+ optional split restriction),
        optionally prune by total annotation count (image_total_counts).
        """
        sql = """
        SELECT i.image_id, i.path, i.width, i.height, i.split,
            COALESCE(itc.total_count, 0) AS total_count,
            COALESCE(irs.review_status, 'na') AS review_status,
            COALESCE(irs.num_annotators, 0) AS num_annotators,
            COALESCE(irs.num_point_votes, 0) AS num_point_votes
        FROM images i
        LEFT JOIN image_total_counts itc ON itc.image_id = i.image_id
        LEFT JOIN image_review_stats irs ON irs.image_id = i.image_id
        WHERE i.dataset = ?
        """
        params: List[Any] = [self.dataset]

        # Query filters
        if self.splits is not None:
            ph = ",".join(["?"] * len(self.splits))
            sql += f" AND i.split IN ({ph})"
            params.extend(sorted(self.splits))

        if self.min_total_count is not None:
            sql += " AND COALESCE(itc.total_count, 0) >= ?"
            params.append(int(self.min_total_count))

        if self.max_total_count is not None:
            sql += " AND COALESCE(itc.total_count, 0) <= ?"
            params.append(int(self.max_total_count))

        if self.crowd_reviewed_only:
            sql += " AND COALESCE(irs.review_status, 'na') = 'reviewed'"

        if self.min_annotators is not None:
            sql += " AND COALESCE(irs.num_annotators, 0) >= ?"
            params.append(int(self.min_annotators))

        if self.max_annotators is not None:
            sql += " AND COALESCE(irs.num_annotators, 0) <= ?"
            params.append(int(self.max_annotators))

        if self.min_point_votes is not None:
            sql += " AND COALESCE(irs.num_point_votes, 0) >= ?"
            params.append(int(self.min_point_votes))

        sql += " ORDER BY i.path"

        with _connect(self.index_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return rows

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
            image_id=image_id, total_count=int(row["total_count"])
        )
        target["review_status"] = row["review_status"]
        target["num_annotators"] = int(row["num_annotators"])
        target["num_point_votes"] = int(row["num_point_votes"])
        return img, target

    def _build_target(self, *, image_id: str, total_count: int) -> Dict[str, Any]:
        """
        Returns:
          - counts: {class_key: count}
          - instances: {class_key: [instance, ...]}
        Optionally restricted to self.class_keys.
        """
        counts = self._fetch_counts(image_id)
        instances, aux = self._fetch_instances_and_aux(image_id)

        return {
            "image_id": image_id,
            "dataset": self.dataset,
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
        params: List[Any] = [image_id]

        with _connect(self.index_path) as conn:
            rows = conn.execute(sql, params).fetchall()

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
            geometry_json, score, meta_json, role
        FROM annotations
        WHERE image_id = ?
        ORDER BY role ASC, class_key ASC, instance_index ASC, ann_id ASC
        """
        params: List[Any] = [image_id]

        with _connect(self.index_path) as conn:
            rows = conn.execute(sql, params).fetchall()

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
                "score": r["score"],
                "meta": json.loads(r["meta_json"]),
                "role": role,
            }

            if role == "instance":
                instances.setdefault(ck, []).append(ann)
            else:
                aux.setdefault(role, {}).setdefault(ck, []).append(ann)

        return instances, aux
