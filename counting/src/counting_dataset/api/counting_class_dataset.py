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
    # Read-performance tuning: 64 MB page cache, memory-mapped I/O, temp tables in RAM
    conn.execute("PRAGMA cache_size=-65536;")
    conn.execute("PRAGMA mmap_size=268435456;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


class CountingClassDataset:
    """
    Iterable dataset for a single class_key.
    Supports sample-level pruning based on per-image count of that class.

    preload_annotations (default True): fetch ALL annotation rows for the
    selected images in one bulk query at construction time, then serve them
    from an in-memory dict on every __getitem__.  This eliminates one SQLite
    round-trip per sample, which is the dominant cost when load_images=False.
    Set to False only when memory is very tight (many thousands of images with
    dense annotations).
    """

    def __init__(
        self,
        *,
        index_path: Path,
        class_key: str,
        splits: Optional[Set[str]],
        load_images: bool,
        target_format: str,
        # sample-level pruning for this class:
        min_count: Optional[int] = None,
        max_count: Optional[int] = None,
        # metadata filters (JSON_EXTRACT on images.meta_json):
        meta_filter: Optional[Dict[str, Any]] = None,
        # yield order:
        natural_sort: Optional[bool] = False,
        # performance:
        preload_annotations: bool = True,
    ):
        self.index_path = Path(index_path)
        self.class_key = class_key
        self.splits = splits
        self.load_images = load_images
        self.target_format = target_format
        self.min_count = min_count
        self.max_count = max_count
        self.meta_filter = meta_filter
        self.natural_sort = natural_sort
        self.preload_annotations = preload_annotations

        self._image_rows = self._fetch_image_rows()
        if self.natural_sort:
            self._image_rows = natsorted(
                self._image_rows, key=lambda r: Path(r["path"]).name
            )

        # Bulk-load annotations for all images in one query.
        # _ann_cache: image_id -> list of raw annotation rows (as dicts)
        self._ann_cache: Optional[Dict[str, List[Dict[str, Any]]]] = None
        if self.preload_annotations and self._image_rows:
            self._ann_cache = self._preload_all_annotations()

    def _fetch_image_rows(self) -> List[sqlite3.Row]:
        sql = """
        SELECT i.image_id, i.path, i.width, i.height, i.split,
            icc.count AS class_count,
            COALESCE(irs.review_status, 'na') AS review_status,
            COALESCE(irs.num_annotators, 0) AS num_annotators,
            COALESCE(irs.num_point_votes, 0) AS num_point_votes
        FROM images i
        JOIN image_class_counts icc
        ON icc.image_id = i.image_id
        LEFT JOIN image_review_stats irs
        ON irs.image_id = i.image_id
        WHERE icc.class_key = ?
        """
        params: List[Any] = [self.class_key]

        if self.splits is not None:
            ph = ",".join(["?"] * len(self.splits))
            sql += f" AND i.split IN ({ph})"
            params.extend(sorted(self.splits))

        if self.min_count is not None:
            sql += " AND icc.count >= ?"
            params.append(int(self.min_count))

        if self.max_count is not None:
            sql += " AND icc.count <= ?"
            params.append(int(self.max_count))

        if self.meta_filter:
            for key in sorted(self.meta_filter):
                sql += f" AND JSON_EXTRACT(i.meta_json, '$.{key}') = ?"
                params.append(str(self.meta_filter[key]))

        sql += " ORDER BY i.path"

        with _connect(self.index_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return rows

    def _preload_all_annotations(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch all annotations for the selected images in a single query.
        Returns a dict keyed by image_id.
        """
        image_ids = [r["image_id"] for r in self._image_rows]
        ph = ",".join(["?"] * len(image_ids))
        sql = f"""
        SELECT ann_id, image_id, ann_type, source, instance_index,
               geometry_json, score, meta_json, role
        FROM annotations
        WHERE image_id IN ({ph}) AND class_key = ?
        ORDER BY image_id ASC, role ASC, instance_index ASC, ann_id ASC
        """
        with _connect(self.index_path) as conn:
            rows = conn.execute(sql, image_ids + [self.class_key]).fetchall()

        cache: Dict[str, List[Dict[str, Any]]] = {r["image_id"]: [] for r in self._image_rows}
        for r in rows:
            cache[r["image_id"]].append(dict(r))
        return cache

    def __len__(self) -> int:
        return len(self._image_rows)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        row = self._image_rows[idx]
        image_id = row["image_id"]
        path = row["path"]
        class_count = int(row["class_count"])

        if self.load_images:
            img = Image.open(path).convert("RGB")
        else:
            img = path

        target = self._build_target(
            image_id=image_id,
            class_count=class_count,
            width=row["width"],
            height=row["height"],
            review_status=row["review_status"],
            num_annotators=int(row["num_annotators"]),
            num_point_votes=int(row["num_point_votes"]),
        )
        return img, target

    def _build_target(
        self,
        *,
        image_id: str,
        class_count: int,
        width: Optional[int],
        height: Optional[int],
        review_status: str,
        num_annotators: int,
        num_point_votes: int,
    ) -> Dict[str, Any]:
        if self._ann_cache is not None:
            raw_rows = self._ann_cache.get(image_id, [])
        else:
            sql = """
            SELECT ann_id, ann_type, source, instance_index, geometry_json, score, meta_json, role
            FROM annotations
            WHERE image_id = ? AND class_key = ?
            ORDER BY role ASC, instance_index ASC, ann_id ASC
            """
            with _connect(self.index_path) as conn:
                raw_rows = [dict(r) for r in conn.execute(sql, [image_id, self.class_key]).fetchall()]

        instances: List[Dict[str, Any]] = []
        aux: Dict[str, List[Dict[str, Any]]] = {}

        for r in raw_rows:
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
                instances.append(ann)
            else:
                aux.setdefault(role, []).append(ann)

        # Important: keep count from icc for speed and consistency.
        # But note: icc.count only tracks role='instance' by design, so it matches len(instances).
        return {
            "image_id": image_id,
            "class_key": self.class_key,
            "width": width,
            "height": height,
            "count": class_count,
            "instances": instances,  # role == "instance" only
            "aux": aux,  # role != "instance", class-local view
            "review_status": review_status,
            "num_annotators": num_annotators,
            "num_point_votes": num_point_votes,
        }
