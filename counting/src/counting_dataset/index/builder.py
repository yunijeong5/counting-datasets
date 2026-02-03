from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from counting_dataset.adapters.base import AdapterContext
from counting_dataset.core.schema import (
    ClassRecord,
    ImageRecord,
    InstanceAnnotationRecord,
)
from counting_dataset.core.splits import normalize_split

from .schema_sql import SCHEMA_SQL


def _to_json(obj) -> str:
    """Stable JSON serialization for dataclasses or dicts."""
    if is_dataclass(obj):
        obj = asdict(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    return conn


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_str(x: Any, default: str = "na") -> str:
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default


def _extract_crowd_stats(meta_json: str) -> Dict[str, Any]:
    """
    Extract crowd stats from images.meta_json, if present.
    Expected meta schema: {"crowd": {...}} as produced by PenguinAdapter.
    """
    try:
        meta = json.loads(meta_json) if meta_json else {}
    except Exception:
        meta = {}

    crowd = meta.get("crowd")
    if not isinstance(crowd, dict):
        return {
            "review_status": "na",
            "num_annotators": 0,
            "num_empty_votes": 0,
            "num_point_votes": 0,
            "num_points_total": 0,
        }

    return {
        "review_status": _safe_str(crowd.get("review_status"), default="na"),
        "num_annotators": _safe_int(crowd.get("num_annotator_entries"), default=0),
        "num_empty_votes": _safe_int(crowd.get("num_empty_votes"), default=0),
        "num_point_votes": _safe_int(crowd.get("num_point_votes"), default=0),
        "num_points_total": _safe_int(crowd.get("num_points_total"), default=0),
    }


def _chunked(it, n: int):
    it = iter(it)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk


def _adapter_bar(total: int, enabled: bool):
    if not enabled:
        return None
    return tqdm(
        total=total,
        desc="IndexBuilder",
        unit="step",
        dynamic_ncols=True,
        file=sys.stdout,
        disable=not sys.stdout.isatty(),
    )


def _iter_with_inner_progress(
    iterable,
    *,
    show_progress: bool,
    desc: str,
    print_every: int = 50_000,
):
    """
    Yields items from iterable while providing "heartbeat" progress:

    - If interactive (stdout is a TTY): show an inner tqdm counter (unknown total).
    - Otherwise: print periodic status lines every `print_every` items.
    """
    if not show_progress:
        yield from iterable
        return

    interactive = sys.stdout.isatty()

    if interactive:
        inner = tqdm(
            iterable,
            desc=desc,
            unit="record",
            dynamic_ncols=True,
            file=sys.stdout,
            leave=False,
            mininterval=0.2,
        )
        yield from inner
        return

    # Non-interactive / captured output: periodic prints
    n = 0
    for x in iterable:
        n += 1
        if print_every and (n % print_every == 0):
            print(f"  ... {desc}: processed {n:,} records", flush=True)
        yield x


class IndexBuilder:
    """
    Builds counting/data/index.sqlite from one or more dataset adapters.

    Robustness note (important for HPC / network filesystems):
      - SQLite can fail on NFS/Lustre-like mounts due to journaling/locking semantics.
      - To be robust, the builder can create the DB in a local temp directory
        (TMPDIR or /tmp) and then atomically install it into out_root/index.sqlite.
    """

    def __init__(
        self,
        raw_root: Path,
        out_root: Path,
        *,
        build_in_tmp: bool = True,
        tmp_root: Optional[Path] = None,
    ):
        self.raw_root = Path(raw_root)
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)

        self.build_in_tmp = bool(build_in_tmp)
        self.tmp_root = Path(tmp_root) if tmp_root is not None else None

    @property
    def index_path(self) -> Path:
        return self.out_root / "index.sqlite"

    # -----------------------
    # Temp build + install helpers
    # -----------------------

    def _choose_tmp_base(self) -> Path:
        """
        Choose a local temp base directory.

        Priority:
          1) user-provided tmp_root
          2) $TMPDIR (cluster node-local scratch)
          3) /tmp
        """
        if self.tmp_root is not None:
            return self.tmp_root

        env_tmp = os.environ.get("TMPDIR")
        if env_tmp:
            return Path(env_tmp)

        return Path("/tmp")

    @staticmethod
    def _integrity_check(db_path: Path) -> None:
        """Raise RuntimeError if PRAGMA integrity_check is not 'ok'."""
        conn = _connect(db_path)
        try:
            row = conn.execute("PRAGMA integrity_check;").fetchone()
            msg = row[0] if row else None
            if msg != "ok":
                raise RuntimeError(
                    f"SQLite integrity_check failed for {db_path}: {msg}"
                )
        finally:
            conn.close()

    def _atomic_install(self, src_db: Path) -> Path:
        """
        Copy src_db to out_root/index.sqlite atomically (best-effort).
        Works even when src and dst are on different filesystems.
        """
        dst = self.index_path
        tmp_dst = dst.with_suffix(dst.suffix + ".tmp")

        self.out_root.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src_db, tmp_dst)

        # fsync the copied file
        with open(tmp_dst, "rb") as f:
            os.fsync(f.fileno())

        # Atomic replace
        os.replace(tmp_dst, dst)

        # fsync the directory entry (best-effort)
        try:
            dir_fd = os.open(str(self.out_root), os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            pass

        return dst

    # -----------------------
    # DB init/build
    # -----------------------

    def init_db_at(self, db_path: Path, overwrite: bool = False) -> None:
        """Initialize schema at a specific db_path."""
        if overwrite and db_path.exists():
            db_path.unlink()

        conn = _connect(db_path)
        try:
            conn.executescript(SCHEMA_SQL)
            conn.commit()
        finally:
            conn.close()

    def init_db(self, overwrite: bool = False) -> None:
        """
        Initialize schema at the final index_path (compatibility method).
        When build_in_tmp=True, initialization happens inside build() on the temp path.
        """
        self.init_db_at(self.index_path, overwrite=overwrite)

    def build(
        self,
        adapters: Sequence[object],
        *,
        overwrite: bool = False,
        compute_counts: bool = True,
        show_progress: bool = False,
        chunk_size: int = 5_000,
        print_every: int = 50_000,
    ) -> Path:
        """
        Build the SQLite index from dataset adapters.

        - show_progress: enable adapter-level tqdm + inner per-record heartbeat
        - chunk_size: batch size for INSERT executemany (memory + speed tuning)
        - print_every: for non-interactive logs, print every N records within a step
        """
        ctx = AdapterContext(raw_root=self.raw_root)

        if not self.build_in_tmp:
            self.init_db(overwrite=overwrite)
            conn = _connect(self.index_path)
            try:
                self._insert_all(
                    conn,
                    ctx,
                    adapters,
                    show_progress=show_progress,
                    chunk_size=chunk_size,
                    print_every=print_every,
                )
                if compute_counts:
                    self._compute_image_class_counts(conn)
                    self._compute_image_total_counts(conn)
                    self._compute_image_review_stats(conn)
                conn.commit()
            finally:
                conn.close()
            return self.index_path

        tmp_base = self._choose_tmp_base()
        tmp_dir = Path(tempfile.mkdtemp(prefix="counting_index_", dir=str(tmp_base)))
        tmp_db = tmp_dir / "index.sqlite"

        try:
            self.init_db_at(tmp_db, overwrite=True)

            conn = _connect(tmp_db)
            try:
                self._insert_all(
                    conn,
                    ctx,
                    adapters,
                    show_progress=show_progress,
                    chunk_size=chunk_size,
                    print_every=print_every,
                )
                if compute_counts:
                    self._compute_image_class_counts(conn)
                    self._compute_image_total_counts(conn)
                    self._compute_image_review_stats(conn)
                conn.commit()
            finally:
                conn.close()

            self._integrity_check(tmp_db)

            if overwrite and self.index_path.exists():
                try:
                    self.index_path.unlink()
                except Exception:
                    pass

            final_path = self._atomic_install(tmp_db)
            self._integrity_check(final_path)
            return final_path
        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass

    # -----------------------
    # Insertions
    # -----------------------

    def _insert_all(
        self,
        conn: sqlite3.Connection,
        ctx: AdapterContext,
        adapters: Sequence[object],
        *,
        show_progress: bool = False,
        chunk_size: int = 5_000,
        print_every: int = 50_000,
    ) -> None:
        """
        Insert classes/images/annotations for each adapter.

        Progress behavior:
          - Outer bar: 3 * len(adapters) steps (classes/images/annotations).
          - Inner heartbeat: per-record tqdm counter if interactive; otherwise periodic prints.
        """

        def ds_name(ad) -> str:
            return getattr(ad, "dataset", ad.__class__.__name__)

        bar = _adapter_bar(total=len(adapters) * 3, enabled=show_progress)

        def tick(msg: str) -> None:
            if bar is not None:
                bar.set_postfix_str(msg)
                bar.update(1)
            elif show_progress:
                print(msg, flush=True)

        def insert_stream(kind: str, ad, iterator, insert_fn) -> None:
            desc = f"[{ds_name(ad)}] {kind}"
            it = _iter_with_inner_progress(
                iterator,
                show_progress=show_progress,
                desc=desc,
                print_every=print_every,
            )
            for chunk in _chunked(it, chunk_size):
                insert_fn(conn, chunk)

        # 1) classes
        for ad in adapters:
            tick(f"{ds_name(ad)}: classes")
            insert_stream("classes", ad, ad.iter_classes(ctx), self._insert_classes)

        # 2) images
        for ad in adapters:
            tick(f"{ds_name(ad)}: images")
            insert_stream("images", ad, ad.iter_images(ctx), self._insert_images)

        # 3) annotations
        for ad in adapters:
            tick(f"{ds_name(ad)}: annotations")
            insert_stream(
                "annotations", ad, ad.iter_annotations(ctx), self._insert_annotations
            )

        if bar is not None:
            bar.close()

    def _insert_classes(
        self, conn: sqlite3.Connection, classes: List[ClassRecord]
    ) -> None:
        sql = """
        INSERT OR REPLACE INTO classes (class_key, dataset, name, meta_json)
        VALUES (?, ?, ?, ?)
        """
        rows = [(c.class_key, c.dataset, c.name, _to_json(c.meta)) for c in classes]
        if rows:
            conn.executemany(sql, rows)

    def _insert_images(
        self, conn: sqlite3.Connection, images: List[ImageRecord]
    ) -> None:
        sql = """
        INSERT OR REPLACE INTO images (
          image_id, dataset, split, path, width, height,
          original_relpath, original_filename, original_id, sha1, size_bytes,
          meta_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = []
        for im in images:
            prov = im.provenance
            rows.append(
                (
                    im.image_id,
                    prov.dataset,
                    normalize_split(im.split.value),
                    im.path,
                    int(im.width),
                    int(im.height),
                    prov.original_relpath,
                    prov.original_filename,
                    prov.original_id,
                    prov.sha1,
                    prov.size_bytes,
                    _to_json(im.meta),
                )
            )
        if rows:
            conn.executemany(sql, rows)

    def _insert_annotations(
        self, conn: sqlite3.Connection, anns: List[InstanceAnnotationRecord]
    ) -> None:
        sql = """
        INSERT OR REPLACE INTO annotations (
          ann_id, image_id, class_key,
          ann_type, source, instance_index,
          geometry_json, score, meta_json, role
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = []
        for a in anns:
            rows.append(
                (
                    a.ann_id,
                    a.image_id,
                    a.class_key,
                    a.ann_type.value,
                    a.source.value,
                    a.instance_index,
                    _to_json(a.geometry),
                    a.score,
                    _to_json(a.meta),
                    a.role,
                )
            )
        if rows:
            conn.executemany(sql, rows)

    # -----------------------
    # Aggregates
    # -----------------------

    def _compute_image_class_counts(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM image_class_counts;")
        conn.execute(
            """
            INSERT INTO image_class_counts (image_id, class_key, count)
            SELECT image_id, class_key, COUNT(*)
            FROM annotations
            WHERE role = 'instance'
            GROUP BY image_id, class_key
            """
        )

    def _compute_image_total_counts(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM image_total_counts;")
        conn.execute(
            """
            INSERT INTO image_total_counts (image_id, total_count)
            SELECT image_id, COUNT(*)
            FROM annotations
            WHERE role = 'instance'
            GROUP BY image_id
            """
        )

    def _compute_image_review_stats(self, conn: sqlite3.Connection) -> None:
        conn.execute("DELETE FROM image_review_stats;")

        rows = conn.execute("SELECT image_id, meta_json FROM images").fetchall()

        to_insert = []
        for image_id, meta_json in rows:
            stats = _extract_crowd_stats(meta_json or "")
            to_insert.append(
                (
                    image_id,
                    stats["review_status"],
                    stats["num_annotators"],
                    stats["num_empty_votes"],
                    stats["num_point_votes"],
                    stats["num_points_total"],
                )
            )

        conn.executemany(
            """
            INSERT INTO image_review_stats
            (image_id, review_status, num_annotators, num_empty_votes, num_point_votes, num_points_total)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            to_insert,
        )
