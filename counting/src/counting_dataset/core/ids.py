from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .schema import (
    AnnId,
    AnnType,
    ClassKey,
    Geometry,
    ImageId,
    SourceType,
)


# ----------------------------
# Path normalization
# ----------------------------


def normalize_relpath(p: str) -> str:
    """
    Normalize a relative path to a stable, cross-platform representation.
    - always uses forward slashes
    - removes redundant separators / '.' segments
    - strips leading './'
    - disallows absolute paths
    """
    if not isinstance(p, str) or not p:
        raise ValueError("original_relpath must be a non-empty string")

    # Convert backslashes and normalize
    p2 = p.replace("\\", "/")

    # Strip leading "./"
    while p2.startswith("./"):
        p2 = p2[2:]

    # Disallow absolute paths
    if p2.startswith("/") or (len(p2) >= 2 and p2[1] == ":" and p2[0].isalpha()):
        raise ValueError(f"Expected a relative path, got: {p!r}")

    # Collapse ".." etc; Path() will use OS separators, so re-normalize to '/'
    norm = os.path.normpath(p2).replace("\\", "/")

    # normpath can return ".", treat that as invalid (should point to a file)
    if norm in (".", ""):
        raise ValueError(f"Invalid relative path: {p!r}")

    return norm


def safe_join(root: Path, relpath: str) -> Path:
    """
    Join root + relpath safely (prevents path traversal outside root).
    """
    rel = normalize_relpath(relpath)
    out = (root / rel).resolve()
    root_resolved = root.resolve()

    # Ensure out is within root
    try:
        out.relative_to(root_resolved)
    except ValueError as e:
        raise ValueError(
            f"Path traversal detected: root={root} relpath={relpath}"
        ) from e

    return out


# ----------------------------
# Hash helpers
# ----------------------------


def _blake2b_hex(data: bytes, digest_size: int = 16) -> str:
    return hashlib.blake2b(data, digest_size=digest_size).hexdigest()


def make_image_id(dataset: str, original_relpath: str) -> ImageId:
    """
    Stable deterministic image ID from dataset name + normalized original relative path.
    """
    if not dataset:
        raise ValueError("dataset must be a non-empty string")
    rel = normalize_relpath(original_relpath)

    payload = f"{dataset}\n{rel}".encode("utf-8")
    return f"img_{_blake2b_hex(payload)}"


def _geometry_to_canonical(geometry: Geometry) -> Any:
    """
    Convert geometry into a canonical JSON-serializable structure.
    This must be stable across runs and Python versions.
    """
    # dataclasses (Point/HBB/OBB/Polygon/MaskRef) are frozen in schema.py
    if is_dataclass(geometry):
        d = asdict(geometry)
        # Normalize tuples/lists for stability where needed
        # (asdict turns tuples into lists; that's fine as long as we keep it consistent)
        return {"__type__": type(geometry).__name__, **d}

    # Non-dataclass geometry types are handled here.
    raise TypeError(f"Unsupported geometry type: {type(geometry)}")


def make_ann_id(
    image_id: ImageId,
    class_key: ClassKey,
    ann_type: AnnType,
    geometry: Geometry,
    source: SourceType,
    *,
    instance_index: Optional[int] = None,
    salt: str = "",
) -> AnnId:
    """
    Stable deterministic annotation ID based on content.

    Uniqueness: If two annotations in the same image happen to have identical geometry
    (possible with crowd labels or rounding), include a deterministic `instance_index`
    (or dataset-native id via `salt`) to disambiguate.

    Recommended adapter rule:
      - if dataset has native annotation id => salt = native_id
      - else => instance_index = per-image (or per-image+class) ordinal starting at 0
    """
    if not image_id or not class_key:
        raise ValueError("image_id and class_key must be non-empty")
    if instance_index is not None and instance_index < 0:
        raise ValueError("instance_index must be >= 0")

    payload_obj: Dict[str, Any] = {
        "image_id": image_id,
        "class_key": class_key,
        "ann_type": str(ann_type),
        "source": str(source),
        "geometry": _geometry_to_canonical(geometry),
        # Disambiguators:
        "instance_index": instance_index,
        "salt": salt,
    }

    payload = json.dumps(payload_obj, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return f"ann_{_blake2b_hex(payload)}"


def file_sha1(path: Path, chunk_size: int = 1 << 20) -> str:
    """
    Compute SHA1 for a file (debugging/dedup/provenance).
    Not required for IDs, but useful to store in Provenance.
    """
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
