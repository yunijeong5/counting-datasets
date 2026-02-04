from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Literal, Optional, Tuple, Union


# ----------------------------
# Enums / basic types (and type alias)
# ----------------------------


class SplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    UNSPECIFIED = "unspecified"


class AnnType(str, Enum):
    POINT = "point"
    HBB = "hbb"  # axis-aligned horizontal bbox
    OBB = "obb"  # oriented bbox
    POLYGON = "polygon"
    MASK = "mask"


class SourceType(str, Enum):
    ORIGINAL = "original"  # from dataset authors
    HUMAN = "human"  # manually labeled by our annotators
    CROWDSOURCE = "crowd"  # crowd-sourced
    GENERATED = "generated"  # SAM/other auto labels


# Dataset-scoped class keys; primary class identity.
# Example: "fsc147/bird", "dota/plane", "malaria/red_blood_cell".
ClassKey = str

# Global, stable, deterministic image id. hash(dataset + original_relpath).
ImageId = str

# Optional stable annotation id (hash(image_id + geometry + class_key + source)).
AnnId = str


# ----------------------------
# Geometry payloads
# ----------------------------


@dataclass(frozen=True)
class Point:
    # coordinate: absolute pixels in image coordinate system
    x: float
    y: float


@dataclass(frozen=True)
class HBB:
    # axis-aligned bbox in xywh
    x: float
    y: float
    w: float
    h: float


@dataclass(frozen=True)
class OBB:
    # oriented bounding box as 4 corners (clockwise or counterclockwise)
    # (x1,y1,x2,y2,x3,y3,x4,y4)
    corners: Tuple[float, float, float, float, float, float, float, float]


@dataclass(frozen=True)
class Polygon:
    # flat list [x1,y1,x2,y2,...]
    points: Tuple[float, ...]


@dataclass(frozen=True)
class MaskRef:
    # reference to a stored mask (rle path, png path, etc.)
    kind: Literal["rle", "png", "npz", "other"]
    ref: str
    data: Optional[Dict[str, Any]] = None


Geometry = Union[Point, HBB, OBB, Polygon, MaskRef]


# ----------------------------
# Provenance / traceability
# ----------------------------


@dataclass(frozen=True)
class Provenance:
    dataset: str  # e.g. "dota"
    original_relpath: str  # relative to raw dataset root
    original_filename: str
    original_id: Optional[str] = None  # if dataset defines a stable id
    # Optional fingerprinting for traceability / dedup
    sha1: Optional[str] = None
    size_bytes: Optional[int] = None


# ----------------------------
# Core records
# ----------------------------


@dataclass
class ClassRecord:
    class_key: ClassKey  # "{dataset}/{slugified_class_name}"
    dataset: str  # duplicated for convenience
    name: str  # "bird", "elephant", "cell", ...
    # Optional: link dataset-specific taxonomy if needed later
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageRecord:
    image_id: ImageId
    path: str  # path to the image file (under <root>/raw/)
    width: int
    height: int
    split: SplitType
    provenance: Provenance

    # Cached derived values for fast queries (sparse: only present classes)
    counts: Dict[ClassKey, int] = field(default_factory=dict)

    # Optional per-image metadata (sensor, location, capture time, etc.)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InstanceAnnotationRecord:
    ann_id: AnnId  # primary key
    image_id: ImageId  # foreign key (-> ImageRecord)
    class_key: ClassKey  # foreign key (-> ClassRecord)
    ann_type: AnnType  # how to interpret geometry
    geometry: Geometry  # actual spacial information
    role: str = "instance"  # "instance" | "aux" | ...
    instance_index: Optional[int] = None  # use as salt if needed (when making ann_id)

    source: SourceType = (
        SourceType.ORIGINAL
    )  # who and what produced this instance record

    # Optional quality/provenance details (who labeled, model version, vote count, etc.)
    score: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)
