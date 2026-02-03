from __future__ import annotations

from typing import Iterable, Optional, Set


_CANON = {
    "train": {"train", "training", "tr"},
    "val": {"val", "valid", "validation", "dev"},
    "test": {"test", "testing", "te"},
}


def normalize_split(split: Optional[str]) -> Optional[str]:
    """
    Normalize common split spellings/capitalization to {"train","val","test"}.

    Returns:
      - canonical split string if recognized
      - None if input is None
      - raises ValueError if unrecognized
    """
    if split is None:
        return None
    s = split.strip().lower()
    for canon, alts in _CANON.items():
        if s == canon or s in alts:
            return canon
    raise ValueError(
        f"Unrecognized split {split!r}. Expected one of: train/val/test (case-insensitive)."
    )


def normalize_splits(splits: Optional[Iterable[str]]) -> Optional[Set[str]]:
    if splits is None:
        return None
    out: Set[str] = set()
    for s in splits:
        out.add(normalize_split(s))
    return out
