from dataclasses import dataclass
from typing import Optional, Set


@dataclass(frozen=True)
class FilterPolicy:
    """
    Dataset curation policy applied by CountingDatasetIndex.

    All fields are optional; if None, that constraint is not applied.
    """

    # Filter classes by the number of distinct images that contain at least one instance of that class.
    min_images_per_class: Optional[int] = None
    max_images_per_class: Optional[int] = None

    # Filters classes by the total number of annotated object instances across all images for the class.
    min_instances_per_class: Optional[int] = None
    max_instances_per_class: Optional[int] = None

    # If provided, only keep classes from these dataset keys (e.g., {"malaria", "penguin"}).
    allowed_datasets: Optional[Set[str]] = None

    # If provided, only keep classes that have at least one instance with ann_type in this set.
    # Example: {"point"} or {"hbb","obb"}
    allowed_ann_types: Optional[Set[str]] = None

    # If provided, only consider images from these splits when loading datasets.
    # Example: {"train","test"}
    allowed_splits: Optional[Set[str]] = None

    def class_allowed(
        self,
        *,
        dataset: str,
        num_images: int,
        num_instances: int,
        ann_types: Set[str],
    ) -> bool:
        if self.allowed_datasets is not None and dataset not in self.allowed_datasets:
            return False

        if (
            self.min_images_per_class is not None
            and num_images < self.min_images_per_class
        ):
            return False
        if (
            self.max_images_per_class is not None
            and num_images > self.max_images_per_class
        ):
            return False

        if (
            self.min_instances_per_class is not None
            and num_instances < self.min_instances_per_class
        ):
            return False
        if (
            self.max_instances_per_class is not None
            and num_instances > self.max_instances_per_class
        ):
            return False

        if self.allowed_ann_types is not None:
            if len(ann_types.intersection(self.allowed_ann_types)) == 0:
                return False

        return True
