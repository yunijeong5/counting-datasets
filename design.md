# Counting Dataset API: Design & Architecture

This document explains the **internal design, data model, and architectural choices** behind the Counting Dataset API. It is intended for users who want to understand how the system works beyond the surface-level API. If you are looking for installation instructions or a quickstart, see the main `README.md`.

Here is a simple diagram of the project structure:
```
Raw datasets
   ↓ (Adapters)
Canonical Records (Class / Image / Annotation)
   ↓ (IndexBuilder)
SQLite Index (base tables + derived tables)
   ↓ (Query-time filtering)
CountingDatasetIndex
   ↓
CountingImageDataset / CountingClassDataset
```

## 1. Project Overview

The Counting Dataset API is a **unified indexing and access layer for object counting datasets**.

Its primary goals are:

- To **normalize heterogeneous counting datasets**, including not only diverse annotation types (e.g., points, boxes, crowdsourced labels, exemplars), but also incompatible raw data formats.
  - In practice, the supported datasets use fundamentally different representations such as custom `annotation.json` files, COCO-style JSON, CSV-based metadata, or dataset-specific text formats. The API hides these differences behind a single, consistent schema.
- To **support both image-centric and class-centric workflows**
- To **enable dataset-scale filtering and querying** without rewriting dataset-specific logic
- To **remain extensible** as new datasets, annotation types, and workflows are added

The project is designed to support research in object counting and related tasks, particularly settings where models must reason about object quantity rather than classification alone. It is also motivated by the need to benchmark counting methods across datasets with very different annotation conventions.

## 2. Core Architectural Idea

The system separates concerns into **three logical layers**:

1. **Adapters**
   Dataset-specific logic that reads raw files and emits normalized records.

2. **Index (SQLite)**
   A global, read-only index that stores normalized images, annotations, and derived statistics in a single relational database.

3. **User-facing Datasets / APIs**
   Lightweight query and iteration interfaces that read from the index and present data in forms suitable for training, evaluation, or analysis.

This separation ensures that dataset-specific quirks are isolated within adapters, that filtering and querying logic is centralized and consistent across datasets, and that expensive parsing of raw files happens only once at build time rather than repeatedly during experimentation. The sections below describe the concrete data structures and components that implement each layer.

## 3. Canonical Schema

All datasets are normalized into a shared schema defined in `core/schema.py`. The schema consists of three primary record types: `ClassRecord`, `ImageRecord`, and `InstanceAnnotationRecord`.

### 3.1 ClassRecord

Represents a dataset-scoped semantic class.

Key fields:

- `class_key`: a stable identifier of the form `{dataset}/{slugified_name}`
  - Example: `fsc147/bird`, `dota/plane`, `penguin/penguin`
- `name`: human-readable class label; original class name that is not necessarily slugified
- `meta`: optional dataset-specific taxonomy or metadata

Classes are **not global across datasets** by default. For example, a class named `"car"` in one dataset is treated as distinct from `"car"` in another dataset unless explicitly merged downstream.

### 3.2 ImageRecord

Represents a single image.

Key fields:

- `image_id`: stable, deterministic hash of `(dataset, original_relpath)`
- `path`: absolute path to the image file
- `width`, `height`: image dimensions (resolved at build time)
- `split`: train / val / test / unspecified
- `provenance`: traceability back to the raw dataset (original filenames, IDs, etc.)
- `counts`: cached per-class instance counts for fast queries
- `meta`: dataset-specific metadata (sensor information, crowd statistics, etc.)

Images are **dataset-scoped**, but globally addressable via `image_id`. The `counts` field is populated during index construction based on associated annotations with `role="instance"`.

Although images are presented to users in a human-friendly order (by filename/path), `image_id` remains the canonical internal identifier used for joins, deduplication, and reproducibility.

### 3.3 InstanceAnnotationRecord

Represents a single annotation instance.

Key fields:

- `ann_id`: stable, deterministic hash derived from the image ID, class key, annotation type, geometry, and optional salt (e.g., instance index)
- `image_id`: foreign key to `ImageRecord`
- `class_key`: dataset-scoped class identifier
- `ann_type`: POINT, HBB (axis-aligned box), OBB (oriented box), etc.
- `geometry`: structured geometry payload corresponding to `ann_type`
- `role`: semantic role of the annotation (see below)
- `source`: ORIGINAL, CROWDSOURCE, GENERATED, etc.

#### Annotation Roles

The `role` field distinguishes **counted objects** from auxiliary annotations.

- `role="instance"`
  Objects that contribute to counts and dataset statistics

- `role!="instance"`
  Auxiliary annotations such as:
  - exemplar boxes in FSC147
  - alternative geometries of the _same underlying objects_ (e.g., HBB representations paired with OBB annotations in DOTA v1.5)

All counting logic **only** considers annotations with `role == "instance"`. This allows the index to store rich auxiliary information without corrupting counting semantics.

## 4. Dataset Adapters

Adapters are responsible for translating raw datasets into normalized records.

Each adapter implements the following methods, which emit schema objects:

- `iter_classes(ctx) → Iterable[ClassRecord]`
- `iter_images(ctx) → Iterable[ImageRecord]`
- `iter_annotations(ctx) → Iterable[InstanceAnnotationRecord]`

The `ctx` argument is an `AdapterContext` object that provides shared configuration such as the raw dataset root directory.

Adapters are intentionally simple: they read files, interpret dataset-specific formats, and emit records. They do not access the database directly.

### Design Principles for Adapters

Adapters are designed as pure readers with minimal assumptions. They do not perform database operations, they do not apply user-level filtering, and they avoid embedding downstream experimental logic. Their responsibility is limited to faithfully translating raw data into the canonical schema.

Adapters may expose **dataset-level configuration options** when strictly necessary (e.g., controlling indexing scope), but they should not implement query-time filtering or policy decisions.

#### Adapter Example: PenguinAdapter and Indexing Scope

The Penguin dataset contains ~80k images, but only ~5k images appear in `annotation.json`. Opening every image just to read its dimensions is slow and largely unhelpful for most use cases.

To address this, the adapter exposes:

```python
PenguinAdapter(include_unlabeled=False)
```

This parameter controls the **indexing scope**, not query-time filtering.

- `include_unlabeled=False` (default):
  Only images that appear in `annotation.json` with `xy != null` are indexed. Here, `xy == null` indicates that no labeler has reviewed the image at all.

- `include_unlabeled=True`:
  All images listed in `split.json` are indexed, regardless of annotation status.

This approach preserves adapter simplicity while avoiding unnecessarily large indices.

## 5. IndexBuilder and SQLite Index

The **IndexBuilder** is responsible for creating the global SQLite index from one or more adapters.

### 5.1 Build Pipeline

The build process proceeds in three phases:

1. **Initialize schema**
   Create all tables, indices, and constraints defined in the SQL schema.

2. **Insert canonical records**
   - class records
   - image records
   - annotation records

3. **Compute derived tables**
   - per-image per-class counts
   - total instance counts per image
   - dataset-specific statistics (e.g., crowd review metadata)

The resulting index is immutable in practice: it is not incrementally updated. Any change in datasets, adapter logic, or schema requires rebuilding the index from scratch.

Typical build would look like:

```python
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
```

### 5.2 HPC / Filesystem Robustness

SQLite can behave unreliably on networked or distributed filesystems (e.g., NFS or Lustre) due to locking and journaling semantics.

To mitigate this, the IndexBuilder builds the database in a **local temporary directory** (e.g., `$TMPDIR` or `/tmp`), performs integrity checks, and then atomically installs the final database into `counting/data/index.sqlite`.

### 5.3 Progress Reporting

IndexBuilder supports adapter-level progress reporting using a simple `tqdm` loop.

Each adapter contributes three high-level steps (classes, images, annotations). Since these steps vary widely in cost across datasets, the progress bar reflects logical progress rather than uniform time slices. For long-running steps, lightweight status messages provide intermediate feedback.

## 6. Derived Tables and Statistics

To make queries efficient and expressive, the index stores several derived tables computed deterministically from annotations.

Key tables include:

- `image_class_counts`: number of instance annotations per `(image_id, class_key)`
- `image_total_counts`: total number of instance annotations per image
- `image_review_stats`: dataset-specific metadata such as crowd review status (currently used by the Penguin dataset)

These tables eliminate the need for repeated aggregation at query time.

## 7. FilterPolicy

Filtering is centralized in `index/policy.py` via `FilterPolicy`.

A FilterPolicy allows users to restrict datasets based on criteria such as:

- allowed datasets or splits
- allowed annotation types
- minimum or maximum object counts
- minimum number of images per class

For example, to query the samples with at least 10 images per class with HBB or POINT annotation type from either test or validataion splits:

```python
policy = FilterPolicy(
    min_images_per_class=10,
    allowed_splits={"test", "val"},
    allowed_ann_types={"hbb", "point"},
)
```

The motivation for FilterPolicy is twofold: first, to support common experimental constraints (e.g., excluding rare classes or empty images), and second, to ensure that all datasets are filtered consistently under a single policy.

**Important:**
Filters are applied at _query time_, not adapter time. This ensures that the indexed data remains complete and that filtering semantics are uniform across datasets.

## 8. User-Facing APIs

The user-facing API is designed around a **single indexed representation** of all datasets, exposed through lightweight dataset views tailored to different experimental needs.

At the top level, users interact with a `CountingDatasetIndex`, which is responsible for discovery, filtering, and dataset construction.

### 8.1 CountingDatasetIndex

`CountingDatasetIndex` is the primary entry point into the system.

It is resposible for identifying which datasets, splits, and classes are available, and applying `FilterPolicy` constraints consistently across datasets. It also instantiates dataset views, either image-centric or class-centric depending on use case.

A typical workflow looks like:

```python
index = CountingDatasetIndex(root="counting/data", policy=policy)
classes = index.get_classes(datasets=["dota", "fsc147"])
dataset = index.load_class(classes[0]["class_key"], split="train")
```

Key methods include:

* `available_datasets()`: list all indexed datasets
* `available_splits()`: discover splits globally or per dataset
* `get_classes()`: retrieve class metadata and summary statistics
* `load_dataset()`: construct an image-centric dataset (`CountingImageDataset`)
* `load_class()`: construct a class-centric dataset (`CountingClassDataset`)

Importantly, `CountingDatasetIndex` itself does **not** load images or annotations eagerly. It only issues SQL queries and constructs iterable dataset objects.

### 8.2 CountingImageDataset (Image-Centric)

`CountingImageDataset` provides an **image-centric view** of the indexed data.

Iteration is performed over images, and each sample aggregates all relevant annotation information for that image across classes.

This view is particularly well suited for:

* multi-class counting or detection models,
* per-image statistics (e.g., total count, class co-occurrence),
* image-level filtering (e.g., minimum total count, review status).

Example usage:

```python
ds = index.load_dataset("dota", split="train")
img, target = ds[0]
```

The returned `target` dictionary summarizes all annotations associated with the image:

```python
{
  "image_id": "img_632c96ac63a0071f9de957e6243c4d9c",
  "total_count": 18,
  "counts": {
    "dota/harbor": 3,
    "dota/ship": 15
  },
  "instances": {
    "dota/harbor": List[InstanceAnnotationRecord],
    "dota/ship": List[InstanceAnnotationRecord]
  },
  "aux": {
    "hbb": {
      "dota/harbor": List[InstanceAnnotationRecord],
      "dota/ship": List[InstanceAnnotationRecord]
    }
  },
  "review_status": "reviewed"
}
```
In this view, multiple classes may be present in a single sample. Only annotations with `role="instance"` contribute to counts, and auxiliary annotations (e.g., exemplar boxes or alternative geometries) are grouped under `aux` by role.

### 8.3 CountingClassDataset (Class-Centric)

`CountingClassDataset` provides a **class-centric view** of the data.

Iteration is performed over images *conditioned on a single semantic class*. Only images containing at least one instance of the specified class are included.

This view is ideal for:

* class-specific counting tasks,
* exemplar-based or prompt-based methods,
* per-class evaluation and error analysis.

Example usage:

```python
ds = index.load_class("dota/harbor", split="train")
img, target = ds[0]
```

The returned `target` focuses exclusively on the selected class:

```python
{
  "image_id": "img_632c96ac63a0071f9de957e6243c4d9c",
  "class_key": "dota/harbor",
  "count": 3,
  "instances": List[InstanceAnnotationRecord],
  "aux": {
    "hbb": List[InstanceAnnotationRecord]
  },
  "review_status": "reviewed"
}
```

In this view, exactly one semantic class is represented per dataset instance. The counts and instances refer **only** to the requested class, and auxiliary annotations are restricted to alternative annotations for that class.

## 9. Ordering and Determinism

- Internal identity uses hashed IDs (`image_id`, `ann_id`)
- Presentation order defaults to lexicographic path order
- Optional `natural_sort=True` enables human-friendly ordering

All iteration is deterministic.

## 10. Extensibility

The system is designed to support new datasets via adapters, new annotation types and roles, export formats such as COCO, and integration with active measurement workflows. Both adapters and the index schema are intentionally conservative to minimize breaking changes.

## 11. Summary of Key Design Decisions

- **Adapters are simple and dataset-scoped**
- **All filtering happens at query time**
- **Counts are derived exclusively from `role="instance"`**
- **SQLite index is immutable and rebuildable**
- **User-facing APIs are thin wrappers over indexed data**
