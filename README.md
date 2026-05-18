# Counting Dataset API

A unified, extensible API for **object counting datasets**, providing a normalized index, consistent schemas, and flexible query interfaces across heterogeneous data sources.

This project is designed to support research in **object counting** and **dataset benchmarking** by abstracting away dataset-specific quirks while preserving rich annotation semantics.

## Key Features

- **Unified indexing layer** for diverse counting datasets
- **Support for heterogeneous annotations**
  - points
  - axis-aligned bounding boxes (HBB)
  - oriented bounding boxes (OBB)
  - crowd-sourced annotations
  - auxiliary annotations (exemplars, alternative geometries)
- **Immutable SQLite index** for fast, reproducible queries
- **Adapter-based architecture** for easy dataset extension
- **Image-centric and class-centric dataset views**
- **HPC-friendly build process** (robust to network filesystems)

## Integrated Datasets

The API currently integrates the following datasets:

- Birds (Tree Swallows Flock)
- Aerial Elephant Dataset
- DOTA v1.5
- FSC-147
- Kenyan Wildlife Aerial Survey
- Malaria Infected Human Blood Smears
- Penguins (crowd-sourced)

A detailed overview, including dataset statistics, licenses, and citations, is available in  👉 **[`datasets.md`](datasets.md)**

## Project Structure

```
counting-datasets/
├── counting/
│   ├── src/            # Core library code
│   ├── data/           # Built SQLite index (optional, not tracked)
│   └── tests/
├── raw/                # Raw datasets (NOT tracked)
│   └── birds/
│       └── generate_bird_bbox_cache.py  # one-time pre-processing for Birds
├── build_index.py      # Convenience build script with per-dataset toggles
├── design.md           # Architecture & design rationale
├── datasets.md         # Dataset descriptions & statistics
├── usage_example.py    # End-to-end example
└── pyproject.toml
```

Raw datasets are **not included** in the repository. See below for guidance.


## Installation

This project uses a standard Python packaging layout. To use the package, first clone the repository and install it with pip.

```bash
git clone https://github.com/yunijeong5/counting-datasets.git
cd counting-datasets
pip install -e .
```

Python ≥ 3.9 is recommended.

## Quick Start

### 1. Prepare raw datasets

Download raw datasets into the `raw/` directory using the the following expected layouts.

[TODO: Add download instructions for raw data]

Dataset-specific directory structures are also documented in the adapters and in `datasets.md`.


### 2. Build the index

The easiest way to build is via the provided script, which has per-dataset toggles and prints a post-build summary:

```bash
python build_index.py
```

To control which datasets are included, edit the `DATASETS` dict at the top of `build_index.py`.

**Note for the Birds dataset:** run `python raw/birds/generate_bird_bbox_cache.py` once before building to pre-compute the pseudo-bounding box cache (`raw/birds/pseudo_bboxes_cache.json`).

You can also invoke the builder directly:

```python
from pathlib import Path
from counting_dataset.index.builder import IndexBuilder
from counting_dataset.adapters.birds import BirdsAdapter
from counting_dataset.adapters.dota import DOTAAdapter
from counting_dataset.adapters.kenyan_wildlife import KenyanWildlifeAdapter
from counting_dataset.adapters.malaria import MalariaAdapter

builder = IndexBuilder(raw_root=Path("raw"), out_root=Path("counting/data"))
db_path = builder.build(
    adapters=[BirdsAdapter(), DOTAAdapter(), KenyanWildlifeAdapter(), MalariaAdapter()],
    overwrite=True,
    show_progress=True,
)
print("Index built at:", db_path)
```

This produces a single SQLite database (`counting/data/index.sqlite`) containing normalized images, annotations, and derived statistics.


### 3. Load datasets

#### Dataset views

The API exposes two complementary dataset views, depending on whether your workflow is image-oriented or class-oriented.

| View              | Loader           | Iterates over                      | Best suited for                                                       |
| ----------------- | ---------------- | ---------------------------------- | --------------------------------------------------------------------- |
| **Image-centric** | `load_dataset()` | Images                             | Multi-class training, image-level statistics, dataset-wide filtering  |
| **Class-centric** | `load_class()`   | Images containing a specific class | Class-specific counting, exemplar-based methods, per-class evaluation |

Image-centric datasets return aggregated annotations across _all classes_ present in each image. Class-centric datasets restrict attention to _one semantic class_ and only return images where that class appears.

Both views are backed by the same indexed data and share identical filtering semantics; choosing between them is purely a matter of how an experiment is structured.

You can load **image-centric** datasets with `index.load_dataset()`:

```python
from counting_dataset import CountingDatasetIndex

index = CountingDatasetIndex("counting/data")  # root directory containing index.sqlite

dataset = index.load_dataset(
    "dota",
    splits={"train"},
    min_total_count=100,
)

for img, target in dataset:
    print(target["width"], target["height"])  # image dimensions
    print(target["counts"])
    break
```

You can load **class-centric** datasets with `index.load_class()`:

```python
cls_ds = index.load_class(
    class_key="dota/small_vehicle",
    splits={"train"},
)

img, target = cls_ds[0]
print(target["count"])
```

See [design.md](./design.md#81-countingdatasetindex) for more details on the two dataset views.

## End-to-End Example

The repository includes a fully working end-to-end script as a sample: `usage_example.py`.

This script demonstrates:

- building the SQLite index from raw datasets,
- inspecting available datasets, splits, and classes,
- loading image-centric and class-centric datasets,
- iterating over samples and inspecting targets.

### Running the example

After downloading raw datasets into `raw/`, simply run:

```bash
python usage_example.py
```

## Design Philosophy

* **Adapters are pure readers**
  * No database access
  * No filtering based on experimental intent
* **All filtering happens at query time**
* **Counting semantics are explicit**
  * Only `role="instance"` annotations contribute to counts
* **The index is immutable**
  * Rebuild to change data or logic
* **Traceability is preserved**
  * Raw identifiers and metadata are retained

For a detailed explanation of the architecture, schema, and design decisions, see
👉 **[`design.md`](design.md)**

## Extending the Project

To add a new dataset:

1. Implement a dataset adapter exposing:

   * `iter_classes(ctx)`
   * `iter_images(ctx)`
   * `iter_annotations(ctx)`
2. Register the adapter in your build script
3. Rebuild the index

A template adapter is provided in the repository for reference.

## Export Utilities

[TODO: export in COCO/JSON format]

## Example Training Loop 

[TODO: sample script using dataset loaders to train/test model]


## What This Project Is (and Is Not)

**This project provides:**

* Dataset normalization
* Indexing and querying
* Dataset views for counting research

**This project does NOT (yet) provide:**

* Training loops
* Model implementations
* Evaluation scripts

Those are intentionally left to downstream research code.

## Licensing

This repository contains **no raw datasets**.

Each dataset is distributed under its original license.
Please consult **[`datasets.md`](datasets.md)** for per-dataset licensing and citations.


## References

This project integrates and builds upon work by many dataset authors and research groups.
Full references are provided in `datasets.md`.
