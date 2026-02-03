# Counting Large Numbers

---

## Schema Overview

Our internal canonical schema has three levels: class, image, and instance annotation.

### 1. Class (category) level

Represents a concept like “FSC147 bird” or “DOTA plane”. Associated object is `ClassRecord`. Primary key is `class_key`, like "FSC147/bird" or "dota_v15/plane". That is, one `ClassRecord` per one `class_key`.

-> Unique by (dataset, class name)

### 2. Image level

Represents one image file (e.g., PNG, JPG). Associated object is `ImageRecord`. Primary key is `image_id` (hashed string). That is, one `ImageRecord` per unique <dataset, original_relpath>

-> Unique by (dataset, normalized original relative path)

In this schema, counts are derived summaries at the image level. So for most cases:

`ImageRecord.counts[class_key] == number of InstanceAnnotationRecord rows with same image_id and class_key`

Exception is when the dataset provides counts without instances. Then we store the provided count and optionally have zero instances.


### 3. Instance annotation level

Represents one labeled object instance in one image for one class. Associated object is `InstanceAnnotationRecord`. Primary key is `ann_id`. Each instance is associated with two foreign keys: `image_id` and `class_key`. That is, one `InstanceAnnotationRecord` per object instance. 

-> Unique by (`image_id`, `class_key`, `ann_type`, `geometry`, `source`, `salt`)

NOTE: To guarantee uniqueness even with duplicate geometry, adapters should always set `salt` deterministically.


---

## Adapters

### What an adapter does (per-dataset, “extractor” layer)

- Reads raw dataset files with all their quirks
- Emits canonical records:
    - ImageRecord (image metadata + provenance + split)
    - ClassRecord (dataset-scoped classes)
    - InstanceAnnotationRecord (instances with geometry)
- Keeps ordering deterministic

Adapters are dataset-specific and do not implement global policy. E.g., they don’t drop “small classes”; that’s curation. This means that adapters can stay simple and do not need cross-dataset knowledge.

## Index Builders

### What the index builder does (“compiler” layer)

- Consumes all adapters
- Validates records + resolves collisions (if any)
- Writes everything to SQLite / manifests
- Computes *derived* things:
    - counts (per image/class)
    - per-class num_images, mean count, etc.
- Applies filtering policy (min_images, allowed ann types, etc.) to produce the curated “view”
- Provides versioning / build metadata

### Why use BLAKE2b hashing for IDs?