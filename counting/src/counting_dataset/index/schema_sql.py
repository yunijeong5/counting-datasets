from __future__ import annotations

SCHEMA_SQL = r"""
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- -----------------------
-- Core tables
-- -----------------------

CREATE TABLE IF NOT EXISTS images (
  image_id            TEXT PRIMARY KEY,
  dataset             TEXT NOT NULL,
  split               TEXT NOT NULL,
  path                TEXT NOT NULL,
  width               INTEGER NOT NULL,
  height              INTEGER NOT NULL,

  -- provenance
  original_relpath    TEXT NOT NULL,
  original_filename   TEXT NOT NULL,
  original_id         TEXT,
  sha1                TEXT,
  size_bytes          INTEGER,

  -- extras
  meta_json           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_images_dataset ON images(dataset);
CREATE INDEX IF NOT EXISTS idx_images_split   ON images(split);

CREATE TABLE IF NOT EXISTS classes (
  class_key           TEXT PRIMARY KEY,
  dataset             TEXT NOT NULL,
  name                TEXT NOT NULL,
  meta_json           TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_classes_dataset ON classes(dataset);

CREATE TABLE IF NOT EXISTS annotations (
  ann_id              TEXT PRIMARY KEY,
  image_id            TEXT NOT NULL,
  class_key           TEXT NOT NULL,

  ann_type            TEXT NOT NULL,
  source              TEXT NOT NULL,
  instance_index      INTEGER,

  -- store geometry in JSON for flexibility (point/hbb/obb/etc.)
  geometry_json       TEXT NOT NULL,

  score               REAL,
  meta_json           TEXT NOT NULL,

  role           TEXT NOT NULL DEFAULT 'instance',

  FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE,
  FOREIGN KEY(class_key) REFERENCES classes(class_key) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_ann_image  ON annotations(image_id);
CREATE INDEX IF NOT EXISTS idx_ann_class  ON annotations(class_key);
CREATE INDEX IF NOT EXISTS idx_ann_type   ON annotations(ann_type);
CREATE INDEX IF NOT EXISTS idx_annotations_role ON annotations(role);
CREATE INDEX IF NOT EXISTS idx_annotations_image_role ON annotations(image_id, role);
CREATE INDEX IF NOT EXISTS idx_annotations_class_role ON annotations(class_key, role);

-- Derived aggregate: counts per (image, class)
CREATE TABLE IF NOT EXISTS image_class_counts (
  image_id            TEXT NOT NULL,
  class_key           TEXT NOT NULL,
  count               INTEGER NOT NULL,
  PRIMARY KEY (image_id, class_key),
  FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE,
  FOREIGN KEY(class_key) REFERENCES classes(class_key) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_icc_class ON image_class_counts(class_key);

-- Derived aggregate: total annotation count per image (across all classes)
CREATE TABLE IF NOT EXISTS image_total_counts (
  image_id            TEXT PRIMARY KEY,
  total_count         INTEGER NOT NULL,
  FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_itc_total_count ON image_total_counts(total_count);

-- Derived aggregate: crowd review/annotator participation per image
-- For non-crowd datasets, we store defaults (review_status='na', counts=0).
CREATE TABLE IF NOT EXISTS image_review_stats (
  image_id            TEXT PRIMARY KEY,
  review_status       TEXT NOT NULL,   -- 'reviewed' | 'unreviewed' | 'malformed' | 'missing_in_annotation_json' | 'na'
  num_annotators      INTEGER NOT NULL,
  num_empty_votes     INTEGER NOT NULL,
  num_point_votes     INTEGER NOT NULL,
  num_points_total    INTEGER NOT NULL,
  FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_irs_review_status ON image_review_stats(review_status);
CREATE INDEX IF NOT EXISTS idx_irs_num_annotators ON image_review_stats(num_annotators);
CREATE INDEX IF NOT EXISTS idx_irs_num_points_total ON image_review_stats(num_points_total);
"""
