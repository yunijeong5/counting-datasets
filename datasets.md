# Integrated Datasets

This document lists the datasets currently integrated into the Counting Dataset API, along with brief descriptions, statistics, licensing information, and references to original sources.

All datasets are indexed through dataset-specific adapters and normalized into a shared schema. The statistics reported below are derived using dataset inspection scripts and reflect the indexed version of each dataset.

## Summary Table

| Dataset                       | Domain          | # Images | Annotation Type            | # Classes |Images per Class (mean)| Total Objects |Objects per Image (mean)| Multi-class Images | License         |
| ----------------------------- | --------------- | -------- | -------------------------- | --------- | --------------------- | ------------- | ---------------------- | ------------------ | --------------- |
| Aerial Elephant Dataset       | Aerial wildlife | 2,074    | Points                     | 1         | –                     | 15,581        | 7.51                   | No                 | CC0             |
| DOTA v1.5                     | Aerial urban    | 1,869    | Bounding boxes (OBB + HBB) | 16        | 343.38                | 280,196       | 149.92                 | Yes                | Apache 2.0      |
| FSC-147                       | Web images      | 6,135    | Points + exemplars (HBB)   | 147       | 41.73                 | 343,693       | 56.02                  | No                 | MIT             |
| Kenyan Wildlife Aerial Survey | Aerial wildlife | 561      | Bounding boxes (HBB)       | 3         | 197.67                | 4,304         | 7.67                   | Yes                | CC0             |
| Malaria Infected Blood Smears | Microscopy      | 1,328    | Bounding boxes (HBB)       | 7         | 434.00                | 86,035        | 64.79                  | Yes                | CC BY-NC-SA 3.0 |
| Penguins                      | Wildlife        | 81,941   | Crowd-sourced points       | 1         | –                     | 68,901        | 17.84                  | No                 | CC BY 4.0       |

### Notes on interpretation

- **Images per Class (mean)** is not meaningful for single-class datasets and hence omitted.
- **Total Objects** and **Objects / Image** are computed using instance-only annotations (`role="instance"`). See [design.md](design.md#annotation-roles).

## Dataset Descriptions

### 🐘 Aerial Elephant Dataset

This dataset is designed for large-scale wildlife counting from aerial imagery. It is strictly single-class and single-label, making it well suited for benchmarking counting accuracy without classification ambiguity.

* **Domain:** Aerial wildlife monitoring
* **Images:** 2,074 high-resolution RGB aerial images
* **Annotations:** Point annotations marking individual elephants
* **Classes:** 1 (elephant)
* **Image resolution:** ~5,500 × 3,700 pixels
* **Objects per image:** Mean ~7.5, Max 78
* **License:** CC0 (Public Domain)

**Reference:** Naudé, Johannes J. and Deon Joubert. “The Aerial Elephant Dataset: A New Public Benchmark for Aerial Object Detection.” CVPR Workshops (2019). [[pdf]](https://openaccess.thecvf.com/content_CVPRW_2019/papers/DOAI/Naude_The_Aerial_Elephant_Dataset_A_New_Public_Benchmark_for_Aerial_CVPRW_2019_paper.pdf) [[dataset download]](https://www.kaggle.com/datasets/davidrpugh/aerial-elephant-dataset)


### ✈️ DOTA v1.5

DOTA is a dense, large-scale detection dataset featuring extreme object counts, heavy class imbalance, and frequent multi-class co-occurrence. In this API, OBBs are treated as canonical counting instances, with HBBs stored as auxiliary alternative geometry. 

* **Domain:** Aerial urban scenes
* **Images:** 1,869
* **Annotations:** Oriented bounding boxes (OBB) with paired axis-aligned boxes (HBB)
* **Classes:** 16 object categories (harbor, small vehicle, baseball diamond, etc.)
* **Image resolution:** ~420 × 350 to ~12,000 × 5,000 pixels
* **Objects per image:** Mean ~150, Max >10,000
* **License:** Apache License 2.0

**Reference:** J. Ding, N. Xue, Y. Long, G. -S. Xia and Q. Lu, "Learning RoI Transformer for Oriented Object Detection in Aerial Images," 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA, 2019, pp. 2844-2853, doi: 10.1109/CVPR.2019.00296. [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8953881) [[dataset download]](https://captain-whu.github.io/DOTA/dataset.html)


### 🖥️ FSC-147

Each image in FSC-147 is associated with exactly one class, even if other objects appear visually. Exemplar boxes are provided as auxiliary annotations and are explicitly marked as non-counting (`role="exemplar"`).

* **Domain:** Web images (crowd counting)
* **Images:** 6,135
* **Annotations:** Point annotations + exemplar bounding boxes (HBB)
* **Classes:** 147 common objects (bird, beads, apple, etc.)
* **Image resolution:** Fixed height (384 px), variable width
* **Objects per image:** Mean ~56, Max 3,701
* **License:** MIT

**Reference:** Viresh Ranjan, Udbhav Sharma, Thu Nguyen and Minh Hoai. "Learning To Count Everything", Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. [[pdf]](https://openaccess.thecvf.com/content/CVPR2021/papers/Ranjan_Learning_To_Count_Everything_CVPR_2021_paper.pdf) [[dataset download]](https://github.com/cvlab-stonybrook/LearningToCountEverything/tree/master)


### 🦒 Kenyan Wildlife Aerial Survey

This dataset is provided in COCO format and contains multi-class aerial scenes of wildlife. It is representative of real-world aerial survey conditions.

* **Domain:** Aerial wildlife monitoring
* **Images:** 561
* **Annotations:** Axis-aligned bounding boxes (HBB)
* **Classes:** 3 (elephants, giraffes, zebras)
* **Image resolution:** ~5,000 × 3,000 pixels
* **Objects per image:** Mean ~7.7, Max 108
* **License:** CC0 (Public Domain)

**Reference:** Eikelboom JAJ, Wind J, van de Ven E, et al. Improving the precision and accuracy of animal population estimates with aerial image object detection. Methods Ecol Evol. 2019; 10: 1875–1887. https://doi.org/10.1111/2041-210X.13277 [[pdf]](https://besjournals.onlinelibrary.wiley.com/doi/epdf/10.1111/2041-210X.13277) [[dataset download]](https://www.kaggle.com/datasets/davidrpugh/kenyan-wildlife-aerial-survey)


### 🦠 Malaria Infected Human Blood Smears
This dataset features dense object distributions and significant class imbalance. It is well suited for evaluating counting methods in high-density microscopy settings.

* **Domain:** Microscopy / biomedical imaging
* **Images:** 1,328
* **Annotations:** Axis-aligned bounding boxes (HBB)
* **Classes:** 7 cell types
* **Image resolution:** ~1,600 × 1,200 to ~1,900 × 1,400 pixels
* **Objects per image:** Mean ~65, Max 223
* ***License:** Creative Commons Attribution-NonCommercial-ShareAlike 3.0

**Reference:** Image set BBBC041v1 from the Broad Bioimage Benchmark Collection [[article]](https://www.nature.com/articles/nmeth.2083) [[dataset download]](https://bbbc.broadinstitute.org/BBBC041)


### 🐧 Penguins

Only a small subset of images contains annotations; the remainder are unreviewed. The adapter exposes an `include_unlabeled` option to control indexing scope (see [design.md](./design.md#adapter-example-penguinadapter-and-indexing-scope)). Crowd metadata (review status, annotator counts, vote statistics) is preserved and queryable.

* **Domain:** Wildlife monitoring
* **Images:** 81,941 (≈5,200 annotated)
* **Annotations:** Crowd-sourced point annotations
* **Classes:** 1 (penguin)
* **Image resolution:** ~1,900 × 1,000 to ~2,000 × 1,500 pixels
* **Objects per image (annotated subset):** Mean ~18, Max 213
* ***License:** CC BY 4.0

**Reference:** C. Arteta, V. Lempitsky, A. Zisserman. Counting in the Wild, European Conference on Computer Vision, 2016 [[pdf]](https://www.robots.ox.ac.uk/~vgg/publications/2016/Arteta16/arteta16.pdf) [[download dataset]](https://www.robots.ox.ac.uk/~vgg/data/penguins/#)

## Remarks on Dataset Diversity

Together, these datasets span:

* aerial, web, microscopy, and wildlife domains,
* sparse to extremely dense object distributions,
* single-class and multi-class settings,
* single-annotator and crowd-sourced labeling,
* point-based and bounding-box-based annotation styles.
