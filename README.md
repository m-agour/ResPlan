# ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential Floor Plans

[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code: MIT](https://img.shields.io/badge/Code-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

**ResPlan** is a dataset of **17,000 residential floor plans** derived from
publicly accessible online real-estate listings, providing vector geometry,
room-connectivity graphs, and metric-scale coordinates.

- **Vector geometry** for walls, doors, windows, rooms and balconies
- **Room-connectivity graphs** with four typed edges (`via_door`, `adjacency`,
  `via_window`, `direct`)
- **Semantic labels** across a 17-category taxonomy
- **Metric-scale coordinates** in metres
- **Canonical splits**, stratified by bedroom count

---

## Example

![Sample Plan](assets/sample_plan.png)
![Plan with Graph](assets/plan_graph.png)

---

## Key statistics

Measured on the released files.

| Property | Value |
|---|---|
| Total plans | 17,000 |
| Train / val / test | 13,053 / 1,632 / 1,632 (plus 683 augmented) |
| Avg functional rooms per plan | 8.1 |
| Avg graph nodes per plan | 9.2 |
| Avg graph edges per plan | 12.9 |
| Median floor area | 110 m² |
| Edge types | `via_door` 54.2%, `adjacency` 35.2%, `direct` 7.6%, `via_window` 3.0% |
| Room polygons | 137,131 (43.2% rectangular) |

---

## Contents

```
├── ResPlan.zip            ← Dataset pickle, geometry only (unzip first)
├── split.json             ← Canonical splits
├── croissant.json         ← Croissant metadata (JSON-LD)
├── resplan_utils.py       ← Loading, plotting, graph construction, conversions
├── ResPlan_demo.ipynb     ← Demo notebook
├── LICENSE                ← CC BY 4.0 (data) and MIT (code)
├── TAKEDOWN.md            ← Removal-request policy
└── baselines/             ← Reproducible baseline experiments + result JSONs
```

The pickle in this repository carries **geometry only**. Build the connectivity
graph on demand (see below). The
[Kaggle release](https://www.kaggle.com/datasets/resplan/resplan) ships the same
geometry with `plan["graph"]` already populated.

---

## Quick start

```bash
pip install -r requirements.txt
unzip ResPlan.zip
```

```python
import pickle
from resplan_utils import plot_plan, plan_to_graph, add_adjacency_edges

with open("ResPlan.pkl", "rb") as f:
    data = pickle.load(f)

plan = data[0]
plot_plan(plan)

G = plan_to_graph(plan)          # strict definition, see note below
G = add_adjacency_edges(G)       # paper / Kaggle definition
```

### Two graph definitions, read this before benchmarking

`plan_to_graph` implements a **strict** definition: two rooms are linked only
when their shared boundary is genuinely open, which it labels `via_opening`.

The **paper, the Kaggle release, and every benchmark number below** use a
broader `adjacency` relation that also fires for rooms separated only by a wall.
Calling `add_adjacency_edges` converts the strict graph to that taxonomy: it
relabels `via_opening` and `fallback` as `adjacency`, and adds `adjacency` edges
between rooms lying within one wall thickness of each other.

Without that second call you get roughly 8.7 edges per plan across five types
instead of the 12.9 edges across four types the benchmarks assume, and results
will not be comparable.

### Using the splits

`split.json` has four keys. The `augmented` list holds 683 plans that are
geometric augmentations (rotations, flips, scales) of 667 originals, kept
separate so you can decide whether to include them.

```python
import json
splits = json.load(open("split.json"))
train_ids = set(splits["train"])      # 13,053
val_ids   = set(splits["val"])        #  1,632
test_ids  = set(splits["test"])       #  1,632
aug_ids   = set(splits["augmented"])  #    683
```

---

## Benchmark tasks and results

Numbering matches the paper. Measured on the released data, 3 seeds, 500 epochs.

**Task 1, semantic room labeling.** Classify each room node into one of five
categories from graph structure and geometric features.

| Method | Accuracy | Macro F1 |
|---|---|---|
| Rule-based (DT) | 0.800 | 0.769 |
| Gradient Boosting | 0.859 | 0.848 |
| Random Forest | 0.867 | 0.856 |
| GCN (3-layer) | 0.713±0.002 | 0.734±0.002 |
| GraphSAGE (3-layer) | 0.944±0.001 | 0.941±0.001 |
| RGCN (typed edges) | 0.954±0.001 | 0.951±0.001 |
| **GraphGPS** | **0.955±0.001** | **0.954±0.001** |

Typed edges matter: removing the four typed-edge degree features costs 2.5
accuracy points (0.944 to 0.919).

**Task 2, constrained floor plan generation.** Generate a plan given a boundary,
room counts, and an adjacency graph.

**Task 3, plan-to-graph extraction.** Recover the typed connectivity graph from
geometry alone.

| Method | Precision | Recall | F1 | Type acc. |
|---|---|---|---|---|
| Proximity | 0.582 | 0.900 | 0.707 | 0.544 |
| Shared boundary | 0.969 | 0.976 | 0.972 | 0.544 |
| Shared boundary + GB | 0.969 | 0.976 | 0.972 | 0.867 |

**Cross-dataset transfer** (GraphSAGE, 8 shared features)

| Train to test | Accuracy |
|---|---|
| ResPlan to ResPlan | 0.918 |
| RPLAN to RPLAN | 0.909 |
| RPLAN to ResPlan | 0.592 |
| ResPlan to RPLAN | 0.664 |

Reproduce with:

```bash
cd baselines && bash reproduce.sh
```

---

## Known limitations

- **Regional scope.** All plans come from South Asian residential markets, so
  layout conventions are not representative of other regions. The cross-dataset
  results above quantify the gap.
- **Single floor, no furniture or 3D.**
- **Wall thickness is normalised per plan**, so within-plan variation between
  structural walls and thin partitions is not preserved; 99.3% of plans fall in
  the 10 to 40 cm range.
- **Vectorisation artefacts.** A small tail of room polygons retains jagged
  traced contours: 0.52% exceed 30 vertices, 0.02% exceed 100.
- **Near-duplicate plans.** Listings are sometimes republished. A geometry-based
  scan finds 1,170 redundant plans (6.9%) in 931 clusters, and 154 of the 1,632
  test plans have a near-duplicate in the training split. The effect on
  benchmark results is small, 0.15 accuracy points on Task 1.
- **Semantic label accuracy** rests on a stratified 500-plan manual audit rather
  than exhaustive verification.

---

## Provenance

Plans derive from publicly accessible real-estate listing pages. Only public,
non-paywalled pages were accessed, with no circumvention of rate limits, login
walls or other access controls, and after review of platform terms of service.
Source platform identities are withheld to comply with those terms.

The release contains **no** source images, listing text, prices, addresses,
geolocation or personally identifying information: only polygon coordinates and
connectivity graphs. See `LICENSE` for the scope of the CC BY 4.0 grant and
`TAKEDOWN.md` for the removal process.

---

## Citation

Citation details are withheld while the accompanying paper is under peer review.
