#!/usr/bin/env python3
"""
Task 1: Constrained Floor Plan Generation — Retrieval Baselines

Three retrieval baselines:
  1. Random: uniformly random training plan
  2. Count retrieval: nearest by room-count L1 distance
  3. Count + Graph: among top-50 by count, pick highest Jaccard on adjacency edges

Metrics:
  - Room Count Accuracy: fraction of room types with matching count
  - Adjacency Satisfaction: fraction of target adjacency edges present in retrieved plan
  - Boundary IoU: IoU of bounding boxes (proxy for boundary match)

Usage:
  python task1_retrieval.py
  python task1_retrieval.py --data ../ResPlan.pkl --split ../split.json

Outputs:
  results/task1_results.json
"""
import argparse, json, os, pickle, time, warnings
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from collections import Counter

warnings.filterwarnings("ignore")

ROOM_TYPES = ["bedroom", "bathroom", "kitchen", "living", "balcony"]


def get_room_count_vector(plan):
    """Get room count vector for a plan."""
    counts = []
    for rt in ROOM_TYPES:
        geo = plan.get(rt)
        if geo is None or not hasattr(geo, "is_empty") or geo.is_empty:
            counts.append(0)
        elif isinstance(geo, MultiPolygon):
            counts.append(len(list(geo.geoms)))
        elif isinstance(geo, Polygon):
            counts.append(1)
        else:
            counts.append(0)
    return np.array(counts)


def get_adj_edges(plan):
    """Get adjacency edge set from graph (room-type pair set)."""
    G = plan.get("graph")
    if G is None:
        return set()
    edges = set()
    for u, v in G.edges():
        tu = G.nodes[u].get("type", "")
        tv = G.nodes[v].get("type", "")
        if tu in ROOM_TYPES and tv in ROOM_TYPES:
            edges.add(frozenset([u, v]))
    return edges


def get_adj_type_multiset(plan):
    """Get adjacency edge multiset as sorted (type, type) tuples.

    Since plans can have multiple rooms of the same type, we track edges as
    a Counter of (type_a, type_b) pairs. This allows cross-plan comparison
    because it's type-level, not node-level.
    """
    G = plan.get("graph")
    if G is None:
        return Counter()
    edge_types = Counter()
    for u, v in G.edges():
        tu = G.nodes[u].get("type", "")
        tv = G.nodes[v].get("type", "")
        if tu in ROOM_TYPES and tv in ROOM_TYPES:
            pair = tuple(sorted([tu, tv]))
            edge_types[pair] += 1
    return edge_types


def multiset_jaccard(a: Counter, b: Counter):
    """Jaccard similarity for multisets (Counters)."""
    all_keys = set(a) | set(b)
    if not all_keys:
        return 1.0
    intersection = sum(min(a[k], b[k]) for k in all_keys)
    union = sum(max(a[k], b[k]) for k in all_keys)
    return intersection / union if union > 0 else 0.0


def get_boundary_bbox(plan):
    """Get bounding box [x1, y1, x2, y2] of inner polygon."""
    inner = plan.get("inner")
    if inner and hasattr(inner, "bounds") and not inner.is_empty:
        return inner.bounds  # (minx, miny, maxx, maxy)
    return None


def bbox_iou(box_a, box_b):
    """IoU of two bounding boxes (minx, miny, maxx, maxy)."""
    if box_a is None or box_b is None:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def rescaled_bbox_iou(box_a, box_b):
    """IoU after rescaling both boxes to unit origin (captures aspect ratio match).

    Since retrieved plans are rescaled to match target boundary, we compare
    normalized shapes (shifted to origin).
    """
    if box_a is None or box_b is None:
        return 0.0
    # Normalize both to origin, keeping aspect ratio
    wa = box_a[2] - box_a[0]
    ha = box_a[3] - box_a[1]
    wb = box_b[2] - box_b[0]
    hb = box_b[3] - box_b[1]
    if wa <= 0 or ha <= 0 or wb <= 0 or hb <= 0:
        return 0.0
    # Rescale b to match a's dimensions, compute aspect-ratio similarity
    # Simpler: use min(w_ratio, 1/w_ratio) * min(h_ratio, 1/h_ratio) as a proxy
    wr = wa / wb
    hr = ha / hb
    return min(wr, 1/wr) * min(hr, 1/hr)


def jaccard(set_a, set_b):
    """Set Jaccard similarity."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def main():
    parser = argparse.ArgumentParser(description="Task 1: Retrieval Baselines")
    parser.add_argument("--data", default="../ResPlan.pkl")
    parser.add_argument("--split", default="../split.json")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for count+graph retrieval")
    parser.add_argument("--output", default="results/task1_results.json")
    args = parser.parse_args()

    np.random.seed(42)

    print(f"Loading {args.data}...")
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    with open(args.split) as f:
        splits = json.load(f)
    train_ids = {int(x) for x in splits["train"]}
    test_ids  = {int(x) for x in splits["test"]}

    train_plans, test_plans = [], []
    for plan in data:
        pid = int(plan.get("id", -1))
        if pid in train_ids:
            train_plans.append(plan)
        elif pid in test_ids:
            test_plans.append(plan)
    del data
    print(f"Train: {len(train_plans)}, Test: {len(test_plans)}")

    # Precompute train features
    print("Precomputing features...")
    train_counts = np.array([get_room_count_vector(p) for p in train_plans])
    train_bboxes = [get_boundary_bbox(p) for p in train_plans]
    train_adj    = [get_adj_type_multiset(p) for p in train_plans]

    methods = {
        "random": {"count_acc": [], "adj_sat": [], "boundary_iou": []},
        "count_retrieval": {"count_acc": [], "adj_sat": [], "boundary_iou": []},
        "count_graph_retrieval": {"count_acc": [], "adj_sat": [], "boundary_iou": []},
    }

    print("Evaluating...")
    t0 = time.time()
    for idx, target in enumerate(test_plans):
        if (idx + 1) % 500 == 0:
            print(f"  {idx+1}/{len(test_plans)}")

        t_count = get_room_count_vector(target)
        t_adj   = get_adj_type_multiset(target)
        t_bbox  = get_boundary_bbox(target)

        # --- Random ---
        ri = np.random.randint(len(train_plans))
        r_count = train_counts[ri]
        r_adj   = train_adj[ri]
        r_bbox  = train_bboxes[ri]
        count_acc = float(np.mean(t_count == r_count))
        adj_sat = multiset_jaccard(t_adj, r_adj)
        b_iou = rescaled_bbox_iou(t_bbox, r_bbox)
        methods["random"]["count_acc"].append(count_acc)
        methods["random"]["adj_sat"].append(adj_sat)
        methods["random"]["boundary_iou"].append(b_iou)

        # --- Count retrieval ---
        dists = np.abs(train_counts - t_count).sum(axis=1)
        min_dist = dists.min()
        candidates = np.where(dists == min_dist)[0]
        # Break ties by boundary aspect-ratio similarity
        best_i = candidates[np.argmax([rescaled_bbox_iou(t_bbox, train_bboxes[c]) for c in candidates])]
        r_count = train_counts[best_i]
        r_adj   = train_adj[best_i]
        r_bbox  = train_bboxes[best_i]
        count_acc = float(np.mean(t_count == r_count))
        adj_sat = multiset_jaccard(t_adj, r_adj)
        b_iou = rescaled_bbox_iou(t_bbox, r_bbox)
        methods["count_retrieval"]["count_acc"].append(count_acc)
        methods["count_retrieval"]["adj_sat"].append(adj_sat)
        methods["count_retrieval"]["boundary_iou"].append(b_iou)

        # --- Count + Graph retrieval ---
        top_k_idx = np.argsort(dists)[:args.top_k]
        best_jacc = -1
        best_j = top_k_idx[0]
        for j in top_k_idx:
            j_val = multiset_jaccard(t_adj, train_adj[j])
            if j_val > best_jacc:
                best_jacc = j_val
                best_j = j
        r_count = train_counts[best_j]
        r_adj   = train_adj[best_j]
        r_bbox  = train_bboxes[best_j]
        count_acc = float(np.mean(t_count == r_count))
        adj_sat = multiset_jaccard(t_adj, r_adj)
        b_iou = rescaled_bbox_iou(t_bbox, r_bbox)
        methods["count_graph_retrieval"]["count_acc"].append(count_acc)
        methods["count_graph_retrieval"]["adj_sat"].append(adj_sat)
        methods["count_graph_retrieval"]["boundary_iou"].append(b_iou)

    results = {}
    print(f"\n{'Method':<25} {'Count Acc':>10} {'Adj Sat':>10} {'Boundary IoU':>12}")
    print("-" * 60)
    for name in ["random", "count_retrieval", "count_graph_retrieval"]:
        m = methods[name]
        res = {
            "room_count_acc": float(np.mean(m["count_acc"])),
            "adj_satisfaction": float(np.mean(m["adj_sat"])),
            "boundary_iou": float(np.mean(m["boundary_iou"])),
        }
        results[name] = res
        print(f"{name:<25} {res['room_count_acc']:>10.3f} {res['adj_satisfaction']:>10.3f} "
              f"{res['boundary_iou']:>12.3f}")

    results["_meta"] = {
        "n_train": len(train_plans),
        "n_test": len(test_plans),
        "top_k": args.top_k,
        "seed": 42,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
