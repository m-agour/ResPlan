#!/usr/bin/env python3
"""
Task 3: Plan-to-Graph Extraction — Baselines

Evaluates how well room adjacency graphs can be recovered from geometry:
  1. Proximity: connect rooms whose centroid distance ≤ per-plan median
  2. Shared boundary: connect rooms whose buffered polygons overlap
  3. Complete graph: connect all room pairs
  4. plan_to_graph (our pipeline): perfect by construction

Metrics: Edge P/R/F1, edge-type accuracy (majority-class baseline for type-unaware methods)

Usage:
  python task3_plan2graph.py
  python task3_plan2graph.py --data ../ResPlan.pkl --split ../split.json

Outputs:
  results/task3_results.json
"""
import argparse, json, os, pickle, time, warnings
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from collections import Counter

warnings.filterwarnings("ignore")

ROOM_TYPES = ["bedroom", "bathroom", "kitchen", "living", "balcony", "front_door"]


def get_gt_edges(plan):
    """Get ground truth edges and types from graph."""
    G = plan.get("graph")
    if G is None:
        return set(), {}
    edges = set()
    edge_types = {}
    for u, v, d in G.edges(data=True):
        tu = G.nodes[u].get("type", "")
        tv = G.nodes[v].get("type", "")
        if tu in ROOM_TYPES and tv in ROOM_TYPES:
            edge = frozenset([u, v])
            edges.add(edge)
            edge_types[edge] = d.get("type", "adjacency")
    return edges, edge_types


def get_room_nodes(plan):
    """Get room node IDs and their properties from graph."""
    G = plan.get("graph")
    if G is None:
        return {}
    nodes = {}
    for nd in G.nodes:
        d = G.nodes[nd]
        if d.get("type", "") in ROOM_TYPES:
            geo = d.get("geometry")
            if geo and not geo.is_empty:
                nodes[nd] = {
                    "type": d["type"],
                    "centroid": (geo.centroid.x, geo.centroid.y),
                    "geometry": geo,
                    "area": geo.area,
                }
    return nodes


def main():
    parser = argparse.ArgumentParser(description="Task 3: Plan-to-Graph baselines")
    parser.add_argument("--data", default="../ResPlan.pkl")
    parser.add_argument("--split", default="../split.json")
    parser.add_argument("--buffer", type=float, default=2.0, help="Buffer size for shared-boundary")
    parser.add_argument("--output", default="results/task3_results.json")
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    with open(args.split) as f:
        splits = json.load(f)
    test_ids = {int(x) for x in splits["test"]}
    test_plans = [p for p in data if int(p.get("id", -1)) in test_ids]
    del data
    print(f"Test plans: {len(test_plans)}")

    counters = {
        "proximity":        {"tp": 0, "fp": 0, "fn": 0},
        "shared_boundary":  {"tp": 0, "fp": 0, "fn": 0},
        "complete":         {"tp": 0, "fp": 0, "fn": 0},
    }
    all_gt_types = Counter()
    n_plans = 0

    t0 = time.time()
    for plan in test_plans:
        nodes = get_room_nodes(plan)
        if len(nodes) < 2:
            continue
        gt_edges, gt_types = get_gt_edges(plan)
        if not gt_edges:
            continue
        n_plans += 1
        all_gt_types.update(gt_types.values())

        node_ids = list(nodes.keys())
        n = len(node_ids)

        # Per-plan centroid distance threshold
        dists = []
        for i in range(n):
            for j in range(i + 1, n):
                ci = nodes[node_ids[i]]["centroid"]
                cj = nodes[node_ids[j]]["centroid"]
                d = np.sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)
                dists.append(d)
        threshold = np.median(dists) if dists else 0.0

        # 1. Proximity
        prox_edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                ci = nodes[node_ids[i]]["centroid"]
                cj = nodes[node_ids[j]]["centroid"]
                d = np.sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)
                if d <= threshold:
                    prox_edges.add(frozenset([node_ids[i], node_ids[j]]))
        tp = len(prox_edges & gt_edges)
        counters["proximity"]["tp"] += tp
        counters["proximity"]["fp"] += len(prox_edges - gt_edges)
        counters["proximity"]["fn"] += len(gt_edges - prox_edges)

        # 2. Shared boundary
        shared_edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                gi = nodes[node_ids[i]]["geometry"]
                gj = nodes[node_ids[j]]["geometry"]
                try:
                    if gi.buffer(args.buffer).intersects(gj.buffer(args.buffer)):
                        shared_edges.add(frozenset([node_ids[i], node_ids[j]]))
                except Exception:
                    pass
        tp = len(shared_edges & gt_edges)
        counters["shared_boundary"]["tp"] += tp
        counters["shared_boundary"]["fp"] += len(shared_edges - gt_edges)
        counters["shared_boundary"]["fn"] += len(gt_edges - shared_edges)

        # 3. Complete graph
        complete_edges = set()
        for i in range(n):
            for j in range(i + 1, n):
                complete_edges.add(frozenset([node_ids[i], node_ids[j]]))
        tp = len(complete_edges & gt_edges)
        counters["complete"]["tp"] += tp
        counters["complete"]["fp"] += len(complete_edges - gt_edges)
        counters["complete"]["fn"] += len(gt_edges - complete_edges)

    print(f"\nEvaluated {n_plans} test plans ({time.time()-t0:.1f}s)\n")

    # Compute P/R/F1
    results = {}
    print(f"{'Method':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Type Acc':>10}")
    print("-" * 62)

    total_gt = sum(all_gt_types.values())
    majority_type = all_gt_types.most_common(1)[0][0] if all_gt_types else "via_door"
    type_acc = all_gt_types[majority_type] / total_gt if total_gt > 0 else 0.0

    for method in ["proximity", "shared_boundary", "complete"]:
        c = counters[method]
        prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0
        rec  = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        results[method] = {
            "precision": float(prec), "recall": float(rec), "f1": float(f1),
            "type_acc": float(type_acc),
        }
        print(f"{method:<20} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {type_acc:>10.3f}")

    results["plan_to_graph"] = {
        "precision": 1.0, "recall": 1.0, "f1": 1.0, "type_acc": 1.0,
    }
    print(f"{'plan_to_graph (ours)':<20} {'1.000':>10} {'1.000':>10} {'1.000':>10} {'1.000':>10}")

    # Edge type distribution
    results["edge_type_distribution"] = {et: int(cnt) for et, cnt in all_gt_types.most_common()}
    results["total_test_edges"] = int(total_gt)
    results["majority_type"] = majority_type
    results["_meta"] = {"n_plans": n_plans, "buffer": args.buffer}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
