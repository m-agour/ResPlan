#!/usr/bin/env python3
"""
Task 3: Learned Edge-Type Classification

Given ground-truth room adjacency edges, classify each edge into one of four
types: via_door, adjacency, via_window, direct.

Approach:
  1. Extract pairwise geometric + room-type features per edge
  2. Train classifiers (Random Forest, Gradient Boosting, MLP)
  3. Evaluate on held-out test edges
  4. Also evaluate in the pipeline setting: shared-boundary detection + classifier

Features per edge:
  - Geometric: centroid distance, shared boundary area/length at buffers 0/2/5,
    bbox overlap, min-distance, angle between centroids
  - Room: area of each room, area ratio, room-type pair (one-hot encoded)
  - Topology: degree of each room in the plan graph

Usage:
  python task3_edge_classifier.py
  python task3_edge_classifier.py --data ../ResPlan.pkl --split ../split.json --output results/task3_edge_type.json

Outputs:
  results/task3_edge_type.json
"""
import argparse, json, os, pickle, time, warnings
import numpy as np
from collections import Counter
from shapely.geometry import Polygon
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

ROOM_TYPES = ["bedroom", "bathroom", "kitchen", "living", "balcony", "front_door"]
EDGE_TYPES = ["via_door", "adjacency", "via_window", "direct"]


# ── Feature extraction ────────────────────────────────────────────────────────

def room_type_pair_features(t1, t2):
    """One-hot encode the ordered room-type pair (15 unique unordered pairs)."""
    # Alphabetical ordering for consistency
    types = sorted(ROOM_TYPES)
    pairs = []
    for i, a in enumerate(types):
        for j, b in enumerate(types):
            if j >= i:
                pairs.append((a, b))
    # Sort the pair alphabetically
    pair = tuple(sorted([t1, t2]))
    feat = [0.0] * len(pairs)
    if pair in pairs:
        feat[pairs.index(pair)] = 1.0
    return feat, [f"pair_{a}_{b}" for a, b in pairs]


def extract_edge_features(gu, gv, type_u, type_v, deg_u, deg_v):
    """Extract features for a single edge."""
    feat = {}

    # Centroid distance
    cu, cv = gu.centroid, gv.centroid
    dx = cv.x - cu.x
    dy = cv.y - cu.y
    dist = np.sqrt(dx ** 2 + dy ** 2)
    feat["centroid_dist"] = dist
    feat["centroid_dx"] = dx
    feat["centroid_dy"] = dy
    feat["centroid_angle"] = np.arctan2(dy, dx)

    # Room areas (ordered: smaller first)
    a1, a2 = min(gu.area, gv.area), max(gu.area, gv.area)
    feat["area_min"] = a1
    feat["area_max"] = a2
    feat["area_ratio"] = a1 / a2 if a2 > 0 else 0.0
    feat["area_sum"] = a1 + a2

    # Bounding box overlap
    b1, b2 = gu.bounds, gv.bounds  # (minx, miny, maxx, maxy)
    overlap_x = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
    overlap_y = max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
    feat["bbox_overlap_x"] = overlap_x
    feat["bbox_overlap_y"] = overlap_y
    feat["bbox_overlap_area"] = overlap_x * overlap_y

    # Width/height of rooms
    w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
    w2, h2 = b2[2] - b2[0], b2[3] - b2[1]
    feat["aspect_ratio_u"] = w1 / h1 if h1 > 0 else 1.0
    feat["aspect_ratio_v"] = w2 / h2 if h2 > 0 else 1.0

    # Shared boundary at different buffer sizes
    for buf_size in [0.0, 1.0, 2.0, 5.0]:
        try:
            if buf_size == 0:
                inter = gu.intersection(gv)
            else:
                inter = gu.buffer(buf_size).intersection(gv.buffer(buf_size))
            feat[f"shared_area_buf{buf_size:.0f}"] = inter.area
            feat[f"shared_len_buf{buf_size:.0f}"] = inter.length
        except Exception:
            feat[f"shared_area_buf{buf_size:.0f}"] = 0.0
            feat[f"shared_len_buf{buf_size:.0f}"] = 0.0

    # Min distance between geometries
    try:
        feat["min_distance"] = gu.distance(gv)
    except Exception:
        feat["min_distance"] = dist

    # Degree features
    feat["degree_u"] = deg_u
    feat["degree_v"] = deg_v
    feat["degree_min"] = min(deg_u, deg_v)
    feat["degree_max"] = max(deg_u, deg_v)

    # Room type pair (one-hot)
    pair_feat, pair_names = room_type_pair_features(type_u, type_v)
    for name, val in zip(pair_names, pair_feat):
        feat[name] = val

    return feat


def extract_plan_edges(plan, room_types=ROOM_TYPES):
    """Extract all GT edge features and labels from one plan."""
    G = plan.get("graph")
    if G is None:
        return [], []

    # Get room nodes
    nodes = {}
    for nd in G.nodes:
        d = G.nodes[nd]
        if d.get("type", "") in room_types:
            geo = d.get("geometry")
            if geo and not geo.is_empty:
                nodes[nd] = {
                    "type": d["type"],
                    "geometry": geo,
                    "degree": G.degree(nd),
                }

    features = []
    labels = []
    for u, v, d in G.edges(data=True):
        etype = d.get("type", "")
        if etype not in EDGE_TYPES:
            continue
        if u not in nodes or v not in nodes:
            continue

        nu, nv = nodes[u], nodes[v]
        feat = extract_edge_features(
            nu["geometry"], nv["geometry"],
            nu["type"], nv["type"],
            nu["degree"], nv["degree"],
        )
        features.append(feat)
        labels.append(EDGE_TYPES.index(etype))

    return features, labels


# ── Shared-boundary edge detection (from task3_plan2graph.py) ─────────────────

def detect_edges_shared_boundary(plan, buffer_size=2.0):
    """Detect edges using shared-boundary method, returning predicted edge pairs."""
    G = plan.get("graph")
    if G is None:
        return set(), {}

    nodes = {}
    for nd in G.nodes:
        d = G.nodes[nd]
        if d.get("type", "") in ROOM_TYPES:
            geo = d.get("geometry")
            if geo and not geo.is_empty:
                nodes[nd] = {
                    "type": d["type"],
                    "geometry": geo,
                    "degree": G.degree(nd),
                }

    detected_edges = set()
    edge_feats = {}
    node_ids = list(nodes.keys())
    n = len(node_ids)

    for i in range(n):
        for j in range(i + 1, n):
            gi = nodes[node_ids[i]]["geometry"]
            gj = nodes[node_ids[j]]["geometry"]
            try:
                if gi.buffer(buffer_size).intersects(gj.buffer(buffer_size)):
                    edge = frozenset([node_ids[i], node_ids[j]])
                    detected_edges.add(edge)

                    # Extract features for this detected edge
                    feat = extract_edge_features(
                        gi, gj,
                        nodes[node_ids[i]]["type"], nodes[node_ids[j]]["type"],
                        nodes[node_ids[i]]["degree"], nodes[node_ids[j]]["degree"],
                    )
                    edge_feats[edge] = feat
            except Exception:
                pass

    return detected_edges, edge_feats


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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Task 3: Learned Edge-Type Classifier")
    parser.add_argument("--data", default="../ResPlan.pkl")
    parser.add_argument("--split", default="../split.json")
    parser.add_argument("--buffer", type=float, default=2.0)
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs (-1 = all cores)")
    parser.add_argument("--output", default="results/task3_edge_type.json")
    args = parser.parse_args()

    print(f"Loading {args.data}...")
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    with open(args.split) as f:
        splits = json.load(f)

    train_ids = {int(x) for x in splits["train"]}
    val_ids = {int(x) for x in splits["val"]}
    test_ids = {int(x) for x in splits["test"]}

    train_plans = [p for p in data if int(p.get("id", -1)) in train_ids]
    val_plans = [p for p in data if int(p.get("id", -1)) in val_ids]
    test_plans = [p for p in data if int(p.get("id", -1)) in test_ids]

    print(f"Plans: {len(train_plans)} train, {len(val_plans)} val, {len(test_plans)} test")

    # ── Step 1: Extract features ──────────────────────────────────────────────
    print("\n=== Feature Extraction ===")
    t0 = time.time()

    def _extract(plans):
        results = Parallel(n_jobs=args.n_jobs, prefer="threads")(
            delayed(extract_plan_edges)(p) for p in plans
        )
        all_feats, all_labels = [], []
        for feats, labels in results:
            all_feats.extend(feats)
            all_labels.extend(labels)
        return all_feats, all_labels

    train_feats, train_labels = _extract(train_plans)
    val_feats, val_labels = _extract(val_plans)
    test_feats, test_labels = _extract(test_plans)

    print(f"Edges: {len(train_feats)} train, {len(val_feats)} val, {len(test_feats)} test")
    print(f"Feature extraction: {time.time()-t0:.1f}s")

    # Convert to arrays
    feature_names = sorted(train_feats[0].keys())
    n_features = len(feature_names)
    print(f"Features: {n_features}")

    def to_array(feats):
        return np.array([[f[k] for k in feature_names] for f in feats], dtype=np.float32)

    X_train = to_array(train_feats)
    y_train = np.array(train_labels)
    X_val = to_array(val_feats)
    y_val = np.array(val_labels)
    X_test = to_array(test_feats)
    y_test = np.array(test_labels)

    # Handle NaN/Inf
    for X in [X_train, X_val, X_test]:
        X[~np.isfinite(X)] = 0.0

    # Class distribution
    print("\nEdge type distribution (train):")
    for i, et in enumerate(EDGE_TYPES):
        n = (y_train == i).sum()
        print(f"  {et}: {n} ({100*n/len(y_train):.1f}%)")

    # ── Step 2: Train classifiers ─────────────────────────────────────────────
    print("\n=== Training Classifiers ===")
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from sklearn.preprocessing import StandardScaler

    # Scale features for MLP
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    classifiers = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=5,
            class_weight="balanced", n_jobs=args.n_jobs, random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            min_samples_leaf=10, random_state=42,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=500,
            early_stopping=True, validation_fraction=0.1,
            random_state=42,
        ),
    }

    results = {}
    best_clf = None
    best_acc = 0.0
    best_name = ""

    # Also compute majority-class baseline
    majority_class = Counter(y_train).most_common(1)[0][0]
    maj_acc_test = (y_test == majority_class).mean()
    maj_f1_test = f1_score(y_test, [majority_class] * len(y_test), average="macro")
    print(f"\nMajority class baseline: acc={maj_acc_test:.4f}, macro_f1={maj_f1_test:.4f}")
    results["majority_class"] = {
        "accuracy": float(maj_acc_test),
        "macro_f1": float(maj_f1_test),
        "class": EDGE_TYPES[majority_class],
    }

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        t1 = time.time()

        X_tr = X_train_s if "MLP" in name else X_train
        X_te = X_test_s if "MLP" in name else X_test
        X_va = X_val_s if "MLP" in name else X_val

        clf.fit(X_tr, y_train)
        t_train = time.time() - t1

        y_pred_val = clf.predict(X_va)
        y_pred_test = clf.predict(X_te)

        val_acc = accuracy_score(y_val, y_pred_val)
        test_acc = accuracy_score(y_test, y_pred_test)
        test_f1 = f1_score(y_test, y_pred_test, average="macro")
        test_wf1 = f1_score(y_test, y_pred_test, average="weighted")

        print(f"  Train: {t_train:.1f}s")
        print(f"  Val  acc: {val_acc:.4f}")
        print(f"  Test acc: {test_acc:.4f}, macro F1: {test_f1:.4f}, weighted F1: {test_wf1:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred_test, target_names=EDGE_TYPES, digits=3))

        # Per-class F1
        per_class_f1 = {}
        per_class = f1_score(y_test, y_pred_test, average=None)
        for i, et in enumerate(EDGE_TYPES):
            per_class_f1[et] = float(per_class[i])

        results[name] = {
            "accuracy": float(test_acc),
            "macro_f1": float(test_f1),
            "weighted_f1": float(test_wf1),
            "val_accuracy": float(val_acc),
            "per_class_f1": per_class_f1,
            "train_time_s": float(t_train),
        }

        if test_acc > best_acc:
            best_acc = test_acc
            best_clf = clf
            best_name = name

        # Feature importances (for tree-based)
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            top_k = 10
            idx = np.argsort(importances)[::-1][:top_k]
            print(f"\n  Top {top_k} features ({name}):")
            for rank, fi in enumerate(idx):
                print(f"    {rank+1}. {feature_names[fi]}: {importances[fi]:.4f}")
            results[name]["top_features"] = [
                {"name": feature_names[fi], "importance": float(importances[fi])}
                for fi in idx
            ]

    # ── Step 3: Pipeline evaluation ───────────────────────────────────────────
    print(f"\n=== Pipeline Evaluation (shared-boundary + {best_name}) ===")

    # For each test plan: detect edges → classify types → compare to GT
    pipeline_counters = {"tp": 0, "fp": 0, "fn": 0}
    pipeline_type_correct = 0
    pipeline_type_total = 0
    majority_type_correct = 0
    n_plans_eval = 0

    for plan in test_plans:
        gt_edges, gt_edge_types = get_gt_edges(plan)
        if not gt_edges:
            continue
        n_plans_eval += 1

        # Detect edges
        detected_edges, edge_feats = detect_edges_shared_boundary(plan, args.buffer)

        tp_edges = detected_edges & gt_edges
        pipeline_counters["tp"] += len(tp_edges)
        pipeline_counters["fp"] += len(detected_edges - gt_edges)
        pipeline_counters["fn"] += len(gt_edges - detected_edges)

        # Classify types on correctly detected edges (TP)
        if tp_edges and edge_feats:
            feat_list = []
            gt_types_list = []
            for edge in tp_edges:
                if edge in edge_feats:
                    feat_list.append(edge_feats[edge])
                    gt_types_list.append(EDGE_TYPES.index(gt_edge_types[edge]))

            if feat_list:
                X_pipe = np.array([[f[k] for k in feature_names] for f in feat_list], dtype=np.float32)
                X_pipe[~np.isfinite(X_pipe)] = 0.0

                if "MLP" in best_name:
                    X_pipe = scaler.transform(X_pipe)

                pred_types = best_clf.predict(X_pipe)
                correct = (pred_types == np.array(gt_types_list)).sum()
                pipeline_type_correct += correct
                pipeline_type_total += len(gt_types_list)

                # Majority baseline on same edges
                majority_type_correct += (np.array(gt_types_list) == majority_class).sum()

    # Pipeline metrics
    c = pipeline_counters
    prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0
    rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    pipe_type_acc = pipeline_type_correct / pipeline_type_total if pipeline_type_total > 0 else 0
    maj_type_acc = majority_type_correct / pipeline_type_total if pipeline_type_total > 0 else 0

    print(f"\nPipeline results ({n_plans_eval} plans):")
    print(f"  Edge detection: P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")
    print(f"  Type accuracy (majority):    {maj_type_acc:.4f}")
    print(f"  Type accuracy ({best_name}): {pipe_type_acc:.4f}")
    print(f"  Improvement:                 +{100*(pipe_type_acc - maj_type_acc):.1f}pp")

    results["pipeline"] = {
        "edge_detection": {"precision": float(prec), "recall": float(rec), "f1": float(f1)},
        "type_accuracy_majority": float(maj_type_acc),
        "type_accuracy_classifier": float(pipe_type_acc),
        "classifier": best_name,
        "improvement_pp": float(100 * (pipe_type_acc - maj_type_acc)),
        "n_typed_edges": pipeline_type_total,
    }

    # ── Save results ──────────────────────────────────────────────────────────
    results["_meta"] = {
        "n_features": n_features,
        "feature_names": feature_names,
        "edge_types": EDGE_TYPES,
        "n_train_edges": len(y_train),
        "n_val_edges": len(y_val),
        "n_test_edges": len(y_test),
        "buffer_size": args.buffer,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
