#!/usr/bin/env python3
"""
Runtime benchmarks and statistical tests for Task 2.

Measures:
  1. Training time for each method (DT, RF, GB, GCN, GraphSAGE)
  2. Inference time per plan
  3. Wilcoxon signed-rank test: GraphSAGE vs each other method (plan-level accuracy)

Usage:
  python runtime_and_stats.py --data ../ResPlan.pkl --split ../split.json

Outputs:
  results/runtime_stats.json
"""
import argparse, json, os, pickle, time, warnings
import numpy as np
from collections import Counter

warnings.filterwarnings("ignore")

ROOM_TYPES_5 = ["bedroom", "bathroom", "kitchen", "living", "balcony"]


def build_flat_features(plans):
    """Build 11-dim node features (flat, for sklearn)."""
    label_map = {t: i for i, t in enumerate(ROOM_TYPES_5)}
    X, y, plan_ids = [], [], []

    for plan in plans:
        G = plan.get("graph")
        if G is None:
            continue

        room_nodes = [n for n in G.nodes if G.nodes[n].get("type", "") in ROOM_TYPES_5]
        if len(room_nodes) < 2:
            continue

        node_map = {n: i for i, n in enumerate(room_nodes)}

        for nd in room_nodes:
            d = G.nodes[nd]
            geo = d.get("geometry")
            feat = np.zeros(11, dtype=np.float32)

            feat[0] = geo.area if geo else 0.0
            for nbr in G.neighbors(nd):
                etype = G.edges[nd, nbr].get("type", "adjacency")
                if etype == "via_door": feat[1] += 1
                elif etype == "adjacency": feat[2] += 1
                elif etype == "via_window": feat[3] += 1
                elif etype == "direct": feat[4] += 1

            nbr_areas = [G.nodes[nbr].get("geometry").area for nbr in G.neighbors(nd)
                         if nbr in node_map and G.nodes[nbr].get("geometry")]
            if nbr_areas:
                feat[5] = np.mean(nbr_areas)
                feat[6] = np.min(nbr_areas)
                feat[7] = np.max(nbr_areas)

            total_area = sum(G.nodes[m].get("geometry").area for m in room_nodes
                            if G.nodes[m].get("geometry"))
            feat[8] = feat[0] / total_area if total_area > 0 else 0.0
            if geo:
                feat[9] = geo.centroid.x
                feat[10] = geo.centroid.y

            X.append(feat)
            y.append(label_map[d["type"]])
            plan_ids.append(plan.get("id", -1))

    return np.array(X), np.array(y), plan_ids


def build_pyg_data(plans):
    """Build PyG data list."""
    import torch
    from torch_geometric.data import Data

    label_map = {t: i for i, t in enumerate(ROOM_TYPES_5)}
    data_list, plan_ids = [], []

    for plan in plans:
        G = plan.get("graph")
        if G is None:
            continue

        room_nodes = [n for n in G.nodes if G.nodes[n].get("type", "") in ROOM_TYPES_5]
        if len(room_nodes) < 2:
            continue

        node_map = {n: i for i, n in enumerate(room_nodes)}
        n = len(room_nodes)
        feats = np.zeros((n, 11), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int64)

        for i, nd in enumerate(room_nodes):
            d = G.nodes[nd]
            geo = d.get("geometry")
            labels[i] = label_map[d["type"]]
            feats[i, 0] = geo.area if geo else 0.0
            for nbr in G.neighbors(nd):
                etype = G.edges[nd, nbr].get("type", "adjacency")
                if etype == "via_door": feats[i, 1] += 1
                elif etype == "adjacency": feats[i, 2] += 1
                elif etype == "via_window": feats[i, 3] += 1
                elif etype == "direct": feats[i, 4] += 1
            nbr_areas = [G.nodes[nbr].get("geometry").area for nbr in G.neighbors(nd)
                         if nbr in node_map and G.nodes[nbr].get("geometry")]
            if nbr_areas:
                feats[i, 5] = np.mean(nbr_areas)
                feats[i, 6] = np.min(nbr_areas)
                feats[i, 7] = np.max(nbr_areas)
            total_area = sum(G.nodes[m].get("geometry").area for m in room_nodes
                             if G.nodes[m].get("geometry"))
            feats[i, 8] = feats[i, 0] / total_area if total_area > 0 else 0.0
            if geo:
                feats[i, 9] = geo.centroid.x
                feats[i, 10] = geo.centroid.y

        src, dst = [], []
        for u, v in G.edges():
            if u in node_map and v in node_map:
                src.extend([node_map[u], node_map[v]])
                dst.extend([node_map[v], node_map[u]])
        if not src:
            continue

        data = Data(
            x=torch.tensor(feats), y=torch.tensor(labels),
            edge_index=torch.tensor([src, dst], dtype=torch.long),
        )
        data_list.append(data)
        plan_ids.append(plan.get("id", -1))

    return data_list, plan_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../ResPlan.pkl")
    parser.add_argument("--split", default="../split.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/runtime_stats.json")
    args = parser.parse_args()

    import torch
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import SAGEConv, GCNConv
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from scipy import stats

    device = torch.device(args.device)
    print(f"Device: {device}")

    print(f"Loading {args.data}...")
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    with open(args.split) as f:
        splits = json.load(f)

    train_ids = {int(x) for x in splits["train"]}
    test_ids = {int(x) for x in splits["test"]}

    train_plans = [p for p in data if int(p.get("id", -1)) in train_ids]
    test_plans = [p for p in data if int(p.get("id", -1)) in test_ids]
    del data

    # ── Feature extraction ────────────────────────────────────────────────────
    print("Building features...")
    X_train, y_train, train_pids = build_flat_features(train_plans)
    X_test, y_test, test_pids = build_flat_features(test_plans)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # PyG data
    train_pyg, train_pyg_pids = build_pyg_data(train_plans)
    test_pyg, test_pyg_pids = build_pyg_data(test_plans)

    # Normalize for GNNs
    all_x = torch.cat([d.x for d in train_pyg], dim=0)
    mu = all_x.mean(0)
    std = all_x.std(0).clamp(min=1e-6)
    for d in train_pyg + test_pyg:
        d.x = (d.x - mu) / std

    # Class weights
    all_y = torch.cat([d.y for d in train_pyg])
    counts = torch.bincount(all_y, minlength=5).float()
    weights = (1.0 / counts.clamp(min=1)).to(device)
    weights /= weights.sum()

    results = {}

    # ── sklearn methods ───────────────────────────────────────────────────────
    sklearn_methods = {
        "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42),
    }

    plan_accs = {}  # method -> {plan_id: acc}

    for name, clf in sklearn_methods.items():
        print(f"\n{name}:")
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        y_pred = clf.predict(X_test)
        infer_time = time.time() - t0

        acc = (y_pred == y_test).mean()
        infer_per_plan = infer_time / len(test_plans) * 1000  # ms

        print(f"  Train: {train_time:.2f}s, Inference: {infer_time:.3f}s ({infer_per_plan:.2f}ms/plan)")
        print(f"  Accuracy: {acc:.4f}")

        results[name] = {
            "train_time_s": round(train_time, 2),
            "inference_time_s": round(infer_time, 3),
            "inference_ms_per_plan": round(infer_per_plan, 2),
            "accuracy": round(float(acc), 4),
        }

        # Per-plan accuracy
        plan_accs[name] = {}
        idx = 0
        for plan in test_plans:
            G = plan.get("graph")
            if G is None:
                continue
            room_nodes = [n for n in G.nodes if G.nodes[n].get("type", "") in ROOM_TYPES_5]
            if len(room_nodes) < 2:
                continue
            n = len(room_nodes)
            plan_pred = y_pred[idx:idx+n]
            plan_true = y_test[idx:idx+n]
            pid = plan.get("id", -1)
            plan_accs[name][pid] = (plan_pred == plan_true).mean()
            idx += n

    # ── GNN methods ───────────────────────────────────────────────────────────
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(11, 128)
            self.conv2 = GCNConv(128, 128)
            self.conv3 = GCNConv(128, 5)

        def forward(self, x, edge_index):
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.nn.functional.dropout(x, p=0.3, training=self.training)
            x = torch.relu(self.conv2(x, edge_index))
            x = torch.nn.functional.dropout(x, p=0.3, training=self.training)
            return self.conv3(x, edge_index)

    class GraphSAGE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = SAGEConv(11, 128)
            self.conv2 = SAGEConv(128, 128)
            self.conv3 = SAGEConv(128, 5)

        def forward(self, x, edge_index):
            x = torch.relu(self.conv1(x, edge_index))
            x = torch.nn.functional.dropout(x, p=0.3, training=self.training)
            x = torch.relu(self.conv2(x, edge_index))
            x = torch.nn.functional.dropout(x, p=0.3, training=self.training)
            return self.conv3(x, edge_index)

    gnn_methods = {"GCN": GCN, "GraphSAGE": GraphSAGE}

    train_loader = DataLoader(train_pyg, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_pyg, batch_size=1, shuffle=False)

    for name, ModelClass in gnn_methods.items():
        print(f"\n{name}:")
        torch.manual_seed(42)
        model = ModelClass().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        # Training
        t0 = time.time()
        for epoch in range(1, 501):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
            scheduler.step()
        train_time = time.time() - t0

        # Inference
        model.eval()
        t0 = time.time()
        plan_accs[name] = {}
        all_correct, all_total = 0, 0
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                batch = batch.to(device)
                pred = model(batch.x, batch.edge_index).argmax(dim=1)
                correct = (pred == batch.y).sum().item()
                total = batch.y.size(0)
                all_correct += correct
                all_total += total
                plan_accs[name][test_pyg_pids[i]] = correct / total

        infer_time = time.time() - t0
        acc = all_correct / all_total
        infer_per_plan = infer_time / len(test_pyg) * 1000

        print(f"  Train (500 epochs): {train_time:.1f}s, Inference: {infer_time:.3f}s ({infer_per_plan:.2f}ms/plan)")
        print(f"  Accuracy: {acc:.4f}")

        results[name] = {
            "train_time_s": round(train_time, 1),
            "inference_time_s": round(infer_time, 3),
            "inference_ms_per_plan": round(infer_per_plan, 2),
            "accuracy": round(float(acc), 4),
        }

    # ── Statistical Tests ─────────────────────────────────────────────────────
    print("\n=== Statistical Tests ===")
    stat_results = {}

    # Get common plan IDs across all methods
    common_ids = set(plan_accs["GraphSAGE"].keys())
    for method in plan_accs:
        common_ids &= set(plan_accs[method].keys())
    common_ids = sorted(common_ids)
    print(f"Common plans for comparison: {len(common_ids)}")

    sage_accs = np.array([plan_accs["GraphSAGE"][pid] for pid in common_ids])

    for method in ["Decision Tree", "Random Forest", "Gradient Boosting", "GCN"]:
        other_accs = np.array([plan_accs[method][pid] for pid in common_ids])
        diff = sage_accs - other_accs

        # Wilcoxon signed-rank test
        try:
            stat, p_value = stats.wilcoxon(diff[diff != 0])
        except ValueError:
            stat, p_value = 0.0, 1.0

        # Effect size (mean difference)
        mean_diff = diff.mean()
        n_sage_wins = (diff > 0).sum()
        n_ties = (diff == 0).sum()
        n_other_wins = (diff < 0).sum()

        sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "n.s."))

        print(f"\n  GraphSAGE vs {method}:")
        print(f"    Wilcoxon W={stat:.0f}, p={p_value:.2e} {sig}")
        print(f"    Mean diff: +{mean_diff:.4f}")
        print(f"    SAGE wins: {n_sage_wins}, ties: {n_ties}, other wins: {n_other_wins}")

        stat_results[f"GraphSAGE_vs_{method.replace(' ', '_')}"] = {
            "wilcoxon_W": float(stat),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "mean_difference": float(mean_diff),
            "sage_wins": int(n_sage_wins),
            "ties": int(n_ties),
            "other_wins": int(n_other_wins),
        }

    # Confidence intervals for GraphSAGE
    print("\n  GraphSAGE 95% CI:")
    n = len(sage_accs)
    mean = sage_accs.mean()
    se = sage_accs.std() / np.sqrt(n)
    ci_lo = mean - 1.96 * se
    ci_hi = mean + 1.96 * se
    print(f"    Plan-level accuracy: {mean:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    stat_results["GraphSAGE_CI"] = {
        "mean": float(mean),
        "se": float(se),
        "ci_95_lo": float(ci_lo),
        "ci_95_hi": float(ci_hi),
    }

    results["statistical_tests"] = stat_results

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
