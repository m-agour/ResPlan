#!/usr/bin/env python3
"""
Error Analysis for Task 2: Semantic Room Labeling

Generates:
  1. Confusion matrix for GraphSAGE
  2. Performance vs. plan complexity (# rooms)
  3. Per-unit-type performance (Apartment, BuilderFloor, Villa, IndependentHouse)
  4. Per-plan accuracy distribution (histogram)

Usage:
  python error_analysis.py
  python error_analysis.py --data ../ResPlan.pkl --split ../split.json --output results/error_analysis.json

Outputs:
  results/error_analysis.json
"""
import argparse, json, os, pickle, time, warnings
import numpy as np
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

ROOM_TYPES_5 = ["bedroom", "bathroom", "kitchen", "living", "balcony"]


def build_features(plan_graphs, feature_dim=11):
    """Build 11-dim node features and labels, same as task2_baselines.py."""
    import torch
    from torch_geometric.data import Data

    label_map = {t: i for i, t in enumerate(ROOM_TYPES_5)}
    data_list = []
    plan_ids = []

    for plan in plan_graphs:
        G = plan.get("graph")
        if G is None:
            continue

        room_nodes = [n for n in G.nodes if G.nodes[n].get("type", "") in ROOM_TYPES_5]
        if len(room_nodes) < 2:
            continue

        node_map = {n: i for i, n in enumerate(room_nodes)}
        n = len(room_nodes)

        # Node features (11-dim)
        feats = np.zeros((n, feature_dim), dtype=np.float32)
        labels = np.zeros(n, dtype=np.int64)

        for i, nd in enumerate(room_nodes):
            d = G.nodes[nd]
            geo = d.get("geometry")
            labels[i] = label_map[d["type"]]

            # Area
            feats[i, 0] = geo.area if geo else 0.0

            # Edge-type degrees
            for nbr in G.neighbors(nd):
                etype = G.edges[nd, nbr].get("type", "adjacency")
                if etype == "via_door":
                    feats[i, 1] += 1
                elif etype == "adjacency":
                    feats[i, 2] += 1
                elif etype == "via_window":
                    feats[i, 3] += 1
                elif etype == "direct":
                    feats[i, 4] += 1

            # Neighbour area stats
            nbr_areas = []
            for nbr in G.neighbors(nd):
                if nbr in node_map:
                    ng = G.nodes[nbr].get("geometry")
                    if ng:
                        nbr_areas.append(ng.area)
            if nbr_areas:
                feats[i, 5] = np.mean(nbr_areas)
                feats[i, 6] = np.min(nbr_areas)
                feats[i, 7] = np.max(nbr_areas)

            # Area ratio
            total_area = sum(
                G.nodes[m].get("geometry").area
                for m in room_nodes
                if G.nodes[m].get("geometry")
            )
            feats[i, 8] = feats[i, 0] / total_area if total_area > 0 else 0.0

            # Centroid
            if geo:
                feats[i, 9] = geo.centroid.x
                feats[i, 10] = geo.centroid.y

        # Build edges (undirected)
        src, dst = [], []
        for u, v in G.edges():
            if u in node_map and v in node_map:
                src.extend([node_map[u], node_map[v]])
                dst.extend([node_map[v], node_map[u]])

        if not src:
            continue

        data = Data(
            x=torch.tensor(feats),
            y=torch.tensor(labels),
            edge_index=torch.tensor([src, dst], dtype=torch.long),
        )
        data_list.append(data)
        plan_ids.append(plan.get("id", -1))

    return data_list, plan_ids


def main():
    parser = argparse.ArgumentParser(description="Error analysis for Task 2")
    parser.add_argument("--data", default="../ResPlan.pkl")
    parser.add_argument("--split", default="../split.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--output", default="results/error_analysis.json")
    args = parser.parse_args()

    import torch
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import SAGEConv

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load data
    print(f"Loading {args.data}...")
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    with open(args.split) as f:
        splits = json.load(f)

    # Build plan metadata index
    plan_meta = {}
    for p in data:
        pid = p.get("id", -1)
        G = p.get("graph")
        room_nodes = [n for n in G.nodes if G.nodes[n].get("type", "") in ROOM_TYPES_5] if G else []
        plan_meta[pid] = {
            "unitType": p.get("unitType", "unknown"),
            "n_rooms": len(room_nodes),
            "area": p.get("area", 0),
        }

    train_ids = {int(x) for x in splits["train"]}
    val_ids = {int(x) for x in splits["val"]}
    test_ids = {int(x) for x in splits["test"]}

    train_plans = [p for p in data if int(p.get("id", -1)) in train_ids]
    val_plans = [p for p in data if int(p.get("id", -1)) in val_ids]
    test_plans = [p for p in data if int(p.get("id", -1)) in test_ids]
    del data

    print(f"Plans: {len(train_plans)} train, {len(val_plans)} val, {len(test_plans)} test")

    # Build PyG data
    print("Building features...")
    train_data, train_pids = build_features(train_plans)
    val_data, val_pids = build_features(val_plans)
    test_data, test_pids = build_features(test_plans)
    print(f"Graphs: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Normalize features
    all_x = torch.cat([d.x for d in train_data], dim=0)
    mu = all_x.mean(0)
    std = all_x.std(0).clamp(min=1e-6)
    for d in train_data + val_data + test_data:
        d.x = (d.x - mu) / std

    # Class weights
    all_y = torch.cat([d.y for d in train_data])
    counts = torch.bincount(all_y, minlength=5).float()
    weights = (1.0 / counts.clamp(min=1)).to(device)
    weights /= weights.sum()

    # Train GraphSAGE
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    class GraphSAGE(torch.nn.Module):
        def __init__(self, in_dim, hidden, out_dim, layers=3, dropout=0.3):
            super().__init__()
            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(in_dim, hidden))
            for _ in range(layers - 2):
                self.convs.append(SAGEConv(hidden, hidden))
            self.convs.append(SAGEConv(hidden, out_dim))
            self.dropout = dropout

        def forward(self, x, edge_index):
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = torch.relu(x)
                x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            return self.convs[-1](x, edge_index)

    model = GraphSAGE(11, 128, 5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=512)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)  # batch_size=1 for per-plan analysis

    print(f"\nTraining GraphSAGE (seed={args.seed})...")
    best_val_acc = 0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 50 == 0 or epoch == args.epochs:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred = model(batch.x, batch.edge_index).argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
            val_acc = correct / total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if epoch % 100 == 0:
                print(f"  Epoch {epoch}: val_acc={val_acc:.4f} (best={best_val_acc:.4f})")

    model.load_state_dict(best_state)
    model.eval()

    # ── Per-plan evaluation ───────────────────────────────────────────────────
    print("\n=== Per-Plan Analysis ===")

    # Collect per-plan predictions
    plan_results = []
    confusion = np.zeros((5, 5), dtype=np.int64)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            pred = out.argmax(dim=1).cpu().numpy()
            true = batch.y.cpu().numpy()

            pid = test_pids[i]
            n_rooms = len(true)
            correct = (pred == true).sum()
            acc = correct / n_rooms

            # Update confusion matrix
            for t, p in zip(true, pred):
                confusion[t][p] += 1

            plan_results.append({
                "plan_id": int(pid),
                "n_rooms": n_rooms,
                "accuracy": float(acc),
                "n_correct": int(correct),
                "unitType": plan_meta.get(pid, {}).get("unitType", "unknown"),
            })

    # Overall test accuracy
    total_correct = sum(r["n_correct"] for r in plan_results)
    total_nodes = sum(r["n_rooms"] for r in plan_results)
    overall_acc = total_correct / total_nodes
    print(f"Overall test accuracy: {overall_acc:.4f}")

    # ── 1. Confusion Matrix ──────────────────────────────────────────────────
    print("\nConfusion Matrix (rows=true, cols=pred):")
    header = "            " + "  ".join(f"{t[:4]:>6}" for t in ROOM_TYPES_5)
    print(header)
    for i, t in enumerate(ROOM_TYPES_5):
        row = f"  {t:<10}" + "  ".join(f"{confusion[i][j]:>6}" for j in range(5))
        total = confusion[i].sum()
        recall = confusion[i][i] / total if total > 0 else 0
        row += f"  (recall={recall:.3f})"
        print(row)

    # Normalized confusion matrix
    conf_norm = confusion.astype(float)
    for i in range(5):
        s = conf_norm[i].sum()
        if s > 0:
            conf_norm[i] /= s

    # ── 2. Performance vs Plan Complexity ─────────────────────────────────────
    print("\nPerformance vs. Plan Complexity (# rooms):")
    complexity_bins = [(5, 7), (8, 9), (10, 12), (13, 16), (17, 40)]
    complexity_results = {}
    for lo, hi in complexity_bins:
        plans_in_bin = [r for r in plan_results if lo <= r["n_rooms"] <= hi]
        if plans_in_bin:
            accs = [r["accuracy"] for r in plans_in_bin]
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            n_plans = len(plans_in_bin)
            label = f"{lo}-{hi}" if lo != hi else str(lo)
            complexity_results[label] = {
                "mean_accuracy": float(mean_acc),
                "std_accuracy": float(std_acc),
                "n_plans": n_plans,
                "n_nodes": sum(r["n_rooms"] for r in plans_in_bin),
            }
            print(f"  {label:>6} rooms: acc={mean_acc:.4f}±{std_acc:.4f} ({n_plans} plans)")

    # ── 3. Per-Unit-Type Performance ──────────────────────────────────────────
    print("\nPerformance by Unit Type:")
    unit_type_results = {}
    for ut in ["Apartment", "BuilderFloor", "Villa", "IndependentHouse"]:
        plans_ut = [r for r in plan_results if r["unitType"] == ut]
        if plans_ut:
            accs = [r["accuracy"] for r in plans_ut]
            total_c = sum(r["n_correct"] for r in plans_ut)
            total_n = sum(r["n_rooms"] for r in plans_ut)
            node_acc = total_c / total_n if total_n > 0 else 0
            unit_type_results[ut] = {
                "plan_accuracy_mean": float(np.mean(accs)),
                "plan_accuracy_std": float(np.std(accs)),
                "node_accuracy": float(node_acc),
                "n_plans": len(plans_ut),
                "n_nodes": total_n,
                "mean_rooms": float(np.mean([r["n_rooms"] for r in plans_ut])),
            }
            print(f"  {ut:<18}: node_acc={node_acc:.4f}, plan_acc={np.mean(accs):.4f}±{np.std(accs):.4f} ({len(plans_ut)} plans, avg {np.mean([r['n_rooms'] for r in plans_ut]):.1f} rooms)")

    # ── 4. Per-Plan Accuracy Distribution ─────────────────────────────────────
    print("\nPer-Plan Accuracy Distribution:")
    accs = [r["accuracy"] for r in plan_results]
    print(f"  Mean: {np.mean(accs):.4f}")
    print(f"  Median: {np.median(accs):.4f}")
    print(f"  Std: {np.std(accs):.4f}")
    print(f"  Min: {np.min(accs):.4f}")
    print(f"  Max: {np.max(accs):.4f}")

    # Histogram bins
    hist_bins = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.001]
    hist_labels = ["<0.5", "0.5-0.7", "0.7-0.8", "0.8-0.9", "0.9-0.95", "0.95-1.0", "1.0"]
    hist_counts = np.histogram(accs, bins=hist_bins)[0]
    for label, cnt in zip(hist_labels, hist_counts):
        print(f"  {label:>10}: {cnt} plans ({100*cnt/len(accs):.1f}%)")

    # ── 5. Hardest/Easiest plans ──────────────────────────────────────────────
    plan_results.sort(key=lambda r: r["accuracy"])
    print("\n10 Hardest plans:")
    for r in plan_results[:10]:
        print(f"  Plan {r['plan_id']}: acc={r['accuracy']:.3f}, {r['n_rooms']} rooms, {r['unitType']}")
    print("\n10 Easiest plans (with 10+ rooms):")
    easy = [r for r in reversed(plan_results) if r["n_rooms"] >= 10]
    for r in easy[:10]:
        print(f"  Plan {r['plan_id']}: acc={r['accuracy']:.3f}, {r['n_rooms']} rooms, {r['unitType']}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "overall_accuracy": float(overall_acc),
        "confusion_matrix": {
            "labels": ROOM_TYPES_5,
            "matrix": confusion.tolist(),
            "normalized": conf_norm.tolist(),
        },
        "complexity": complexity_results,
        "unit_type": unit_type_results,
        "plan_accuracy_distribution": {
            "mean": float(np.mean(accs)),
            "median": float(np.median(accs)),
            "std": float(np.std(accs)),
            "min": float(np.min(accs)),
            "max": float(np.max(accs)),
            "histogram": {label: int(cnt) for label, cnt in zip(hist_labels, hist_counts)},
        },
        "_meta": {
            "model": "GraphSAGE (3L, 128H)",
            "seed": args.seed,
            "epochs": args.epochs,
            "n_test_plans": len(plan_results),
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
