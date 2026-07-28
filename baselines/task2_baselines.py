#!/usr/bin/env python3
"""
Task 2: Semantic Room Labeling — All Baselines (Reproducible, GPU-enabled)

Methods:
  1. Rule-based (Decision Tree, depth 5)
  2. Random Forest (200 trees, depth 20)
  3. Gradient Boosting (200 trees, depth 6)
  4. GCN (3-layer, 128-dim, 500 epochs, 3 seeds)
  5. GraphSAGE (3-layer, 128-dim, 500 epochs, 3 seeds)

Features (11-dim per node):
  [area, degree_via_door, degree_adjacency, degree_via_window, degree_direct,
   neighbor_area_mean, neighbor_area_min, neighbor_area_max, area_ratio, cx, cy]

Usage:
  python task2_baselines.py                    # auto-detect GPU
  python task2_baselines.py --device cpu       # force CPU
  python task2_baselines.py --seeds 42 123 7 0 # custom seeds

Outputs:
  results/task2_results.json   — all metrics
  results/task2_summary.txt    — human-readable summary
"""
import argparse, json, os, pickle, sys, time, warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.loader import DataLoader as PyGLoader
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["bedroom", "bathroom", "kitchen", "living", "balcony"]
RESPLAN_MAP = {"bedroom": 0, "bathroom": 1, "kitchen": 2, "living": 3, "balcony": 4}
EDGE_TYPES  = ["via_door", "adjacency", "via_window", "direct"]
NUM_CLASSES = 5
HIDDEN_DIM  = 128
NUM_LAYERS  = 3
DROPOUT     = 0.3
LR          = 0.01
WD          = 5e-4
BATCH_TRAIN = 256
BATCH_EVAL  = 512
EPOCHS      = 500

# ─── Deterministic seeding ────────────────────────────────────────────────────
def seed_everything(seed: int):
    """Set all random seeds for full reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─── Feature extraction ──────────────────────────────────────────────────────
def build_graph(plan):
    """Convert a ResPlan plan dict into a PyG Data object with 11-dim features."""
    G = plan.get("graph")
    if G is None or len(G.nodes) < 2:
        return None
    valid_nodes = [nd for nd in G.nodes if G.nodes[nd].get("type", "") in RESPLAN_MAP]
    if len(valid_nodes) < 2:
        return None

    n = len(valid_nodes)
    node2idx = {nd: i for i, nd in enumerate(valid_nodes)}

    labels = []
    areas  = []
    cx_list, cy_list = [], []
    deg_by_type = {et: np.zeros(n) for et in EDGE_TYPES}

    for nd in valid_nodes:
        d = G.nodes[nd]
        labels.append(RESPLAN_MAP[d["type"]])
        areas.append(float(d.get("area", 1.0)))
        geo = d.get("geometry")
        if geo and not geo.is_empty:
            c = geo.centroid
            cx_list.append(c.x)
            cy_list.append(c.y)
        else:
            cx_list.append(0.0)
            cy_list.append(0.0)

    areas = np.array(areas)
    cxs = np.array(cx_list)
    cys = np.array(cy_list)

    edge_list = []
    for u, v, edata in G.edges(data=True):
        if u in node2idx and v in node2idx:
            i, j = node2idx[u], node2idx[v]
            edge_list.append([i, j])
            edge_list.append([j, i])
            et = edata.get("type", "adjacency")
            if et in deg_by_type:
                deg_by_type[et][i] += 1
                deg_by_type[et][j] += 1

    if not edge_list:
        for i in range(n):
            for j in range(i + 1, n):
                edge_list.append([i, j])
                edge_list.append([j, i])

    # Neighbor area statistics
    adj = [set() for _ in range(n)]
    for e in edge_list:
        adj[e[0]].add(e[1])
    neigh_mean = np.zeros(n)
    neigh_min  = np.zeros(n)
    neigh_max  = np.zeros(n)
    for i in range(n):
        if adj[i]:
            na = [areas[j] for j in adj[i]]
            neigh_mean[i] = np.mean(na)
            neigh_min[i]  = np.min(na)
            neigh_max[i]  = np.max(na)

    total_area = areas.sum() + 1e-8

    # Normalize centroids per plan
    all_coords = np.concatenate([cxs, cys])
    cmin, cmax = all_coords.min(), all_coords.max()
    if cmax > cmin:
        cxs = (cxs - cmin) / (cmax - cmin)
        cys = (cys - cmin) / (cmax - cmin)

    # Stack 11 features
    feats = np.stack([
        areas,
        deg_by_type["via_door"],
        deg_by_type["adjacency"],
        deg_by_type["via_window"],
        deg_by_type["direct"],
        neigh_mean,
        neigh_min,
        neigh_max,
        areas / total_area,
        cxs,
        cys,
    ], axis=1)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(
        x=torch.tensor(feats, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
    )


# ─── Models ──────────────────────────────────────────────────────────────────
class GCN3Layer(torch.nn.Module):
    def __init__(self, in_dim=11):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            GCNConv(in_dim, HIDDEN_DIM),
            GCNConv(HIDDEN_DIM, HIDDEN_DIM),
            GCNConv(HIDDEN_DIM, NUM_CLASSES),
        ])
        self.bns = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(HIDDEN_DIM),
            torch.nn.BatchNorm1d(HIDDEN_DIM),
        ])

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = F.dropout(F.relu(bn(conv(x, ei))), DROPOUT, training=self.training)
        return self.convs[-1](x, ei)


class GraphSAGE3Layer(torch.nn.Module):
    def __init__(self, in_dim=11):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            SAGEConv(in_dim, HIDDEN_DIM),
            SAGEConv(HIDDEN_DIM, HIDDEN_DIM),
            SAGEConv(HIDDEN_DIM, NUM_CLASSES),
        ])
        self.bns = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(HIDDEN_DIM),
            torch.nn.BatchNorm1d(HIDDEN_DIM),
        ])

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for conv, bn in zip(self.convs[:-1], self.bns):
            x = F.dropout(F.relu(bn(conv(x, ei))), DROPOUT, training=self.training)
        return self.convs[-1](x, ei)


# ─── GNN training loop ──────────────────────────────────────────────────────
def train_gnn(model_cls, train_gs, val_gs, test_gs, raw_train, raw_val, raw_test,
              gmean, gstd, device, seed, epochs=EPOCHS, name="GNN"):
    """Train a GNN model and return test metrics."""
    seed_everything(seed)

    # Normalize features (on device)
    for g, r in zip(train_gs, raw_train):
        g.x = (r - gmean) / gstd
    for g, r in zip(val_gs, raw_val):
        g.x = (r - gmean) / gstd
    for g, r in zip(test_gs, raw_test):
        g.x = (r - gmean) / gstd

    # Class-balanced loss weights
    all_y = torch.cat([g.y for g in train_gs])
    cnt = torch.bincount(all_y, minlength=NUM_CLASSES).float()
    w = 1.0 / cnt.clamp(min=1)
    w = (w / w.sum() * NUM_CLASSES).to(device)

    model = model_cls().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_loader = PyGLoader(train_gs, batch_size=BATCH_TRAIN, shuffle=True)
    val_loader   = PyGLoader(val_gs, batch_size=BATCH_EVAL)
    test_loader  = PyGLoader(test_gs, batch_size=BATCH_EVAL)

    best_val_acc = 0.0
    best_state   = None

    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(batch), batch.y, weight=w)
            loss.backward()
            opt.step()
        sch.step()

        if (ep + 1) % 10 == 0:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    preds.append(model(batch).argmax(1).cpu())
                    trues.append(batch.y.cpu())
            val_acc = accuracy_score(torch.cat(trues).numpy(), torch.cat(preds).numpy())
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Evaluate on test set
    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            preds.append(model(batch).argmax(1).cpu())
            trues.append(batch.y.cpu())
    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    acc = float(accuracy_score(trues, preds))
    mf1 = float(f1_score(trues, preds, average="macro"))
    wf1 = float(f1_score(trues, preds, average="weighted"))
    pc  = f1_score(trues, preds, average=None, labels=list(range(NUM_CLASSES)))

    # Restore raw features
    for g, r in zip(train_gs, raw_train):
        g.x = r.clone()
    for g, r in zip(val_gs, raw_val):
        g.x = r.clone()
    for g, r in zip(test_gs, raw_test):
        g.x = r.clone()

    print(f"  [{name}] seed={seed}  Acc={acc:.4f}  MF1={mf1:.4f}  WF1={wf1:.4f}")
    return {"acc": acc, "macro_f1": mf1, "weighted_f1": wf1,
            "per_class_f1": {CLASS_NAMES[i]: float(pc[i]) for i in range(NUM_CLASSES)}}


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Task 2: Semantic Room Labeling Baselines")
    parser.add_argument("--data", default="../ResPlan.pkl", help="Path to ResPlan.pkl")
    parser.add_argument("--split", default="../split.json", help="Path to split.json")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Device for GNN training")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7],
                        help="Random seeds for GNN training")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="GNN training epochs")
    parser.add_argument("--output", default="results/task2_results.json", help="Output JSON path")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load data ──
    t0 = time.time()
    print(f"Loading {args.data}...")
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    with open(args.split) as f:
        splits = json.load(f)
    train_ids = {int(x) for x in splits["train"]}
    val_ids   = {int(x) for x in splits["val"]}
    test_ids  = {int(x) for x in splits["test"]}

    train_gs, val_gs, test_gs = [], [], []
    for plan in data:
        g = build_graph(plan)
        if g is None:
            continue
        pid = int(plan.get("id", -1))
        if pid in train_ids:
            train_gs.append(g)
        elif pid in val_ids:
            val_gs.append(g)
        elif pid in test_ids:
            test_gs.append(g)
    del data
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Split: {len(train_gs)} train / {len(val_gs)} val / {len(test_gs)} test")

    # Compute global normalization stats (train set)
    raw_train = [g.x.clone() for g in train_gs]
    raw_val   = [g.x.clone() for g in val_gs]
    raw_test  = [g.x.clone() for g in test_gs]
    all_x = torch.cat(raw_train, 0)
    gmean = all_x.mean(0)
    gstd  = all_x.std(0).clamp(min=1e-6)

    # Flatten for sklearn
    def flatten(gs):
        X = torch.cat([g.x for g in gs], 0).numpy()
        y = torch.cat([g.y for g in gs], 0).numpy()
        return X, y

    X_train, y_train = flatten(train_gs)
    X_test, y_test   = flatten(test_gs)
    gmean_np = gmean.numpy()
    gstd_np  = gstd.numpy()
    X_train_n = (X_train - gmean_np) / gstd_np
    X_test_n  = (X_test - gmean_np) / gstd_np

    print(f"\nTest nodes: {len(y_test)}")
    print(f"Class dist: {dict(zip(CLASS_NAMES, np.bincount(y_test, minlength=5)))}")

    results = {}

    # ── 1. Rule-based (Decision Tree) ──
    print("\n=== Rule-based (DT depth 5) ===")
    seed_everything(42)
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train_n, y_train)
    p = dt.predict(X_test_n)
    pc = f1_score(y_test, p, average=None, labels=list(range(5)))
    results["rule_based"] = {
        "acc": float(accuracy_score(y_test, p)),
        "macro_f1": float(f1_score(y_test, p, average="macro")),
        "weighted_f1": float(f1_score(y_test, p, average="weighted")),
        "per_class_f1": {CLASS_NAMES[i]: float(pc[i]) for i in range(5)},
    }
    print(f"  Acc={results['rule_based']['acc']:.4f}  MF1={results['rule_based']['macro_f1']:.4f}")

    # ── 2. Random Forest ──
    print("\n=== Random Forest (200 trees, depth 20) ===")
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train_n, y_train)
    p = rf.predict(X_test_n)
    pc = f1_score(y_test, p, average=None, labels=list(range(5)))
    results["random_forest"] = {
        "acc": float(accuracy_score(y_test, p)),
        "macro_f1": float(f1_score(y_test, p, average="macro")),
        "weighted_f1": float(f1_score(y_test, p, average="weighted")),
        "per_class_f1": {CLASS_NAMES[i]: float(pc[i]) for i in range(5)},
    }
    print(f"  Acc={results['random_forest']['acc']:.4f}  MF1={results['random_forest']['macro_f1']:.4f}")

    # ── 3. Gradient Boosting ──
    print("\n=== Gradient Boosting (200 trees, depth 6) ===")
    gb = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=42)
    gb.fit(X_train_n, y_train)
    p = gb.predict(X_test_n)
    pc = f1_score(y_test, p, average=None, labels=list(range(5)))
    results["gradient_boosting"] = {
        "acc": float(accuracy_score(y_test, p)),
        "macro_f1": float(f1_score(y_test, p, average="macro")),
        "weighted_f1": float(f1_score(y_test, p, average="weighted")),
        "per_class_f1": {CLASS_NAMES[i]: float(pc[i]) for i in range(5)},
    }
    print(f"  Acc={results['gradient_boosting']['acc']:.4f}  MF1={results['gradient_boosting']['macro_f1']:.4f}")

    # ── 4. GCN (multi-seed) ──
    print(f"\n=== GCN (3-layer, 128-dim, {args.epochs} epochs, seeds={args.seeds}) ===")
    gcn_runs = []
    for seed in args.seeds:
        r = train_gnn(GCN3Layer, train_gs, val_gs, test_gs,
                       raw_train, raw_val, raw_test, gmean, gstd,
                       device, seed, epochs=args.epochs, name="GCN")
        gcn_runs.append(r)
    gcn_accs = [r["acc"] for r in gcn_runs]
    results["gcn"] = {
        "runs": gcn_runs,
        "mean_acc": float(np.mean(gcn_accs)),
        "std_acc":  float(np.std(gcn_accs)),
        "mean_macro_f1": float(np.mean([r["macro_f1"] for r in gcn_runs])),
        "std_macro_f1":  float(np.std([r["macro_f1"] for r in gcn_runs])),
        "seeds": args.seeds,
    }
    print(f"  GCN  MEAN: Acc={results['gcn']['mean_acc']:.4f}±{results['gcn']['std_acc']:.4f}")

    # ── 5. GraphSAGE (multi-seed) ──
    print(f"\n=== GraphSAGE (3-layer, 128-dim, {args.epochs} epochs, seeds={args.seeds}) ===")
    sage_runs = []
    for seed in args.seeds:
        r = train_gnn(GraphSAGE3Layer, train_gs, val_gs, test_gs,
                       raw_train, raw_val, raw_test, gmean, gstd,
                       device, seed, epochs=args.epochs, name="SAGE")
        sage_runs.append(r)
    sage_accs = [r["acc"] for r in sage_runs]
    results["graphsage"] = {
        "runs": sage_runs,
        "mean_acc": float(np.mean(sage_accs)),
        "std_acc":  float(np.std(sage_accs)),
        "mean_macro_f1": float(np.mean([r["macro_f1"] for r in sage_runs])),
        "std_macro_f1":  float(np.std([r["macro_f1"] for r in sage_runs])),
        "seeds": args.seeds,
    }
    print(f"  SAGE MEAN: Acc={results['graphsage']['mean_acc']:.4f}±{results['graphsage']['std_acc']:.4f}")

    # ── Save results ──
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    results["_meta"] = {
        "device": str(device),
        "epochs": args.epochs,
        "seeds": args.seeds,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_train": len(train_gs),
        "n_val": len(val_gs),
        "n_test": len(test_gs),
        "n_test_nodes": int(len(y_test)),
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # ── Human-readable summary ──
    summary_path = args.output.replace(".json", "_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Task 2: Semantic Room Labeling — Results Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Method':<25} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12}\n")
        f.write("-" * 60 + "\n")
        for name, key in [("Rule-based", "rule_based"), ("Random Forest", "random_forest"),
                          ("Gradient Boosting", "gradient_boosting")]:
            r = results[key]
            f.write(f"{name:<25} {r['acc']:>10.4f} {r['macro_f1']:>10.4f} {r['weighted_f1']:>12.4f}\n")
        for name, key in [("GCN", "gcn"), ("GraphSAGE", "graphsage")]:
            r = results[key]
            f.write(f"{name:<25} {r['mean_acc']:>10.4f}±{r['std_acc']:.3f} "
                    f"{r['mean_macro_f1']:>6.4f}±{r['std_macro_f1']:.3f}\n")
        f.write("\n\nPer-class F1:\n")
        f.write(f"{'Method':<25} " + " ".join(f"{c:>10}" for c in CLASS_NAMES) + "\n")
        f.write("-" * 80 + "\n")
        for name, key in [("Rule-based", "rule_based"), ("Random Forest", "random_forest"),
                          ("Gradient Boosting", "gradient_boosting")]:
            r = results[key]
            f.write(f"{name:<25} " + " ".join(f"{r['per_class_f1'][c]:>10.4f}" for c in CLASS_NAMES) + "\n")
        for name, key in [("GCN", "gcn"), ("GraphSAGE", "graphsage")]:
            # Average per-class across seeds
            for c in CLASS_NAMES:
                vals = [run["per_class_f1"][c] for run in results[key]["runs"]]
            f.write(f"{name:<25} ")
            for c in CLASS_NAMES:
                vals = [run["per_class_f1"][c] for run in results[key]["runs"]]
                f.write(f"{np.mean(vals):>10.4f}")
            f.write("\n")
    print(f"Summary saved to {summary_path}")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
