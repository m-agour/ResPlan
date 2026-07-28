#!/usr/bin/env python3
"""
Task 2: Feature & Architecture Ablation Study (GPU-enabled, Reproducible)

Part A — Feature Ablation (GraphSAGE 3-layer, 128-dim):
  1. All 11 features (baseline)
  2. 8 features (drop edge-type degrees → shows typed-edge value)
  3. 4 features (area + total-degree + area-ratio + centroid)
  4. 1 feature  (area only → lower bound on feature utility)
  5. 0 features (one-hot identity → pure graph structure)

Part B — Architecture Ablation (all 11 features):
  1. GraphSAGE: 1, 2, 3, 4 layers
  2. GraphSAGE hidden dim: 32, 64, 128, 256
  3. GAT  (Graph Attention Network, 3-layer, 128-dim)
  4. GIN  (Graph Isomorphism Network, 3-layer, 128-dim)

All experiments use 3 random seeds and report mean ± std.

Usage:
  python task2_ablation.py                     # auto-detect GPU
  python task2_ablation.py --device cuda        # force GPU
  python task2_ablation.py --part A             # feature ablation only
  python task2_ablation.py --part B             # architecture ablation only

Outputs:
  results/task2_ablation.json
"""
import argparse, json, os, pickle, sys, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GINConv
from torch_geometric.loader import DataLoader as PyGLoader
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────
CLASS_NAMES = ["bedroom", "bathroom", "kitchen", "living", "balcony"]
RESPLAN_MAP = {"bedroom": 0, "bathroom": 1, "kitchen": 2, "living": 3, "balcony": 4}
EDGE_TYPES  = ["via_door", "adjacency", "via_window", "direct"]
NUM_CLASSES = 5
DROPOUT     = 0.3
LR          = 0.01
WD          = 5e-4
BATCH_TRAIN = 256
BATCH_EVAL  = 512
EPOCHS      = 500

# Feature set definitions
# Full 11 features:
#   [0] area
#   [1] degree_via_door, [2] degree_adjacency, [3] degree_via_window, [4] degree_direct
#   [5] neigh_area_mean, [6] neigh_area_min, [7] neigh_area_max
#   [8] area_ratio, [9] cx, [10] cy

FEATURE_SETS = {
    "all_11":     list(range(11)),                      # full 11 features
    "no_etype_8": [0, 5, 6, 7, 8, 9, 10],              # drop edge-type degrees (7 feat)
    "basic_4":    [0, 8, 9, 10],                        # area, area_ratio, cx, cy
    "area_only":  [0],                                  # just area
    "structure":  [],                                   # empty → one-hot identity (pure graph)
}
# Note: "no_etype_8" is 7 features, but we also add total degree for fair comparison → 8 features
# We'll compute total degree on the fly


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Feature extraction (same as task2_baselines.py) ─────────────────────────
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
        areas,                         # [0]
        deg_by_type["via_door"],       # [1]
        deg_by_type["adjacency"],      # [2]
        deg_by_type["via_window"],     # [3]
        deg_by_type["direct"],         # [4]
        neigh_mean,                    # [5]
        neigh_min,                     # [6]
        neigh_max,                     # [7]
        areas / total_area,            # [8]
        cxs,                           # [9]
        cys,                           # [10]
    ], axis=1)

    # Also store total degree for the no-etype ablation
    total_deg = sum(deg_by_type[et] for et in EDGE_TYPES)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    data = Data(
        x=torch.tensor(feats, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long),
    )
    data.total_deg = torch.tensor(total_deg, dtype=torch.float).unsqueeze(1)
    data.n_nodes = n
    return data


def select_features(graphs, feat_name):
    """Return copies of graphs with only selected features."""
    new_gs = []
    for g in graphs:
        gn = g.clone()
        if feat_name == "structure":
            # Constant 1-dim feature (all ones): GNN learns purely from
            # graph topology via message passing, no node attributes.
            n = gn.x.size(0)
            gn.x = torch.ones(n, 1, dtype=torch.float)
        elif feat_name == "no_etype_8":
            # area, total_deg, neigh_mean/min/max, area_ratio, cx, cy = 8 features
            selected = torch.cat([
                gn.x[:, [0]],          # area
                gn.total_deg,           # total degree (1 dim)
                gn.x[:, [5, 6, 7]],    # neigh_mean, neigh_min, neigh_max
                gn.x[:, [8, 9, 10]],   # area_ratio, cx, cy
            ], dim=1)
            gn.x = selected
        else:
            indices = FEATURE_SETS[feat_name]
            gn.x = gn.x[:, indices]
        new_gs.append(gn)
    return new_gs


# ─── Flexible GNN Models ─────────────────────────────────────────────────────
class FlexGraphSAGE(nn.Module):
    """GraphSAGE with configurable layers and hidden dim."""
    def __init__(self, in_dim, hidden_dim=128, num_layers=3, out_dim=NUM_CLASSES):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList([
            SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, ei)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, DROPOUT, training=self.training)
        return self.convs[-1](x, ei)


class FlexGCN(nn.Module):
    """GCN with configurable layers and hidden dim."""
    def __init__(self, in_dim, hidden_dim=128, num_layers=3, out_dim=NUM_CLASSES):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList([
            GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, ei)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, DROPOUT, training=self.training)
        return self.convs[-1](x, ei)


class FlexGAT(nn.Module):
    """GAT with configurable layers and hidden dim."""
    def __init__(self, in_dim, hidden_dim=128, num_layers=3, out_dim=NUM_CLASSES,
                 heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        # First layer
        self.convs.append(GATConv(in_dim, hidden_dim // heads, heads=heads,
                                   dropout=DROPOUT))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads,
                                       heads=heads, dropout=DROPOUT))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        # Output layer (single head)
        self.convs.append(GATConv(hidden_dim, out_dim, heads=1, concat=False,
                                   dropout=DROPOUT))

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, ei)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, DROPOUT, training=self.training)
        return self.convs[-1](x, ei)


class FlexGIN(nn.Module):
    """GIN (Graph Isomorphism Network) with configurable layers."""
    def __init__(self, in_dim, hidden_dim=128, num_layers=3, out_dim=NUM_CLASSES):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        # First layer
        mlp0 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                              nn.Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(mlp0))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        # Middle layers
        for _ in range(num_layers - 2):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        # Output layer
        mlp_out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, out_dim))
        self.convs.append(GINConv(mlp_out))

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, ei)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, DROPOUT, training=self.training)
        return self.convs[-1](x, ei)


# ─── Training Loop ────────────────────────────────────────────────────────────
def train_and_eval(model, train_gs, val_gs, test_gs, device, seed,
                   epochs=EPOCHS, label=""):
    """Train a GNN model and return test metrics."""
    seed_everything(seed)

    # Compute normalization stats from training set
    all_x = torch.cat([g.x for g in train_gs], 0)
    gmean = all_x.mean(0)
    gstd  = all_x.std(0).clamp(min=1e-6)

    # Save raw features, apply normalization
    raw_feats = {id(g): g.x.clone() for g in train_gs + val_gs + test_gs}
    for g in train_gs + val_gs + test_gs:
        g.x = (g.x - gmean) / gstd

    # Class-balanced loss
    all_y = torch.cat([g.y for g in train_gs])
    cnt = torch.bincount(all_y, minlength=NUM_CLASSES).float()
    w = 1.0 / cnt.clamp(min=1)
    w = (w / w.sum() * NUM_CLASSES).to(device)

    model = model.to(device)
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
            val_acc = accuracy_score(torch.cat(trues).numpy(),
                                      torch.cat(preds).numpy())
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}

    # Evaluate on test
    if best_state:
        model.load_state_dict({k: v.to(device)
                               for k, v in best_state.items()})
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
    for g in train_gs + val_gs + test_gs:
        g.x = raw_feats[id(g)]

    return {"acc": acc, "macro_f1": mf1, "weighted_f1": wf1,
            "per_class_f1": {CLASS_NAMES[i]: float(pc[i])
                             for i in range(NUM_CLASSES)}}


def run_multi_seed(model_fn, train_gs, val_gs, test_gs, device, seeds, label=""):
    """Run model across multiple seeds, return aggregated results."""
    runs = []
    for seed in seeds:
        model = model_fn()
        r = train_and_eval(model, train_gs, val_gs, test_gs, device, seed,
                           label=f"{label}/s{seed}")
        runs.append(r)
        print(f"    seed={seed}  Acc={r['acc']:.4f}  MF1={r['macro_f1']:.4f}")

    accs = [r["acc"] for r in runs]
    mf1s = [r["macro_f1"] for r in runs]
    return {
        "runs": runs,
        "mean_acc": float(np.mean(accs)),
        "std_acc":  float(np.std(accs)),
        "mean_macro_f1": float(np.mean(mf1s)),
        "std_macro_f1":  float(np.std(mf1s)),
    }


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Task 2: Feature & Architecture Ablation")
    parser.add_argument("--data", default="../ResPlan.pkl")
    parser.add_argument("--split", default="../split.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--part", default="AB", help="A=feature, B=architecture, AB=both")
    parser.add_argument("--output", default="results/task2_ablation.json")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Memory: {mem:.1f} GB")

    # ── Load data ──
    t0 = time.time()
    print(f"\nLoading {args.data}...")
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    with open(args.split) as f:
        splits = json.load(f)
    train_ids = {int(x) for x in splits["train"]}
    val_ids   = {int(x) for x in splits["val"]}
    test_ids  = {int(x) for x in splits["test"]}

    all_graphs = {"train": [], "val": [], "test": []}
    for plan in data:
        g = build_graph(plan)
        if g is None:
            continue
        pid = int(plan.get("id", -1))
        if pid in train_ids:
            all_graphs["train"].append(g)
        elif pid in val_ids:
            all_graphs["val"].append(g)
        elif pid in test_ids:
            all_graphs["test"].append(g)
    del data
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Split: {len(all_graphs['train'])} train / "
          f"{len(all_graphs['val'])} val / {len(all_graphs['test'])} test")

    results = {}

    # ══════════════════════════════════════════════════════════════════════════
    # Part A: Feature Ablation (GraphSAGE 3-layer, 128-dim)
    # ══════════════════════════════════════════════════════════════════════════
    if "A" in args.part.upper():
        print("\n" + "=" * 70)
        print("PART A: Feature Ablation (GraphSAGE 3-layer, 128-dim)")
        print("=" * 70)

        feature_configs = [
            ("all_11",      "All 11 features"),
            ("no_etype_8",  "No edge-type degrees (8 feat)"),
            ("basic_4",     "Area + ratio + centroid (4 feat)"),
            ("area_only",   "Area only (1 feat)"),
            ("structure",   "Graph structure only (identity)"),
        ]

        results["feature_ablation"] = {}
        for feat_name, desc in feature_configs:
            print(f"\n  ── {desc} ({feat_name}) ──")
            # Select features
            train_fs = select_features(all_graphs["train"], feat_name)
            val_fs   = select_features(all_graphs["val"], feat_name)
            test_fs  = select_features(all_graphs["test"], feat_name)

            in_dim = train_fs[0].x.size(1)
            print(f"    Input dim: {in_dim}")

            def make_model(in_d=in_dim):
                return FlexGraphSAGE(in_d, hidden_dim=128, num_layers=3)

            r = run_multi_seed(make_model, train_fs, val_fs, test_fs,
                               device, args.seeds, label=feat_name)
            r["description"] = desc
            r["in_dim"] = in_dim
            results["feature_ablation"][feat_name] = r
            print(f"    → Acc={r['mean_acc']:.4f}±{r['std_acc']:.4f}  "
                  f"MF1={r['mean_macro_f1']:.4f}±{r['std_macro_f1']:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Part B: Architecture Ablation (all 11 features)
    # ══════════════════════════════════════════════════════════════════════════
    if "B" in args.part.upper():
        print("\n" + "=" * 70)
        print("PART B: Architecture Ablation (all 11 features)")
        print("=" * 70)

        # Use full-feature graphs
        train_gs = select_features(all_graphs["train"], "all_11")
        val_gs   = select_features(all_graphs["val"], "all_11")
        test_gs  = select_features(all_graphs["test"], "all_11")
        in_dim = 11

        # B.1: Layer depth ablation (GraphSAGE, 128-dim)
        print("\n  ── Layer Depth (GraphSAGE, 128-dim) ──")
        results["layer_ablation"] = {}
        for n_layers in [1, 2, 3, 4]:
            print(f"\n    {n_layers} layer(s):")
            def make_model(nl=n_layers):
                return FlexGraphSAGE(in_dim, hidden_dim=128, num_layers=nl)
            r = run_multi_seed(make_model, train_gs, val_gs, test_gs,
                               device, args.seeds, label=f"sage_{n_layers}L")
            r["num_layers"] = n_layers
            results["layer_ablation"][f"{n_layers}_layer"] = r
            print(f"    → Acc={r['mean_acc']:.4f}±{r['std_acc']:.4f}")

        # B.2: Hidden dimension ablation (GraphSAGE, 3-layer)
        print("\n  ── Hidden Dim (GraphSAGE, 3-layer) ──")
        results["hidden_dim_ablation"] = {}
        for hdim in [32, 64, 128, 256]:
            print(f"\n    hidden={hdim}:")
            def make_model(hd=hdim):
                return FlexGraphSAGE(in_dim, hidden_dim=hd, num_layers=3)
            r = run_multi_seed(make_model, train_gs, val_gs, test_gs,
                               device, args.seeds, label=f"sage_h{hdim}")
            r["hidden_dim"] = hdim
            results["hidden_dim_ablation"][f"h{hdim}"] = r
            print(f"    → Acc={r['mean_acc']:.4f}±{r['std_acc']:.4f}")

        # B.3: Architecture comparison (3-layer, 128-dim)
        print("\n  ── Architecture Comparison (3-layer, 128-dim) ──")
        results["architecture"] = {}

        arch_configs = [
            ("GraphSAGE", lambda: FlexGraphSAGE(in_dim, 128, 3)),
            ("GCN",       lambda: FlexGCN(in_dim, 128, 3)),
            ("GAT",       lambda: FlexGAT(in_dim, 128, 3)),
            ("GIN",       lambda: FlexGIN(in_dim, 128, 3)),
        ]

        for arch_name, model_fn in arch_configs:
            print(f"\n    {arch_name}:")
            r = run_multi_seed(model_fn, train_gs, val_gs, test_gs,
                               device, args.seeds, label=arch_name)
            r["architecture"] = arch_name
            results["architecture"][arch_name] = r
            print(f"    → Acc={r['mean_acc']:.4f}±{r['std_acc']:.4f}  "
                  f"MF1={r['mean_macro_f1']:.4f}±{r['std_macro_f1']:.4f}")

    # ── Save results ──
    results["_meta"] = {
        "device": str(device),
        "epochs": args.epochs,
        "seeds": args.seeds,
        "part": args.part,
        "torch_version": torch.__version__,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # ── Print summary tables ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if "feature_ablation" in results:
        print("\nFeature Ablation (GraphSAGE 3L/128):")
        print(f"  {'Features':<35} {'Dim':>4} {'Acc':>12} {'Macro F1':>12}")
        print("  " + "-" * 65)
        for k, v in results["feature_ablation"].items():
            print(f"  {v['description']:<35} {v['in_dim']:>4} "
                  f"{v['mean_acc']:.4f}±{v['std_acc']:.3f} "
                  f"{v['mean_macro_f1']:.4f}±{v['std_macro_f1']:.3f}")

    if "layer_ablation" in results:
        print("\nLayer Depth (GraphSAGE, 128-dim):")
        print(f"  {'Layers':>6} {'Acc':>12}")
        print("  " + "-" * 20)
        for k, v in results["layer_ablation"].items():
            print(f"  {v['num_layers']:>6} {v['mean_acc']:.4f}±{v['std_acc']:.3f}")

    if "hidden_dim_ablation" in results:
        print("\nHidden Dim (GraphSAGE, 3-layer):")
        print(f"  {'Dim':>6} {'Acc':>12}")
        print("  " + "-" * 20)
        for k, v in results["hidden_dim_ablation"].items():
            print(f"  {v['hidden_dim']:>6} {v['mean_acc']:.4f}±{v['std_acc']:.3f}")

    if "architecture" in results:
        print("\nArchitecture Comparison (3L/128, 11 features):")
        print(f"  {'Arch':<12} {'Acc':>12} {'Macro F1':>12}")
        print("  " + "-" * 38)
        for k, v in results["architecture"].items():
            print(f"  {k:<12} {v['mean_acc']:.4f}±{v['std_acc']:.3f} "
                  f"{v['mean_macro_f1']:.4f}±{v['std_macro_f1']:.3f}")

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()
