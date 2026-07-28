#!/usr/bin/env python3
"""
Task 2: Cross-Dataset Transfer (ResPlan ↔ RPLAN)

Trains GraphSAGE on one dataset and evaluates on the other using 8 common
features (area, total_degree, neighbor_area_mean/min/max, area_ratio, cx, cy).
RPLAN lacks typed edges, so edge-type degree features are excluded.

RPLAN data: Graph2Plan's data_train_converted.pkl
Category mapping: {0→living, 1→bedroom, 2→kitchen, 3→bathroom, 4→living,
                   5-8→bedroom, 9→balcony, 10→living, 11,12→excluded}

Usage:
  python task2_cross_dataset.py                             # auto GPU
  python task2_cross_dataset.py --device cpu                # force CPU
  python task2_cross_dataset.py --rplan PATH_TO_RPLAN.pkl   # custom RPLAN path

Outputs:
  results/task2_cross_dataset.json — all metrics
"""
import argparse, json, os, pickle, sys, time, warnings
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader as PyGLoader
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

CLASS_NAMES = ["bedroom", "bathroom", "kitchen", "living", "balcony"]
RPLAN_MAP   = {0: 3, 1: 0, 2: 2, 3: 1, 4: 3, 5: 0, 6: 0, 7: 0, 8: 0,
               9: 4, 10: 3, 11: -1, 12: -1}
RESPLAN_MAP = {"bedroom": 0, "bathroom": 1, "kitchen": 2, "living": 3, "balcony": 4}
HIDDEN_DIM  = 128
NUM_CLASSES = 5
DROPOUT     = 0.3
EPOCHS_DEFAULT = 300

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_features_8dim(areas, edge_list, n, cxs, cys):
    """Build 8-dim feature vectors (common features for cross-dataset)."""
    total_area = areas.sum() + 1e-8

    # Total degree (1 per undirected edge per endpoint)
    deg = np.zeros(n)
    seen = set()
    for e in edge_list:
        edge_key = (min(e[0], e[1]), max(e[0], e[1]))
        if edge_key not in seen:
            seen.add(edge_key)
            deg[e[0]] += 1
            deg[e[1]] += 1

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

    return np.stack([areas, deg, neigh_mean, neigh_min, neigh_max,
                     areas / total_area, cxs, cys], axis=1)


class FlexSAGE(torch.nn.Module):
    def __init__(self, in_dim=8):
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


def load_rplan(path):
    """Load RPLAN data from Graph2Plan PKL and convert to PyG graphs."""
    print(f"Loading RPLAN from {path}...")
    with open(path, "rb") as f:
        rplan_raw = pickle.load(f, encoding="latin1")

    graphs = []
    for sample in rplan_raw["data"]:
        boxes = sample.box
        edges_raw = sample.edge
        nr = len(boxes)
        labels = []
        valid  = []
        for i in range(nr):
            m = RPLAN_MAP.get(int(boxes[i, 4]), -1)
            labels.append(m)
            valid.append(m >= 0)
        if sum(valid) < 2:
            continue

        old2new = {}
        new_labels = []
        new_boxes  = []
        idx = 0
        for i in range(nr):
            if valid[i]:
                old2new[i] = idx
                new_labels.append(labels[i])
                new_boxes.append(boxes[i, :4].astype(float))
                idx += 1

        n = len(new_labels)
        nb = np.array(new_boxes)
        el = []
        for e in range(len(edges_raw)):
            i_e = int(edges_raw[e, 0])
            j_e = int(edges_raw[e, 1])
            if i_e in old2new and j_e in old2new:
                ni, nj = old2new[i_e], old2new[j_e]
                el.append([ni, nj])
                el.append([nj, ni])
        if not el:
            for i in range(n):
                for j in range(i + 1, n):
                    el.append([i, j])
                    el.append([j, i])

        areas = (nb[:, 2] - nb[:, 0]) * (nb[:, 3] - nb[:, 1])
        cxs = (nb[:, 0] + nb[:, 2]) / 2.0
        cys = (nb[:, 1] + nb[:, 3]) / 2.0
        ac = nb.flatten()
        cmin, cmax = ac.min(), ac.max()
        if cmax > cmin:
            cxs = (cxs - cmin) / (cmax - cmin)
            cys = (cys - cmin) / (cmax - cmin)

        feats = build_features_8dim(areas, el, n, cxs, cys)
        ei = torch.tensor(el, dtype=torch.long).t().contiguous()
        graphs.append(Data(
            x=torch.tensor(feats, dtype=torch.float),
            edge_index=ei,
            y=torch.tensor(new_labels, dtype=torch.long),
        ))
    del rplan_raw
    print(f"  Valid RPLAN graphs: {len(graphs)}")
    return graphs


def load_resplan(data_path, split_path):
    """Load ResPlan data and return train/val/test graphs with 8-dim features."""
    print(f"Loading ResPlan from {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    with open(split_path) as f:
        splits = json.load(f)
    train_ids = set(splits["train"])
    val_ids   = set(splits["val"])
    test_ids  = set(splits["test"])

    train_gs, val_gs, test_gs = [], [], []
    for plan in data:
        G = plan.get("graph")
        if G is None or len(G.nodes) < 2:
            continue
        vn = [nd for nd in G.nodes if G.nodes[nd].get("type", "") in RESPLAN_MAP]
        if len(vn) < 2:
            continue

        n = len(vn)
        n2i = {nd: i for i, nd in enumerate(vn)}
        labels, areas, cxl, cyl = [], [], [], []
        for nd in vn:
            d = G.nodes[nd]
            labels.append(RESPLAN_MAP[d["type"]])
            areas.append(float(d.get("area", 1.0)))
            geo = d.get("geometry")
            if geo and not geo.is_empty:
                c = geo.centroid
                cxl.append(c.x)
                cyl.append(c.y)
            else:
                cxl.append(0.0)
                cyl.append(0.0)
        areas = np.array(areas)
        cxs = np.array(cxl)
        cys = np.array(cyl)
        ac = np.concatenate([cxs, cys])
        cmin, cmax = ac.min(), ac.max()
        if cmax > cmin:
            cxs = (cxs - cmin) / (cmax - cmin)
            cys = (cys - cmin) / (cmax - cmin)

        el = []
        for u, v in G.edges():
            if u in n2i and v in n2i:
                i, j = n2i[u], n2i[v]
                el.append([i, j])
                el.append([j, i])
        if not el:
            for i in range(n):
                for j in range(i + 1, n):
                    el.append([i, j])
                    el.append([j, i])

        feats = build_features_8dim(areas, el, n, cxs, cys)
        ei = torch.tensor(el, dtype=torch.long).t().contiguous()
        g = Data(
            x=torch.tensor(feats, dtype=torch.float),
            edge_index=ei,
            y=torch.tensor(labels, dtype=torch.long),
        )
        pid = plan.get("id")
        if pid in train_ids:
            train_gs.append(g)
        elif pid in val_ids:
            val_gs.append(g)
        elif pid in test_ids:
            test_gs.append(g)
    del data
    print(f"  Split: {len(train_gs)}/{len(val_gs)}/{len(test_gs)}")
    return train_gs, val_gs, test_gs


def train_eval(train_gs, val_gs, test_gs, device, label, seed=42, epochs=300):
    """Train GraphSAGE and evaluate. Returns dict of metrics."""
    seed_everything(seed)

    # Normalize from train set
    ax = torch.cat([g.x for g in train_gs], 0)
    m = ax.mean(0)
    s = ax.std(0).clamp(min=1e-6)

    tr_orig = [g.x.clone() for g in train_gs]
    va_orig = [g.x.clone() for g in val_gs]
    te_orig = [g.x.clone() for g in test_gs]
    for g in train_gs:
        g.x = (g.x - m) / s
    for g in val_gs:
        g.x = (g.x - m) / s
    for g in test_gs:
        g.x = (g.x - m) / s

    # Class-balanced loss
    all_y = torch.cat([g.y for g in train_gs])
    cnt = torch.bincount(all_y, minlength=5).float()
    w = 1.0 / cnt.clamp(min=1)
    w = (w / w.sum() * 5).to(device)

    model = FlexSAGE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_va = 0.0
    best_st = None
    for ep in range(epochs):
        model.train()
        for b in PyGLoader(train_gs, batch_size=256, shuffle=True):
            b = b.to(device)
            opt.zero_grad()
            F.cross_entropy(model(b), b.y, weight=w).backward()
            opt.step()
        sch.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            ps, ts = [], []
            with torch.no_grad():
                for b in PyGLoader(val_gs, batch_size=512):
                    b = b.to(device)
                    ps.append(model(b).argmax(1).cpu())
                    ts.append(b.y.cpu())
            va_a = accuracy_score(torch.cat(ts).numpy(), torch.cat(ps).numpy())
            if va_a > best_va:
                best_va = va_a
                best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_st:
        model.load_state_dict({k: v.to(device) for k, v in best_st.items()})
    model.eval()
    ps, ts = [], []
    with torch.no_grad():
        for b in PyGLoader(test_gs, batch_size=512):
            b = b.to(device)
            ps.append(model(b).argmax(1).cpu())
            ts.append(b.y.cpu())
    ps = torch.cat(ps).numpy()
    ts = torch.cat(ts).numpy()

    acc = float(accuracy_score(ts, ps))
    mf1 = float(f1_score(ts, ps, average="macro"))
    wf1 = float(f1_score(ts, ps, average="weighted"))
    pc  = f1_score(ts, ps, average=None, labels=list(range(5)))
    print(f"  [{label}] Acc={acc:.3f}  MF1={mf1:.3f}  WF1={wf1:.3f}")
    print(f"    Per-class: " + ", ".join(f"{CLASS_NAMES[i]}={pc[i]:.3f}" for i in range(5)))

    # Restore
    for g, ox in zip(train_gs, tr_orig):
        g.x = ox
    for g, ox in zip(val_gs, va_orig):
        g.x = ox
    for g, ox in zip(test_gs, te_orig):
        g.x = ox

    return {
        "acc": acc, "macro_f1": mf1, "weighted_f1": wf1,
        "per_class_f1": {CLASS_NAMES[i]: float(pc[i]) for i in range(5)},
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset transfer: ResPlan ↔ RPLAN")
    parser.add_argument("--data", default="../ResPlan.pkl")
    parser.add_argument("--split", default="../split.json")
    parser.add_argument("--rplan", default="../../rplan_data/Interface/static/Data/data_train_converted.pkl",
                        help="Path to RPLAN Graph2Plan PKL")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    parser.add_argument("--output", default="results/task2_cross_dataset.json")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load datasets
    resplan_train, resplan_val, resplan_test = load_resplan(args.data, args.split)

    rplan_all = load_rplan(args.rplan)
    seed_everything(args.seed)
    perm = np.random.permutation(len(rplan_all))
    nt = int(0.8 * len(perm))
    nv = int(0.1 * len(perm))
    rplan_train = [rplan_all[i] for i in perm[:nt]]
    rplan_val   = [rplan_all[i] for i in perm[nt:nt + nv]]
    rplan_test  = [rplan_all[i] for i in perm[nt + nv:]]
    del rplan_all
    print(f"  RPLAN split: {len(rplan_train)}/{len(rplan_val)}/{len(rplan_test)}")

    results = {}

    print("\n===== EXP 1: ResPlan → ResPlan =====")
    results["resplan_resplan"] = train_eval(
        resplan_train, resplan_val, resplan_test, device, "ResPlan→ResPlan",
        seed=args.seed, epochs=args.epochs)

    print("\n===== EXP 2: RPLAN → RPLAN =====")
    results["rplan_rplan"] = train_eval(
        rplan_train, rplan_val, rplan_test, device, "RPLAN→RPLAN",
        seed=args.seed, epochs=args.epochs)

    print("\n===== EXP 3: RPLAN → ResPlan =====")
    results["rplan_resplan"] = train_eval(
        rplan_train, rplan_val, resplan_test, device, "RPLAN→ResPlan",
        seed=args.seed, epochs=args.epochs)

    print("\n===== EXP 4: ResPlan → RPLAN =====")
    results["resplan_rplan"] = train_eval(
        resplan_train, resplan_val, rplan_test, device, "ResPlan→RPLAN",
        seed=args.seed, epochs=args.epochs)

    # Summary
    r1 = results["resplan_resplan"]
    r2 = results["rplan_rplan"]
    r3 = results["rplan_resplan"]
    r4 = results["resplan_rplan"]
    print(f"\n{'='*60}")
    print(f"{'Setting':<30} {'Acc':>7} {'MF1':>7}")
    print(f"{'-'*45}")
    for lbl, r in [("ResPlan→ResPlan", r1), ("RPLAN→RPLAN", r2),
                   ("RPLAN→ResPlan", r3), ("ResPlan→RPLAN", r4)]:
        print(f"{lbl:<30} {r['acc']:>7.3f} {r['macro_f1']:>7.3f}")
    print(f"\nGap RPLAN→ResPlan: {r1['acc']-r3['acc']:+.3f} acc")
    print(f"Gap ResPlan→RPLAN: {r2['acc']-r4['acc']:+.3f} acc")

    results["_meta"] = {
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "torch_version": torch.__version__,
        "rplan_category_mapping": RPLAN_MAP,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
