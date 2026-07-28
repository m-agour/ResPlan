#!/usr/bin/env python3
"""
Task 2: Domain Adaptation (RPLAN → ResPlan fine-tuning)

Pre-train GraphSAGE on RPLAN, then fine-tune on varying fractions of ResPlan
training data (0%, 1%, 5%, 10%, 50%, 100%), evaluating on ResPlan test set.

Demonstrates:
  1. How quickly the cross-dataset gap closes with fine-tuning
  2. Practical value of pre-training on RPLAN for low-data ResPlan scenarios
  3. Comparison with training from scratch on same data fractions

Uses 8 common features (no edge-type degrees) for cross-dataset compatibility.

Usage:
  python task2_domain_adapt.py --device auto
  python task2_domain_adapt.py --data ../ResPlan.pkl --split ../split.json

Outputs:
  results/task2_domain_adapt.json
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
RPLAN_MAP = {0: 3, 1: 0, 2: 2, 3: 1, 4: 3, 5: 0, 6: 0, 7: 0, 8: 0,
             9: 4, 10: 3, 11: -1, 12: -1}
RESPLAN_MAP = {"bedroom": 0, "bathroom": 1, "kitchen": 2, "living": 3, "balcony": 4}
HIDDEN_DIM = 128
NUM_CLASSES = 5
DROPOUT = 0.3


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_features_8dim(areas, edge_list, n, cxs, cys):
    """Build 8-dim feature vectors (area, total_deg, neigh stats, ratio, cx, cy)."""
    total_area = areas.sum() + 1e-8
    deg = np.zeros(n)
    seen = set()
    for e in edge_list:
        edge_key = (min(e[0], e[1]), max(e[0], e[1]))
        if edge_key not in seen:
            seen.add(edge_key)
            deg[e[0]] += 1
            deg[e[1]] += 1

    adj = [set() for _ in range(n)]
    for e in edge_list:
        adj[e[0]].add(e[1])
    neigh_mean = np.zeros(n)
    neigh_min = np.zeros(n)
    neigh_max = np.zeros(n)
    for i in range(n):
        if adj[i]:
            na = [areas[j] for j in adj[i]]
            neigh_mean[i] = np.mean(na)
            neigh_min[i] = np.min(na)
            neigh_max[i] = np.max(na)

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
    """Load RPLAN data and convert to PyG graphs with 8 common features."""
    print(f"Loading RPLAN from {path}...")
    with open(path, "rb") as f:
        rplan_raw = pickle.load(f, encoding="latin1")

    graphs = []
    for sample in rplan_raw["data"]:
        boxes = sample.box
        edges_raw = sample.edge
        nr = len(boxes)
        labels, valid = [], []
        for i in range(nr):
            m = RPLAN_MAP.get(int(boxes[i, 4]), -1)
            labels.append(m)
            valid.append(m >= 0)
        if sum(valid) < 2:
            continue

        old2new, new_labels, new_boxes = {}, [], []
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
            i_e, j_e = int(edges_raw[e, 0]), int(edges_raw[e, 1])
            if i_e in old2new and j_e in old2new:
                ni, nj = old2new[i_e], old2new[j_e]
                el.append([ni, nj])
                el.append([nj, ni])
        if not el:
            for i in range(n):
                for j in range(i + 1, n):
                    el.extend([[i, j], [j, i]])

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
    """Load ResPlan and return train/val/test with 8-dim features."""
    print(f"Loading ResPlan from {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    with open(split_path) as f:
        splits = json.load(f)
    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])

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
        areas_arr = np.array(areas)
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
                el.extend([[i, j], [j, i]])
        if not el:
            for i in range(n):
                for j in range(i + 1, n):
                    el.extend([[i, j], [j, i]])

        feats = build_features_8dim(areas_arr, el, n, cxs, cys)
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


def normalize_graphs(train_gs, *other_gs_list):
    """Normalize features based on training set statistics."""
    ax = torch.cat([g.x for g in train_gs], 0)
    mu = ax.mean(0)
    std = ax.std(0).clamp(min=1e-6)
    for g in train_gs:
        g.x = (g.x - mu) / std
    for gs in other_gs_list:
        for g in gs:
            g.x = (g.x - mu) / std
    return mu, std


def evaluate(model, graphs, device):
    """Evaluate model on a list of graphs."""
    model.eval()
    all_pred, all_true = [], []
    loader = PyGLoader(graphs, batch_size=512, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).argmax(dim=1)
            all_pred.append(pred.cpu())
            all_true.append(batch.y.cpu())
    all_pred = torch.cat(all_pred).numpy()
    all_true = torch.cat(all_true).numpy()
    acc = accuracy_score(all_true, all_pred)
    mf1 = f1_score(all_true, all_pred, average="macro")
    return acc, mf1


def train_model(model, train_gs, val_gs, device, epochs, lr=0.01):
    """Train model, return best state dict."""
    all_y = torch.cat([g.y for g in train_gs])
    cnt = torch.bincount(all_y, minlength=5).float()
    w = 1.0 / cnt.clamp(min=1)
    w = (w / w.sum() * 5).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = PyGLoader(train_gs, batch_size=256, shuffle=True)

    best_val_acc = 0.0
    best_state = None
    for ep in range(1, epochs + 1):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            F.cross_entropy(model(batch), batch.y, weight=w).backward()
            opt.step()
        sch.step()

        if ep % 50 == 0 or ep == epochs:
            val_acc, _ = evaluate(model, val_gs, device)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Domain Adaptation: RPLAN → ResPlan")
    parser.add_argument("--data", default="../ResPlan.pkl")
    parser.add_argument("--split", default="../split.json")
    parser.add_argument("--rplan", default="../../rplan_data/Interface/static/Data/data_train_converted.pkl")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pretrain-epochs", type=int, default=300)
    parser.add_argument("--finetune-epochs", type=int, default=200)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--output", default="results/task2_domain_adapt.json")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load datasets
    rplan_all = load_rplan(args.rplan)
    rp_train, rp_val, rp_test = load_resplan(args.data, args.split)

    # Split RPLAN into train/val (90/10)
    n_rplan = len(rplan_all)
    np.random.seed(42)
    idx = np.random.permutation(n_rplan)
    n_val = n_rplan // 10
    rplan_val = [rplan_all[i] for i in idx[:n_val]]
    rplan_train = [rplan_all[i] for i in idx[n_val:]]
    print(f"RPLAN split: {len(rplan_train)} train, {len(rplan_val)} val")

    fractions = [0.0, 0.01, 0.05, 0.10, 0.50, 1.00]
    results = {}

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}")
        print(f"{'='*60}")
        seed_everything(seed)

        # ── Pre-train on RPLAN ────────────────────────────────────────────
        print("\n--- Pre-training on RPLAN ---")
        # Normalize RPLAN features
        rplan_train_copy = [Data(x=g.x.clone(), edge_index=g.edge_index, y=g.y) for g in rplan_train]
        rplan_val_copy = [Data(x=g.x.clone(), edge_index=g.edge_index, y=g.y) for g in rplan_val]
        normalize_graphs(rplan_train_copy, rplan_val_copy)

        model = FlexSAGE().to(device)
        pretrained_state, val_acc = train_model(
            model, rplan_train_copy, rplan_val_copy, device, args.pretrain_epochs
        )
        print(f"  RPLAN val acc: {val_acc:.4f}")

        # ── Fine-tune on ResPlan fractions ────────────────────────────────
        for frac in fractions:
            print(f"\n--- Fine-tune fraction: {frac*100:.0f}% ---")

            # Deep copy ResPlan data
            rp_train_copy = [Data(x=g.x.clone(), edge_index=g.edge_index, y=g.y) for g in rp_train]
            rp_val_copy = [Data(x=g.x.clone(), edge_index=g.edge_index, y=g.y) for g in rp_val]
            rp_test_copy = [Data(x=g.x.clone(), edge_index=g.edge_index, y=g.y) for g in rp_test]

            if frac == 0.0:
                # Zero-shot: just evaluate pre-trained RPLAN model on ResPlan
                # Normalize ResPlan with RPLAN statistics
                normalize_graphs(
                    [Data(x=g.x.clone(), edge_index=g.edge_index, y=g.y) for g in rplan_train],
                    rp_test_copy,
                )
                model = FlexSAGE().to(device)
                model.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()})
                acc, mf1 = evaluate(model, rp_test_copy, device)
                print(f"  RPLAN→ResPlan (0-shot): acc={acc:.4f}, mf1={mf1:.4f}")
            else:
                # Subsample training data
                n_sub = max(1, int(len(rp_train_copy) * frac))
                np.random.seed(seed)
                sub_idx = np.random.permutation(len(rp_train_copy))[:n_sub]
                rp_train_sub = [rp_train_copy[i] for i in sub_idx]

                # Normalize with subsample statistics
                normalize_graphs(rp_train_sub, rp_val_copy, rp_test_copy)

                # Fine-tuned model (from RPLAN pre-trained weights)
                model_ft = FlexSAGE().to(device)
                model_ft.load_state_dict({k: v.to(device) for k, v in pretrained_state.items()})
                ft_state, ft_val = train_model(
                    model_ft, rp_train_sub, rp_val_copy, device,
                    args.finetune_epochs, lr=0.005  # lower LR for fine-tuning
                )
                model_ft.load_state_dict({k: v.to(device) for k, v in ft_state.items()})
                acc_ft, mf1_ft = evaluate(model_ft, rp_test_copy, device)

                # From-scratch model (same data, random init)
                rp_train_sub2 = [Data(x=g.x.clone(), edge_index=g.edge_index, y=g.y)
                                 for g in [rp_train[i] for i in sub_idx]]
                rp_val_copy2 = [Data(x=g.x.clone(), edge_index=g.edge_index, y=g.y) for g in rp_val]
                rp_test_copy2 = [Data(x=g.x.clone(), edge_index=g.edge_index, y=g.y) for g in rp_test]
                normalize_graphs(rp_train_sub2, rp_val_copy2, rp_test_copy2)

                model_sc = FlexSAGE().to(device)
                sc_state, sc_val = train_model(
                    model_sc, rp_train_sub2, rp_val_copy2, device,
                    args.finetune_epochs + 100  # more epochs for scratch
                )
                model_sc.load_state_dict({k: v.to(device) for k, v in sc_state.items()})
                acc_sc, mf1_sc = evaluate(model_sc, rp_test_copy2, device)

                acc, mf1 = acc_ft, mf1_ft
                print(f"  Fine-tuned:    acc={acc_ft:.4f}, mf1={mf1_ft:.4f} ({n_sub} plans)")
                print(f"  From scratch:  acc={acc_sc:.4f}, mf1={mf1_sc:.4f}")
                print(f"  Improvement:   +{100*(acc_ft-acc_sc):.1f}pp")

            # Store results
            key = f"frac_{frac:.2f}"
            if key not in results:
                results[key] = {"finetuned": [], "scratch": [], "fraction": frac,
                                "n_plans": int(len(rp_train) * frac) if frac > 0 else 0}

            if frac == 0.0:
                results[key]["finetuned"].append({"acc": float(acc), "mf1": float(mf1)})
                results[key]["scratch"].append({"acc": 0.2, "mf1": 0.2})  # random baseline
            else:
                results[key]["finetuned"].append({"acc": float(acc_ft), "mf1": float(mf1_ft)})
                results[key]["scratch"].append({"acc": float(acc_sc), "mf1": float(mf1_sc)})

    # ── Aggregate results ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY (mean ± std over seeds)")
    print(f"{'='*60}")
    print(f"{'Fraction':>10} {'N plans':>8} {'Fine-tuned':>16} {'From scratch':>16} {'Δ':>8}")
    print("-" * 62)

    summary = {}
    for frac in fractions:
        key = f"frac_{frac:.2f}"
        r = results[key]
        ft_accs = [x["acc"] for x in r["finetuned"]]
        sc_accs = [x["acc"] for x in r["scratch"]]
        ft_mean, ft_std = np.mean(ft_accs), np.std(ft_accs)
        sc_mean, sc_std = np.mean(sc_accs), np.std(sc_accs)
        delta = ft_mean - sc_mean

        n_plans = r["n_plans"]
        ft_mf1s = [x["mf1"] for x in r["finetuned"]]
        sc_mf1s = [x["mf1"] for x in r["scratch"]]

        summary[key] = {
            "fraction": frac,
            "n_plans": n_plans,
            "finetuned_acc_mean": float(ft_mean),
            "finetuned_acc_std": float(ft_std),
            "scratch_acc_mean": float(sc_mean),
            "scratch_acc_std": float(sc_std),
            "finetuned_mf1_mean": float(np.mean(ft_mf1s)),
            "scratch_mf1_mean": float(np.mean(sc_mf1s)),
            "delta_pp": float(100 * delta),
        }

        label_frac = f"{frac*100:.0f}%" if frac > 0 else "0%"
        print(f"{label_frac:>10} {n_plans:>8} {ft_mean:.4f}±{ft_std:.4f}  {sc_mean:.4f}±{sc_std:.4f}  {'+' if delta >= 0 else ''}{100*delta:.1f}pp")

    results["summary"] = summary
    results["_meta"] = {
        "pretrain_epochs": args.pretrain_epochs,
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr": 0.005,
        "seeds": args.seeds,
        "device": str(device),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
