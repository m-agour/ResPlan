#!/usr/bin/env python3
"""
Task 1 (semantic room labeling): modern graph architectures.

Adds four post-2020 architectures to the Task-1 architecture comparison, in
response to the reviewer observation that GraphSAGE (2017) and GAT (2018) are
not representative of current graph learning:

  * GATv2      (Brody et al., ICLR 2022)    -- dynamic attention, fixes GAT
  * Graph Transformer / UniMP
               (Shi et al., IJCAI 2021)     -- full attention over neighbours
  * GraphGPS   (Rampasek et al., NeurIPS 2022) -- local MPNN + global attention
  * RGCN       (Schlichtkrull et al., 2018) -- relation-specific message passing
                                               over ResPlan's 4 typed edges

Everything else (features, split, normalisation, class-balanced loss, optimiser,
schedule, epochs, seeds, model selection on val accuracy) is inherited from
task2_ablation.py so numbers are directly comparable to the published table.

Usage:
  python task1_modern_arch.py --device cuda
Outputs:
  results/task1_modern_arch.json
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import (GATv2Conv, GPSConv, RGCNConv, SAGEConv,
                                TransformerConv)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task2_ablation import (CLASS_NAMES, DROPOUT, EDGE_TYPES, EPOCHS,
                            NUM_CLASSES, RESPLAN_MAP, FlexGraphSAGE,
                            train_and_eval, seed_everything)

NUM_RELATIONS = len(EDGE_TYPES)


# ─── Graph construction (same 11 features, plus edge_type for RGCN) ───────────
def build_graph_typed(plan):
    """Same as task2_ablation.build_graph but retains per-edge relation ids."""
    G = plan.get("graph")
    if G is None or len(G.nodes) < 2:
        return None
    valid_nodes = [nd for nd in G.nodes if G.nodes[nd].get("type", "") in RESPLAN_MAP]
    if len(valid_nodes) < 2:
        return None

    n = len(valid_nodes)
    node2idx = {nd: i for i, nd in enumerate(valid_nodes)}
    et2id = {et: i for i, et in enumerate(EDGE_TYPES)}

    labels, areas, cx_list, cy_list = [], [], [], []
    deg_by_type = {et: np.zeros(n) for et in EDGE_TYPES}

    for nd in valid_nodes:
        d = G.nodes[nd]
        labels.append(RESPLAN_MAP[d["type"]])
        areas.append(float(d.get("area", 1.0)))
        geo = d.get("geometry")
        if geo is not None and not geo.is_empty:
            c = geo.centroid
            cx_list.append(c.x)
            cy_list.append(c.y)
        else:
            cx_list.append(0.0)
            cy_list.append(0.0)

    areas = np.array(areas)
    cxs, cys = np.array(cx_list), np.array(cy_list)

    edge_list, edge_type = [], []
    for u, v, edata in G.edges(data=True):
        if u in node2idx and v in node2idx:
            i, j = node2idx[u], node2idx[v]
            et = edata.get("type", "adjacency")
            rid = et2id.get(et, et2id["adjacency"])
            edge_list += [[i, j], [j, i]]
            edge_type += [rid, rid]
            if et in deg_by_type:
                deg_by_type[et][i] += 1
                deg_by_type[et][j] += 1

    if not edge_list:
        for i in range(n):
            for j in range(i + 1, n):
                edge_list += [[i, j], [j, i]]
                edge_type += [et2id["adjacency"]] * 2

    adj = [set() for _ in range(n)]
    for e in edge_list:
        adj[e[0]].add(e[1])
    neigh_mean, neigh_min, neigh_max = (np.zeros(n) for _ in range(3))
    for i in range(n):
        if adj[i]:
            na = [areas[j] for j in adj[i]]
            neigh_mean[i], neigh_min[i], neigh_max[i] = np.mean(na), np.min(na), np.max(na)

    total_area = areas.sum() + 1e-8
    all_coords = np.concatenate([cxs, cys])
    cmin, cmax = all_coords.min(), all_coords.max()
    if cmax > cmin:
        cxs = (cxs - cmin) / (cmax - cmin)
        cys = (cys - cmin) / (cmax - cmin)

    feats = np.stack([
        areas,
        deg_by_type["via_door"], deg_by_type["adjacency"],
        deg_by_type["via_window"], deg_by_type["direct"],
        neigh_mean, neigh_min, neigh_max,
        areas / total_area, cxs, cys,
    ], axis=1)

    data = Data(
        x=torch.tensor(feats, dtype=torch.float),
        edge_index=torch.tensor(edge_list, dtype=torch.long).t().contiguous(),
        y=torch.tensor(labels, dtype=torch.long),
    )
    data.edge_type = torch.tensor(edge_type, dtype=torch.long)
    return data


# ─── Architectures ───────────────────────────────────────────────────────────
class GATv2(nn.Module):
    """GATv2 (ICLR 2022): dynamic attention."""
    def __init__(self, in_dim, hidden_dim=128, num_layers=3, out_dim=NUM_CLASSES,
                 heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GATv2Conv(in_dim, hidden_dim // heads, heads=heads,
                                    dropout=DROPOUT))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim // heads,
                                        heads=heads, dropout=DROPOUT))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(GATv2Conv(hidden_dim, out_dim, heads=1, concat=False,
                                    dropout=DROPOUT))

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(F.relu(self.bns[i](conv(x, ei))), DROPOUT,
                          training=self.training)
        return self.convs[-1](x, ei)


class GraphTransformer(nn.Module):
    """UniMP-style graph transformer (TransformerConv, IJCAI 2021)."""
    def __init__(self, in_dim, hidden_dim=128, num_layers=3, out_dim=NUM_CLASSES,
                 heads=4):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(TransformerConv(in_dim, hidden_dim // heads,
                                          heads=heads, dropout=DROPOUT))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(hidden_dim, hidden_dim // heads,
                                              heads=heads, dropout=DROPOUT))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.convs.append(TransformerConv(hidden_dim, out_dim, heads=1,
                                          concat=False, dropout=DROPOUT))

    def forward(self, data):
        x, ei = data.x, data.edge_index
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(F.relu(self.bns[i](conv(x, ei))), DROPOUT,
                          training=self.training)
        return self.convs[-1](x, ei)


class GPS(nn.Module):
    """GraphGPS (NeurIPS 2022): local message passing + global self-attention."""
    def __init__(self, in_dim, hidden_dim=128, num_layers=3, out_dim=NUM_CLASSES,
                 heads=4):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GPSConv(hidden_dim,
                    SAGEConv(hidden_dim, hidden_dim),
                    heads=heads, dropout=DROPOUT)
            for _ in range(num_layers)
        ])
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x = self.inp(data.x)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for layer in self.layers:
            x = layer(x, data.edge_index, batch)
        return self.out(x)


class RGCN(nn.Module):
    """Relational GCN over ResPlan's four typed edges."""
    def __init__(self, in_dim, hidden_dim=128, num_layers=3, out_dim=NUM_CLASSES):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs = nn.ModuleList([
            RGCNConv(dims[i], dims[i + 1], num_relations=NUM_RELATIONS)
            for i in range(num_layers)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)
        ])

    def forward(self, data):
        x, ei, et = data.x, data.edge_index, data.edge_type
        for i, conv in enumerate(self.convs[:-1]):
            x = F.dropout(F.relu(self.bns[i](conv(x, ei, et))), DROPOUT,
                          training=self.training)
        return self.convs[-1](x, ei, et)


def run_multi_seed_epochs(model_fn, train_gs, val_gs, test_gs, device, seeds,
                          epochs, label=""):
    """Like task2_ablation.run_multi_seed but forwards the epoch budget."""
    runs = []
    for seed in seeds:
        r = train_and_eval(model_fn(), train_gs, val_gs, test_gs, device, seed,
                           epochs=epochs, label=f"{label}/s{seed}")
        runs.append(r)
        print(f"    seed={seed}  Acc={r['acc']:.4f}  MF1={r['macro_f1']:.4f}",
              flush=True)
    accs = [r["acc"] for r in runs]
    mf1s = [r["macro_f1"] for r in runs]
    return {
        "runs": runs,
        "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
        "mean_macro_f1": float(np.mean(mf1s)), "std_macro_f1": float(np.std(mf1s)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../ResPlan.pkl")
    ap.add_argument("--split", default="../split.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default="results/task1_modern_arch.json")
    args = ap.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print(f"device: {device}", flush=True)

    plans = pickle.load(open(args.data, "rb"))
    splits = json.load(open(args.split))
    want = {n: {int(x) for x in splits[n]} for n in ("train", "val", "test")}

    gs = {"train": [], "val": [], "test": []}
    for plan in plans:
        pid = plan.get("id")
        for name in gs:
            if pid in want[name]:
                d = build_graph_typed(plan)
                if d is not None:
                    gs[name].append(d)
                break
    print({k: len(v) for k, v in gs.items()}, flush=True)

    in_dim = gs["train"][0].x.shape[1]
    archs = {
        "GraphSAGE (reference)": lambda: FlexGraphSAGE(in_dim, 128, 3),
        "GATv2":                 lambda: GATv2(in_dim, 128, 3),
        "GraphTransformer":      lambda: GraphTransformer(in_dim, 128, 3),
        "GraphGPS":              lambda: GPS(in_dim, 128, 3),
        "RGCN (typed edges)":    lambda: RGCN(in_dim, 128, 3),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = {}
    for name, fn in archs.items():
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        n_params = sum(p.numel() for p in fn().parameters())
        res = run_multi_seed_epochs(fn, gs["train"], gs["val"], gs["test"],
                                    device, args.seeds, args.epochs, label=name)
        res["params"] = int(n_params)
        res["train_seconds_total"] = round(time.time() - t0, 1)
        out[name] = res
        print(f"  {name}: acc={res['mean_acc']:.4f}+-{res['std_acc']:.4f} "
              f"mF1={res['mean_macro_f1']:.4f} params={n_params} "
              f"({res['train_seconds_total']}s)", flush=True)
        json.dump(out, open(args.out, "w"), indent=1)

    print("\n===== SUMMARY =====")
    for k, v in sorted(out.items(), key=lambda kv: -kv[1]["mean_acc"]):
        print(f"{k:24s} acc={v['mean_acc']:.4f}+-{v['std_acc']:.4f} "
              f"mF1={v['mean_macro_f1']:.4f}+-{v['std_macro_f1']:.4f} "
              f"params={v['params']:,}")
    json.dump(out, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
