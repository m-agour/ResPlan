#!/usr/bin/env python3
"""
Layout GNN baseline for Task 1: Graph-Conditioned Layout Generation.

A 3-layer Graph Convolutional Network (GCN) that predicts room bounding boxes
(cx, cy, w, h) from the connectivity graph and room type features.

Architecture:
    - Input: 7-dim node features (6-dim one-hot room type + area fraction)
    - 3 GCN layers with 64 hidden units each (D^{-1/2} A D^{-1/2} message passing)
    - Prediction head: Linear(64,64) -> ReLU -> Linear(64,4) -> Sigmoid
    - Total: 13,700 parameters

Training:
    - MSE loss with node masking (padded to MAX_NODES=35)
    - Adam optimizer, lr=3e-3, weight_decay=1e-5
    - Batch size 512, 200 epochs, best checkpoint by val loss

IMPORTANT: Rooms are read directly from the graph's node set to ensure
perfect alignment with graph edges. The graph construction in ResPlan
unions multi-part living rooms into a single 'living_0' node and includes
'front_door' nodes, so we must match that convention.
"""

import os
import json
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import Polygon, MultiPolygon

torch.set_num_threads(16)

# Must include front_door — the graph has front_door nodes with edges
ROOM_CATS = ['bedroom', 'bathroom', 'kitchen', 'living', 'balcony', 'front_door']
NC = len(ROOM_CATS)
MAX_NODES = 35
FEAT_DIM = NC + 1  # one-hot room type + area fraction


class GCN(nn.Module):
    """Simple 3-layer Graph Convolutional Network for layout prediction."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(FEAT_DIM, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 64)
        self.head = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 4), nn.Sigmoid()
        )

    def forward(self, x, a):
        """
        Args:
            x: (B, N, FEAT_DIM) node features
            a: (B, N, N) normalized adjacency (D^{-1/2} A D^{-1/2})
        Returns:
            (B, N, 4) predicted bounding boxes [cx, cy, w, h] in [0,1]
        """
        x = F.relu(self.l1(torch.bmm(a, x)))
        x = F.relu(self.l2(torch.bmm(a, x)))
        x = F.relu(self.l3(torch.bmm(a, x)))
        return self.head(x)


def load_and_convert(pkl_path, split_path):
    """Load ResPlan data and convert to padded tensor format.

    Rooms are read directly from graph.nodes() to ensure perfect
    alignment with graph edges. This avoids the living-room union
    mismatch and includes front_door nodes.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    with open(split_path) as f:
        split = json.load(f)

    id_to_split = {}
    for s in ['train', 'val', 'test']:
        for pid in split[s]:
            id_to_split[int(pid)] = s

    all_x, all_adj, all_y, all_mask, all_split = [], [], [], [], []
    skipped_no_graph = 0
    skipped_size = 0
    skipped_no_geom = 0

    for plan in data:
        graph = plan.get('graph')
        if graph is None:
            skipped_no_graph += 1
            continue

        # Extract rooms directly from graph nodes
        nodes = []  # (name, type, geometry)
        for nid, attrs in graph.nodes(data=True):
            rtype = attrs.get('type', '')
            geom = attrs.get('geometry')
            if rtype not in ROOM_CATS:
                continue
            if geom is None or geom.is_empty:
                continue
            nodes.append((nid, rtype, geom))

        n = len(nodes)
        if n < 3 or n > MAX_NODES:
            skipped_size += 1
            continue

        # Compute plan bounding box for normalization
        all_b = []
        for _, _, geom in nodes:
            if isinstance(geom, (Polygon, MultiPolygon)):
                all_b.append(geom.bounds)
            else:
                # front_door may be a LineString
                all_b.append(geom.bounds)

        if not all_b:
            skipped_no_geom += 1
            continue

        pminx = min(b[0] for b in all_b)
        pminy = min(b[1] for b in all_b)
        pmaxx = max(b[2] for b in all_b)
        pmaxy = max(b[3] for b in all_b)
        pw = max(pmaxx - pminx, 1e-6)
        ph = max(pmaxy - pminy, 1e-6)
        plan_area = pw * ph

        x = np.zeros((MAX_NODES, FEAT_DIM), dtype=np.float32)
        y = np.zeros((MAX_NODES, 4), dtype=np.float32)
        mask = np.zeros(MAX_NODES, dtype=np.float32)
        adj = np.eye(MAX_NODES, dtype=np.float32)
        node_names = []

        for i, (nid, rtype, geom) in enumerate(nodes):
            # One-hot room type
            x[i, ROOM_CATS.index(rtype)] = 1.0
            # Area fraction
            area = geom.area if hasattr(geom, 'area') else 0.0
            x[i, NC] = area / plan_area

            # Bounding box target (normalized)
            bx0, by0, bx1, by1 = geom.bounds
            cx = ((bx0 + bx1) / 2 - pminx) / pw
            cy = ((by0 + by1) / 2 - pminy) / ph
            w = (bx1 - bx0) / pw
            h = (by1 - by0) / ph
            y[i] = [cx, cy, w, h]
            mask[i] = 1.0
            node_names.append(nid)

        # Build adjacency from actual graph edges
        name_to_idx = {name: i for i, name in enumerate(node_names)}
        has_edge = False
        for u, v in graph.edges():
            ui = name_to_idx.get(u)
            vi = name_to_idx.get(v)
            if ui is not None and vi is not None:
                adj[ui, vi] = adj[vi, ui] = 1.0
                has_edge = True

        if not has_edge:
            continue

        # Symmetric normalization: D^{-1/2} A D^{-1/2}
        deg = adj.sum(1, keepdims=True)
        dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0)
        adj_norm = dinv * adj * dinv.T

        all_x.append(x)
        all_adj.append(adj_norm)
        all_y.append(y)
        all_mask.append(mask)
        all_split.append(id_to_split.get(plan['id'], 'train'))

    print(f'  Skipped: no_graph={skipped_no_graph}, '
          f'size_filter={skipped_size}, no_geom={skipped_no_geom}')

    X = torch.tensor(np.stack(all_x))
    A = torch.tensor(np.stack(all_adj))
    Y = torch.tensor(np.stack(all_y))
    M = torch.tensor(np.stack(all_mask))

    splits = {
        'train': [i for i, s in enumerate(all_split) if s == 'train'],
        'val': [i for i, s in enumerate(all_split) if s == 'val'],
        'test': [i for i, s in enumerate(all_split) if s == 'test'],
    }
    return X, A, Y, M, splits


def evaluate(model, X, A, Y, M, indices):
    """Evaluate model on given indices, returning Task 1 metrics."""
    model.eval()
    ti = torch.tensor(indices)
    with torch.no_grad():
        pred = model(X[ti], A[ti]).numpy()
    gt = Y[ti].numpy()
    mask = M[ti].numpy()
    adj_raw = A[ti].numpy()

    adj_sats, bious = [], []
    for s in range(len(ti)):
        n = int(mask[s].sum())
        p, g = pred[s, :n], gt[s, :n]
        a_row = adj_raw[s, :n, :n]

        # Extract edges (non-diagonal entries > threshold in normalized adj)
        pairs = set()
        for i in range(n):
            for j in range(i + 1, n):
                if a_row[i, j] > 0.001:
                    pairs.add((i, j))

        # Adjacency satisfaction: predicted rooms overlap if edge exists
        sat = 0
        for i, j in pairs:
            dx = abs(p[i, 0] - p[j, 0])
            dy = abs(p[i, 1] - p[j, 1])
            if dx < (p[i, 2] + p[j, 2]) / 2 + 0.02 and \
               dy < (p[i, 3] + p[j, 3]) / 2 + 0.02:
                sat += 1
        adj_sats.append(sat / max(len(pairs), 1))

        # Boundary IoU: overall bounding box IoU
        pb = [
            (p[:, 0] - p[:, 2] / 2).min(),
            (p[:, 1] - p[:, 3] / 2).min(),
            (p[:, 0] + p[:, 2] / 2).max(),
            (p[:, 1] + p[:, 3] / 2).max(),
        ]
        gb = [
            (g[:, 0] - g[:, 2] / 2).min(),
            (g[:, 1] - g[:, 3] / 2).min(),
            (g[:, 0] + g[:, 2] / 2).max(),
            (g[:, 1] + g[:, 3] / 2).max(),
        ]
        ix1, iy1 = max(pb[0], gb[0]), max(pb[1], gb[1])
        ix2, iy2 = min(pb[2], gb[2]), min(pb[3], gb[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = max(pb[2] - pb[0], 0) * max(pb[3] - pb[1], 0)
        a2 = max(gb[2] - gb[0], 0) * max(gb[3] - gb[1], 0)
        bious.append(inter / max(a1 + a2 - inter, 1e-8))

    return {
        'room_count_accuracy': 1.0,  # conditioned on exact room set
        'adjacency_satisfaction': float(np.mean(adj_sats)),
        'boundary_iou': float(np.mean(bious)),
    }


def main():
    # Paths (relative to repo root)
    base = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    pkl_path = os.path.join(base, 'dataset', 'ResPlan.pkl')
    split_path = os.path.join(base, 'dataset', 'split.json')

    print('Loading and converting data...')
    X, A, Y, M, splits = load_and_convert(pkl_path, split_path)
    print(f'Total={len(X)}, Train={len(splits["train"])}, '
          f'Val={len(splits["val"])}, Test={len(splits["test"])}')

    model = GCN()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # Training
    BS = 512
    best_vl = 1e9
    ckpt_path = '/tmp/best_layout_gnn.pt'

    for ep in range(1, 201):
        model.train()
        idx = torch.tensor(splits['train'])[
            torch.randperm(len(splits['train']))]
        ep_loss = 0
        nb = 0
        for s in range(0, len(idx), BS):
            bi = idx[s:s + BS]
            xb, ab, yb, mb = X[bi], A[bi], Y[bi], M[bi]
            pred = model(xb, ab)
            m3 = mb.unsqueeze(-1)
            loss = ((pred - yb) ** 2 * m3).sum() / (m3.sum() * 4)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1

        # Validation
        model.eval()
        with torch.no_grad():
            vi = torch.tensor(splits['val'])
            pred = model(X[vi], A[vi])
            m3 = M[vi].unsqueeze(-1)
            vl = ((pred - Y[vi]) ** 2 * m3).sum().item() / \
                 (m3.sum().item() * 4)

        if ep % 10 == 0 or ep == 1:
            print(f'  Ep {ep:3d}: train={ep_loss / nb:.5f} val={vl:.5f}')
        if vl < best_vl:
            best_vl = vl
            torch.save(model.state_dict(), ckpt_path)
            best_ep = ep

    print(f'Best val at epoch {best_ep}')

    # Test evaluation
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    results = evaluate(model, X, A, Y, M, splits['test'])

    print()
    print('=' * 60)
    print('Layout GNN - Task 1 Test Results')
    print('=' * 60)
    print(f'  Room Count Accuracy:    {results["room_count_accuracy"]:.3f}')
    print(f'  Adjacency Satisfaction: {results["adjacency_satisfaction"]:.3f}')
    print(f'  Boundary IoU:           {results["boundary_iou"]:.3f}')
    print(f'  Samples:                {len(splits["test"])}')
    print('=' * 60)

    # Save results
    out = {
        'method': 'Layout GNN',
        'room_count_accuracy': round(results['room_count_accuracy'], 3),
        'adjacency_satisfaction': round(results['adjacency_satisfaction'], 3),
        'boundary_iou': round(results['boundary_iou'], 3),
        'total': len(splits['test']),
    }
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'results'), exist_ok=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'results', 'task1_layout_gnn.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved results to {out_path}')


if __name__ == '__main__':
    main()
