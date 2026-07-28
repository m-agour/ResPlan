#!/usr/bin/env python3
"""
Effect of near-duplicate train/test leakage on the Task-1 benchmark.

A geometry-based near-duplicate scan (best-of-8 dihedral label-raster IoU >= 0.90)
finds that 154 of the 1,632 canonical test plans have a near-duplicate in the
training split. This script trains the reference GraphSAGE baseline unchanged and
evaluates it on:

  (a) the full canonical test split          (the published protocol)
  (b) the leak-free subset                    (test minus the 154 leaked plans)
  (c) the leaked subset alone                 (the 154 plans, for contrast)

If (a) and (b) agree, the published benchmark numbers are not inflated by
duplication.

Usage:
  python task1_leakage_eval.py --device cuda
Outputs:
  results/task1_leakage_eval.json
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.loader import DataLoader as PyGLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from task2_ablation import (CLASS_NAMES, EPOCHS, NUM_CLASSES, FlexGraphSAGE,
                            build_graph, train_and_eval)


def evaluate(model, graphs, device, gmean, gstd):
    if not graphs:
        return None
    raw = [g.x.clone() for g in graphs]
    for g in graphs:
        g.x = (g.x - gmean) / gstd
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in PyGLoader(graphs, batch_size=512):
            batch = batch.to(device)
            preds.append(model(batch).argmax(1).cpu())
            trues.append(batch.y.cpu())
    for g, r in zip(graphs, raw):
        g.x = r
    p = torch.cat(preds).numpy()
    t = torch.cat(trues).numpy()
    pc = f1_score(t, p, average=None, labels=list(range(NUM_CLASSES)),
                  zero_division=0)
    return {
        "n_plans": len(graphs),
        "n_nodes": int(len(t)),
        "acc": float(accuracy_score(t, p)),
        "macro_f1": float(f1_score(t, p, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(t, p, average="weighted", zero_division=0)),
        "per_class_f1": {CLASS_NAMES[i]: float(pc[i]) for i in range(NUM_CLASSES)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../ResPlan.pkl")
    ap.add_argument("--split", default="../split.json")
    ap.add_argument("--leaked", default="../../leaked_test_ids.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default="results/task1_leakage_eval.json")
    args = ap.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    print("device:", device, flush=True)

    plans = pickle.load(open(args.data, "rb"))
    splits = json.load(open(args.split))
    leaked = set(json.load(open(args.leaked)))
    want = {n: {int(x) for x in splits[n]} for n in ("train", "val", "test")}

    train_gs, val_gs, test_all, test_clean, test_leak = [], [], [], [], []
    for plan in plans:
        pid = plan.get("id")
        d = None
        if pid in want["train"] or pid in want["val"] or pid in want["test"]:
            d = build_graph(plan)
            if d is None:
                continue
        else:
            continue
        if pid in want["train"]:
            train_gs.append(d)
        elif pid in want["val"]:
            val_gs.append(d)
        else:
            test_all.append(d)
            (test_leak if pid in leaked else test_clean).append(d)

    print(f"train={len(train_gs)} val={len(val_gs)} test={len(test_all)} "
          f"clean={len(test_clean)} leaked={len(test_leak)}", flush=True)

    in_dim = train_gs[0].x.shape[1]
    runs = []
    for seed in args.seeds:
        model = FlexGraphSAGE(in_dim, 128, 3)
        # train_and_eval performs training + model selection on val
        base = train_and_eval(model, train_gs, val_gs, test_all, device, seed,
                              epochs=args.epochs, label=f"leak/s{seed}")
        # recompute normalisation exactly as train_and_eval did, then re-evaluate
        all_x = torch.cat([g.x for g in train_gs], 0)
        gmean = all_x.mean(0)
        gstd = all_x.std(0).clamp(min=1e-6)
        r = {
            "seed": seed,
            "full_test": base,
            "leak_free_test": evaluate(model, test_clean, device, gmean, gstd),
            "leaked_only": evaluate(model, test_leak, device, gmean, gstd),
        }
        runs.append(r)
        print(f"  seed={seed} full={r['full_test']['acc']:.4f} "
              f"clean={r['leak_free_test']['acc']:.4f} "
              f"leaked={r['leaked_only']['acc']:.4f}", flush=True)

    def agg(key, metric="acc"):
        v = [r[key][metric] for r in runs if r[key]]
        return {"mean": float(np.mean(v)), "std": float(np.std(v))}

    out = {
        "n_leaked_test_plans": len(test_leak),
        "n_clean_test_plans": len(test_clean),
        "runs": runs,
        "summary": {
            k: {"acc": agg(k, "acc"), "macro_f1": agg(k, "macro_f1")}
            for k in ("full_test", "leak_free_test", "leaked_only")
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
