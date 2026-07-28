#!/usr/bin/env python3
"""
task1_housediffusion_eval.py
Evaluate HouseDiffusion on ResPlan Task 1 metrics:
  - Room Count Accuracy (RCA)
  - Adjacency Satisfaction (AdjSat)  
  - Boundary IoU (BIoU)
  - FID (from HouseDiffusion's built-in)
  - Compatibility (graph errors)

Usage:
  python task1_housediffusion_eval.py --model_path <path_to_checkpoint>
"""

import argparse
import os
import sys
import json
import numpy as np
import torch as th
from collections import defaultdict
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tqdm import tqdm

# Add HouseDiffusion to path
HOUSEDIFF_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'house_diffusion')
sys.path.insert(0, HOUSEDIFF_DIR)
sys.path.insert(0, os.path.join(HOUSEDIFF_DIR, 'scripts'))

from house_diffusion import dist_util, logger
from house_diffusion.rplanhg_datasets import load_rplanhg_data
from house_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    update_arg_parser,
)


# ResPlan room type mapping (matching the converter)
RPLAN_TO_RESPLAN = {
    1: 'living',
    2: 'bedroom',
    3: 'kitchen',
    4: 'bathroom',
    10: 'balcony',
    17: 'front_door',
}


def extract_rooms_from_sample(sample, model_kwargs, prefix='syn_'):
    """Extract room polygons and types from a generated sample.
    
    Args:
        sample: (seq_len, 2) tensor — generated corner coordinates
        model_kwargs: dict with room_types, room_indices, src_key_padding_mask
        prefix: 'syn_' for generated, '' for ground truth
        
    Returns:
        list of (room_type_int, Polygon) tuples
    """
    padding_mask = model_kwargs[f'{prefix}src_key_padding_mask']  # (seq_len,)
    room_types = model_kwargs[f'{prefix}room_types']  # (seq_len, 25)
    room_indices = model_kwargs[f'{prefix}room_indices']  # (seq_len, 32)
    
    # Group corners by room index
    rooms = {}
    for j in range(len(padding_mask)):
        if padding_mask[j] == 1:  # 1 = padded (invalid)
            continue
        ri = int(th.argmax(room_indices[j]).item())
        rt = int(th.argmax(room_types[j]).item())
        coord = sample[j].cpu().numpy()  # (2,)
        
        if ri not in rooms:
            rooms[ri] = {'type': rt, 'coords': []}
        rooms[ri]['coords'].append(coord)
    
    result = []
    for ri in sorted(rooms.keys()):
        coords = np.array(rooms[ri]['coords'])
        rt = rooms[ri]['type']
        if len(coords) >= 3:
            try:
                poly = Polygon(coords)
                if poly.is_valid and poly.area > 1e-6:
                    result.append((rt, poly))
                else:
                    # Try to fix
                    poly = poly.buffer(0)
                    if isinstance(poly, Polygon) and poly.area > 1e-6:
                        result.append((rt, poly))
            except Exception:
                pass
    return result


def compute_rca(gt_rooms, pred_rooms):
    """Room Count Accuracy: fraction of room types with matching count."""
    gt_counts = defaultdict(int)
    pred_counts = defaultdict(int)
    
    for rt, _ in gt_rooms:
        gt_counts[rt] += 1
    for rt, _ in pred_rooms:
        pred_counts[rt] += 1
    
    all_types = set(gt_counts.keys()) | set(pred_counts.keys())
    if not all_types:
        return 1.0
    
    correct = sum(1 for t in all_types if gt_counts[t] == pred_counts[t])
    return correct / len(all_types)


def compute_adj_satisfaction(pred_rooms, graph, num_rooms):
    """Adjacency Satisfaction: fraction of graph edges satisfied in generated layout.
    
    Two rooms are adjacent if their polygons touch or overlap.
    """
    # Build adjacency from graph
    expected_edges = set()
    expected_non_edges = set()
    for row in graph:
        i, conn, j = int(row[0]), int(row[1]), int(row[2])
        if i == 0 and conn == 0 and j == 0:
            continue  # padding
        if conn == 1:
            expected_edges.add((min(i, j), max(i, j)))
        elif conn == -1:
            expected_non_edges.add((min(i, j), max(i, j)))
    
    if not expected_edges:
        return 1.0
    
    # Check adjacency in generated layout
    n = len(pred_rooms)
    pred_adj = set()
    for i in range(n):
        for j in range(i + 1, n):
            _, pi = pred_rooms[i]
            _, pj = pred_rooms[j]
            try:
                # Rooms are adjacent if they intersect or are very close
                if pi.intersects(pj) or pi.distance(pj) < 0.05:
                    pred_adj.add((i, j))
            except Exception:
                pass
    
    # Count satisfied edges
    satisfied = 0
    total = len(expected_edges)
    for edge in expected_edges:
        if edge in pred_adj:
            satisfied += 1
    
    return satisfied / total if total > 0 else 1.0


def compute_biou(gt_rooms, pred_rooms):
    """Boundary IoU: IoU between GT and predicted room bounding boxes, averaged over rooms.
    
    For each GT room, find the best-matching predicted room (same type) by IoU.
    """
    if not gt_rooms or not pred_rooms:
        return 0.0
    
    ious = []
    for gt_rt, gt_poly in gt_rooms:
        if gt_rt in [11, 12, 13, 17]:  # skip door types
            continue
        best_iou = 0.0
        for pred_rt, pred_poly in pred_rooms:
            if pred_rt != gt_rt:
                continue
            try:
                inter = gt_poly.intersection(pred_poly).area
                union = gt_poly.union(pred_poly).area
                if union > 0:
                    iou = inter / union
                    best_iou = max(best_iou, iou)
            except Exception:
                pass
        ious.append(best_iou)
    
    return np.mean(ious) if ious else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_samples', type=int, default=1699, help='Number of eval samples')
    parser.add_argument('--target_set', type=int, default=0)
    parser.add_argument('--output', default='results/task1_housediffusion.json')
    args = parser.parse_args()
    
    # Change to scripts dir for data loading
    os.chdir(os.path.join(HOUSEDIFF_DIR, 'scripts'))
    
    # Ensure symlink
    if not os.path.exists('processed_rplan'):
        os.symlink('../processed_rplan', 'processed_rplan')
    
    # Setup
    os.environ['OPENAI_LOGDIR'] = '/tmp/housediff_eval'
    os.makedirs('/tmp/housediff_eval', exist_ok=True)
    
    dist_util.setup_dist()
    
    # Create model
    class Args:
        dataset = 'rplan'
        analog_bit = False
    
    model_args = Args()
    update_arg_parser(model_args)
    
    model_kwargs = {k: v for k, v in model_and_diffusion_defaults().items()}
    model_kwargs.update({
        'input_channels': model_args.input_channels,
        'condition_channels': model_args.condition_channels,
        'out_channels': model_args.out_channels,
        'use_unet': model_args.use_unet,
        'dataset': 'rplan',
    })
    
    model, diffusion = create_model_and_diffusion(**model_kwargs)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    
    # Load eval data
    data = load_rplanhg_data(
        batch_size=args.batch_size,
        analog_bit=False,
        set_name='eval',
        target_set=args.target_set,
    )
    
    # Evaluate
    rca_scores = []
    adj_scores = []
    biou_scores = []
    total_processed = 0
    
    print(f"Evaluating {args.num_samples} samples...")
    
    while total_processed < args.num_samples:
        data_sample, mkwargs = next(data)
        for key in mkwargs:
            mkwargs[key] = mkwargs[key].cuda()
        
        # Generate samples
        with th.no_grad():
            sample = diffusion.p_sample_loop(
                model,
                data_sample.shape,
                clip_denoised=True,
                model_kwargs=mkwargs,
                analog_bit=False,
            )
        
        # sample shape: (timesteps, batch, channels, seq_len)
        # Take final timestep
        final_sample = sample[-1]  # (batch, 2, 200)
        final_sample = final_sample.permute(0, 2, 1)  # (batch, 200, 2)
        
        gt_sample = data_sample.cuda().permute(0, 2, 1)  # (batch, 200, 2)
        
        batch_size = final_sample.shape[0]
        
        for i in range(batch_size):
            if total_processed >= args.num_samples:
                break
            
            # Extract rooms
            gt_rooms = extract_rooms_from_sample(gt_sample[i], mkwargs, prefix='')
            pred_rooms = extract_rooms_from_sample(final_sample[i], mkwargs, prefix='syn_')
            graph = mkwargs['graph'][i].cpu().numpy()
            
            # Compute metrics
            rca = compute_rca(gt_rooms, pred_rooms)
            adj_sat = compute_adj_satisfaction(pred_rooms, graph, len(pred_rooms))
            biou = compute_biou(gt_rooms, pred_rooms)
            
            rca_scores.append(rca)
            adj_scores.append(adj_sat)
            biou_scores.append(biou)
            
            total_processed += 1
            
            if total_processed % 100 == 0:
                print(f"  {total_processed}/{args.num_samples}: "
                      f"RCA={np.mean(rca_scores):.3f}, "
                      f"AdjSat={np.mean(adj_scores):.3f}, "
                      f"BIoU={np.mean(biou_scores):.3f}")
    
    # Summary
    results = {
        'method': 'HouseDiffusion',
        'model_path': args.model_path,
        'num_samples': total_processed,
        'RCA': float(np.mean(rca_scores)),
        'RCA_std': float(np.std(rca_scores)),
        'AdjSat': float(np.mean(adj_scores)),
        'AdjSat_std': float(np.std(adj_scores)),
        'BIoU': float(np.mean(biou_scores)),
        'BIoU_std': float(np.std(biou_scores)),
    }
    
    print("\n" + "=" * 60)
    print("HouseDiffusion on ResPlan — Task 1 Results")
    print("=" * 60)
    print(f"  Room Count Accuracy:    {results['RCA']:.3f} ± {results['RCA_std']:.3f}")
    print(f"  Adjacency Satisfaction: {results['AdjSat']:.3f} ± {results['AdjSat_std']:.3f}")
    print(f"  Boundary IoU:           {results['BIoU']:.3f} ± {results['BIoU_std']:.3f}")
    print("=" * 60)
    
    # Save
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
