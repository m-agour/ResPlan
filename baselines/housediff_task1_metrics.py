#!/usr/bin/env python3
"""
housediff_task1_metrics.py
Compute Task 1 metrics (Room Count Accuracy, Adjacency Satisfaction, Boundary IoU)
from HouseDiffusion generated samples on the ResPlan test set.

This script:
1. Loads the trained HouseDiffusion model
2. Runs inference on the eval (test) set
3. Extracts per-sample room type counts and adjacency from generated polygons
4. Compares to ground truth conditioning (room counts + graph)
5. Reports metrics matching Table 8 format

Usage (GPU):
  cd house_diffusion/scripts
  python ../../resplan_release/baselines/housediff_task1_metrics.py \
    --model_path ../logs_resplan/ema_0.9999_200000.pt \
    --dataset rplan --set_name eval --target_set 0 --batch_size 16
"""

import argparse, os, sys, json
import numpy as np
import torch as th
from collections import defaultdict, Counter
from shapely.geometry import Polygon
from shapely.validation import make_valid
from tqdm import tqdm

# Add house_diffusion to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../house_diffusion'))

from house_diffusion.rplanhg_datasets import load_rplanhg_data
from house_diffusion import dist_util, logger
from house_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
    update_arg_parser,
)

# HouseDiffusion room type IDs → ResPlan categories
# In the converter: living=1, bedroom=2, kitchen=3, bathroom=4, balcony=10, front_door=17→12
HD_TO_RESPLAN = {
    1: 'living',
    2: 'bedroom',
    3: 'kitchen',
    4: 'bathroom',
    10: 'balcony',
    # 11, 12, 13, 17 are doors — skip for room counting
}

ROOM_TYPES_FOR_COUNTING = {1, 2, 3, 4, 10}  # non-door types
DOOR_TYPES = {11, 12, 13}  # front_door=12, exterior_wall=11, interior_wall=13


def extract_room_counts_from_types(room_types_onehot, padding_mask, room_indices_onehot):
    """Extract per-type room counts from the conditioning data."""
    valid = padding_mask > 0.5
    if valid.sum() == 0:
        return Counter()

    types = np.argmax(room_types_onehot[valid], axis=1)
    room_idxs = np.argmax(room_indices_onehot[valid], axis=1)

    # Group corners by room index, then count distinct rooms per type
    room_type_map = {}
    for t, ri in zip(types, room_idxs):
        room_type_map[ri] = t

    counts = Counter()
    for ri, t in room_type_map.items():
        if t in ROOM_TYPES_FOR_COUNTING:
            counts[t] += 1
    return counts


def extract_polys_and_types(sample, model_kwargs, batch_idx, prefix='syn_'):
    """Extract polygons and their types from a generated sample."""
    polys = []
    types = []

    for j in range(sample.shape[1]):  # iterate over corners
        if model_kwargs[f'{prefix}src_key_padding_mask'][batch_idx][j] == 1:
            continue

        point = sample[batch_idx, j].cpu().numpy()
        point = point / 2 + 0.5
        point = point * 256  # scale to 256x256 pixel space

        if j == 0:
            poly = []
        if j > 0 and (model_kwargs[f'{prefix}room_indices'][batch_idx, j] !=
                       model_kwargs[f'{prefix}room_indices'][batch_idx, j - 1]).any():
            polys.append(np.array(poly))
            types.append(c)
            poly = []

        poly.append((point[0], point[1]))
        c = int(np.argmax(model_kwargs[f'{prefix}room_types'][batch_idx][j - 1 if j > 0 else 0].cpu().numpy()))

    if poly:
        polys.append(np.array(poly))
        types.append(c)

    return polys, types


def count_rooms_from_generated(polys, types):
    """Count rooms per type from generated polygons (skip doors)."""
    counts = Counter()
    for t in types:
        if t in ROOM_TYPES_FOR_COUNTING:
            counts[t] += 1
    return counts


def estimate_adjacency_from_polys(polys, types):
    """Estimate room adjacency from generated polygons using IoU overlap."""
    n = len(polys)
    edges = set()

    rooms_inds = [i for i in range(n) if types[i] in ROOM_TYPES_FOR_COUNTING]
    door_inds = [i for i in range(n) if types[i] in DOOR_TYPES]

    # Build Shapely polygons
    shapely_polys = {}
    for i in range(n):
        if len(polys[i]) >= 3:
            try:
                p = Polygon(polys[i])
                if not p.is_valid:
                    p = make_valid(p)
                shapely_polys[i] = p
            except:
                pass

    # Method 1: door-mediated connections (like HouseDiffusion's estimate_graph)
    for d in door_inds:
        if d not in shapely_polys:
            continue
        dp = shapely_polys[d]
        connected = []
        for r in rooms_inds:
            if r not in shapely_polys:
                continue
            rp = shapely_polys[r]
            try:
                union_area = dp.union(rp).area
                if union_area > 0:
                    iou = dp.intersection(rp).area / union_area
                    if 0 < iou < 0.2:
                        connected.append((r, iou))
            except:
                pass
        connected.sort(key=lambda x: x[1], reverse=True)
        if len(connected) >= 2:
            edges.add((min(connected[0][0], connected[1][0]),
                       max(connected[0][0], connected[1][0])))

    # Method 2: direct polygon overlap/touching for rooms
    for i_idx, i in enumerate(rooms_inds):
        if i not in shapely_polys:
            continue
        for j in rooms_inds[i_idx + 1:]:
            if j not in shapely_polys:
                continue
            try:
                if shapely_polys[i].intersects(shapely_polys[j]):
                    area_i = shapely_polys[i].area
                    area_j = shapely_polys[j].area
                    inter = shapely_polys[i].intersection(shapely_polys[j]).area
                    min_area = min(area_i, area_j)
                    if min_area > 0 and inter / min_area < 0.5:
                        edges.add((min(i, j), max(i, j)))
            except:
                pass

    return edges


def extract_gt_adjacency(graph_triples, types_gt):
    """Extract GT adjacency edges (room_i, room_j) from graph triples."""
    edges = set()
    for triple in graph_triples:
        i, conn, j = int(triple[0]), int(triple[1]), int(triple[2])
        if conn == 1:  # connected
            # Only count if both are room types (not doors)
            edges.add((min(i, j), max(i, j)))
    return edges


def compute_boundary_iou(polys_gen, types_gen, polys_gt, types_gt):
    """Compute IoU between generated and GT bounding boxes."""
    def get_bbox(polys, types):
        all_pts = []
        for p, t in zip(polys, types):
            if t in ROOM_TYPES_FOR_COUNTING and len(p) >= 3:
                all_pts.extend(p.tolist() if isinstance(p, np.ndarray) else p)
        if not all_pts:
            return None
        pts = np.array(all_pts)
        return [pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()]

    bbox_gen = get_bbox(polys_gen, types_gen)
    bbox_gt = get_bbox(polys_gt, types_gt)

    if bbox_gen is None or bbox_gt is None:
        return 0.0

    # Compute IoU of bounding boxes
    x1 = max(bbox_gen[0], bbox_gt[0])
    y1 = max(bbox_gen[1], bbox_gt[1])
    x2 = min(bbox_gen[2], bbox_gt[2])
    y2 = min(bbox_gen[3], bbox_gt[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_gen = (bbox_gen[2] - bbox_gen[0]) * (bbox_gen[3] - bbox_gen[1])
    area_gt = (bbox_gt[2] - bbox_gt[0]) * (bbox_gt[3] - bbox_gt[1])
    union = area_gen + area_gt - inter

    return inter / union if union > 0 else 0.0


def main():
    args = create_argparser().parse_args()
    update_arg_parser(args)

    dist_util.setup_dist()
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    logger.log("Loading eval data...")
    data = load_rplanhg_data(
        batch_size=args.batch_size,
        analog_bit=args.analog_bit,
        set_name='eval',
        target_set=args.target_set,
    )

    # Determine total number of eval samples
    # Load the dataset directly to get the count
    eval_npz = np.load(f'processed_rplan/rplan_eval_{args.target_set}.npz', allow_pickle=True)
    total_samples = len(eval_npz['houses'])
    logger.log(f"Total eval samples: {total_samples}")

    # Metrics accumulators
    room_count_correct = 0
    adj_satisfaction_total = 0.0
    boundary_iou_total = 0.0
    total = 0
    compatibility_errors = []

    logger.log("Running inference...")
    tmp_count = 0
    pbar = tqdm(total=total_samples, desc="Evaluating")

    while tmp_count < total_samples:
        data_sample, model_kwargs = next(data)

        # Move to GPU
        for key in model_kwargs:
            model_kwargs[key] = model_kwargs[key].cuda()

        # Run diffusion sampling
        sample_fn = diffusion.p_sample_loop
        sample = sample_fn(
            model,
            data_sample.shape,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            analog_bit=args.analog_bit,
        )

        # sample shape: [T, B, 2, num_points] → take last timestep
        sample = sample[-1]  # [B, 2, num_points]
        sample = sample.permute(0, 2, 1)  # [B, num_points, 2]

        # GT sample
        sample_gt = data_sample.cuda()  # [B, 2, num_points]
        sample_gt = sample_gt.permute(0, 2, 1)  # [B, num_points, 2]

        batch_size = sample.shape[0]

        for i in range(batch_size):
            if tmp_count + i >= total_samples:
                break

            # Extract GT room counts from conditioning
            gt_room_types = model_kwargs['room_types'][i].cpu().numpy()  # [num_points, 25]
            gt_padding = model_kwargs['src_key_padding_mask'][i].cpu().numpy()  # [num_points]
            gt_room_indices = model_kwargs['room_indices'][i].cpu().numpy()  # [num_points, 32]
            gt_graph = model_kwargs['graph'][i].cpu().numpy()  # [200, 3]

            gt_counts = extract_room_counts_from_types(gt_room_types, 1 - gt_padding, gt_room_indices)

            # Extract generated polygons and types
            polys_gen, types_gen = extract_polys_and_types(
                sample, model_kwargs, i, prefix='syn_'
            )

            # Extract GT polygons and types
            polys_gt, types_gt = extract_polys_and_types(
                sample_gt, model_kwargs, i, prefix=''
            )

            # Metric 1: Room Count Accuracy
            gen_counts = count_rooms_from_generated(polys_gen, types_gen)
            count_match = (gt_counts == gen_counts)
            if count_match:
                room_count_correct += 1

            # Metric 2: Adjacency Satisfaction
            gt_edges = extract_gt_adjacency(gt_graph, types_gt)
            gen_edges = estimate_adjacency_from_polys(polys_gen, types_gen)

            if len(gt_edges) > 0:
                satisfied = len(gt_edges & gen_edges)
                adj_sat = satisfied / len(gt_edges)
            else:
                adj_sat = 1.0
            adj_satisfaction_total += adj_sat

            # Metric 3: Boundary IoU
            biou = compute_boundary_iou(polys_gen, types_gen, polys_gt, types_gt)
            boundary_iou_total += biou

            total += 1

        tmp_count += batch_size
        pbar.update(batch_size)

    pbar.close()

    # Compute final metrics
    results = {
        'method': 'HouseDiffusion',
        'room_count_accuracy': room_count_correct / total,
        'adjacency_satisfaction': adj_satisfaction_total / total,
        'boundary_iou': boundary_iou_total / total,
        'total_samples': total,
        'fid': 56.75,  # from eval job
        'compatibility': 11.45,  # from eval job
    }

    print("\n" + "=" * 60)
    print("HouseDiffusion Task 1 Results on ResPlan Test Set")
    print("=" * 60)
    print(f"Room Count Accuracy:    {results['room_count_accuracy']:.3f}")
    print(f"Adjacency Satisfaction: {results['adjacency_satisfaction']:.3f}")
    print(f"Boundary IoU:           {results['boundary_iou']:.3f}")
    print(f"FID:                    {results['fid']:.2f}")
    print(f"Compatibility:          {results['compatibility']:.2f}")
    print(f"Total samples:          {results['total_samples']}")
    print("=" * 60)

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'results', 'task1_housediffusion.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


def create_argparser():
    defaults = dict(
        dataset='rplan',
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        draw_graph=False,
        save_svg=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
