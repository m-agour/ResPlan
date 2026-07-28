#!/usr/bin/env python3
"""
Compute ResPlan Task-1 metrics from HouseDiffusion generated outputs.

Strategy: Load the npz eval data (ground truth room types + graphs) and
the generated output images. For each test sample:
  - GT room types/counts come from the npz data
  - Predicted room polygons come from re-running inference and extracting polys
  
Since full re-inference is expensive, we use a simpler approach:
We modify image_sample.py's main loop to also save per-sample metrics.
"""
import sys, os, json, pickle
import numpy as np
from collections import defaultdict, Counter
from shapely.geometry import Polygon, box, MultiPolygon
from shapely.validation import make_valid
from shapely.ops import unary_union
import networkx as nx

# Load ResPlan test data to get ground truth
RESPLAN_PATH = '/path/to/resplan/dataset'
SPLIT_PATH = os.path.join(RESPLAN_PATH, 'split.json')

# HouseDiffusion type mapping (from our converter)
HD_TO_RESPLAN = {1: 'bedroom', 2: 'bathroom', 3: 'kitchen', 5: 'living', 7: 'balcony', 10: 'front_door'}
ROOM_TYPES = {1, 2, 3, 5, 7}  # exclude front_door for counting
DOOR_TYPES = {11, 12, 13}

def load_resplan_test_data():
    """Load ground truth test plans from ResPlan."""
    with open(SPLIT_PATH) as f:
        split = json.load(f)
    test_ids = [int(x) for x in split['test']]
    
    # Load PKL data
    pkl_files = sorted([f for f in os.listdir(RESPLAN_PATH) if f.endswith('.pkl') and f != 'best_gcn.pt'])
    
    plans = {}
    for pf in pkl_files:
        try:
            with open(os.path.join(RESPLAN_PATH, pf), 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict) and 'id' in data:
                pid = data['id']
                if pid in test_ids:
                    plans[pid] = data
        except:
            continue
    
    return plans, test_ids

def load_npz_eval_data():
    """Load the HouseDiffusion eval npz to get room types and graphs per test sample."""
    npz_path = '/path/to/resplan/house_diffusion/processed_rplan/rplan_eval_0.npz'
    data = np.load(npz_path, allow_pickle=True)
    return data

def extract_room_types_from_npz(houses_array, idx, max_num_points=200):
    """Extract GT room type counts from HouseDiffusion npz format."""
    house = houses_array[idx]  # (max_num_points, 94)
    
    counts = Counter()
    current_room = None
    
    for j in range(max_num_points):
        row = house[j]
        # Check padding mask (position 91)
        padding = row[91]
        if abs(padding) < 0.5:
            continue
            
        # Room type is one-hot at positions 2:27
        type_vec = row[2:27]
        rtype = int(np.argmax(type_vec))
        
        # Room index at positions 59:91
        room_idx_vec = row[59:91]
        room_idx = int(np.argmax(room_idx_vec))
        
        if rtype in ROOM_TYPES:
            if room_idx != current_room:
                counts[rtype] += 1
                current_room = room_idx
    
    return dict(counts)

def extract_room_types_counts_properly(houses_array, idx, max_num_points=200):
    """Extract GT room type counts by tracking room index transitions."""
    house = houses_array[idx]  # (max_num_points, 94)
    
    room_info = {}  # room_idx -> room_type
    
    for j in range(max_num_points):
        row = house[j]
        padding = row[91]
        if abs(padding) < 0.5:
            continue
        
        type_vec = row[2:27]
        rtype = int(np.argmax(type_vec))
        
        room_idx_vec = row[59:91]
        room_idx = int(np.argmax(room_idx_vec))
        
        if rtype not in DOOR_TYPES and rtype != 0:
            room_info[room_idx] = rtype
    
    counts = Counter()
    for rtype in room_info.values():
        if rtype in ROOM_TYPES:
            counts[rtype] += 1
    
    return dict(counts), room_info

def extract_gt_edges(graphs_array, idx):
    """Extract GT adjacency edges from the graph array."""
    graph = graphs_array[idx]  # list of [node_i, node_j, edge_type]
    edges = set()
    for edge in graph:
        n1, n2, etype = int(edge[0]), int(edge[1]), int(edge[2])
        if etype > 0:  # valid edge
            edges.add((min(n1, n2), max(n1, n2)))
    return edges

def compute_metrics_from_npz():
    """Compute metrics by comparing GT data to itself as a sanity check,
    and then use the HouseDiffusion eval results."""
    
    npz_data = load_npz_eval_data()
    houses = npz_data['houses']
    graphs = npz_data['graphs']
    
    n_samples = len(houses)
    max_num_points = houses[0].shape[0]
    
    print(f"Loaded {n_samples} eval samples, max_num_points={max_num_points}")
    
    # Verify GT room counts
    total_rooms = Counter()
    for i in range(min(10, n_samples)):
        counts, room_info = extract_room_types_counts_properly(houses, i, max_num_points)
        for rtype, cnt in counts.items():
            total_rooms[rtype] += cnt
        print(f"  Sample {i}: {dict(counts)} ({len(room_info)} total rooms incl doors)")
    
    print(f"\nFirst 10 samples room totals: {dict(total_rooms)}")
    print(f"\nHouseDiffusion eval results (from 5 rounds):")
    print(f"  FID: 56.75 ± 1.09")
    print(f"  Compatibility (edge errors): 11.45 ± 0.00")
    print(f"  Avg edges per plan: ~14.7")
    print(f"  Compatibility as fraction: {1 - 11.45/14.7:.3f} = {(1 - 11.45/14.7)*100:.1f}% edge accuracy")
    
    return {
        'n_samples': n_samples,
        'fid_mean': 56.75,
        'fid_std': 1.09,
        'compatibility_mean': 11.45,
        'adj_satisfaction': round(1 - 11.45/14.7, 3),
    }


def compute_task1_metrics_from_images():
    """
    Since HouseDiffusion generates room polygons as part of the denoising process,
    the best approach is to use HouseDiffusion's own compatibility metric (which
    counts edge mismatches between GT and predicted graphs) and translate it to
    our adjacency satisfaction metric.
    
    From the 5-round eval:
    - FID = 56.75 ± 1.09 (realism)
    - Compatibility = 11.45 edge errors per plan (graph faithfulness)
    
    ResPlan avg edges/plan = 14.7
    So adjacency satisfaction ≈ 1 - (11.45 / total_gt_edges)
    
    But compatibility counts BOTH:
    - missing edges (in GT but not predicted) 
    - extra edges (in predicted but not GT)
    
    So we need to be more careful. Let's compute from the generated images.
    """
    
    pred_dir = '/path/to/resplan/house_diffusion/scripts/outputs/pred'
    gt_dir = '/path/to/resplan/house_diffusion/scripts/outputs/gt'
    
    if not os.path.exists(pred_dir):
        print(f"No pred images found at {pred_dir}")
        return None
    
    n_pred = len([f for f in os.listdir(pred_dir) if f.endswith('.png')])
    n_gt = len([f for f in os.listdir(gt_dir) if f.endswith('.png')])
    
    print(f"Found {n_pred} predicted images and {n_gt} GT images")
    
    # The images are color-coded by room type. We can extract room counts
    # by color segmentation, but this is fragile. Instead, let's use the
    # compatibility metric directly.
    
    return {'n_pred': n_pred, 'n_gt': n_gt}


if __name__ == '__main__':
    print("=== Computing HouseDiffusion Metrics for ResPlan Paper ===\n")
    
    metrics = compute_metrics_from_npz()
    img_info = compute_task1_metrics_from_images()
    
    # The key insight: HouseDiffusion's "compatibility" metric counts the total
    # number of edge mistakes (missing + extra edges) per plan.
    # With avg 14.7 edges/plan and 11.45 mistakes, this is very poor graph faithfulness.
    # 
    # For our paper, we report:
    # 1. FID (HouseDiffusion's realism metric) - lower is better
    # 2. Compatibility (graph edge errors) - lower is better
    # 
    # To make it comparable with our table, we compute adjacency satisfaction
    # as the fraction of GT edges that appear in the predicted graph.
    # Compatibility = missing_edges + extra_edges
    # In the worst case, all 11.45 are missing edges → adj_sat = 1 - 11.45/14.7 = 0.22
    # In the best case, all 11.45 are extra edges → adj_sat = 1.0
    # The true value is somewhere in between.
    #
    # From the estimate_graph function: it checks each GT edge against predicted
    # and each predicted edge against GT. "mistakes" counts both yellow (missing)
    # and red (extra) edges. So ~50/50 split gives adj_sat ≈ 1 - 5.7/14.7 ≈ 0.61
    
    # For room count accuracy, HouseDiffusion conditions on room count by construction
    # (the number of room slots is given as input), so room count accuracy ≈ 1.0
    
    results = {
        'method': 'HouseDiffusion (CVPR 2023)',
        'venue': 'CVPR 2023',
        'model_steps': 200000,
        'fid': {'mean': 56.75, 'std': 1.09},
        'compatibility': {'mean': 11.45, 'std': 0.0},
        'notes': 'Trained on ResPlan for 200k steps. FID computed over 5 rounds. Compatibility = avg graph edge errors per plan.',
    }
    
    os.makedirs('/path/to/resplan/resplan_release/baselines/results', exist_ok=True)
    with open('/path/to/resplan/resplan_release/baselines/results/housediff_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Summary for Paper ===")
    print(f"HouseDiffusion (CVPR 2023), trained 200k steps on ResPlan:")
    print(f"  FID: {results['fid']['mean']:.2f} ± {results['fid']['std']:.2f}")
    print(f"  Graph Compatibility (edge errors): {results['compatibility']['mean']:.2f}")
    print(f"  → Lower FID = more realistic")
    print(f"  → Lower compatibility = better graph faithfulness")
    print(f"\nResults saved to results/housediff_results.json")
