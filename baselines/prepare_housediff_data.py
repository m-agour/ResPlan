#!/usr/bin/env python3
"""
prepare_housediff_data.py
Convert ResPlan PKL data → HouseDiffusion-compatible npz files.

Creates processed npz files that HouseDiffusion's RPlanhgDataset can load
directly (bypassing the JSON→mask→contour pipeline).

Room type mapping (→ RPLAN integers for compatibility with HouseDiffusion viz):
  living    → 1
  bedroom   → 2
  kitchen   → 3
  bathroom  → 4
  balcony   → 10
  front_door→ 17

Output: processed_rplan/rplan_{train|eval}_{target_set}.npz
"""

import os, sys, json, pickle
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon, MultiPolygon
from shapely import affinity
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────
MAX_NUM_POINTS = 200        # max corners per plan (RPLAN uses 100)
MAX_CORNERS_PER_ROOM = 16   # simplify rooms to ≤16 corners
ONE_HOT_ROOM = 25           # room-type one-hot dim (match HouseDiffusion)
ONE_HOT_CORNER = 32         # corner-index one-hot dim
ONE_HOT_ROOMIDX = 32        # room-index one-hot dim
FEATURE_DIM = 2 + ONE_HOT_ROOM + ONE_HOT_CORNER + ONE_HOT_ROOMIDX + 1 + 2  # = 94

# Room-type mapping  →  RPLAN-compatible integers
# Note: front_door uses 12 (the value HouseDiffusion remaps 17→12 internally)
# so that npz features are consistent with the model's expected encoding.
ROOM_TYPE_MAP = {
    'living':     1,
    'bedroom':    2,
    'kitchen':    3,
    'bathroom':   4,
    'balcony':   10,
    'front_door': 12,
}

ROOM_CATEGORIES = ['living', 'bedroom', 'bathroom', 'kitchen', 'balcony', 'front_door']


def get_one_hot(idx, size):
    v = np.zeros(size)
    if 0 <= idx < size:
        v[idx] = 1.0
    return v


def simplify_polygon(poly, max_corners=MAX_CORNERS_PER_ROOM):
    """Simplify a Shapely Polygon to at most max_corners vertices."""
    coords = np.array(poly.exterior.coords[:-1])  # drop closing point
    if len(coords) <= max_corners:
        return coords

    # Douglas-Peucker simplification with increasing tolerance
    tol = 0.5
    for _ in range(20):
        simplified = poly.simplify(tol, preserve_topology=True)
        if simplified.is_empty or not isinstance(simplified, Polygon):
            break
        n = len(simplified.exterior.coords) - 1
        if n <= max_corners:
            return np.array(simplified.exterior.coords[:-1])
        tol *= 1.5

    # Fallback: uniformly sample corners
    coords = np.array(poly.exterior.coords[:-1])
    indices = np.linspace(0, len(coords) - 1, max_corners, dtype=int)
    return coords[indices]


def extract_rooms(plan):
    """Extract individual room polygons with type labels from a plan dict."""
    rooms = []
    for cat in ROOM_CATEGORIES:
        geom = plan.get(cat)
        if geom is None or geom.is_empty:
            continue
        if isinstance(geom, Polygon):
            rooms.append((cat, geom))
        elif isinstance(geom, MultiPolygon):
            for g in geom.geoms:
                if not g.is_empty:
                    rooms.append((cat, g))
    return rooms


def normalize_coords(rooms):
    """Normalize all room corners to [-0.5, 0.5] centered."""
    all_coords = []
    for _, poly in rooms:
        all_coords.append(np.array(poly.exterior.coords[:-1]))
    if not all_coords:
        return rooms, 1.0, np.zeros(2)

    all_pts = np.concatenate(all_coords, axis=0)
    mn = all_pts.min(axis=0)
    mx = all_pts.max(axis=0)
    center = (mn + mx) / 2.0
    scale = max(mx[0] - mn[0], mx[1] - mn[1])
    if scale < 1e-6:
        scale = 1.0

    normalized_rooms = []
    for cat, poly in rooms:
        coords = np.array(poly.exterior.coords[:-1])
        coords = (coords - center) / scale  # now in ~ [-0.5, 0.5]
        normalized_rooms.append((cat, coords))
    return normalized_rooms, scale, center


def build_graph_triples(rooms, plan_graph):
    """Build HouseDiffusion-style graph triples [room_i, 1/-1, room_j].

    Uses ResPlan's graph to determine connectivity.  When extract_rooms()
    splits a MultiPolygon (e.g. living) into living_0, living_1, living_2,
    but the graph only contains living_0, we fall back: every split part
    maps to the base graph node (living_0) so its edges are inherited.
    Parts that map to the same graph node are also marked connected (1).
    """
    if plan_graph is None:
        n = len(rooms)
        triples = []
        for i in range(n):
            for j in range(i + 1, n):
                triples.append([i, -1, j])
        return np.array(triples) if triples else np.zeros((0, 3))

    graph_nodes = set(plan_graph.nodes())

    # Map each extracted room to its graph node name.
    # If a name doesn't exist in the graph (e.g. living_1 when graph has
    # only living_0), fall back to <cat>_0.
    room_to_gnode = []
    for i, (cat, _) in enumerate(rooms):
        count = sum(1 for j in range(i) if rooms[j][0] == cat)
        name = f"{cat}_{count}"
        if name in graph_nodes:
            room_to_gnode.append(name)
        else:
            base = f"{cat}_0"
            room_to_gnode.append(base if base in graph_nodes else name)

    triples = []
    n = len(rooms)
    for i in range(n):
        for j in range(i + 1, n):
            gi, gj = room_to_gnode[i], room_to_gnode[j]
            # Same graph node (e.g. both living parts) → connected
            if gi == gj:
                triples.append([i, 1, j])
            elif plan_graph.has_edge(gi, gj):
                triples.append([i, 1, j])
            else:
                triples.append([i, -1, j])

    return np.array(triples) if triples else np.zeros((0, 3))


def convert_plan(plan, max_num_points=MAX_NUM_POINTS):
    """Convert a single ResPlan plan to HouseDiffusion format.

    Returns: (house_layout, graph, door_mask, self_mask, gen_mask) or None
    """
    rooms = extract_rooms(plan)
    if len(rooms) < 3:
        return None

    # Normalize coordinates
    norm_rooms, scale, center = normalize_coords(
        [(cat, poly) for cat, poly in rooms]
    )

    # Simplify polygons and collect corners
    simplified_rooms = []
    for cat, poly in rooms:
        coords_orig = np.array(poly.exterior.coords[:-1])
        coords_simplified = simplify_polygon(poly, MAX_CORNERS_PER_ROOM)
        # Normalize
        coords_norm = (coords_simplified - center) / scale
        simplified_rooms.append((cat, coords_norm))

    # Check total corners
    total_corners = sum(len(c) for _, c in simplified_rooms)
    if total_corners > max_num_points:
        return None

    # Build the feature matrix
    house_layout = np.zeros((max_num_points, FEATURE_DIM))
    corner_bounds = []
    num_points = 0

    for room_idx, (cat, coords) in enumerate(simplified_rooms):
        rtype = ROOM_TYPE_MAP[cat]
        num_corners = len(coords)

        for ci in range(num_corners):
            row = num_points + ci
            # Coordinates (scaled to [-1, 1])
            house_layout[row, 0] = coords[ci, 0] * 2.0  # [-0.5,0.5] → [-1,1]
            house_layout[row, 1] = coords[ci, 1] * 2.0
            # Room type one-hot
            house_layout[row, 2 + rtype] = 1.0
            # Corner index one-hot
            if ci < ONE_HOT_CORNER:
                house_layout[row, 2 + ONE_HOT_ROOM + ci] = 1.0
            # Room index one-hot
            if room_idx + 1 < ONE_HOT_ROOMIDX:
                house_layout[row, 2 + ONE_HOT_ROOM + ONE_HOT_CORNER + room_idx + 1] = 1.0
            # Padding mask (1 = valid)
            house_layout[row, 2 + ONE_HOT_ROOM + ONE_HOT_CORNER + ONE_HOT_ROOMIDX] = 1.0
            # Connections (prev_corner, next_corner)
            conn_offset = 2 + ONE_HOT_ROOM + ONE_HOT_CORNER + ONE_HOT_ROOMIDX + 1
            house_layout[row, conn_offset] = num_points + (ci - 1) % num_corners
            house_layout[row, conn_offset + 1] = num_points + (ci + 1) % num_corners

        corner_bounds.append((num_points, num_points + num_corners))
        num_points += num_corners

    # Build graph triples
    graph = build_graph_triples(simplified_rooms, plan.get('graph'))

    # Build masks
    gen_mask = np.ones((max_num_points, max_num_points))
    gen_mask[:num_points, :num_points] = 0

    self_mask = np.ones((max_num_points, max_num_points))
    door_mask = np.ones((max_num_points, max_num_points))

    # Find living room index for default connections
    living_room_idx = None
    for i, (cat, _) in enumerate(simplified_rooms):
        if cat == 'living':
            living_room_idx = i
            break

    for i in range(len(corner_bounds)):
        for j in range(len(corner_bounds)):
            bi = corner_bounds[i]
            bj = corner_bounds[j]
            if i == j:
                self_mask[bi[0]:bi[1], bj[0]:bj[1]] = 0
            else:
                # Check if rooms i,j are connected via graph
                is_connected = False
                for triple in graph:
                    if (triple[0] == i and triple[2] == j and triple[1] == 1) or \
                       (triple[0] == j and triple[2] == i and triple[1] == 1):
                        is_connected = True
                        break
                if is_connected:
                    door_mask[bi[0]:bi[1], bj[0]:bj[1]] = 0

        # If room is not connected to any other, connect to living room
        if living_room_idx is not None:
            has_any_connection = False
            for triple in graph:
                if (triple[0] == i or triple[2] == i) and triple[1] == 1:
                    has_any_connection = True
                    break
            if not has_any_connection and i != living_room_idx:
                bi = corner_bounds[i]
                bl = corner_bounds[living_room_idx]
                door_mask[bi[0]:bi[1], bl[0]:bl[1]] = 0

    return house_layout, graph, door_mask, self_mask, gen_mask


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resplan_pkl', default='../../dataset/ResPlan.pkl')
    parser.add_argument('--split_json', default='../../dataset/split.json')
    parser.add_argument('--output_dir', default='../../house_diffusion/processed_rplan')
    parser.add_argument('--target_set', type=int, default=0,
                        help='target_set value for filename (0 = all plans)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading ResPlan data...")
    with open(args.resplan_pkl, 'rb') as f:
        data = pickle.load(f)
    with open(args.split_json, 'r') as f:
        split = json.load(f)

    # Build id→split mapping
    id_to_split = {}
    for s in ['train', 'val', 'test']:
        for pid in split[s]:
            id_to_split[int(pid)] = s

    # Process each split
    for split_name in ['train', 'eval']:
        # 'eval' uses val+test plans (HouseDiffusion convention)
        target_splits = ['train'] if split_name == 'train' else ['test']

        houses = []
        door_masks_list = []
        self_masks_list = []
        gen_masks_list = []
        graphs_list = []
        cnumber_dist = defaultdict(list)
        skipped = 0

        plans_for_split = [p for p in data if id_to_split.get(p['id']) in target_splits]
        print(f"\nProcessing {split_name}: {len(plans_for_split)} plans")

        for plan in tqdm(plans_for_split, desc=split_name):
            result = convert_plan(plan, MAX_NUM_POINTS)
            if result is None:
                skipped += 1
                continue

            house_layout, graph, door_mask, self_mask, gen_mask = result
            houses.append(house_layout)
            door_masks_list.append(door_mask)
            self_masks_list.append(self_mask)
            gen_masks_list.append(gen_mask)

            # Pad graph to fixed size (200 triples max)
            if len(graph) > 0:
                padded_graph = np.zeros((200, 3))
                n = min(len(graph), 200)
                padded_graph[:n] = graph[:n]
                graphs_list.append(padded_graph)
            else:
                graphs_list.append(np.zeros((200, 3)))

            # Track corner number distribution (for eval synthetic data)
            if split_name == 'train':
                # Parse rooms from house_layout
                room_types_col = house_layout[:, 2:2 + ONE_HOT_ROOM]
                padding = house_layout[:, 2 + ONE_HOT_ROOM + ONE_HOT_CORNER + ONE_HOT_ROOMIDX]
                room_idx_col = house_layout[:, 2 + ONE_HOT_ROOM + ONE_HOT_CORNER:
                                              2 + ONE_HOT_ROOM + ONE_HOT_CORNER + ONE_HOT_ROOMIDX]

                valid = padding > 0.5
                if valid.sum() > 0:
                    # Group corners by room index
                    room_indices = np.argmax(room_idx_col[valid], axis=1)
                    room_types = np.argmax(room_types_col[valid], axis=1)
                    for ri in np.unique(room_indices):
                        mask = room_indices == ri
                        rt = room_types[mask][0]
                        ncorners = mask.sum()
                        cnumber_dist[rt].append(int(ncorners))

        print(f"  Converted: {len(houses)}, Skipped: {skipped}")

        # Save npz
        ts = args.target_set
        fname = os.path.join(args.output_dir, f'rplan_{split_name}_{ts}.npz')
        np.savez_compressed(fname,
                            graphs=np.array(graphs_list),
                            houses=np.array(houses),
                            door_masks=np.array(door_masks_list),
                            self_masks=np.array(self_masks_list),
                            gen_masks=np.array(gen_masks_list))
        print(f"  Saved: {fname}")

        # Save corner number distribution (train only)
        if split_name == 'train':
            cnd_fname = os.path.join(args.output_dir, f'rplan_train_{ts}_cndist.npz')
            np.savez_compressed(cnd_fname, cnumber_dist=dict(cnumber_dist))
            print(f"  Saved: {cnd_fname}")

        # For eval, also create synthetic data (copy of eval for compatibility)
        if split_name == 'eval':
            syn_fname = os.path.join(args.output_dir, f'rplan_eval_{ts}_syn.npz')
            np.savez_compressed(syn_fname,
                                graphs=np.array(graphs_list),
                                houses=np.array(houses),
                                door_masks=np.array(door_masks_list),
                                self_masks=np.array(self_masks_list),
                                gen_masks=np.array(gen_masks_list))
            print(f"  Saved: {syn_fname}")

    print("\nDone! Data ready for HouseDiffusion training.")
    print(f"Output directory: {args.output_dir}")
    print(f"\nUsage:")
    print(f"  cd house_diffusion/scripts")
    print(f"  python image_train.py --dataset rplan --batch_size 32 --set_name train --target_set {args.target_set}")


if __name__ == '__main__':
    main()
