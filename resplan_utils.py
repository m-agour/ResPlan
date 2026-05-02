"""
resplan_utils.py — Non–deep-learning helpers for the ResPlan-style floorplan datasets.

Dependencies (install as needed):
    pip install shapely geopandas matplotlib networkx numpy opencv-python

Contents:
    - Color maps and constants
    - Geometry utilities (get_geometries, centroid, perturb, noise)
    - Mask conversion (geometry_to_mask)
    - Augmentations (rotate/flip/scale)
    - Buffer helpers (shrink→expand, expand→shrink)
    - Plan plotting (plot_plan)
    - Plan→graph (plan_to_graph) + graph overlay plotting (plot_plan_and_graph)
    - Dataset helpers (normalize_keys, get_plan_width)
"""

from __future__ import annotations
import math
from typing import Iterable, List, Dict, Any, Tuple, Optional, Union

import numpy as np
import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString, Point, GeometryCollection, base, box
)
from shapely.ops import unary_union
from shapely import affinity

# -----------------------------
# Colors & constants
# -----------------------------

CATEGORY_COLORS: Dict[str, str] = {
    "living": "#d9d9d9",    # light gray
    "bedroom": "#66c2a5",    # greenish
    "bathroom": "#fc8d62",   # orange
    "kitchen": "#8da0cb",    # blue
    "door": "#e78ac3",       # pink
    "window": "#a6d854",     # lime
    "wall": "#ffd92f",       # yellow
    "front_door": "#a63603", # dark reddish-brown
    "balcony": "#b3b3b3",    # dark gray
    "storage": "#a37c52",    # brown
    "stair": "#9e9ac8",      # purple
}

DEFAULT_CANVAS_SIZE = (256, 256)  # (H, W)

# -----------------------------
# Dataset helpers
# -----------------------------

def normalize_keys(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common key typos / variations in-place (balacony→balcony)."""
    if "balacony" in plan and "balcony" not in plan:
        plan["balcony"] = plan.pop("balacony")
    return plan

def get_plan_width(plan: Dict[str, Any]) -> float:
    """Returns the max(width, height) of the inner polygon bounds."""
    inner = plan.get("inner")
    if inner is None or inner.is_empty:
        return 0.0
    x1, y1, x2, y2 = inner.bounds
    return max(x2 - x1, y2 - y1)

# -----------------------------
# Geometry utilities
# -----------------------------

def get_geometries(geom_data: Any) -> List[Any]:
    """Safely extract individual geometries from single/multi/collections."""
    if geom_data is None:
        return []
    if isinstance(geom_data, (Polygon, LineString, Point)):
        return [] if geom_data.is_empty else [geom_data]
    if isinstance(geom_data, (MultiPolygon, MultiLineString, GeometryCollection)):
        return [g for g in geom_data.geoms if g is not None and not g.is_empty]
    return []

def centroid(poly: Union[Polygon, MultiPolygon]) -> Point:
    """Centroid for Polygon/MultiPolygon (largest part if multi)."""
    if isinstance(poly, Polygon):
        return poly.centroid
    if isinstance(poly, MultiPolygon) and len(poly.geoms) > 0:
        largest = max(poly.geoms, key=lambda p: p.area)
        return largest.centroid
    return Point(-1e6, -1e6)

def perturb_polygon(polygon: Polygon, x_range: Tuple[float, float]=(-2, 2),
                    y_range: Tuple[float, float]=(-2, 2)) -> Polygon:
    """Apply random per-vertex perturbation to a polygon."""
    coords = np.asarray(polygon.exterior.coords, dtype=float)
    dx = np.random.uniform(x_range[0], x_range[1], size=len(coords))
    dy = np.random.uniform(y_range[0], y_range[1], size=len(coords))
    perturbed = np.column_stack([coords[:,0] + dx, coords[:,1] + dy])
    return Polygon(perturbed)

def noise(point: Point, noise_scale: float = 10.0) -> Point:
    """Jitter a point by uniform noise within ±noise_scale."""
    x, y = point.x, point.y
    return Point(x + np.random.uniform(-noise_scale, noise_scale),
                 y + np.random.uniform(-noise_scale, noise_scale))

# -----------------------------
# Augmentations
# -----------------------------

def augment_geom(geom: base.BaseGeometry,
                 degree: float = 0.0,
                 flip_vertical: bool = False,
                 scale: float = 1.0,
                 size: int = 256) -> base.BaseGeometry:
    """Rotate around image center, optional vertical flip (via negative y-scale), and scale."""
    if geom is None:
        return Point(-1e6, -1e6)
    g = affinity.rotate(geom, degree, origin=(size/2, size/2))
    flip = -1.0 if flip_vertical else 1.0
    return affinity.scale(g, xfact=scale, yfact=scale * flip, origin=(size/2, size/2))

# -----------------------------
# Buffer helpers
# -----------------------------

def buffer_shrink_expand(geom: base.BaseGeometry, w: float,
                         join_style: int = 2, cap_style: int = 2) -> base.BaseGeometry:
    """Shrink then expand by w (useful for cleaning)."""
    return geom.buffer(-w, join_style=join_style, cap_style=cap_style)                   .buffer(+w, join_style=join_style, cap_style=cap_style)

def buffer_expand_shrink(geom: base.BaseGeometry, w: float,
                         join_style: int = 2, cap_style: int = 2) -> base.BaseGeometry:
    """Expand then shrink by w (useful for filling tiny gaps)."""
    return geom.buffer(+w, join_style=join_style, cap_style=cap_style)                   .buffer(-w, join_style=join_style, cap_style=cap_style)

# -----------------------------
# Geometry → mask
# -----------------------------

def _poly_to_mask(poly: Polygon, shape: Tuple[int, int], line_thickness: int = 0) -> np.ndarray:
    h, w = shape
    img = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(poly.exterior.coords, dtype=np.int32)
    if line_thickness > 0:
        cv2.polylines(img, [pts], isClosed=True, color=255, thickness=line_thickness)
    else:
        cv2.fillPoly(img, [pts], color=255)
    for interior in poly.interiors:
        pts_in = np.array(interior.coords, dtype=np.int32)
        if line_thickness > 0:
            cv2.polylines(img, [pts_in], isClosed=True, color=0, thickness=line_thickness)
        else:
            cv2.fillPoly(img, [pts_in], color=0)
    return img

def geometry_to_mask(geom: Any,
                     shape: Tuple[int, int] = DEFAULT_CANVAS_SIZE,
                     point_radius: int = 5,
                     line_thickness: int = 0) -> np.ndarray:
    """Rasterize Polygon/MultiPolygon/LineString/Point/iterables to a binary mask [0,255]."""
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)

    # Single geometry
    if isinstance(geom, Polygon):
        return _poly_to_mask(geom, shape, line_thickness)
    if isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            out = np.maximum(out, _poly_to_mask(p, shape, line_thickness))
        return out
    if isinstance(geom, LineString):
        pts = np.array(geom.coords, dtype=np.int32)
        cv2.polylines(out, [pts], isClosed=False, color=255, thickness=max(1, line_thickness or 1))
        return out
    if isinstance(geom, MultiLineString):
        for ls in geom.geoms:
            pts = np.array(ls.coords, dtype=np.int32)
            cv2.polylines(out, [pts], isClosed=False, color=255, thickness=max(1, line_thickness or 1))
        return out
    if isinstance(geom, Point):
        cx, cy = int(round(geom.x)), int(round(geom.y))
        cv2.circle(out, (cx, cy), point_radius, 255, -1)
        return out
    if isinstance(geom, Iterable):
        for g in geom:
            out = np.maximum(out, geometry_to_mask(g, shape, point_radius, line_thickness))
        return out
    # Unrecognized → empty
    return out

# -----------------------------
# Plotting
# -----------------------------

def plot_plan(plan: Dict[str, Any],
              categories: Optional[List[str]] = None,
              colors: Dict[str, str] = CATEGORY_COLORS,
              ax: Optional[plt.Axes] = None,
              legend: bool = True,
              title: Optional[str] = None,
              tight: bool = True) -> plt.Axes:
    """Plot a single plan with colored layers."""
    plan = normalize_keys(plan)
    if categories is None:
        categories = ["living","bedroom","bathroom","kitchen","storage","stair","door","window","wall","front_door","balcony"]

    geoms, color_list, present = [], [], []
    for key in categories:
        geom = plan.get(key)
        if geom is None:
            continue
        parts = get_geometries(geom)
        if not parts:
            continue
        geoms.extend(parts)
        color_list.extend([colors.get(key, "#000000")] * len(parts))
        present.append(key)

    if not geoms:
        raise ValueError("No geometries to plot.")

    gseries = gpd.GeoSeries(geoms)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    gseries.plot(ax=ax, color=color_list, edgecolor="black", linewidth=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    if title:
        ax.set_title(title)

    if legend:
        from matplotlib.patches import Patch
        uniq_present = list(dict.fromkeys(present))  # preserve order
        handles = [Patch(facecolor=colors.get(k, "#000000"), edgecolor="black", label=k.replace("_"," ")) for k in uniq_present]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1,1), frameon=False)

    if tight:
        plt.tight_layout()
    return ax

# -----------------------------
# Plan → Graph
# -----------------------------

def plan_to_graph(plan: Dict[str, Any],
                  buffer_factor: float = 0.007,
                  open_passage_min_factor: float = 2.0) -> nx.Graph:
    """Create a room connectivity graph from a floor plan.

    Nodes represent functional spaces (living, kitchen, bedroom, bathroom,
    balcony, storage, stair, front_door).  Edges encode spatial
    relationships:

    * **via_door** / **via_window** – two rooms bridged by a door or
      window geometry (type-agnostic: any pair of rooms sharing a
      connector).
    * **via_opening** – two rooms whose shared boundary contains a
      contiguous wall-free segment of length ≥ ``open_passage_min_factor
      × wall_depth`` (open kitchens, archways, walk-throughs). Replaces
      the old generic ``adjacency`` edge that fired on any buffered
      overlap and produced spurious connections between rooms separated
      by a wall.
    * **direct** – front-door nodes linked to rooms they physically touch.

    Two rooms separated by a wall with no door / window / open passage
    receive **no edge** (the truthful answer: you cannot walk between
    them). A nearest-neighbour fallback is used only as a last resort to
    keep the graph connected for downstream code that assumes one
    component.

    The buffer size is *scale-aware*: ``buffer_factor × max(plan width,
    plan height)``, which accounts for wall thickness across plans of
    different coordinate scales.

    Parameters
    ----------
    plan : dict
        A ResPlan plan dictionary with Shapely geometry keys.
    buffer_factor : float, optional
        Fraction of plan extent used as the spatial buffer (default 0.007,
        ≈ 1.8 px for a 256-unit canvas – just enough to bridge typical
        walls).
    open_passage_min_factor : float, optional
        Minimum length of contiguous wall-free shared-boundary segment
        required to declare an open passage between two rooms, in
        multiples of ``wall_depth`` (default 2.0). Below this, small
        gaps at corners / discretisation artefacts are ignored.

    Returns
    -------
    nx.Graph
        Room graph with node attributes ``geometry``, ``type``, ``area``
        and edge attribute ``type``.
    """
    plan = normalize_keys(plan)
    G = nx.Graph()

    # --- scale-aware buffer ---
    inner = plan.get("inner")
    if inner is not None and not inner.is_empty:
        x1, y1, x2, y2 = inner.bounds
        plan_size = max(x2 - x1, y2 - y1)
    else:
        plan_size = 256.0
    buf = max(plan_size * buffer_factor, 0.5)
    wd = float(plan.get("wall_depth") or 4.0)
    open_min = open_passage_min_factor * wd

    all_nodes: List[str] = []

    # Living: union all parts into a single node (standard convention)
    living_parts = [p for p in get_geometries(plan.get("living"))
                    if isinstance(p, Polygon) and not p.is_empty]
    if living_parts:
        living_geom = unary_union(living_parts)
        if not living_geom.is_empty:
            G.add_node("living_0", geometry=living_geom,
                       type="living", area=living_geom.area)
            all_nodes.append("living_0")

    # Other room types: one node per polygon part. ``storage`` and
    # ``stair`` were previously omitted, which silently dropped doors
    # leading to those rooms from the graph.
    for room_type in ["kitchen", "bedroom", "bathroom",
                       "balcony", "storage", "stair"]:
        for i, geom in enumerate(get_geometries(plan.get(room_type))):
            if isinstance(geom, Polygon) and not geom.is_empty:
                nid = f"{room_type}_{i}"
                G.add_node(nid, geometry=geom, type=room_type, area=geom.area)
                all_nodes.append(nid)

    # Front door (may be line or polygon)
    for i, geom in enumerate(get_geometries(plan.get("front_door"))):
        nid = f"front_door_{i}"
        G.add_node(nid, geometry=geom, type="front_door",
                   area=getattr(geom, "area", 0.0))
        all_nodes.append(nid)

    if not all_nodes:
        return G

    # Pre-buffer node geometries once
    node_buf = {nid: G.nodes[nid]["geometry"].buffer(buf)
                for nid in all_nodes}

    # --- door / window connections (type-agnostic) ---
    # A door spans exactly one wall and therefore connects exactly two
    # rooms. When a door sits in the corner where ≥3 rooms meet, the
    # naive "any room within buf" test creates spurious edges with the
    # third room (e.g. plan 14926 d2: kitchen↔bedroom). To pick the
    # correct two rooms we score each candidate by *contact length* —
    # the length of the connector's boundary that lies inside the room
    # (after a small buffer to bridge wall thickness). Real connections
    # have a contact line ≈ the door's long side; corner overlaps
    # produce only tiny intersections.
    doors  = get_geometries(plan.get("door"))
    windows = get_geometries(plan.get("window"))

    def _contact_score(cgeom, room_geom):
        try:
            return cgeom.boundary.intersection(room_geom.buffer(buf * 0.6)).length
        except Exception:
            return 0.0

    for conn_list, edge_type in [(doors, "via_door"),
                                 (windows, "via_window")]:
        for cgeom in conn_list:
            touching = [n for n in all_nodes
                        if cgeom.intersects(node_buf[n])]
            if len(touching) <= 1:
                continue
            if len(touching) > 2:
                # Score and keep the two rooms with the longest contact.
                scored = sorted(
                    ((n, _contact_score(cgeom, G.nodes[n]["geometry"]))
                     for n in touching),
                    key=lambda x: x[1], reverse=True)
                # Drop candidates whose contact is < 25% of the best
                # one — those are corner artefacts. Always keep at
                # least the top-2 so we don't lose a valid edge.
                best = scored[0][1]
                kept = [scored[0][0], scored[1][0]]
                for n, s in scored[2:]:
                    if s >= 0.25 * best and s >= buf:
                        kept.append(n)
                touching = kept
            for i in range(len(touching)):
                for j in range(i + 1, len(touching)):
                    if not G.has_edge(touching[i], touching[j]):
                        G.add_edge(touching[i], touching[j],
                                   type=edge_type)

    # --- front door → any room reachable through it (direct) ---
    # The front door is itself a passage; treat it like an interior door:
    # link it to rooms whose geometry it actually overlaps (after a small
    # ``buf`` margin to bridge the wall band). Using a larger buffer here
    # produced spurious links to rooms that merely sit close to the door
    # on the other side of a wall (e.g. plan 11053 front_door↔bedroom_0).
    for nid in all_nodes:
        if G.nodes[nid]["type"] != "front_door":
            continue
        fd_geom = G.nodes[nid]["geometry"]
        fd_buf = fd_geom.buffer(buf)
        for other in all_nodes:
            if other == nid or G.nodes[other]["type"] == "front_door":
                continue
            if fd_buf.intersects(G.nodes[other]["geometry"]):
                if not G.has_edge(nid, other):
                    G.add_edge(nid, other, type="direct")

    # --- via_opening: rooms with a wall-free shared boundary ---
    # The previous "adjacency" edge fired on any buffered overlap, which
    # spuriously linked rooms separated by a wall (e.g. two bedrooms
    # sharing a wall, or living↔kitchen through a 24cm wall band with no
    # door). Real connections require a contiguous gap in the wall band
    # along the shared boundary — this is what an open kitchen or
    # archway looks like geometrically.
    walls = plan.get("wall")
    door_geom = plan.get("door")
    win_geom  = plan.get("window")
    obstacles = [g for g in (walls, door_geom, win_geom) if g is not None and not g.is_empty]
    obstacle_union = unary_union(obstacles) if obstacles else None
    # Doors and windows do not "block" — they are passages — but we want
    # to claim them under via_door/via_window, which already happened
    # above. So for residual via_opening detection we only treat *walls*
    # as obstacles.
    wall_buf_inner = walls.buffer(0.5) if (walls is not None and not walls.is_empty) else None

    room_node_ids = [n for n in all_nodes
                     if G.nodes[n]["type"] != "front_door"]
    for i in range(len(room_node_ids)):
        for j in range(i + 1, len(room_node_ids)):
            a_id, b_id = room_node_ids[i], room_node_ids[j]
            if G.has_edge(a_id, b_id):
                continue
            a = G.nodes[a_id]["geometry"]
            b = G.nodes[b_id]["geometry"]
            # Primary test: do the two room polygons, dilated by ~wd to
            # cover the wall band but with the wall geometry subtracted,
            # overlap on a region wider than ``open_min``? This is the
            # "subtract walls and see if rooms touch" check — robust to
            # boundary discretisation and corner cases.
            edge_added = False
            try:
                a_d = a.buffer(wd * 0.55)
                b_d = b.buffer(wd * 0.55)
                gap = a_d.intersection(b_d)
                if walls is not None and not walls.is_empty:
                    gap = gap.difference(walls.buffer(0.0))
                if not gap.is_empty:
                    pieces = list(gap.geoms) if hasattr(gap, "geoms") else [gap]
                    # An "opening" must be wide enough to walk through:
                    # an extent of at least ``open_min`` in some
                    # direction. Use the longest side of each piece's
                    # bounding box as a cheap proxy.
                    for pc in pieces:
                        if pc.is_empty:
                            continue
                        x1p, y1p, x2p, y2p = pc.bounds
                        extent = max(x2p - x1p, y2p - y1p)
                        if extent >= open_min and pc.area >= wd * wd * 0.25:
                            G.add_edge(a_id, b_id, type="via_opening")
                            edge_added = True
                            break
            except Exception:
                pass
            if edge_added:
                continue

            # Fallback test: shared boundary segment with wall mask.
            try:
                contact = a.boundary.intersection(b.buffer(wd * 0.6))
            except Exception:
                continue
            if contact.is_empty:
                continue
            total = contact.length
            if total < open_min:
                continue
            if wall_buf_inner is not None:
                try:
                    open_seg = contact.difference(wall_buf_inner)
                except Exception:
                    open_seg = contact
            else:
                open_seg = contact
            if open_seg.is_empty:
                continue
            # Largest contiguous open segment must exceed the minimum.
            if hasattr(open_seg, "geoms"):
                seg_lens = [g.length for g in open_seg.geoms]
            else:
                seg_lens = [open_seg.length]
            if max(seg_lens, default=0.0) >= open_min:
                G.add_edge(a_id, b_id, type="via_opening")

    # --- post-fix: a via_window edge between two interior rooms is a
    # misclassified door (real interior windows are rare and ResPlan
    # should not have any after fix_plans.py); promote to via_door so
    # the graph reflects the truth that those rooms are connected by a
    # real walkable opening. ---
    INTERIOR = {"living", "kitchen", "bedroom", "bathroom",
                "storage", "stair"}
    for u, v, d in list(G.edges(data=True)):
        if d.get("type") != "via_window":
            continue
        tu = G.nodes[u].get("type")
        tv = G.nodes[v].get("type")
        if tu in INTERIOR and tv in INTERIOR:
            G[u][v]["type"] = "via_door"

    # --- fallback: connect remaining isolated components ---
    # Only as a last resort to keep the graph in one piece — labelled
    # ``fallback`` so it is distinguishable from genuine connections.
    if G.number_of_nodes() > 1:
        components = list(nx.connected_components(G))
        if len(components) > 1:
            components.sort(key=len, reverse=True)
            main_comp = components[0]
            for comp in components[1:]:
                best_dist = float("inf")
                best_pair = None
                for cn in comp:
                    cg = G.nodes[cn]["geometry"]
                    for mn in main_comp:
                        d = cg.distance(G.nodes[mn]["geometry"])
                        if d < best_dist:
                            best_dist = d
                            best_pair = (cn, mn)
                if best_pair is not None:
                    G.add_edge(best_pair[0], best_pair[1],
                               type="fallback")
                    main_comp = main_comp | comp

    return G

# -----------------------------
# Graph overlay on plan
# -----------------------------

def plot_plan_and_graph(plan: Dict[str, Any],
                        ax: Optional[plt.Axes] = None,
                        node_scale: Tuple[float,float]=(150, 1000),
                        title: Optional[str] = None) -> plt.Axes:
    """Plot plan and overlay the room graph (node size scaled by room area)."""
    G = plan["graph"] if "graph" in plan else plan_to_graph(plan)
    ax = plot_plan(plan, legend=True, ax=ax, title=title)

    # node positions = centroids
    pos = {}
    for n, data in G.nodes(data=True):
        geom = data.get("geometry")
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        pos[n] = (c.x, c.y)

    # style maps
    node_style = {
        "living":    dict(color="white",     shape="o", size=400, edgecolor="black"),
        "bedroom":    dict(color="cyan",      shape="s", size=300, edgecolor="black"),
        "bathroom":   dict(color="magenta",   shape="D", size=260, edgecolor="black"),
        "kitchen":    dict(color="yellow",    shape="^", size=300, edgecolor="black"),
        "balcony":    dict(color="lightgray", shape="X", size=260, edgecolor="black"),
        "storage":    dict(color="#a37c52",   shape="p", size=260, edgecolor="black"),
        "stair":      dict(color="#9e9ac8",   shape="h", size=260, edgecolor="black"),
        "front_door": dict(color="red",       shape="*", size=420, edgecolor="black"),
    }

    # draw nodes per type for shapes
    nodes_plotted = set()
    min_size, max_size = node_scale
    # area-based scaling
    areas = [G.nodes[n].get("area", 0.0) for n in G.nodes]
    a_min = min(areas) if areas else 0.0
    a_max = max(areas) if areas else 1.0
    def scale_size(a):
        if a_max <= a_min:
            return (min_size + max_size) / 2
        t = (a - a_min) / (a_max - a_min)
        return min_size + t * (max_size - min_size)

    for t, style in node_style.items():
        nlist = [n for n, d in G.nodes(data=True) if d.get("type")==t and n in pos]
        if not nlist:
            continue
        sizes = [scale_size(G.nodes[n].get("area", 0.0)) for n in nlist]
        nx.draw_networkx_nodes(
            G, pos, nodelist=nlist, node_size=sizes,
            node_shape=style["shape"], node_color=style["color"],
            edgecolors=style["edgecolor"], linewidths=1.0, ax=ax, alpha=0.9
        )
        nodes_plotted.update(nlist)

    # edges by type
    edge_style = {
        "direct":      dict(color="darkred",   width=2.0,  style="-"),
        "via_door":    dict(color="darkblue",  width=1.2,  style="-"),
        "via_window":  dict(color="orange",    width=1.0,  style=":"),
        "via_opening": dict(color="darkgreen", width=1.5,  style="--"),
        # legacy edge name still rendered for backwards compatibility
        "adjacency":   dict(color="darkgreen", width=1.5,  style="--"),
        "fallback":    dict(color="gray",      width=0.8,  style=":"),
    }
    for etype, style in edge_style.items():
        elist = [(u,v) for u,v,d in G.edges(data=True) if d.get("type")==etype and u in pos and v in pos]
        if not elist:
            continue
        nx.draw_networkx_edges(G, pos, edgelist=elist,
                               width=style["width"], edge_color=style["color"],
                               style=style["style"], ax=ax, alpha=0.8)

    if title:
        ax.set_title(title)
    plt.tight_layout()
    return ax
