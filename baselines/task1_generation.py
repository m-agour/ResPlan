#!/usr/bin/env python3
"""
Task 1: Constrained Floor Plan Generation — Learned Baselines (GPU)

Two conditional VAE baselines for room layout generation:
  1. CVAE:        Conditioned on room counts + boundary aspect ratio
  2. CVAE+Graph:  Additionally conditioned on adjacency type-pair features

Architecture:
  - Layout = 5 types × MAX_PER_TYPE rooms × 4 coords (cx,cy,w,h), sorted by area
  - Coordinates normalised to [0,1] wrt max(plan_width, plan_height) to preserve AR
  - Encoder: layout(Ld)+cond → 512 → 256 → μ(128), σ(128)
  - Decoder: z(128)+cond → 256 → 512 → layout(Ld)
  - β-VAE loss with KL annealing; masked reconstruction on valid slots

Metrics (same as retrieval baselines for direct comparison):
  - Room Count Accuracy:   fraction of types with matching count
  - Adjacency Satisfaction: multiset Jaccard of type-pair adjacency
  - Boundary IoU:           rescaled aspect-ratio match

Usage:
  python task1_generation.py                       # auto-detect GPU
  python task1_generation.py --device cuda          # force GPU
  python task1_generation.py --epochs 500           # custom epochs
  python task1_generation.py --samples 10           # best-of-K sampling

Outputs:
  results/task1_generation.json
"""
import argparse, json, os, pickle, sys, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter

warnings.filterwarnings("ignore")

# ─── Constants ────────────────────────────────────────────────────────────────
ROOM_TYPES = ["bedroom", "bathroom", "kitchen", "living", "balcony"]
NUM_TYPES = 5
MAX_PER_TYPE = 8          # covers >99.5% of plans
COORDS_PER_ROOM = 4       # cx, cy, w, h
LAYOUT_DIM = NUM_TYPES * MAX_PER_TYPE * COORDS_PER_ROOM  # 5×8×4 = 160
COND_DIM_BASE = NUM_TYPES + 1  # room counts (5) + boundary AR (1) = 6
COND_DIM_GRAPH = 15            # upper-triangle of 5×5 type-pair adj matrix
LATENT_DIM = 128
HIDDEN_DIM = 512
ADJ_THRESHOLD = 0.03  # normalised distance threshold for box adjacency


# ─── Determinism ──────────────────────────────────────────────────────────────
def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Data Preprocessing ──────────────────────────────────────────────────────
def extract_plan_features(plan):
    """Extract room bounding boxes, count vector, boundary AR, and graph
    features from a single plan dict.

    Returns:
        rooms_by_type: dict  {type_idx: [(cx, cy, w, h), ...]}  sorted by area↓
        count_vec:     np.array  (5,)
        boundary_ar:   float   (width / height of inner polygon)
        graph_vec:     np.array (15,) type-pair edge counts (upper triangle)
    Returns (None, ...) on degenerate plans.
    """
    from shapely.geometry import Polygon, MultiPolygon

    inner = plan.get("inner")
    if inner is None or not hasattr(inner, "bounds") or inner.is_empty:
        return None, None, None, None

    x1, y1, x2, y2 = inner.bounds
    pw, ph = x2 - x1, y2 - y1
    if pw <= 0 or ph <= 0:
        return None, None, None, None

    scale = max(pw, ph)
    boundary_ar = np.float32(pw / ph)

    rooms_by_type = {i: [] for i in range(NUM_TYPES)}
    for type_idx, rt in enumerate(ROOM_TYPES):
        geo = plan.get(rt)
        if geo is None or not hasattr(geo, "is_empty") or geo.is_empty:
            continue
        polys = list(geo.geoms) if hasattr(geo, "geoms") else [geo]
        for poly in polys:
            if poly.is_empty:
                continue
            bx1, by1, bx2, by2 = poly.bounds
            cx = np.clip(((bx1 + bx2) / 2 - x1) / scale, 0, 1)
            cy = np.clip(((by1 + by2) / 2 - y1) / scale, 0, 1)
            w  = np.clip((bx2 - bx1) / scale, 0, 1)
            h  = np.clip((by2 - by1) / scale, 0, 1)
            if w * h > 1e-6:
                rooms_by_type[type_idx].append((float(cx), float(cy),
                                                 float(w), float(h)))
    # Sort each type by area descending
    for t in rooms_by_type:
        rooms_by_type[t] = sorted(rooms_by_type[t],
                                   key=lambda r: r[2] * r[3], reverse=True)

    count_vec = np.array([len(rooms_by_type[i]) for i in range(NUM_TYPES)],
                         dtype=np.float32)

    # Graph features: type-pair edge counts (upper triangle, 15-dim)
    graph_vec = np.zeros(COND_DIM_GRAPH, dtype=np.float32)
    G = plan.get("graph")
    if G is not None:
        pair_counts = Counter()
        for u, v in G.edges():
            tu = G.nodes[u].get("type", "")
            tv = G.nodes[v].get("type", "")
            if tu in ROOM_TYPES and tv in ROOM_TYPES:
                ti = ROOM_TYPES.index(tu)
                tj = ROOM_TYPES.index(tv)
                pair = (min(ti, tj), max(ti, tj))
                pair_counts[pair] += 1
        idx = 0
        for i in range(NUM_TYPES):
            for j in range(i, NUM_TYPES):
                graph_vec[idx] = pair_counts.get((i, j), 0)
                idx += 1

    return rooms_by_type, count_vec, boundary_ar, graph_vec


def rooms_to_layout(rooms_by_type):
    """Convert rooms_by_type → fixed-size layout vector (LAYOUT_DIM,)."""
    vec = np.zeros(LAYOUT_DIM, dtype=np.float32)
    for type_idx in range(NUM_TYPES):
        rooms = rooms_by_type.get(type_idx, [])
        for j, (cx, cy, w, h) in enumerate(rooms[:MAX_PER_TYPE]):
            offset = (type_idx * MAX_PER_TYPE + j) * COORDS_PER_ROOM
            vec[offset]     = cx
            vec[offset + 1] = cy
            vec[offset + 2] = w
            vec[offset + 3] = h
    return vec


def layout_to_rooms(vec, count_vec):
    """Convert layout vector → list of (type_idx, cx, cy, w, h).
    Only returns rooms that pass the validity threshold."""
    rooms = []
    for type_idx in range(NUM_TYPES):
        n = int(min(count_vec[type_idx], MAX_PER_TYPE))
        for j in range(n):
            offset = (type_idx * MAX_PER_TYPE + j) * COORDS_PER_ROOM
            cx, cy, w, h = vec[offset:offset + 4]
            # Clamp to valid range
            cx = np.clip(cx, 0, 1)
            cy = np.clip(cy, 0, 1)
            w  = np.clip(w, 0.01, 1)
            h  = np.clip(h, 0.01, 1)
            if w * h > 0.001:  # validity threshold
                rooms.append((type_idx, float(cx), float(cy),
                               float(w), float(h)))
    return rooms


def make_mask(count_vec):
    """Create binary mask for valid room slots in layout vector."""
    mask = np.zeros(LAYOUT_DIM, dtype=np.float32)
    for type_idx in range(NUM_TYPES):
        n = int(min(count_vec[type_idx], MAX_PER_TYPE))
        for j in range(n):
            offset = (type_idx * MAX_PER_TYPE + j) * COORDS_PER_ROOM
            mask[offset:offset + COORDS_PER_ROOM] = 1.0
    return mask


# ─── Evaluation Metrics ──────────────────────────────────────────────────────
def multiset_jaccard(a: Counter, b: Counter) -> float:
    """Jaccard similarity for multisets (Counters)."""
    all_keys = set(a) | set(b)
    if not all_keys:
        return 1.0
    intersection = sum(min(a[k], b[k]) for k in all_keys)
    union = sum(max(a[k], b[k]) for k in all_keys)
    return intersection / union if union > 0 else 0.0


def boxes_to_adj_multiset(rooms, threshold=ADJ_THRESHOLD):
    """Build type-pair adjacency multiset from generated bounding boxes.

    Two rooms are considered adjacent if their bounding boxes overlap or
    their edges are within `threshold` (in normalised coords).
    """
    adj = Counter()
    for i in range(len(rooms)):
        ti, cx_i, cy_i, w_i, h_i = rooms[i]
        x1_i, y1_i = cx_i - w_i / 2, cy_i - h_i / 2
        x2_i, y2_i = cx_i + w_i / 2, cy_i + h_i / 2
        for j in range(i + 1, len(rooms)):
            tj, cx_j, cy_j, w_j, h_j = rooms[j]
            x1_j, y1_j = cx_j - w_j / 2, cy_j - h_j / 2
            x2_j, y2_j = cx_j + w_j / 2, cy_j + h_j / 2
            # Overlap with tolerance
            if (x1_i < x2_j + threshold and x2_i > x1_j - threshold and
                    y1_i < y2_j + threshold and y2_i > y1_j - threshold):
                pair = tuple(sorted([ROOM_TYPES[ti], ROOM_TYPES[tj]]))
                adj[pair] += 1
    return adj


def get_target_adj_multiset(plan):
    """Get type-pair adjacency multiset from the ground-truth graph."""
    G = plan.get("graph")
    if G is None:
        return Counter()
    adj = Counter()
    for u, v in G.edges():
        tu = G.nodes[u].get("type", "")
        tv = G.nodes[v].get("type", "")
        if tu in ROOM_TYPES and tv in ROOM_TYPES:
            pair = tuple(sorted([tu, tv]))
            adj[pair] += 1
    return adj


def rescaled_bbox_iou(gen_rooms, target_plan):
    """Aspect-ratio match between generated layout and target boundary.

    Computes the tight bounding box of generated rooms and compares its
    aspect ratio to the target plan's inner polygon.
    """
    if not gen_rooms:
        return 0.0

    # Generated bounding box
    min_x = min(r[1] - r[3] / 2 for r in gen_rooms)
    max_x = max(r[1] + r[3] / 2 for r in gen_rooms)
    min_y = min(r[2] - r[4] / 2 for r in gen_rooms)
    max_y = max(r[2] + r[4] / 2 for r in gen_rooms)
    gw = max_x - min_x
    gh = max_y - min_y
    if gw <= 0 or gh <= 0:
        return 0.0

    # Target bounding box
    inner = target_plan.get("inner")
    if inner is None or inner.is_empty:
        return 0.0
    tx1, ty1, tx2, ty2 = inner.bounds
    tw = tx2 - tx1
    th = ty2 - ty1
    if tw <= 0 or th <= 0:
        return 0.0

    # Compare aspect ratios (same formula as retrieval baselines)
    wr = gw / tw if tw > 0 else 0
    hr = gh / th if th > 0 else 0

    # Normalise both to unit dimensions for fair comparison
    # Since generated coords preserve AR via max-dim scaling, compare AR directly
    gen_ar = gw / gh
    tgt_ar = tw / th
    r = gen_ar / tgt_ar
    return min(r, 1.0 / r) if r > 0 else 0.0


# ─── Dataset ─────────────────────────────────────────────────────────────────
class LayoutDataset(Dataset):
    """Dataset of (layout, condition, mask) tuples for CVAE training."""
    def __init__(self, layouts, conditions, masks):
        self.layouts    = torch.tensor(layouts, dtype=torch.float32)
        self.conditions = torch.tensor(conditions, dtype=torch.float32)
        self.masks      = torch.tensor(masks, dtype=torch.float32)

    def __len__(self):
        return len(self.layouts)

    def __getitem__(self, idx):
        return self.layouts[idx], self.conditions[idx], self.masks[idx]


# ─── CVAE Model ──────────────────────────────────────────────────────────────
class LayoutCVAE(nn.Module):
    """Conditional VAE for room layout generation.

    Encoder: layout + condition → (μ, log σ²)
    Decoder: z + condition → layout
    """
    def __init__(self, layout_dim=LAYOUT_DIM, cond_dim=COND_DIM_BASE,
                 latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(layout_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
        )
        self.mu_head     = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, layout_dim),
            nn.Sigmoid(),  # coordinates in [0, 1]
        )

    def encode(self, x, cond):
        h = self.enc(torch.cat([x, cond], dim=-1))
        return self.mu_head(h), self.logvar_head(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        return self.dec(torch.cat([z, cond], dim=-1))

    def forward(self, x, cond):
        mu, logvar = self.encode(x, cond)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar

    @torch.no_grad()
    def generate(self, cond, n_samples=1):
        """Sample layouts from the prior given conditions."""
        if n_samples > 1:
            cond = cond.unsqueeze(0).expand(n_samples, -1)
        z = torch.randn(cond.size(0), self.latent_dim, device=cond.device)
        return self.decode(z, cond)


# ─── Loss ─────────────────────────────────────────────────────────────────────
def cvae_loss(recon, target, mu, logvar, mask, beta):
    """Masked reconstruction + KL divergence.

    Args:
        recon:  (B, LAYOUT_DIM) reconstructed layout
        target: (B, LAYOUT_DIM) ground-truth layout
        mu:     (B, LATENT_DIM) posterior mean
        logvar: (B, LATENT_DIM) posterior log-variance
        mask:   (B, LAYOUT_DIM) binary mask for valid slots
        beta:   KL weight (annealed during training)
    """
    # Masked MSE (only on valid room slots)
    diff = (recon - target) * mask
    valid = mask.sum().clamp(min=1.0)
    recon_loss = (diff ** 2).sum() / valid

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()


# ─── Training ─────────────────────────────────────────────────────────────────
def train_cvae(model, train_loader, val_loader, device, epochs,
               lr=1e-3, beta_max=0.5, warmup_epochs=100, patience=50):
    """Train CVAE with KL annealing, cosine LR schedule, and early stopping."""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs,
                                                         eta_min=1e-5)
    best_val = float("inf")
    best_state = None
    patience_ctr = 0
    history = {"train_loss": [], "val_loss": [], "recon": [], "kl": []}

    for epoch in range(1, epochs + 1):
        # KL annealing: linear warmup 0 → beta_max
        beta = min(beta_max, beta_max * epoch / warmup_epochs) if warmup_epochs > 0 else beta_max

        model.train()
        epoch_loss, epoch_recon, epoch_kl, n_batches = 0, 0, 0, 0
        for layout, cond, mask in train_loader:
            layout = layout.to(device)
            cond   = cond.to(device)
            mask   = mask.to(device)

            recon, mu, logvar = model(layout, cond)
            loss, rl, kl = cvae_loss(recon, layout, mu, logvar, mask, beta)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()
            epoch_recon += rl
            epoch_kl += kl
            n_batches += 1

        sched.step()

        # Validation — track recon loss separately for early stopping
        model.eval()
        val_loss_total = 0
        val_recon_total = 0
        with torch.no_grad():
            for layout, cond, mask in val_loader:
                layout = layout.to(device)
                cond   = cond.to(device)
                mask   = mask.to(device)
                recon, mu, logvar = model(layout, cond)
                loss, rl, _ = cvae_loss(recon, layout, mu, logvar, mask, beta)
                val_loss_total += loss.item()
                val_recon_total += rl

        avg_train = epoch_loss / max(n_batches, 1)
        avg_val   = val_loss_total / max(len(val_loader), 1)
        avg_val_recon = val_recon_total / max(len(val_loader), 1)
        avg_recon = epoch_recon / max(n_batches, 1)
        avg_kl    = epoch_kl / max(n_batches, 1)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["recon"].append(avg_recon)
        history["kl"].append(avg_kl)

        # Early stopping on RECONSTRUCTION loss, but only start counting
        # patience AFTER warmup finishes.  During warmup the KL weight rises
        # from 0 → beta_max, so the latent space is not yet regularised and
        # the "best" reconstruction checkpoint would have a meaningless prior.
        if epoch >= warmup_epochs:
            if avg_val_recon < best_val:
                best_val = avg_val_recon
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
        else:
            # During warmup: always save if improving, never increment patience
            if avg_val_recon < best_val:
                best_val = avg_val_recon
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4d} | loss {avg_train:.4f} "
                  f"(recon {avg_recon:.4f} + β·KL {beta:.3f}×{avg_kl:.4f}) "
                  f"| val_recon {avg_val_recon:.4f} | best {best_val:.4f}")

        if patience_ctr >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_generative(model, test_plans, test_conditions, test_count_vecs,
                        device, n_samples=1):
    """Evaluate generative model on test set.

    For each test plan, generates layout(s) and computes:
      - Room Count Accuracy
      - Adjacency Satisfaction (multiset Jaccard)
      - Boundary IoU (aspect-ratio match)

    If n_samples > 1: generates K layouts, picks best by adjacency score.
    """
    model.eval()
    count_accs, adj_sats, b_ious = [], [], []

    with torch.no_grad():
        for i in range(len(test_plans)):
            target_plan = test_plans[i]
            cond = torch.tensor(test_conditions[i], dtype=torch.float32,
                                device=device)
            target_count = test_count_vecs[i]
            target_adj = get_target_adj_multiset(target_plan)

            if n_samples == 1:
                # Single sample
                z = torch.randn(1, model.latent_dim, device=device)
                layout_vec = model.decode(z, cond.unsqueeze(0)).squeeze(0)
                layout_np = layout_vec.cpu().numpy()
                gen_rooms = layout_to_rooms(layout_np, target_count)

                # Room count accuracy
                gen_count = np.zeros(NUM_TYPES)
                for (ti, *_) in gen_rooms:
                    gen_count[ti] += 1
                ca = float(np.mean(gen_count == target_count))

                # Adjacency satisfaction
                gen_adj = boxes_to_adj_multiset(gen_rooms)
                adj_s = multiset_jaccard(target_adj, gen_adj)

                # Boundary IoU
                b_iou = rescaled_bbox_iou(gen_rooms, target_plan)

            else:
                # Best-of-K sampling
                best_adj = -1
                best_ca, best_iou = 0, 0
                cond_batch = cond.unsqueeze(0).expand(n_samples, -1)
                z = torch.randn(n_samples, model.latent_dim, device=device)
                layouts = model.decode(z, cond_batch).cpu().numpy()

                for k in range(n_samples):
                    gen_rooms = layout_to_rooms(layouts[k], target_count)
                    gen_count = np.zeros(NUM_TYPES)
                    for (ti, *_) in gen_rooms:
                        gen_count[ti] += 1
                    ca_k = float(np.mean(gen_count == target_count))
                    gen_adj = boxes_to_adj_multiset(gen_rooms)
                    adj_k = multiset_jaccard(target_adj, gen_adj)
                    iou_k = rescaled_bbox_iou(gen_rooms, target_plan)
                    if adj_k > best_adj:
                        best_adj = adj_k
                        best_ca = ca_k
                        best_iou = iou_k

                ca, adj_s, b_iou = best_ca, best_adj, best_iou

            count_accs.append(ca)
            adj_sats.append(adj_s)
            b_ious.append(b_iou)

    return {
        "room_count_acc":   float(np.mean(count_accs)),
        "adj_satisfaction":  float(np.mean(adj_sats)),
        "boundary_iou":      float(np.mean(b_ious)),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Task 1: Learned Generative Baselines (CVAE / CVAE+Graph)")
    parser.add_argument("--data",    default="../ResPlan.pkl")
    parser.add_argument("--split",   default="../split.json")
    parser.add_argument("--device",  default="auto",
                        help="'auto', 'cuda', or 'cpu'")
    parser.add_argument("--epochs",  type=int, default=500)
    parser.add_argument("--batch",   type=int, default=256)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--beta",    type=float, default=0.5,
                        help="Max KL weight (β-VAE)")
    parser.add_argument("--warmup",  type=int, default=100,
                        help="KL annealing warmup epochs")
    parser.add_argument("--latent",  type=int, default=LATENT_DIM,
                        help="Latent dimension")
    parser.add_argument("--hidden",  type=int, default=HIDDEN_DIM,
                        help="Hidden dimension")
    parser.add_argument("--patience", type=int, default=80,
                        help="Early stopping patience")
    parser.add_argument("--samples", type=int, default=1,
                        help="Number of samples per test plan (best-of-K)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--output",  default="results/task1_generation.json")
    args = parser.parse_args()

    # ── Device setup ──
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  Memory: {mem_gb:.1f} GB")

    seed_everything(args.seed)

    # ── Load data ──
    print(f"\nLoading {args.data}...")
    with open(args.data, "rb") as f:
        data = pickle.load(f)
    with open(args.split) as f:
        splits = json.load(f)

    train_ids = {int(x) for x in splits["train"]}
    val_ids   = {int(x) for x in splits["val"]}
    test_ids  = {int(x) for x in splits["test"]}

    # ── Preprocess all plans ──
    print("Extracting room bounding boxes...")
    t0 = time.time()

    split_data = {"train": [], "val": [], "test": []}
    skipped = 0
    max_rooms_per_type = np.zeros(NUM_TYPES, dtype=int)

    for plan in data:
        pid = int(plan.get("id", -1))
        rooms_by_type, count_vec, boundary_ar, graph_vec = extract_plan_features(plan)
        if rooms_by_type is None:
            skipped += 1
            continue

        # Track max rooms per type
        for t in range(NUM_TYPES):
            max_rooms_per_type[t] = max(max_rooms_per_type[t],
                                         len(rooms_by_type[t]))

        entry = {
            "plan": plan,
            "layout": rooms_to_layout(rooms_by_type),
            "count_vec": count_vec,
            "boundary_ar": boundary_ar,
            "graph_vec": graph_vec,
            "mask": make_mask(count_vec),
        }

        if pid in train_ids:
            split_data["train"].append(entry)
        elif pid in val_ids:
            split_data["val"].append(entry)
        elif pid in test_ids:
            split_data["test"].append(entry)

    del data
    print(f"  Train: {len(split_data['train'])}, "
          f"Val: {len(split_data['val'])}, "
          f"Test: {len(split_data['test'])}, "
          f"Skipped: {skipped}")
    print(f"  Max rooms/type: {dict(zip(ROOM_TYPES, max_rooms_per_type.tolist()))}")
    print(f"  Preprocessing: {time.time() - t0:.1f}s")

    # Check MAX_PER_TYPE is sufficient
    for t in range(NUM_TYPES):
        if max_rooms_per_type[t] > MAX_PER_TYPE:
            print(f"  WARNING: {ROOM_TYPES[t]} has up to {max_rooms_per_type[t]} rooms "
                  f"(MAX_PER_TYPE={MAX_PER_TYPE}). Some rooms will be truncated.")

    # ── Build condition vectors ──
    def make_cond_base(entries):
        """Room counts (5) + boundary AR (1) = 6-dim condition."""
        return np.stack([
            np.concatenate([e["count_vec"], [e["boundary_ar"]]])
            for e in entries
        ])

    def make_cond_graph(entries):
        """Room counts (5) + boundary AR (1) + graph features (15) = 21-dim."""
        return np.stack([
            np.concatenate([e["count_vec"], [e["boundary_ar"]], e["graph_vec"]])
            for e in entries
        ])

    # Normalise graph features by max in training set
    train_graph_max = np.max(np.stack([e["graph_vec"] for e in split_data["train"]]),
                             axis=0).clip(min=1.0)

    for split_name in ["train", "val", "test"]:
        for entry in split_data[split_name]:
            entry["graph_vec_norm"] = entry["graph_vec"] / train_graph_max

    def make_cond_graph_norm(entries):
        """Room counts (5) + boundary AR (1) + normalised graph (15) = 21-dim."""
        return np.stack([
            np.concatenate([e["count_vec"], [e["boundary_ar"]],
                            e["graph_vec_norm"]])
            for e in entries
        ])

    # ── Helper: prepare data arrays ──
    def get_arrays(entries):
        layouts = np.stack([e["layout"] for e in entries])
        masks   = np.stack([e["mask"] for e in entries])
        counts  = np.stack([e["count_vec"] for e in entries])
        return layouts, masks, counts

    train_layouts, train_masks, train_counts = get_arrays(split_data["train"])
    val_layouts,   val_masks,   val_counts   = get_arrays(split_data["val"])
    test_layouts,  test_masks,  test_counts  = get_arrays(split_data["test"])

    test_plans = [e["plan"] for e in split_data["test"]]

    results = {}

    # ════════════════════════════════════════════════════════════════════════════
    # Model 1: CVAE (conditioned on room counts + boundary AR)
    # ════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Training CVAE (condition: room counts + boundary AR)")
    print("=" * 70)

    cond_dim = COND_DIM_BASE  # 6
    train_cond = make_cond_base(split_data["train"])
    val_cond   = make_cond_base(split_data["val"])
    test_cond  = make_cond_base(split_data["test"])

    train_ds = LayoutDataset(train_layouts, train_cond, train_masks)
    val_ds   = LayoutDataset(val_layouts, val_cond, val_masks)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=2, pin_memory=True)

    seed_everything(args.seed)
    model_cvae = LayoutCVAE(
        layout_dim=LAYOUT_DIM, cond_dim=cond_dim,
        latent_dim=args.latent, hidden_dim=args.hidden
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model_cvae.parameters()):,}")

    t0 = time.time()
    train_cvae(model_cvae, train_loader, val_loader, device,
               epochs=args.epochs, lr=args.lr, beta_max=args.beta,
               warmup_epochs=args.warmup, patience=args.patience)
    print(f"  Training time: {time.time() - t0:.1f}s")

    print("\n  Evaluating CVAE...")
    res_cvae = evaluate_generative(
        model_cvae, test_plans, test_cond, test_counts,
        device, n_samples=args.samples)
    results["cvae"] = res_cvae
    print(f"  CVAE — Count: {res_cvae['room_count_acc']:.3f}  "
          f"Adj: {res_cvae['adj_satisfaction']:.3f}  "
          f"BIoU: {res_cvae['boundary_iou']:.3f}")

    # ════════════════════════════════════════════════════════════════════════════
    # Model 2: CVAE+Graph (additionally conditioned on adjacency features)
    # ════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("Training CVAE+Graph (condition: counts + AR + adjacency features)")
    print("=" * 70)

    cond_dim_g = COND_DIM_BASE + COND_DIM_GRAPH  # 21
    train_cond_g = make_cond_graph_norm(split_data["train"])
    val_cond_g   = make_cond_graph_norm(split_data["val"])
    test_cond_g  = make_cond_graph_norm(split_data["test"])

    train_ds_g = LayoutDataset(train_layouts, train_cond_g, train_masks)
    val_ds_g   = LayoutDataset(val_layouts, val_cond_g, val_masks)
    train_loader_g = DataLoader(train_ds_g, batch_size=args.batch, shuffle=True,
                                num_workers=4, pin_memory=True, drop_last=True)
    val_loader_g   = DataLoader(val_ds_g, batch_size=args.batch, shuffle=False,
                                num_workers=2, pin_memory=True)

    seed_everything(args.seed)
    model_cvae_g = LayoutCVAE(
        layout_dim=LAYOUT_DIM, cond_dim=cond_dim_g,
        latent_dim=args.latent, hidden_dim=args.hidden
    ).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model_cvae_g.parameters()):,}")

    t0 = time.time()
    train_cvae(model_cvae_g, train_loader_g, val_loader_g, device,
               epochs=args.epochs, lr=args.lr, beta_max=args.beta,
               warmup_epochs=args.warmup, patience=args.patience)
    print(f"  Training time: {time.time() - t0:.1f}s")

    print("\n  Evaluating CVAE+Graph...")
    res_cvae_g = evaluate_generative(
        model_cvae_g, test_plans, test_cond_g, test_counts,
        device, n_samples=args.samples)
    results["cvae_graph"] = res_cvae_g
    print(f"  CVAE+Graph — Count: {res_cvae_g['room_count_acc']:.3f}  "
          f"Adj: {res_cvae_g['adj_satisfaction']:.3f}  "
          f"BIoU: {res_cvae_g['boundary_iou']:.3f}")

    # ════════════════════════════════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"{'Method':<25} {'Count Acc':>10} {'Adj Sat':>10} {'Boundary IoU':>13}")
    print("-" * 60)
    for name, res in results.items():
        if name.startswith("_"):
            continue
        print(f"{name:<25} {res['room_count_acc']:>10.3f} "
              f"{res['adj_satisfaction']:>10.3f} "
              f"{res['boundary_iou']:>13.3f}")

    # Metadata
    results["_meta"] = {
        "n_train": len(split_data["train"]),
        "n_val":   len(split_data["val"]),
        "n_test":  len(split_data["test"]),
        "max_per_type": MAX_PER_TYPE,
        "layout_dim": LAYOUT_DIM,
        "latent_dim": args.latent,
        "hidden_dim": args.hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "beta_max": args.beta,
        "warmup_epochs": args.warmup,
        "patience": args.patience,
        "seed": args.seed,
        "samples": args.samples,
        "device": str(device),
        "max_rooms_per_type": dict(zip(ROOM_TYPES,
                                        max_rooms_per_type.tolist())),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
