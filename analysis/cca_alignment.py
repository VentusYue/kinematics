"""
cca_alignment.py — v0 → current 

- Global normalization **default**:
  - global grid-unit estimator (`axis_mode`) + global scale (q=0.95) to fit trajectories into ridge grid (target_radius=9).
  - avoids per-episode scale mismatch / out-of-frame ridge embeddings.
- Best cycle per route: pick the highest `match_ratio` cycle for each route (1 cycle ↔ 1 behavior).
- Cycle-level mean pooling: aggregate hidden states over the cycle to produce one neural vector per cycle.
- Figure-5 variants: save `fig5_by_length.png`, `fig5_by_displacement.png`, `fig5_by_angle.png`.
- 3D interactive: save `alignment_3d.html` (CM0/CM1/CM2, neural vs behavior).
- Ridge radius tuning: expose `--ridge_radius_scale`; **default 0.6**.
"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr, svd, solve_triangular
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis.ridge_embedding import build_ridge_vector

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def _estimate_grid_unit_median_euclid(paths: list[np.ndarray], tol: float = 0.1) -> float:
    """
    Robustly estimate a global step size using Euclidean step lengths.
    Uses median-of-medians: per-path median step -> global median.
    """
    step_medians: list[float] = []
    for p in paths:
        if p is None or len(p) < 2:
            continue
        diffs = np.linalg.norm(p[1:] - p[:-1], axis=1)
        diffs = diffs[np.isfinite(diffs)]
        diffs = diffs[diffs > tol]
        if diffs.size > 0:
            step_medians.append(float(np.median(diffs)))
    if not step_medians:
        return 1.0
    return float(np.median(step_medians))


def _estimate_grid_unit_axis_mode(
    paths: list[np.ndarray],
    tol: float = 0.1,
    round_decimals: int = 2,
    min_cluster_frac: float = 0.05,
) -> tuple[float, np.ndarray]:
    """
    Mode-like estimator using |dx| and |dy| step components (robust when motion is axis-aligned).

    Returns (grid_unit, values_used) where values_used are the axis step magnitudes.
    """
    vals: list[np.ndarray] = []
    for p in paths:
        if p is None or len(p) < 2:
            continue
        d = p[1:] - p[:-1]
        dx = np.abs(d[:, 0])
        dy = np.abs(d[:, 1])
        a = np.concatenate([dx, dy], axis=0)
        a = a[np.isfinite(a)]
        a = a[a > tol]
        if a.size > 0:
            vals.append(a)

    if not vals:
        return 1.0, np.array([], dtype=np.float64)

    a = np.concatenate(vals, axis=0).astype(np.float64)
    if a.size == 0:
        return 1.0, a

    # "Mode-like": quantize then take the most frequent bin value, then refine by local median.
    q = np.round(a, decimals=round_decimals)
    uniq, cnt = np.unique(q, return_counts=True)
    if uniq.size == 0:
        return float(np.median(a)), a

    k = int(np.argmax(cnt))
    mode_val = float(uniq[k])

    # refine: take values near that mode bin
    bin_w = 10.0 ** (-round_decimals)
    mask = np.abs(a - mode_val) <= (1.5 * bin_w)
    if mask.mean() < min_cluster_frac:
        # fallback to median if the "mode" is too weak (no clear peak)
        return float(np.median(a)), a
    return float(np.median(a[mask])), a


def _compute_global_scale_factor(
    paths: list[np.ndarray],
    grid_unit: float,
    target_radius: float = 9.0,
    quantile: float = 0.95,
) -> tuple[float, np.ndarray]:
    """
    Compute a global scale so that the `quantile` of trajectory extents fits within target_radius
    (for 21x21, radius~10; default 9 leaves a margin).

    Returns (scale, extents_grid_units).
    """
    grid_unit = float(grid_unit)
    if not np.isfinite(grid_unit) or grid_unit <= 1e-8:
        grid_unit = 1.0

    extents: list[float] = []
    for p in paths:
        if p is None or len(p) == 0:
            continue
        p0 = (p - p[0]) / grid_unit
        e = float(np.max(np.abs(p0)))
        if np.isfinite(e):
            extents.append(e)

    if not extents:
        return 1.0, np.array([], dtype=np.float64)

    ext = np.asarray(extents, dtype=np.float64)
    cutoff = float(np.quantile(ext, quantile))
    if not np.isfinite(cutoff) or cutoff < 1e-8:
        return 1.0, ext

    scale = float(target_radius) / cutoff
    return scale, ext


def _rotate_path_to_disp(path: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Rotate a (T,2) path (assumed centered at origin) so that its net displacement points to +x.
    """
    if path is None or len(path) == 0:
        return path
    d = path[-1]  # since start is (0,0)
    dx, dy = float(d[0]), float(d[1])
    if (dx * dx + dy * dy) < eps:
        return path
    theta = float(np.arctan2(dy, dx))
    c = float(np.cos(-theta))
    s = float(np.sin(-theta))
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (path @ R.T).astype(np.float32)

def plot_3d_interactive(U_means, V_means, metadata, out_path, title="3D Alignment"):
    """
    Create an interactive 3D scatter plot using Plotly.
    U_means: (N, 3) - Neural top 3 modes
    V_means: (N, 3) - Behavior top 3 modes
    metadata: dict of arrays (N,) - e.g. {'Length': ..., 'Angle': ...}
    """
    # Create DataFrame for easier plotting? 
    # Or just use graph_objects for flexibility
    
    n_points = U_means.shape[0]
    
    # Create subplots: 1 row, 2 columns, both 3D specs
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Neural State (CM0-2)", "Behavior Ridge (CM0-2)")
    )
    
    # We'll add traces for each metadata type, but that might be too heavy.
    # Let's just pick the first metadata key as default color, 
    # or create a hovertext that includes all metadata.
    
    hover_text = []
    keys = list(metadata.keys())
    for i in range(n_points):
        text = f"Point {i}<br>"
        for k in keys:
            val = metadata[k][i]
            if isinstance(val, (float, np.floating)):
                text += f"{k}: {val:.2f}<br>"
            else:
                text += f"{k}: {val}<br>"
        hover_text.append(text)
        
    # Color by the first key in metadata (usually Length or Angle)
    color_key = keys[0] if keys else None
    color_vals = metadata[color_key] if color_key else np.zeros(n_points)
    
    # Neural Trace
    fig.add_trace(
        go.Scatter3d(
            x=U_means[:, 0],
            y=U_means[:, 1],
            z=U_means[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=color_vals,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title=color_key, x=0.45)
            ),
            text=hover_text,
            name='Neural'
        ),
        row=1, col=1
    )
    
    # Behavior Trace
    fig.add_trace(
        go.Scatter3d(
            x=V_means[:, 0],
            y=V_means[:, 1],
            z=V_means[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=color_vals,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title=color_key, x=1.0)
            ),
            text=hover_text,
            name='Behavior'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text=title,
        height=800,
        width=1600,
        showlegend=False
    )
    
    # Update scene axes labels
    scene_dict = dict(xaxis_title='CM0', yaxis_title='CM1', zaxis_title='CM2')
    fig.update_scenes(scene_dict)
    
    fig.write_html(out_path)
    print(f"Saved interactive 3D plot to {out_path}")

def plot_alignment_multi(U_means, V_means, metadata, out_dir, prefix="fig5"):
    """
    Generate multiple static 2D alignment plots colored by different metadata features.
    """
    for key, values in metadata.items():
        out_path = os.path.join(out_dir, f"{prefix}_by_{key.lower()}.png")
        plot_alignment(U_means, V_means, values, out_path, title=f"Alignment colored by {key}")
        print(f"Saved alignment plot colored by {key} to {out_path}")


def _zscore_cols(Z: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Column-wise z-score with safe std floor.
    Matches the plotting convention in the reference script: z-score U and V
    before visualizing CM0/CM1.
    """
    Z = np.asarray(Z, dtype=np.float64)
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (Z - mu) / sd


def canoncorr(X0: np.ndarray, Y0: np.ndarray, fullReturn: bool = False) -> np.ndarray:
    """
    Canonical Correlation Analysis (CCA)
    """
    n, p1 = X0.shape
    p2 = Y0.shape[1]
    
    # Data diagnostics
    print(f"  X shape: {X0.shape}, Y shape: {Y0.shape}")
    
    if p1 >= n or p2 >= n:
        logging.warning('Not enough samples, might cause problems')

    # Preprocessing: Standardize the variables
    # Handle constant columns to avoid division by zero
    X_std = np.std(X0, 0)
    Y_std = np.std(Y0, 0)

    # Count constant columns
    x_const = (X_std == 0).sum()
    y_const = (Y_std == 0).sum()
    if x_const > 0 or y_const > 0:
        print(f"  [warn] Constant columns: X={x_const}/{p1}, Y={y_const}/{p2}")

    X_std[X_std == 0] = 1.0
    Y_std[Y_std == 0] = 1.0

    X = (X0 - np.mean(X0, 0)) / X_std
    Y = (Y0 - np.mean(Y0, 0)) / Y_std

    # Factor the inputs, and find a full rank set of columns if necessary
    Q1, T11, perm1 = qr(X, mode='economic', pivoting=True)
    Q2, T22, perm2 = qr(Y, mode='economic', pivoting=True)

    # Determine ranks
    tol = np.finfo(float).eps * 100
    rankX = np.sum(np.abs(np.diagonal(T11)) > tol * np.abs(T11[0, 0]))
    rankY = np.sum(np.abs(np.diagonal(T22)) > tol * np.abs(T22[0, 0]))

    print(f"  Rank of X: {rankX}, Rank of Y: {rankY}")

    if rankX == 0:
        raise ValueError('X has zero rank')
    elif rankX < p1:
        Q1 = Q1[:, :rankX]
        T11 = T11[:rankX, :rankX]

    if rankY == 0:
        raise ValueError('Y has zero rank')
    elif rankY < p2:
        Q2 = Q2[:, :rankY]
        T22 = T22[:rankY, :rankY]

    # Compute canonical coefficients and canonical correlations
    d = min(rankX, rankY)
    L,D,M = svd(Q1.T @ Q2, full_matrices=True, check_finite=True, lapack_driver='gesdd')
    M = M.T

    # Solve triangular systems (more stable than explicit matrix inverse)
    # T11 and T22 are upper triangular from QR.
    A = solve_triangular(T11, L[:, :d], lower=False) * np.sqrt(n - 1)
    B = solve_triangular(T22, M[:, :d], lower=False) * np.sqrt(n - 1)
    r = D[:d]
    
    r = np.clip(r, 0, 1)

    if not fullReturn:
        return r

    # Put coefficients back to their full size and correct order
    A_full = np.zeros((p1, d))
    A_full[perm1, :] = np.vstack((A, np.zeros((p1 - rankX, d))))
    
    B_full = np.zeros((p2, d))
    B_full[perm2, :] = np.vstack((B, np.zeros((p2 - rankY, d))))

    # Compute the canonical variates.
    #
    # For rank-deficient cases, constructing variates via the orthonormal bases
    # is more numerically stable and guarantees corr(U_k, V_k) == r_k (up to sign)
    # on the same data used for fitting.
    U_from_data = X @ A_full
    V_from_data = Y @ B_full
    U_direct = Q1 @ (L[:, :d] * np.sqrt(n - 1))
    V_direct = Q2 @ (M[:, :d] * np.sqrt(n - 1))
    denom_u = np.linalg.norm(U_direct) + 1e-12
    denom_v = np.linalg.norm(V_direct) + 1e-12
    rel_u = np.linalg.norm(U_from_data - U_direct) / denom_u
    rel_v = np.linalg.norm(V_from_data - V_direct) / denom_v
    if rel_u > 1e-3 or rel_v > 1e-3:
        print(f"  [warn] Canonical variates mismatch (rel_u={rel_u:.2e}, rel_v={rel_v:.2e}); using direct Q-based variates.")
        U = U_direct
        V = V_direct
    else:
        U = U_from_data
        V = V_from_data

    return A_full, B_full, r, U, V

def plot_lollipop(scores, out_path, title="Canonical Correlation", xlabel="Mode", ylabel="Correlation"):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(scores))
    ax.vlines(x, 0, scores, colors='gray', lw=1, alpha=0.5)
    ax.plot(x, scores, color='black', marker='o', markersize=8, linestyle='-', linewidth=1)
    for i, score in enumerate(scores):
        ax.annotate(f'{score:.2f}', (i, score), textcoords="offset points", 
                    xytext=(0,10), ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(range(1, len(scores) + 1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_alignment(U_means, V_means, colors, out_path, title="Alignment"):
    """
    Scatter plot of U vs V
    U_means: (N_cycles, d) - typically CM0 vs CM1
    V_means: (N_cycles, d)
    colors: (N_cycles,) for color coding (e.g. length)
    """
    n_points = len(colors)
    print(f"  Plotting {n_points} points")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Add small jitter to reveal overlapping points
    jitter_scale_u = 0.02 * (U_means[:, :2].max() - U_means[:, :2].min() + 1e-6)
    jitter_scale_v = 0.02 * (V_means[:, :2].max() - V_means[:, :2].min() + 1e-6)
    
    jitter_u = np.random.randn(n_points, 2) * jitter_scale_u
    jitter_v = np.random.randn(n_points, 2) * jitter_scale_v

    # Determine colormap based on length range
    if len(colors) > 0:
        c_min, c_max = colors.min(), colors.max()
        c_range = c_max - c_min
        if c_max <= 20:
            cmap = 'tab20'
        elif c_range < 20:
            cmap = 'jet'  # High contrast for small range at high offset
        else:
            cmap = 'turbo' # Better than viridis for distinguishing values
    else:
        cmap = 'viridis'
    
    # Left: Neural (U)
    sc1 = axes[0].scatter(
        U_means[:, 0] + jitter_u[:, 0], 
        U_means[:, 1] + jitter_u[:, 1], 
        c=colors, cmap=cmap, alpha=0.7, s=50, edgecolors='black', linewidths=0.5
    )
    axes[0].set_title(f"Neural State (CM0 vs CM1) - {n_points} points")
    axes[0].set_xlabel("CM 0")
    axes[0].set_ylabel("CM 1")
    plt.colorbar(sc1, ax=axes[0], label='Episode Length')
    
    # Right: Behavior (V)
    sc2 = axes[1].scatter(
        V_means[:, 0] + jitter_v[:, 0], 
        V_means[:, 1] + jitter_v[:, 1], 
        c=colors, cmap=cmap, alpha=0.7, s=50, edgecolors='black', linewidths=0.5
    )
    axes[1].set_title(f"Behavior Ridge (CM0 vs CM1) - {n_points} points")
    axes[1].set_xlabel("CM 0")
    axes[1].set_ylabel("CM 1")
    plt.colorbar(sc2, ax=axes[1], label='Episode Length')
    
    plt.suptitle(f"{title} (n={n_points})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles_npz", required=True)
    parser.add_argument("--routes_npz", type=str, default=None) # Optional if integrated
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_modes", type=int, default=10)
    parser.add_argument("--filter_outliers", action="store_true", help="Filter out extreme points in CM space")
    parser.add_argument("--cca_level", type=str, default="cycle", choices=["cycle", "step"],
                        help="What constitutes an observation for CCA. "
                             "'cycle' = one sample per PKD cycle (recommended for per-trajectory ridge vectors). "
                             "'step' = one sample per time-step (note: ridge is constant within a cycle, so this can inflate in-sample correlations).")
    parser.add_argument("--x_agg", type=str, default="mean", choices=["mean", "first", "last"],
                        help="How to aggregate hidden states into a per-cycle vector when --cca_level=cycle.")
    parser.add_argument("--noise_eps", type=float, default=1e-3,
                        help="Small uniform noise added to X/Y before PCA to break degenerate/constant columns (reference-style).")
    parser.add_argument("--noise_seed", type=int, default=0, help="RNG seed for the small-noise injection.")
    parser.add_argument("--drop_const_tol", type=float, default=1e-8,
                        help="Drop features whose across-sample std <= this threshold (done before noise/PCA).")
    parser.add_argument("--pca_dim_x", type=int, default=50,
                        help="[deprecated] Ignored. We always run full PCA (up to rank) to 'straighten' the point cloud.")
    parser.add_argument("--pca_dim_y", type=int, default=50, 
                        help="[deprecated] Ignored. We always run full PCA (up to rank) to 'straighten' the point cloud.")

    # -------------------------------------------------------------------------
    # Update B: trajectory normalization options for ridge embedding
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--ridge_norm",
        type=str,
        default="global",
        choices=["per_episode", "global"],
        help="Trajectory normalization before ridge embedding. "
             "'per_episode' = current heuristic (median euclid step per episode). "
             "'global' = estimate one global grid unit + global scale so trajectories fit inside 21x21.",
    )
    parser.add_argument(
        "--grid_unit_estimator",
        type=str,
        default="axis_mode",
        choices=["median_euclid", "axis_mode"],
        help="How to estimate the fundamental grid unit when --ridge_norm=global.",
    )
    parser.add_argument("--grid_step_tol", type=float, default=0.1, help="Threshold to ignore near-zero steps.")
    parser.add_argument("--global_scale_quantile", type=float, default=0.95, help="Quantile of extents to fit inside target radius.")
    parser.add_argument("--global_target_radius", type=float, default=9.0, help="Target radius in ridge grid units (<=10 for 21x21).")
    parser.add_argument("--rotate_to_disp", action="store_true", help="Optionally rotate each trajectory so its net displacement points to +x before ridge embedding.")
    parser.add_argument("--save_scale_diagnostics", action="store_true", help="Save histograms for step-size/extents when using global normalization.")

    # Ridge embedding knobs
    parser.add_argument("--ridge_aggregate", type=str, default="max", choices=["max", "sum"], help="How to aggregate point radiance fields into ridge image.")
    parser.add_argument("--ridge_normalize_path", action="store_true", help="Also apply ridge_embedding.normalize_path_to_grid before building ridge vector.")
    parser.add_argument("--ridge_grid_size", type=int, default=21, help="Ridge grid size (default 21 => 441D).")
    parser.add_argument("--ridge_radius_scale", type=float, default=0.6, help="Radius scale for ridge radiance field (smaller => more local/contrast).")

    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("="*60)
    print("CCA ALIGNMENT ANALYSIS")
    print("="*60)
    
    print(f"\nLoading cycles from {args.cycles_npz}")
    pkd_data = np.load(args.cycles_npz, allow_pickle=True)
    cycles_hidden = pkd_data['cycles_hidden']
    cycles_route_id = pkd_data['cycles_route_id']
    cycles_match_ratio = pkd_data['cycles_match_ratio']
    
    print(f"Loading routes from {args.routes_npz}")
    routes_data = np.load(args.routes_npz, allow_pickle=True)
    # ---------------------------------------------------------------------
    # Behavior-side inputs (from route collection)
    #
    # This script is the "D) CCA 对齐" stage.
    #
    # What is used here:
    # - routes_xy: (T,2) behavior trajectory per route, later converted to ridge embedding (441D).
    # - routes_ep_len: episode length metadata, used for plotting/coloring diagnostics.
    #
    # Semantics of routes_xy:
    # - Maze: agent(mouse) world coords from procgen `get_state()` entity ents[0].
    # - CoinRun: player world coords from procgen `get_state()` entity ents[0].
    #
    # What is NOT used here (but may exist in routes.npz):
    # - routes_obs/routes_actions: used earlier by PKD sampler, not needed for CCA step.
    # - routes_player_v/routes_ents_count/routes_nearest_ents: optional extra state features
    #   (currently ignored by this CCA script; could be used to build richer behavior embeddings later).
    # ---------------------------------------------------------------------
    routes_xy = routes_data['routes_xy']
    routes_ep_len = routes_data['routes_ep_len']
    
    num_cycles = len(cycles_hidden)
    print(f"\nProcessing {num_cycles} cycles")
    
    if num_cycles == 0:
        print("[ERROR] No cycles found! Cannot run CCA.")
        return
    
    # =========================================================================
    # INPUT DATA DIAGNOSTICS
    # =========================================================================
    print("\n" + "-"*40)
    print("INPUT DATA DIAGNOSTICS")
    print("-"*40)
    
    # Cycle statistics
    cycle_lens = [c.shape[0] for c in cycles_hidden]
    hidden_dims = [c.shape[1] if c.ndim > 1 else 256 for c in cycles_hidden]
    hidden_dim0 = int(hidden_dims[0])
    print(f"\n[Cycles]")
    print(f"  Number: {num_cycles}")
    print(f"  Lengths: min={min(cycle_lens)}, max={max(cycle_lens)}, mean={np.mean(cycle_lens):.1f}")
    print(f"  Hidden dim: {hidden_dims[0]}")
    print(f"  Match ratios: min={cycles_match_ratio.min():.3f}, max={cycles_match_ratio.max():.3f}")
    
    # Route XY statistics
    print(f"\n[Routes XY]")
    unique_routes = np.unique(cycles_route_id)
    print(f"  Unique routes used: {len(unique_routes)}")
    
    xy_variances = []
    xy_ranges = []
    for r_id in unique_routes:
        xy = routes_xy[r_id]
        if xy.shape[0] > 0:
            var = np.var(xy, axis=0).sum()
            xy_variances.append(var)
            xy_ranges.append(xy.max(axis=0) - xy.min(axis=0))
    
    if xy_variances:
        xy_var_arr = np.array(xy_variances)
        print(f"  XY variance: min={xy_var_arr.min():.4f}, max={xy_var_arr.max():.4f}, mean={xy_var_arr.mean():.4f}")
        low_var = (xy_var_arr < 1e-6).sum()
        if low_var > 0:
            print(f"  [WARN] {low_var} routes have near-zero variance!")
    
    # =========================================================================
    # SELECT BEST CYCLE PER ROUTE (One Cycle <-> One Behavior)
    # =========================================================================
    # Group by route_id and pick the one with highest match_ratio
    
    # Store indices of best cycles
    best_indices = {}
    for idx, r_id in enumerate(cycles_route_id):
        ratio = cycles_match_ratio[idx]
        if r_id not in best_indices:
            best_indices[r_id] = idx
        else:
            # Update if better match
            current_best_idx = best_indices[r_id]
            if ratio > cycles_match_ratio[current_best_idx]:
                best_indices[r_id] = idx
    
    selected_indices = sorted(list(best_indices.values()))
    print(f"\n[Filtering] Selected {len(selected_indices)} best cycles from {num_cycles} total candidates (Unique Routes: {len(unique_routes)})")

    # =========================================================================
    # Update B: GLOBAL NORMALIZATION CALIBRATION (optional)
    # =========================================================================
    grid_unit_global = None
    global_scale = None
    axis_step_vals = None
    extents_grid = None
    if args.ridge_norm == "global":
        print("\n" + "-"*40)
        print("CALIBRATING TRAJECTORY NORMALIZATION (GLOBAL)")
        print("-"*40)

        calib_paths: list[np.ndarray] = []
        for i in selected_indices:
            r_id = int(cycles_route_id[i])
            path_xy = routes_xy[r_id]
            h_cycle = cycles_hidden[i]

            # Ensure shape (L, H) to get a reliable cycle length for truncation
            if h_cycle.ndim == 1 and (h_cycle.shape[0] % hidden_dim0 == 0):
                h_cycle = h_cycle.reshape(-1, hidden_dim0)
            if h_cycle.ndim != 2:
                continue

            L = int(h_cycle.shape[0])
            if path_xy is None or len(path_xy) == 0:
                continue
            L_use = int(min(L, path_xy.shape[0]))
            if L_use <= 0:
                continue
            p = np.asarray(path_xy[:L_use], dtype=np.float32)
            if not np.isfinite(p).all():
                continue
            p = p - p[0]
            calib_paths.append(p)

        if args.grid_unit_estimator == "median_euclid":
            grid_unit_global = _estimate_grid_unit_median_euclid(calib_paths, tol=args.grid_step_tol)
            axis_step_vals = None
            print(f"  Grid unit (median_euclid): {grid_unit_global:.4f}")
        else:
            grid_unit_global, axis_step_vals = _estimate_grid_unit_axis_mode(
                calib_paths, tol=args.grid_step_tol, round_decimals=2
            )
            print(f"  Grid unit (axis_mode): {grid_unit_global:.4f}")

        global_scale, extents_grid = _compute_global_scale_factor(
            calib_paths,
            grid_unit=grid_unit_global,
            target_radius=args.global_target_radius,
            quantile=args.global_scale_quantile,
        )
        if extents_grid is not None and extents_grid.size > 0:
            qv = float(np.quantile(extents_grid, args.global_scale_quantile))
            print(f"  Extent quantile q={args.global_scale_quantile:.2f}: {qv:.3f} grid units")
        print(f"  Global scale: {global_scale:.4f} (target_radius={args.global_target_radius})")

        if args.save_scale_diagnostics:
            diag_path = os.path.join(args.out_dir, "ridge_scale_diagnostics.png")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            if axis_step_vals is not None and axis_step_vals.size > 0:
                axes[0].hist(axis_step_vals, bins=80, color="steelblue", alpha=0.8)
                axes[0].axvline(grid_unit_global, color="crimson", lw=2, label=f"grid_unit={grid_unit_global:.3f}")
                axes[0].set_title("Axis step magnitudes (|dx|,|dy|)")
                axes[0].set_xlabel("step size")
                axes[0].set_ylabel("count")
                axes[0].legend(loc="best")
            else:
                axes[0].text(0.5, 0.5, "axis_mode not used", ha="center", va="center")
                axes[0].set_axis_off()

            if extents_grid is not None and extents_grid.size > 0:
                axes[1].hist(extents_grid, bins=60, color="darkorange", alpha=0.8)
                qv = float(np.quantile(extents_grid, args.global_scale_quantile))
                axes[1].axvline(qv, color="crimson", lw=2, label=f"q={args.global_scale_quantile:.2f}: {qv:.2f}")
                axes[1].axvline(args.global_target_radius, color="black", lw=2, ls="--", label=f"target_radius={args.global_target_radius:.1f}")
                axes[1].set_title("Trajectory extents in grid units (max |x| or |y|)")
                axes[1].set_xlabel("extent")
                axes[1].set_ylabel("count")
                axes[1].legend(loc="best")
            else:
                axes[1].text(0.5, 0.5, "no extents", ha="center", va="center")
                axes[1].set_axis_off()

            plt.tight_layout()
            plt.savefig(diag_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"  Saved scaling diagnostics to {diag_path}")
    
    # =========================================================================
    # BUILD DATA MATRICES
    # =========================================================================
    print("\n" + "-"*40)
    print("BUILDING DATA MATRICES")
    print("-"*40)
    
    # Reference-style: one observation per cycle.
    if args.cca_level != "cycle":
        print("  [warn] --cca_level=step is not recommended for ridge-per-cycle features; using cycle-level samples for CCA/plots.")

    X_cycles: list[np.ndarray] = []
    Y_cycles: list[np.ndarray] = []
    cycle_lengths: list[float] = []
    cycle_disps: list[float] = []
    cycle_angles: list[float] = []

    # Ridge embedding statistics
    ridge_vecs: list[np.ndarray] = []
    est_grid_steps: list[float] = []
    all_path_tiles: list[np.ndarray] = []  # Store normalized path_tile for visualization

    skipped_nan = 0

    for i in selected_indices:
        h_cycle = cycles_hidden[i]  # (L, H)
        r_id = int(cycles_route_id[i])

        # Ensure shape (L, H)
        if h_cycle.ndim == 1:
            if h_cycle.shape[0] % hidden_dim0 == 0:
                h_cycle = h_cycle.reshape(-1, hidden_dim0)

        if h_cycle.ndim != 2:
            print(f"[WARN] Cycle {i}: unexpected h_cycle shape={getattr(h_cycle, 'shape', None)}; skipping.")
            continue

        path_xy = routes_xy[r_id]  # (T_route, 2)

        if np.isnan(path_xy).any() or np.isnan(h_cycle).any():
            skipped_nan += 1
            print(f"[WARN] Cycle {i}: NaNs detected in path_xy or h_cycle; skipping.")
            continue

        # Match lengths conservatively (truncate to the shared prefix).
        L = int(h_cycle.shape[0])
        if path_xy.shape[0] != L:
            print(f"[WARN] Cycle {i}: Length mismatch! path_xy={len(path_xy)}, h_cycle={L}; truncating to min length.")
        L_use = int(min(L, path_xy.shape[0]))
        if L_use <= 0:
            print(f"[WARN] Cycle {i}: empty after truncation; skipping.")
            continue
        h_cycle = h_cycle[:L_use]
        path_xy = path_xy[:L_use]

        # Center at origin first
        path_centered = (path_xy - path_xy[0]).astype(np.float32)

        # Pick a normalization unit (grid unit) and (optional) a scale to fit ridge canvas
        if args.ridge_norm == "global":
            if grid_unit_global is None or global_scale is None:
                # Safety fallback (should not happen)
                grid_unit = 1.0
                scale = 1.0
            else:
                grid_unit = float(grid_unit_global)
                scale = float(global_scale)
        else:
            # Original per-episode heuristic
            if len(path_xy) > 1:
                diffs = np.linalg.norm(path_xy[1:] - path_xy[:-1], axis=1)
                diffs = diffs[diffs > args.grid_step_tol]
                grid_unit = float(np.median(diffs)) if len(diffs) > 0 else 1.0
            else:
                grid_unit = 1.0
            grid_unit = max(grid_unit, 1e-3)
            scale = 1.0

        est_grid_steps.append(grid_unit)

        # Convert to (approx) tile/grid units
        path_grid = (path_centered / grid_unit).astype(np.float32)

        # Geometry features computed BEFORE optional rotation/scaling (so Angle remains meaningful)
        d_vec = path_grid[-1] if len(path_grid) > 0 else np.array([0.0, 0.0], dtype=np.float32)
        disp_val = float(np.linalg.norm(d_vec))
        ang_val = float(np.arctan2(float(d_vec[1]), float(d_vec[0])))

        # Optional rotation (can reduce orientation variance, but may remove informative angle structure)
        if args.rotate_to_disp:
            path_grid_for_ridge = _rotate_path_to_disp(path_grid)
        else:
            path_grid_for_ridge = path_grid

        # Global scale (if enabled) to fit the ridge canvas
        path_tile = (path_grid_for_ridge * scale).astype(np.float32)

        # Behavior embedding: ridge image flatten (441D)
        ridge_vec = build_ridge_vector(
            path_tile,
            grid_size=args.ridge_grid_size,
            radius_scale=args.ridge_radius_scale,
            aggregate=args.ridge_aggregate,
            normalize_path=args.ridge_normalize_path,
        )  # (441,)
        ridge_vecs.append(ridge_vec)
        all_path_tiles.append(path_tile.copy())

        # Neural embedding: choose one, like the original "ring center"
        if args.x_agg == "mean":
            x_i = h_cycle.mean(axis=0)
        elif args.x_agg == "first":
            x_i = h_cycle[0]
        elif args.x_agg == "last":
            x_i = h_cycle[-1]
        else:
            raise ValueError(f"Unknown x_agg: {args.x_agg}")

        X_cycles.append(np.asarray(x_i, dtype=np.float64))
        Y_cycles.append(np.asarray(ridge_vec, dtype=np.float64))
        cycle_lengths.append(float(routes_ep_len[r_id]))
        cycle_disps.append(disp_val)
        cycle_angles.append(ang_val)

    if skipped_nan > 0:
        print(f"\n[WARN] Skipped cycles due to NaNs: {skipped_nan}/{num_cycles}")

    X = np.stack(X_cycles, axis=0).astype(np.float64)  # (N, H)
    Y = np.stack(Y_cycles, axis=0).astype(np.float64)  # (N, 441)
    cycle_lengths = np.asarray(cycle_lengths, dtype=np.float64)
    cycle_disps = np.asarray(cycle_disps, dtype=np.float64)
    cycle_angles = np.asarray(cycle_angles, dtype=np.float64)
    sample_cycle_ids = np.arange(X.shape[0])
    
    print(f"\n[Matrix X (Neural)]")
    print(f"  Shape: {X.shape}")
    print(f"  Value range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  Mean: {X.mean():.3f}, Std: {X.std():.3f}")
    
    print(f"\n[Matrix Y (Ridge)]")
    print(f"  Shape: {Y.shape}")
    print(f"  Value range: [{Y.min():.3f}, {Y.max():.3f}]")
    print(f"  Mean: {Y.mean():.3f}, Std: {Y.std():.3f}")
    
    # Check ridge embedding diversity
    ridge_arr = np.array(ridge_vecs)
    ridge_normed = ridge_arr / (np.linalg.norm(ridge_arr, axis=1, keepdims=True) + 1e-8)
    cos_sim = ridge_normed @ ridge_normed.T
    n_obs = int(ridge_normed.shape[0])
    if n_obs >= 2:
        cos_sim_off_diag = cos_sim[np.triu_indices(n_obs, k=1)]
        print(f"\n[Ridge Embedding Diversity]")
        print(f"  Pairwise cosine similarity: min={cos_sim_off_diag.min():.4f}, max={cos_sim_off_diag.max():.4f}, mean={cos_sim_off_diag.mean():.4f}")
        if cos_sim_off_diag.mean() > 0.95:
            print(f"  [WARN] Ridge embeddings are very similar (mean cos_sim > 0.95)!")
            print(f"         This may limit CCA's ability to find meaningful correlations.")
    else:
        cos_sim_off_diag = np.array([], dtype=np.float64)
        print("\n[Ridge Embedding Diversity]")
        print("  Not enough observations to compute off-diagonal cosine similarity.")

    # =========================================================================
    # PRE-PROCESSING (reference-style)
    # - drop near-constant columns (before noise)
    # - add a tiny noise (optional) to break degeneracies
    # - full PCA per view (up to rank) to de-correlate / straighten point clouds
    # =========================================================================
    from sklearn.decomposition import PCA

    if X.shape[0] < 2:
        print("[ERROR] Need at least 2 cycles to run CCA.")
        return

    X_raw = X.copy()
    Y_raw = Y.copy()

    def _drop_near_const_cols(Z: np.ndarray, name: str, tol: float) -> tuple[np.ndarray, np.ndarray]:
        std = Z.std(axis=0)
        keep = std > tol
        dropped = int((~keep).sum())
        if dropped > 0:
            print(f"  [pre] Dropping near-constant cols in {name}: {dropped}/{Z.shape[1]} (tol={tol:g})")
        return Z[:, keep], keep

    X, keep_x = _drop_near_const_cols(X_raw, "X", args.drop_const_tol)
    Y, keep_y = _drop_near_const_cols(Y_raw, "Y", args.drop_const_tol)
    print(f"  [pre] After drop const cols: X={X.shape}, Y={Y.shape}")

    if args.noise_eps and args.noise_eps > 0:
        rng_noise = np.random.default_rng(args.noise_seed)
        eps = float(args.noise_eps)
        X = X + rng_noise.uniform(-eps, eps, size=X.shape)
        Y = Y + rng_noise.uniform(-eps, eps, size=Y.shape)
        print(f"  [pre] Added uniform noise eps={eps:g} (seed={args.noise_seed})")

    ncomp_x = int(min(X.shape[0] - 1, X.shape[1]))
    ncomp_y = int(min(Y.shape[0] - 1, Y.shape[1]))
    if ncomp_x < 1 or ncomp_y < 1:
        print("[ERROR] Not enough rank after preprocessing for PCA/CCA.")
        return

    pca_x = PCA(n_components=ncomp_x, svd_solver="full")
    pca_y = PCA(n_components=ncomp_y, svd_solver="full")
    X = pca_x.fit_transform(X)
    Y = pca_y.fit_transform(Y)
    print(f"  [pre] Full PCA applied: X_pca={X.shape}, Y_pca={Y.shape}")
    print(f"  [pre] Explained variance ratio sum: X={pca_x.explained_variance_ratio_.sum():.4f}, Y={pca_y.explained_variance_ratio_.sum():.4f}")
    
    # =========================================================================
    # RUN CCA
    # =========================================================================
    print("\n" + "-"*40)
    print("RUNNING CCA")
    print("-"*40)

    try:
        A, B, r, U, V = canoncorr(X, Y, fullReturn=True)
    except Exception as e:
        print(f"[ERROR] CCA failed: {e}")
        return
    
    print(f"\n[CCA Results]")
    print(f"  Number of modes: {len(r)}")
    print(f"  Top {min(10, len(r))} correlations: {r[:10]}")
    
    # Analyze correlation distribution
    high_corr = (r > 0.9).sum()
    mid_corr = ((r > 0.5) & (r <= 0.9)).sum()
    low_corr = (r <= 0.5).sum()
    print(f"\n[Correlation Distribution]")
    print(f"  High (>0.9): {high_corr}")
    print(f"  Medium (0.5-0.9): {mid_corr}")
    print(f"  Low (<=0.5): {low_corr}")
    
    if high_corr == len(r) and len(r) > 5:
        print(f"\n[WARN] All correlations are >0.9!")
        print(f"       This usually indicates:")
        print(f"       1. Low sample diversity (too few unique routes)")
        print(f"       2. Ridge embeddings are too similar")
        print(f"       3. Overfitting due to low-rank data")
    
    # Save lollipop
    lollipop_path = os.path.join(args.out_dir, "cca_lollipop.png")
    plot_lollipop(r[:args.num_modes], lollipop_path)
    print(f"\nSaved lollipop plot to {lollipop_path}")
    
    # =========================================================================
    # FIGURE 5 (reference-style): plot U[:,0:2] and V[:,0:2] directly
    # - one point per cycle (already true)
    # - z-score U and V columns before plotting for comparable axes
    # =========================================================================
    print("\n" + "-"*40)
    print("ALIGNMENT PLOT (DIRECT U/V, Z-SCORED)")
    print("-"*40)
    
    U_plot = _zscore_cols(np.asarray(U, dtype=np.float64))
    V_plot = _zscore_cols(np.asarray(V, dtype=np.float64))
    print(f"  U shape: {U_plot.shape}")
    print(f"  V shape: {V_plot.shape}")
    
    # Optional Filtering
    if args.filter_outliers:
        print("\n" + "-"*40)
        print("FILTERING OUTLIERS")
        print("-"*40)
        
        # Calculate robust statistics for U_plot (Neural) in (CM0, CM1)
        u_cm0 = U_plot[:, 0]
        u_cm1 = U_plot[:, 1]
        
        # Using IQR method
        q1_0, q3_0 = np.percentile(u_cm0, [25, 75])
        iqr_0 = q3_0 - q1_0
        lower_0 = q1_0 - 1.5 * iqr_0
        upper_0 = q3_0 + 1.5 * iqr_0
        
        q1_1, q3_1 = np.percentile(u_cm1, [25, 75])
        iqr_1 = q3_1 - q1_1
        lower_1 = q1_1 - 1.5 * iqr_1
        upper_1 = q3_1 + 1.5 * iqr_1
        
        mask_0 = (u_cm0 >= lower_0) & (u_cm0 <= upper_0)
        mask_1 = (u_cm1 >= lower_1) & (u_cm1 <= upper_1)
        
        # Repeat for V_plot (Behavior)
        v_cm0 = V_plot[:, 0]
        v_cm1 = V_plot[:, 1]
        
        q1_v0, q3_v0 = np.percentile(v_cm0, [25, 75])
        iqr_v0 = q3_v0 - q1_v0
        lower_v0 = q1_v0 - 1.5 * iqr_v0
        upper_v0 = q3_v0 + 1.5 * iqr_v0
        
        q1_v1, q3_v1 = np.percentile(v_cm1, [25, 75])
        iqr_v1 = q3_v1 - q1_v1
        lower_v1 = q1_v1 - 1.5 * iqr_v1
        upper_v1 = q3_v1 + 1.5 * iqr_v1
        
        mask_v0 = (v_cm0 >= lower_v0) & (v_cm0 <= upper_v0)
        mask_v1 = (v_cm1 >= lower_v1) & (v_cm1 <= upper_v1)
        
        # Combined mask
        keep_mask = mask_0 & mask_1 & mask_v0 & mask_v1
        
        print(f"  Filtering stats (IQR method):")
        print(f"    U_CM0: IQR={iqr_0:.2f}, Range=[{lower_0:.2f}, {upper_0:.2f}]")
        print(f"    U_CM1: IQR={iqr_1:.2f}, Range=[{lower_1:.2f}, {upper_1:.2f}]")
        print(f"    V_CM0: IQR={iqr_v0:.2f}, Range=[{lower_v0:.2f}, {upper_v0:.2f}]")
        print(f"    V_CM1: IQR={iqr_v1:.2f}, Range=[{lower_v1:.2f}, {upper_v1:.2f}]")
        
        kept_count = keep_mask.sum()
        total_count = len(keep_mask)
        print(f"  Kept {kept_count}/{total_count} points ({kept_count/total_count*100:.1f}%)")
        
        U_plot = U_plot[keep_mask]
        V_plot = V_plot[keep_mask]
        cycle_lengths = cycle_lengths[keep_mask]
        cycle_disps = cycle_disps[keep_mask]
        cycle_angles = cycle_angles[keep_mask]

    # Check spread in canonical space
    u_spread = U_plot[:, :2].std(axis=0)
    v_spread = V_plot[:, :2].std(axis=0)
    print(f"  U spread (CM0, CM1): {u_spread}")
    print(f"  V spread (CM0, CM1): {v_spread}")
    
    # Plot Figure 5 and variants
    metadata = {
        'Length': cycle_lengths,
        'Displacement': cycle_disps,
        'Angle': cycle_angles
    }
    
    # Original plot (Length)
    # alignment_path = os.path.join(args.out_dir, "figure5_alignment.png")
    # plot_alignment(U_plot, V_plot, np.array(cycle_lengths), alignment_path, title="Alignment (z-scored U/V)")
    # print(f"Saved alignment plot to {alignment_path}")
    
    # Multi-feature plots
    plot_alignment_multi(U_plot, V_plot, metadata, args.out_dir, prefix="fig5")
    
    # 3D Interactive Plot
    if U_plot.shape[1] >= 3:
        plot_3d_interactive(
            U_plot[:, :3], 
            V_plot[:, :3], 
            metadata, 
            os.path.join(args.out_dir, "alignment_3d.html")
        )
    else:
        print(f"[WARN] Only {U_plot.shape[1]} modes available, skipping 3D plot.")
    
    # =========================================================================
    # PLOT ALL PATH_TILES OVERLAY (Center-aligned, exclude last point)
    # =========================================================================
    print("\n" + "-"*40)
    print("PLOTTING ALL PATH TRAJECTORIES")
    print("-"*40)
    
    if len(all_path_tiles) > 0:
        from matplotlib.collections import LineCollection
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        print(f"  Plotting {len(all_path_tiles)} path trajectories (center-aligned, last point removed)")
        
        # Prepare line segments for LineCollection
        line_segments = []
        start_points_x = []
        start_points_y = []
        
        # For counting unique trajectories
        path_signatures = []
        
        for path in all_path_tiles:
            if len(path) > 2:  # Need at least 3 points (to have 2+ after removing last)
                # Remove the last point from each path
                path_trimmed = path[:-1]
                
                # Add to line segments
                line_segments.append(path_trimmed)
                
                # Collect start points (should all be at origin)
                start_points_x.append(path_trimmed[0, 0])
                start_points_y.append(path_trimmed[0, 1])
                
                # Create signature for uniqueness check
                # Round to 3 decimal places to handle floating point precision
                path_rounded = np.round(path_trimmed, decimals=3)
                # Convert to tuple of tuples for hashing
                path_signature = tuple(map(tuple, path_rounded))
                path_signatures.append(path_signature)
        
        # Count unique trajectories
        unique_paths = set(path_signatures)
        num_unique = len(unique_paths)
        num_total = len(path_signatures)
        
        print(f"\n  [Trajectory Diversity]")
        print(f"  Total trajectories: {num_total}")
        print(f"  Unique trajectories: {num_unique}")
        print(f"  Duplicate ratio: {(num_total - num_unique)/max(1, num_total)*100:.1f}%")
        print(f"  Diversity ratio: {num_unique/max(1, num_total)*100:.1f}%")
        
        # Use LineCollection to plot all paths at once
        if line_segments:
            lc = LineCollection(line_segments, colors='blue', linewidths=0.5, alpha=0.3)
            ax.add_collection(lc)
        
        # Plot all start points together
        if start_points_x:
            ax.scatter(start_points_x, start_points_y, s=5, c='green', alpha=0.5, 
                      marker='o', label='Start Points (at origin)', zorder=5)
        
        # Set equal aspect ratio (1:1)
        ax.set_aspect('equal', adjustable='box')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Mark the center (0, 0) with a larger marker
        ax.scatter([0], [0], s=100, c='black', marker='o', 
                  label='Origin (All Paths Start Here)', zorder=10, 
                  edgecolors='yellow', linewidths=2)
        
        # Auto-scale to fit all paths
        ax.autoscale()
        
        ax.set_xlabel("X (Grid Units)")
        ax.set_ylabel("Y (Grid Units)")
        ax.set_title(f"All Path Trajectories Overlay (n={len(line_segments)}, Last Point Removed)")
        ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        
        # Save
        all_paths_overlay_path = os.path.join(args.out_dir, "all_paths_overlay.png")
        plt.savefig(all_paths_overlay_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved all paths overlay to {all_paths_overlay_path}")
        print(f"  Total paths plotted: {len(line_segments)} (last point removed from each)")
        print(f"  All paths start from origin (0, 0)")
        print(f"  Aspect ratio: 1:1 (equal)")
    
    # =========================================================================
    # SAVE DETAILED RESULTS
    # =========================================================================
    results_path = os.path.join(args.out_dir, "cca_results.npz")
    ridge_cos_mean = float(cos_sim_off_diag.mean()) if cos_sim_off_diag.size > 0 else float("nan")
    num_cycles_used = int(len(cycle_lengths))
    np.savez_compressed(
        results_path,
        correlations_in_sample=r,
        # For compatibility with older scripts, keep these keys.
        # In this version, these are the *direct* canonical variates (one row per cycle),
        # z-scored column-wise for plotting (reference-style).
        U_means=U_plot,
        V_means=V_plot,
        # Also store raw U/V from canoncorr (on PCA-transformed data).
        U=U,
        V=V,
        cycle_lengths=np.array(cycle_lengths),
        cycle_disps=np.array(cycle_disps),
        cycle_angles=np.array(cycle_angles),
        ridge_cosine_sim_mean=ridge_cos_mean,
        num_cycles=num_cycles_used,
        num_cycles_total=int(num_cycles),
        cca_level=args.cca_level,
        x_agg=args.x_agg,
    )
    print(f"Saved detailed results to {results_path}")
    
    print("\n" + "="*60)
    print("CCA ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
