# udpate for this version:
'''
1. select best cycle per route based on match_ratio
2. cycle-level mean pooling
3. ridge embedding fixed back to sqrt(2)*grid_size
4. X: (Total_Steps, H), Y: ridge, T), change to X:(N_cycles, H), Y: (N_cycles, 441)
'''


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr, svd, solve_triangular
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis.ridge_embedding import build_ridge_vector

def _safe_colwise_corr(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute Pearson correlation for each corresponding column in U and V.
    Returns NaN for columns with near-zero std.
    """
    assert U.shape == V.shape
    u = U - U.mean(axis=0, keepdims=True)
    v = V - V.mean(axis=0, keepdims=True)
    u_std = u.std(axis=0)
    v_std = v.std(axis=0)
    denom = u_std * v_std
    out = np.full((U.shape[1],), np.nan, dtype=np.float64)
    ok = denom > 1e-12
    out[ok] = (u[:, ok] * v[:, ok]).mean(axis=0) / denom[ok]
    return out


def _standardize_fit(X0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(X0, axis=0)
    std = np.std(X0, axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def _standardize_apply(X0: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X0 - mean) / std


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

def plot_lollipop_train_test(train_scores, test_scores, out_path, title="Canonical Correlation (Train vs Test)", xlabel="Mode", ylabel="Correlation"):
    """
    Overlay train and held-out (test) canonical correlations.
    """
    m = min(len(train_scores), len(test_scores))
    train_scores = np.asarray(train_scores)[:m]
    test_scores = np.asarray(test_scores)[:m]
    x = np.arange(m)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.vlines(x, 0, train_scores, colors='gray', lw=1, alpha=0.35)
    ax.plot(x, train_scores, color='gray', marker='o', markersize=6, linestyle='-', linewidth=1, alpha=0.6, label='train (in-sample)')
    ax.plot(x, test_scores, color='black', marker='o', markersize=7, linestyle='-', linewidth=1.5, label='test (held-out)')
    for i, score in enumerate(test_scores):
        if np.isfinite(score):
            ax.annotate(f'{score:.2f}', (i, score), textcoords="offset points",
                        xytext=(0, 10), ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(range(1, m + 1))
    ax.legend(loc='best')
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
    parser.add_argument("--test_frac", type=float, default=0.2,
                        help="Held-out fraction of cycles for reporting out-of-sample canonical correlations.")
    parser.add_argument("--pca_dim_x", type=int, default=50,
                        help="[deprecated] Ignored. We always run full PCA (up to rank) to 'straighten' the point cloud.")
    parser.add_argument("--pca_dim_y", type=int, default=50, 
                        help="[deprecated] Ignored. We always run full PCA (up to rank) to 'straighten' the point cloud.")
    parser.add_argument("--split_seed", type=int, default=0, help="Random seed for train/test split.")
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

    # Ridge embedding statistics
    ridge_vecs: list[np.ndarray] = []
    est_grid_steps: list[float] = []
    all_path_tiles: list[np.ndarray] = []  # Store normalized path_tile for visualization

    hidden_dim0 = int(hidden_dims[0])
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

        # Grid step normalization heuristic
        if len(path_xy) > 1:
            diffs = np.linalg.norm(path_xy[1:] - path_xy[:-1], axis=1)
            diffs = diffs[diffs > 0.1]
            est_grid_step = float(np.median(diffs)) if len(diffs) > 0 else 1.0
        else:
            est_grid_step = 1.0
        est_grid_step = max(est_grid_step, 1e-3)
        est_grid_steps.append(est_grid_step)

        # Normalize to (approx) tile units and align first point to origin
        path_tile = (path_xy / est_grid_step).astype(np.float32)
        path_tile = path_tile - path_tile[0]

        # Behavior embedding: ridge image flatten (441D)
        ridge_vec = build_ridge_vector(path_tile)  # (441,)
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

    if skipped_nan > 0:
        print(f"\n[WARN] Skipped cycles due to NaNs: {skipped_nan}/{num_cycles}")

    X = np.stack(X_cycles, axis=0).astype(np.float64)  # (N, H)
    Y = np.stack(Y_cycles, axis=0).astype(np.float64)  # (N, 441)
    cycle_lengths = np.asarray(cycle_lengths, dtype=np.float64)
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

    # Keep a copy for held-out evaluation (we'll fit PCA on train only).
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
    
    # -------------------------------------------------------------------------
    # Report held-out correlations to avoid in-sample inflation.
    # We split by cycle (route identity) regardless of cca_level to prevent leakage.
    # -------------------------------------------------------------------------
    print("\n[Held-out evaluation]")
    if not (0.0 < args.test_frac < 1.0):
        print("  [warn] test_frac outside (0,1); skipping held-out evaluation.")
        do_eval = False
    else:
        do_eval = True

    r_test = None
    r_train_emp = None
    train_idx = None
    test_idx = None

    if do_eval:
        n_obs_raw = int(X_raw.shape[0])
        if n_obs_raw < 3:
            print("  [warn] Too few cycles for a meaningful train/test split; skipping held-out evaluation.")
        else:
            rng = np.random.default_rng(args.split_seed)
            perm = rng.permutation(n_obs_raw)
            n_test = max(1, int(round(args.test_frac * n_obs_raw)))
            test_idx = perm[:n_test]
            train_idx = perm[n_test:]

            X_tr0, Y_tr0 = X_raw[train_idx], Y_raw[train_idx]
            X_te0, Y_te0 = X_raw[test_idx], Y_raw[test_idx]

            # Drop near-constant columns based on train only, then apply the same mask to test.
            vx = X_tr0.std(axis=0)
            vy = Y_tr0.std(axis=0)
            keep_x_tr = vx > args.drop_const_tol
            keep_y_tr = vy > args.drop_const_tol
            X_tr = X_tr0[:, keep_x_tr]
            X_te = X_te0[:, keep_x_tr]
            Y_tr = Y_tr0[:, keep_y_tr]
            Y_te = Y_te0[:, keep_y_tr]

            # Add tiny noise (same setting as full run) to stabilize degeneracies.
            if args.noise_eps and args.noise_eps > 0:
                rng_noise = np.random.default_rng(args.noise_seed)
                eps = float(args.noise_eps)
                X_tr = X_tr + rng_noise.uniform(-eps, eps, size=X_tr.shape)
                X_te = X_te + rng_noise.uniform(-eps, eps, size=X_te.shape)
                Y_tr = Y_tr + rng_noise.uniform(-eps, eps, size=Y_tr.shape)
                Y_te = Y_te + rng_noise.uniform(-eps, eps, size=Y_te.shape)

            # Full PCA fitted on train only (avoid leakage).
            ncomp_x_tr = int(min(X_tr.shape[0] - 1, X_tr.shape[1]))
            ncomp_y_tr = int(min(Y_tr.shape[0] - 1, Y_tr.shape[1]))
            if ncomp_x_tr < 1 or ncomp_y_tr < 1:
                print("  [warn] Not enough rank in train split for PCA/CCA; skipping held-out evaluation.")
            else:
                pca_x_tr = PCA(n_components=ncomp_x_tr, svd_solver="full")
                pca_y_tr = PCA(n_components=ncomp_y_tr, svd_solver="full")
                X_tr_p = pca_x_tr.fit_transform(X_tr)
                X_te_p = pca_x_tr.transform(X_te)
                Y_tr_p = pca_y_tr.fit_transform(Y_tr)
                Y_te_p = pca_y_tr.transform(Y_te)

                # Standardize by train stats before CCA + compute empirical held-out correlations.
                xm, xs = _standardize_fit(X_tr_p)
                ym, ys = _standardize_fit(Y_tr_p)
                X_tr_z = _standardize_apply(X_tr_p, xm, xs)
                Y_tr_z = _standardize_apply(Y_tr_p, ym, ys)
                X_te_z = _standardize_apply(X_te_p, xm, xs)
                Y_te_z = _standardize_apply(Y_te_p, ym, ys)

                A_z, B_z, _, _, _ = canoncorr(X_tr_z, Y_tr_z, fullReturn=True)
                U_tr = X_tr_z @ A_z
                V_tr = Y_tr_z @ B_z
                U_te = X_te_z @ A_z
                V_te = Y_te_z @ B_z
                r_train_emp = _safe_colwise_corr(U_tr, V_tr)
                r_test = _safe_colwise_corr(U_te, V_te)

                r_train_emp = np.clip(np.abs(r_train_emp), 0, 1)
                r_test = np.clip(np.abs(r_test), 0, 1)

                print(f"  Split by cycles: train={len(train_idx)}, test={len(test_idx)} (test_frac={args.test_frac})")
                print(f"  Top {min(10, len(r_test))} held-out correlations: {r_test[:10]}")

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
    if r_test is not None and r_train_emp is not None:
        plot_lollipop_train_test(r_train_emp[:args.num_modes], r_test[:args.num_modes], lollipop_path)
        print(f"\nSaved train-vs-test lollipop plot to {lollipop_path}")
    else:
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

    # Check spread in canonical space
    u_spread = U_plot[:, :2].std(axis=0)
    v_spread = V_plot[:, :2].std(axis=0)
    print(f"  U spread (CM0, CM1): {u_spread}")
    print(f"  V spread (CM0, CM1): {v_spread}")
    
    # Plot Figure 5
    alignment_path = os.path.join(args.out_dir, "figure5_alignment.png")
    plot_alignment(U_plot, V_plot, np.array(cycle_lengths), alignment_path, title="Alignment (z-scored U/V)")
    print(f"Saved alignment plot to {alignment_path}")
    
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
        correlations_train_empirical=r_train_emp if r_train_emp is not None else np.array([]),
        correlations_test_heldout=r_test if r_test is not None else np.array([]),
        # For compatibility with older scripts, keep these keys.
        # In this version, these are the *direct* canonical variates (one row per cycle),
        # z-scored column-wise for plotting (reference-style).
        U_means=U_plot,
        V_means=V_plot,
        # Also store raw U/V from canoncorr (on PCA-transformed data).
        U=U,
        V=V,
        cycle_lengths=np.array(cycle_lengths),
        ridge_cosine_sim_mean=ridge_cos_mean,
        num_cycles=num_cycles_used,
        num_cycles_total=int(num_cycles),
        cca_level=args.cca_level,
        x_agg=args.x_agg,
        test_frac=args.test_frac,
        split_seed=args.split_seed,
    )
    print(f"Saved detailed results to {results_path}")
    
    print("\n" + "="*60)
    print("CCA ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
