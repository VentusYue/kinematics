
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr, svd, inv
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

    A = inv(T11) @ L[:, :d] * np.sqrt(n - 1)
    B = inv(T22) @ M[:, :d] * np.sqrt(n - 1)
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
    parser.add_argument("--x_agg", type=str, default="mean", choices=["mean", "last"],
                        help="How to aggregate hidden states into a per-cycle vector when --cca_level=cycle.")
    parser.add_argument("--test_frac", type=float, default=0.2,
                        help="Held-out fraction of cycles for reporting out-of-sample canonical correlations.")
    parser.add_argument("--pca_dim_x", type=int, default=50,
                        help="If > 0, apply PCA to X (Neural) before CCA.")
    parser.add_argument("--pca_dim_y", type=int, default=50, 
                        help="If > 0, apply PCA to Y (Ridge) before CCA.")
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
    # BUILD DATA MATRICES
    # =========================================================================
    print("\n" + "-"*40)
    print("BUILDING DATA MATRICES")
    print("-"*40)
    
    # We'll build per-cycle features first, then optionally expand to per-step.
    X_cycle = []
    Y_cycle = []
    cycle_lengths = []
    
    # Ridge embedding statistics
    ridge_vecs = []
    est_grid_steps = []
    all_path_tiles = []  # Store all normalized path_tile data for visualization
    
    for i in range(num_cycles):
        h_cycle = cycles_hidden[i] # (L, H)
        r_id = cycles_route_id[i]
        
        if h_cycle.ndim == 1:
             # Should be (L, H) but sometimes saved as flattened or squeezed?
             # If H is 256
             if h_cycle.shape[0] % 256 == 0:
                 h_cycle = h_cycle.reshape(-1, 256)
        
        path_xy = routes_xy[r_id] # (T_route, 2)

        # [CHECK 4] Assertions
        if len(path_xy) != len(h_cycle):
            print(f"[WARN] Cycle {i}: Length mismatch! path_xy={len(path_xy)}, h_cycle={len(h_cycle)}")
            # Adjust path_xy to match h_cycle if possible (e.g. truncate or error)
            # For now, we'll just warn and proceed, but this is a critical check.
        
        if np.isnan(path_xy).any():
             print(f"[WARN] Cycle {i}: NaNs in path_xy!")
        if np.isnan(h_cycle).any():
             print(f"[WARN] Cycle {i}: NaNs in h_cycle!")
        
        # Grid Step Normalization Heuristic
        # Procgen Maze grid step is usually around 24.0 or similar (depends on resolution).
        # We can check the diffs.
        if len(path_xy) > 1:
            diffs = np.linalg.norm(path_xy[1:] - path_xy[:-1], axis=1)
            # Filter zero diffs
            diffs = diffs[diffs > 0.1]
            if len(diffs) > 0:
                est_grid_step = np.median(diffs)
            else:
                est_grid_step = 1.0
        else:
            est_grid_step = 1.0
        
        est_grid_steps.append(est_grid_step)
            
        # Normalize
        # Avoid division by zero
        if est_grid_step < 1e-3:
            est_grid_step = 1.0
            
        path_tile = path_xy / est_grid_step
        
        # Align first coordinate to origin (0, 0)
        if len(path_tile) > 0:
            path_tile = path_tile - path_tile[0]
        
        # Compute ridge embedding
        ridge_vec = build_ridge_vector(path_tile) # (441,)
        ridge_vecs.append(ridge_vec)
        
        # Save path_tile for later visualization
        all_path_tiles.append(path_tile.copy())
        
        # Build per-cycle X feature
        if args.x_agg == "mean":
            x_i = np.mean(h_cycle, axis=0)
        elif args.x_agg == "last":
            x_i = h_cycle[-1]
        else:
            raise ValueError(f"Unknown x_agg: {args.x_agg}")

        X_cycle.append(x_i)
        Y_cycle.append(ridge_vec)
        cycle_lengths.append(routes_ep_len[r_id])

    X_cycle = np.asarray(X_cycle, dtype=np.float32)
    Y_cycle = np.asarray(Y_cycle, dtype=np.float32)
    cycle_lengths = np.asarray(cycle_lengths)

    if args.cca_level == "cycle":
        X = X_cycle
        Y = Y_cycle
        sample_cycle_ids = np.arange(num_cycles)
    else:
        # Per-step mode: expand cycle-level ridge to match time-steps, and keep time-step hiddens.
        X_samples = []
        Y_samples = []
        sample_cycle_ids = []
        for i in range(num_cycles):
            h_cycle = cycles_hidden[i]
            if h_cycle.ndim == 1:
                if h_cycle.shape[0] % 256 == 0:
                    h_cycle = h_cycle.reshape(-1, 256)
            L = h_cycle.shape[0]
            X_samples.append(h_cycle)
            Y_samples.append(np.tile(Y_cycle[i], (L, 1)))
            sample_cycle_ids.extend([i] * L)
        X = np.concatenate(X_samples, axis=0).astype(np.float32)
        Y = np.concatenate(Y_samples, axis=0).astype(np.float32)
        sample_cycle_ids = np.asarray(sample_cycle_ids)
    
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
    cos_sim_off_diag = cos_sim[np.triu_indices(num_cycles, k=1)]
    print(f"\n[Ridge Embedding Diversity]")
    print(f"  Pairwise cosine similarity: min={cos_sim_off_diag.min():.4f}, max={cos_sim_off_diag.max():.4f}, mean={cos_sim_off_diag.mean():.4f}")
    if cos_sim_off_diag.mean() > 0.95:
        print(f"  [WARN] Ridge embeddings are very similar (mean cos_sim > 0.95)!")
        print(f"         This may limit CCA's ability to find meaningful correlations.")
    
    # =========================================================================
    # OPTIONAL: PCA on X and Y
    # =========================================================================
    from sklearn.decomposition import PCA
    
    if args.pca_dim_x > 0 and args.pca_dim_x < X.shape[1]:
        print("\n" + "-"*40)
        print(f"PRE-PROCESSING: PCA on X (target dim={args.pca_dim_x})")
        print("-"*40)
        pca_x = PCA()
        X = pca_x.fit_transform(X)
        print(f"  X reduced shape: {X.shape}")
        print(f"  Explained variance ratio sum: {pca_x.explained_variance_ratio_.sum():.4f}")

    if args.pca_dim_y > 0 and args.pca_dim_y < Y.shape[1]:
        print("\n" + "-"*40)
        print(f"PRE-PROCESSING: PCA on Y (target dim={args.pca_dim_y})")
        print("-"*40)
        pca_y = PCA()
        Y = pca_y.fit_transform(Y)
        print(f"  Y reduced shape: {Y.shape}")
        print(f"  Explained variance ratio sum: {pca_y.explained_variance_ratio_.sum():.4f}")
    
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
        rng = np.random.default_rng(args.split_seed)
        perm = rng.permutation(num_cycles)
        n_test = max(1, int(round(args.test_frac * num_cycles)))
        test_cycles = perm[:n_test]
        train_cycles = perm[n_test:]

        # Map cycle split to sample indices (for step-level this expands).
        if args.cca_level == "cycle":
            train_idx = train_cycles
            test_idx = test_cycles
        else:
            train_mask = np.isin(sample_cycle_ids, train_cycles)
            test_mask = np.isin(sample_cycle_ids, test_cycles)
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

        X_tr, Y_tr = X[train_idx], Y[train_idx]
        X_te, Y_te = X[test_idx], Y[test_idx]

        # Fit using train-set normalization, apply to train/test, and measure empirical corr.
        xm, xs = _standardize_fit(X_tr)
        ym, ys = _standardize_fit(Y_tr)
        X_tr_z = _standardize_apply(X_tr, xm, xs)
        Y_tr_z = _standardize_apply(Y_tr, ym, ys)
        X_te_z = _standardize_apply(X_te, xm, xs)
        Y_te_z = _standardize_apply(Y_te, ym, ys)

        # Fit CCA on train (using standardized data inside canoncorr is fine as long as we feed z-scored).
        A_z, B_z, r_train, _, _ = canoncorr(X_tr_z, Y_tr_z, fullReturn=True)
        U_tr = X_tr_z @ A_z
        V_tr = Y_tr_z @ B_z
        U_te = X_te_z @ A_z
        V_te = Y_te_z @ B_z
        r_train_emp = _safe_colwise_corr(U_tr, V_tr)
        r_test = _safe_colwise_corr(U_te, V_te)

        # In rare cases, numerical issues can create |r|>1; clamp for display.
        r_train_emp = np.clip(np.abs(r_train_emp), 0, 1)
        r_test = np.clip(np.abs(r_test), 0, 1)

        print(f"  Split by cycles: train={len(train_cycles)}, test={len(test_cycles)} (test_frac={args.test_frac})")
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
    # AGGREGATE FOR FIGURE 5
    # =========================================================================
    print("\n" + "-"*40)
    print("AGGREGATING FOR ALIGNMENT PLOT")
    print("-"*40)
    
    # We want one point per cycle
    if args.cca_level == "cycle":
        U_means = np.asarray(U, dtype=np.float64)
        V_means = np.asarray(V, dtype=np.float64)
    else:
        U_means = []
        V_means = []
        for i in range(num_cycles):
            indices = np.where(sample_cycle_ids == i)[0]
            U_means.append(np.mean(U[indices], axis=0))
            V_means.append(np.mean(V[indices], axis=0))
        U_means = np.asarray(U_means)
        V_means = np.asarray(V_means)
    
    print(f"  U_means shape: {U_means.shape}")
    print(f"  V_means shape: {V_means.shape}")
    
    # Optional Filtering
    if args.filter_outliers:
        print("\n" + "-"*40)
        print("FILTERING OUTLIERS")
        print("-"*40)
        
        # Calculate robust statistics for U_means (Neural)
        # We focus on CM0 and CM1 (first 2 columns)
        u_cm0 = U_means[:, 0]
        u_cm1 = U_means[:, 1]
        
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
        
        # Repeat for V_means (Behavior)
        v_cm0 = V_means[:, 0]
        v_cm1 = V_means[:, 1]
        
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
        
        U_means = U_means[keep_mask]
        V_means = V_means[keep_mask]
        cycle_lengths = cycle_lengths[keep_mask]

    # Check spread in canonical space
    u_spread = U_means[:, :2].std(axis=0)
    v_spread = V_means[:, :2].std(axis=0)
    print(f"  U spread (CM0, CM1): {u_spread}")
    print(f"  V spread (CM0, CM1): {v_spread}")
    
    # Plot Figure 5
    alignment_path = os.path.join(args.out_dir, "figure5_alignment.png")
    plot_alignment(U_means, V_means, np.array(cycle_lengths), alignment_path)
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
    np.savez_compressed(
        results_path,
        correlations_in_sample=r,
        correlations_train_empirical=r_train_emp if r_train_emp is not None else np.array([]),
        correlations_test_heldout=r_test if r_test is not None else np.array([]),
        U_means=U_means,
        V_means=V_means,
        cycle_lengths=np.array(cycle_lengths),
        ridge_cosine_sim_mean=float(cos_sim_off_diag.mean()),
        num_cycles=num_cycles,
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
