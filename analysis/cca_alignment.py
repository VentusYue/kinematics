
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr, svd, inv
import argparse
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis.ridge_embedding import build_ridge_vector

def canoncorr(X0: np.array, Y0: np.array, fullReturn: bool = False) -> np.array:
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

    # Compute the canonical variates
    U = X @ A_full
    V = Y @ B_full

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
    
    # Left: Neural (U)
    sc1 = axes[0].scatter(
        U_means[:, 0] + jitter_u[:, 0], 
        U_means[:, 1] + jitter_u[:, 1], 
        c=colors, cmap='viridis', alpha=0.7, s=50, edgecolors='black', linewidths=0.5
    )
    axes[0].set_title(f"Neural State (CM0 vs CM1) - {n_points} points")
    axes[0].set_xlabel("CM 0")
    axes[0].set_ylabel("CM 1")
    plt.colorbar(sc1, ax=axes[0], label='Episode Length')
    
    # Right: Behavior (V)
    sc2 = axes[1].scatter(
        V_means[:, 0] + jitter_v[:, 0], 
        V_means[:, 1] + jitter_v[:, 1], 
        c=colors, cmap='viridis', alpha=0.7, s=50, edgecolors='black', linewidths=0.5
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
    
    X_samples = []
    Y_samples = []
    sample_cycle_ids = []
    
    # For alignment plot later
    cycle_lengths = []
    
    # Ridge embedding statistics
    ridge_vecs = []
    est_grid_steps = []
    
    for i in range(num_cycles):
        h_cycle = cycles_hidden[i] # (L, H)
        r_id = cycles_route_id[i]
        
        if h_cycle.ndim == 1:
             # Should be (L, H) but sometimes saved as flattened or squeezed?
             # If H is 256
             if h_cycle.shape[0] % 256 == 0:
                 h_cycle = h_cycle.reshape(-1, 256)
        
        path_xy = routes_xy[r_id] # (T_route, 2)
        
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
        
        # Compute ridge embedding
        ridge_vec = build_ridge_vector(path_tile) # (441,)
        ridge_vecs.append(ridge_vec)
        
        # Hiddens
        L = h_cycle.shape[0]
        
        X_samples.append(h_cycle)
        Y_samples.append(np.tile(ridge_vec, (L, 1)))
        
        sample_cycle_ids.extend([i] * L)
        cycle_lengths.append(routes_ep_len[r_id])

    X = np.concatenate(X_samples, axis=0).astype(np.float32)
    Y = np.concatenate(Y_samples, axis=0).astype(np.float32)
    sample_cycle_ids = np.array(sample_cycle_ids)
    
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
    # AGGREGATE FOR FIGURE 5
    # =========================================================================
    print("\n" + "-"*40)
    print("AGGREGATING FOR ALIGNMENT PLOT")
    print("-"*40)
    
    # We want one point per cycle
    U_means = []
    V_means = []
    
    for i in range(num_cycles):
        # Indices for this cycle
        indices = np.where(sample_cycle_ids == i)[0]
        
        # U mean
        u_mean = np.mean(U[indices], axis=0)
        U_means.append(u_mean)
        
        # V mean (should be constant since ridge is constant per cycle)
        v_mean = np.mean(V[indices], axis=0)
        V_means.append(v_mean)
        
    U_means = np.array(U_means)
    V_means = np.array(V_means)
    
    print(f"  U_means shape: {U_means.shape}")
    print(f"  V_means shape: {V_means.shape}")
    
    # Check spread in canonical space
    u_spread = U_means[:, :2].std(axis=0)
    v_spread = V_means[:, :2].std(axis=0)
    print(f"  U spread (CM0, CM1): {u_spread}")
    print(f"  V spread (CM0, CM1): {v_spread}")
    
    # Plot Figure 5
    alignment_path = os.path.join(args.out_dir, "figure5_alignment.png")
    plot_alignment(U_means, V_means, cycle_lengths, alignment_path)
    print(f"Saved alignment plot to {alignment_path}")
    
    # =========================================================================
    # SAVE DETAILED RESULTS
    # =========================================================================
    results_path = os.path.join(args.out_dir, "cca_results.npz")
    np.savez_compressed(
        results_path,
        correlations=r,
        U_means=U_means,
        V_means=V_means,
        cycle_lengths=np.array(cycle_lengths),
        ridge_cosine_sim_mean=float(cos_sim_off_diag.mean()),
        num_cycles=num_cycles,
    )
    print(f"Saved detailed results to {results_path}")
    
    print("\n" + "="*60)
    print("CCA ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
