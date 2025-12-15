
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
    print(f"X shape: {X0.shape}, Y shape: {Y0.shape}")
    
    if p1 >= n or p2 >= n:
        logging.warning('Not enough samples, might cause problems')

    # Preprocessing: Standardize the variables
    # Handle constant columns to avoid division by zero
    X_std = np.std(X0, 0)
    Y_std = np.std(Y0, 0)
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

    print(f"Rank of X: {rankX}, Rank of Y: {rankY}")

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
    U_means: (N_cycles, 2) - typically CM0 vs CM1
    V_means: (N_cycles, 2)
    colors: (N_cycles,) for color coding (e.g. length)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Neural (U)
    sc1 = axes[0].scatter(U_means[:, 0], U_means[:, 1], c=colors, cmap='viridis', alpha=0.7)
    axes[0].set_title("Neural State (CM0 vs CM1)")
    axes[0].set_xlabel("CM 0")
    axes[0].set_ylabel("CM 1")
    plt.colorbar(sc1, ax=axes[0], label='Episode Length')
    
    # Right: Behavior (V)
    sc2 = axes[1].scatter(V_means[:, 0], V_means[:, 1], c=colors, cmap='viridis', alpha=0.7)
    axes[1].set_title("Behavior Ridge (CM0 vs CM1)")
    axes[1].set_xlabel("CM 0")
    axes[1].set_ylabel("CM 1")
    plt.colorbar(sc2, ax=axes[1], label='Episode Length')
    
    plt.suptitle(title)
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
    
    print(f"Loading cycles from {args.cycles_npz}")
    pkd_data = np.load(args.cycles_npz, allow_pickle=True)
    cycles_hidden = pkd_data['cycles_hidden']
    cycles_route_id = pkd_data['cycles_route_id']
    cycles_match_ratio = pkd_data['cycles_match_ratio']
    
    print(f"Loading routes from {args.routes_npz}")
    routes_data = np.load(args.routes_npz, allow_pickle=True)
    routes_xy = routes_data['routes_xy']
    routes_ep_len = routes_data['routes_ep_len']
    
    num_cycles = len(cycles_hidden)
    print(f"Processing {num_cycles} cycles")
    
    X_samples = []
    Y_samples = []
    sample_cycle_ids = []
    
    # For alignment plot later
    cycle_lengths = []
    
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
            
        # Normalize
        # Avoid division by zero
        if est_grid_step < 1e-3:
            est_grid_step = 1.0
            
        path_tile = path_xy / est_grid_step
        
        # Compute ridge embedding
        # We use the whole route path to generate the ridge image for this trial
        # And replicate it for every step in the cycle?
        # Yes, Section 7.1: "Y: stack corresponding ridge_vec... same cycle will repeat same ridge_vec"
        
        ridge_vec = build_ridge_vector(path_tile) # (441,)
        
        # Hiddens
        # h_cycle is (L, H)
        # Add to lists
        L = h_cycle.shape[0]
        
        X_samples.append(h_cycle)
        Y_samples.append(np.tile(ridge_vec, (L, 1)))
        
        sample_cycle_ids.extend([i] * L)
        cycle_lengths.append(routes_ep_len[r_id])

    X = np.concatenate(X_samples, axis=0).astype(np.float32)
    Y = np.concatenate(Y_samples, axis=0).astype(np.float32)
    sample_cycle_ids = np.array(sample_cycle_ids)
    
    print("Running CCA...")
    A, B, r, U, V = canoncorr(X, Y, fullReturn=True)
    
    print(f"Top 5 correlations: {r[:5]}")
    
    # Save lollipop
    plot_lollipop(r[:args.num_modes], os.path.join(args.out_dir, "cca_lollipop.png"))
    
    # Aggregate for Figure 5
    # We want one point per cycle
    U_means = []
    V_means = []
    
    for i in range(num_cycles):
        # Indices for this cycle
        indices = np.where(sample_cycle_ids == i)[0]
        
        # U mean
        u_mean = np.mean(U[indices], axis=0)
        U_means.append(u_mean)
        
        # V mean (should be constant)
        v_mean = np.mean(V[indices], axis=0)
        V_means.append(v_mean)
        
    U_means = np.array(U_means)
    V_means = np.array(V_means)
    
    # Plot Figure 5
    plot_alignment(U_means, V_means, cycle_lengths, os.path.join(args.out_dir, "figure5_alignment.png"))
    
    print("Done!")

if __name__ == "__main__":
    main()

