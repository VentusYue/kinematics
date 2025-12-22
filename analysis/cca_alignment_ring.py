
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr, svd, solve_triangular
import argparse
import logging
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analysis.ridge_embedding import build_ridge_vector

# =============================================================================
# CCA HELPER FUNCTIONS
# =============================================================================

def canoncorr(X0: np.ndarray, Y0: np.ndarray, fullReturn: bool = False) -> np.ndarray:
    """
    Canonical Correlation Analysis (CCA)
    """
    n, p1 = X0.shape
    p2 = Y0.shape[1]
    
    # Preprocessing: Standardize
    X_std = np.std(X0, 0)
    Y_std = np.std(Y0, 0)
    X_std[X_std == 0] = 1.0
    Y_std[Y_std == 0] = 1.0
    X = (X0 - np.mean(X0, 0)) / X_std
    Y = (Y0 - np.mean(Y0, 0)) / Y_std

    # QR Decomposition
    Q1, T11, perm1 = qr(X, mode='economic', pivoting=True)
    Q2, T22, perm2 = qr(Y, mode='economic', pivoting=True)

    tol = np.finfo(float).eps * 100
    rankX = np.sum(np.abs(np.diagonal(T11)) > tol * np.abs(T11[0, 0]))
    rankY = np.sum(np.abs(np.diagonal(T22)) > tol * np.abs(T22[0, 0]))

    if rankX == 0 or rankY == 0:
        raise ValueError('X or Y has zero rank')

    Q1 = Q1[:, :rankX]
    Q2 = Q2[:, :rankY]
    T11 = T11[:rankX, :rankX]
    T22 = T22[:rankY, :rankY]

    # SVD of cross-covariance
    d = min(rankX, rankY)
    L, D, M = svd(Q1.T @ Q2, full_matrices=True, check_finite=True, lapack_driver='gesdd')
    M = M.T
    r = np.clip(D[:d], 0, 1)

    if not fullReturn:
        return r

    # Coefficients
    A = solve_triangular(T11, L[:, :d], lower=False) * np.sqrt(n - 1)
    B = solve_triangular(T22, M[:, :d], lower=False) * np.sqrt(n - 1)

    # Expand to full size
    A_full = np.zeros((p1, d))
    A_full[perm1, :] = np.vstack((A, np.zeros((p1 - rankX, d))))
    B_full = np.zeros((p2, d))
    B_full[perm2, :] = np.vstack((B, np.zeros((p2 - rankY, d))))

    # Variates
    U = X @ A_full
    V = Y @ B_full

    return A_full, B_full, r, U, V

def zscore_cols(Z: np.ndarray) -> np.ndarray:
    eps = 1e-8
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (Z - mu) / sd

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_alignment_with_properties(
    U, V, properties, prop_name, out_path, 
    cmap='viridis', title_suffix=""
):
    """
    Plot U vs V colored by a specific property (e.g. length, angle).
    """
    n_points = len(properties)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Jitter for visibility
    jitter_scale = 0.05
    ju = np.random.randn(n_points, 2) * jitter_scale
    jv = np.random.randn(n_points, 2) * jitter_scale
    
    # Neural
    sc1 = axes[0].scatter(
        U[:, 0] + ju[:, 0], U[:, 1] + ju[:, 1],
        c=properties, cmap=cmap, alpha=0.6, s=30, edgecolors='none'
    )
    axes[0].set_title(f"Neural (CM0 vs CM1)\nColor: {prop_name}")
    axes[0].set_xlabel("CM 0")
    axes[0].set_ylabel("CM 1")
    axes[0].set_aspect('equal', 'datalim')
    plt.colorbar(sc1, ax=axes[0])
    
    # Behavior
    sc2 = axes[1].scatter(
        V[:, 0] + jv[:, 0], V[:, 1] + jv[:, 1],
        c=properties, cmap=cmap, alpha=0.6, s=30, edgecolors='none'
    )
    axes[1].set_title(f"Behavior (CM0 vs CM1)\nColor: {prop_name}")
    axes[1].set_xlabel("CM 0")
    axes[1].set_ylabel("CM 1")
    axes[1].set_aspect('equal', 'datalim')
    plt.colorbar(sc2, ax=axes[1])
    
    plt.suptitle(f"CCA Alignment {title_suffix}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_3d_scatter(U, properties, out_path, title="3D View"):
    """
    Simple 3D scatter of first 3 components
    """
    from mpl_toolkits.mplot3d import Axes3D
    if U.shape[1] < 3:
        return
        
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(U[:, 0], U[:, 1], U[:, 2], c=properties, cmap='viridis', s=20, alpha=0.6)
    ax.set_xlabel('CM 0')
    ax.set_ylabel('CM 1')
    ax.set_zlabel('CM 2')
    ax.set_title(title)
    plt.colorbar(sc, label='Property')
    plt.savefig(out_path, dpi=200)
    plt.close()

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles_npz", required=True)
    parser.add_argument("--routes_npz", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--pca_dim", type=int, default=50, help="PCA dim before CCA")
    parser.add_argument("--filter_iqr", action="store_true", help="Filter outliers using IQR")
    parser.add_argument("--color_by", type=str, default="angle", choices=["length", "angle", "displacement"], help="Primary color coding")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading cycles: {args.cycles_npz}")
    pkd_data = np.load(args.cycles_npz, allow_pickle=True)
    cycles_hidden = pkd_data['cycles_hidden']
    cycles_route_id = pkd_data['cycles_route_id']
    cycles_match_ratio = pkd_data['cycles_match_ratio']
    
    print(f"Loading routes: {args.routes_npz}")
    routes_data = np.load(args.routes_npz, allow_pickle=True)
    routes_xy = routes_data['routes_xy']
    routes_ep_len = routes_data['routes_ep_len']
    
    # -------------------------------------------------------------------------
    # 1. Feature Extraction
    # -------------------------------------------------------------------------
    print("Extracting features...")
    
    X_list = []
    Y_list = []
    
    # Properties for analysis
    props_length = []
    props_angle = []
    props_disp = []
    
    valid_indices = []
    
    for i, h_cycle in enumerate(cycles_hidden):
        if len(h_cycle) == 0: continue
        
        r_id = cycles_route_id[i]
        path_xy = routes_xy[r_id]
        
        # 1.1 Neural Feature: Mean of Limit Cycle (Centroid)
        # Try to capture the "location" of the cycle in neural space
        x_vec = h_cycle.mean(axis=0)
        
        # 1.2 Behavior Feature: Ridge Embedding
        # Normalize path to grid
        if len(path_xy) > 1:
            diffs = np.linalg.norm(path_xy[1:] - path_xy[:-1], axis=1)
            diffs = diffs[diffs > 0.01] # Filter stationary steps
            scale = np.median(diffs) if len(diffs) > 0 else 1.0
        else:
            scale = 1.0
        
        # Standardize scale to ~1.0
        path_norm = path_xy / scale
        
        # Center path (trajectory-centric, invariant to absolute position)
        if len(path_norm) > 0:
            path_norm = path_norm - path_norm[0]
            
        y_vec = build_ridge_vector(path_norm, grid_size=21, radius_scale=1.414)
        
        # 1.3 Properties
        # Length
        l = routes_ep_len[r_id]
        
        # Angle (Start to End)
        if len(path_xy) > 0:
            disp_vec = path_xy[-1] - path_xy[0]
            angle = np.arctan2(disp_vec[1], disp_vec[0]) # -pi to pi
            mag = np.linalg.norm(disp_vec)
        else:
            angle = 0
            mag = 0
            
        X_list.append(x_vec)
        Y_list.append(y_vec)
        props_length.append(l)
        props_angle.append(angle)
        props_disp.append(mag)
        valid_indices.append(i)

    X = np.stack(X_list).astype(np.float64)
    Y = np.stack(Y_list).astype(np.float64)
    
    # Arrays
    p_len = np.array(props_length)
    p_ang = np.array(props_angle)
    p_disp = np.array(props_disp)
    
    print(f"Data Shapes: X={X.shape}, Y={Y.shape}")
    
    # -------------------------------------------------------------------------
    # 2. PCA Preprocessing
    # -------------------------------------------------------------------------
    # Standardize first
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)
    
    pca_x = PCA(n_components=min(args.pca_dim, X.shape[1], X.shape[0]-1))
    pca_y = PCA(n_components=min(args.pca_dim, Y.shape[1], Y.shape[0]-1))
    
    X_pca = pca_x.fit_transform(X)
    Y_pca = pca_y.fit_transform(Y)
    
    print(f"PCA Shapes: X={X_pca.shape}, Y={Y_pca.shape}")
    print(f"Explained Variance: X={pca_x.explained_variance_ratio_.sum():.2f}, Y={pca_y.explained_variance_ratio_.sum():.2f}")

    # -------------------------------------------------------------------------
    # 3. CCA
    # -------------------------------------------------------------------------
    _, _, r, U, V = canoncorr(X_pca, Y_pca, fullReturn=True)
    
    print(f"Top 5 Correlations: {r[:5]}")
    
    # Z-score for plotting
    U_z = zscore_cols(U)
    V_z = zscore_cols(V)
    
    # -------------------------------------------------------------------------
    # 4. Filter Outliers (Optional but recommended for "Ring" visibility)
    # -------------------------------------------------------------------------
    mask = np.ones(len(U_z), dtype=bool)
    if args.filter_iqr:
        for dim in [0, 1]:
            q1, q3 = np.percentile(U_z[:, dim], [25, 75])
            iqr = q3 - q1
            mask &= (U_z[:, dim] >= q1 - 1.5*iqr) & (U_z[:, dim] <= q3 + 1.5*iqr)
        
        print(f"Filtered {len(U_z) - mask.sum()} outliers. Remaining: {mask.sum()}")
        U_z = U_z[mask]
        V_z = V_z[mask]
        p_len = p_len[mask]
        p_ang = p_ang[mask]
        p_disp = p_disp[mask]

    # -------------------------------------------------------------------------
    # 5. Visualization
    # -------------------------------------------------------------------------
    
    print("\nProperty Statistics (Remaining Points):")
    print(f"  Length: min={p_len.min():.1f}, max={p_len.max():.1f}, mean={p_len.mean():.1f}")
    print(f"  Angle:  min={p_ang.min():.2f}, max={p_ang.max():.2f}")
    print(f"  Disp:   min={p_disp.min():.2f}, max={p_disp.max():.2f}")

    # Plot 1: Colored by Length (Original Figure 5 style)
    plot_alignment_with_properties(
        U_z, V_z, p_len, "Path Length", 
        os.path.join(args.out_dir, "fig5_by_length.png"),
        cmap='viridis'
    )
    
    # Plot 2: Colored by Angle (To check for ring topology)
    # Use hsv or twilight for cyclic colormap
    plot_alignment_with_properties(
        U_z, V_z, p_ang, "Displacement Angle", 
        os.path.join(args.out_dir, "fig5_by_angle.png"),
        cmap='twilight'
    )
    
    # Plot 3: Colored by Displacement Magnitude
    plot_alignment_with_properties(
        U_z, V_z, p_disp, "Displacement Mag", 
        os.path.join(args.out_dir, "fig5_by_disp.png"),
        cmap='plasma'
    )
    
    # Plot 3D
    plot_3d_scatter(U_z, p_len, os.path.join(args.out_dir, "fig5_3d_length.png"))
    
    # Plot Correlation Lollipop
    plt.figure(figsize=(8, 4))
    plt.plot(r[:15], 'o-')
    plt.axhline(0, color='gray', lw=0.5)
    plt.ylim(-0.1, 1.1)
    plt.title("Canonical Correlations")
    plt.xlabel("Mode")
    plt.ylabel("Correlation")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(args.out_dir, "cca_spectrum.png"), dpi=300)
    plt.close()
    
    # Save Data
    np.savez(
        os.path.join(args.out_dir, "cca_ring_results.npz"),
        U=U_z, V=V_z,
        props_length=p_len,
        props_angle=p_ang,
        props_disp=p_disp,
        correlations=r
    )
    print("Done.")

if __name__ == "__main__":
    main()

