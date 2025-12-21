#!/usr/bin/env python3
"""
Trajectory Statistics and Diversity Analysis

Analyzes collected routes.npz file and produces:
1. Diversity statistics (unique trajectories, action sequences)
2. Length distribution plots
3. XY trajectory visualizations
4. Seed coverage analysis

Usage:
    python analysis/trajectory_stats.py --routes_npz=experiments/xxx/data/routes.npz --out_dir=experiments/xxx/figures
"""

import os
import sys
import argparse
from collections import Counter
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Set style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


def load_routes(routes_npz: str) -> Dict[str, Any]:
    """Load routes data from npz file."""
    data = np.load(routes_npz, allow_pickle=True)
    return {key: data[key] for key in data.files}


def compute_trajectory_hash(actions: np.ndarray, xy: np.ndarray, use_xy: bool = True) -> str:
    """
    Compute a hash for trajectory uniqueness.
    Uses action sequence and optionally XY positions.
    """
    # Action sequence hash
    action_hash = tuple(actions.tolist()) if actions.size > 0 else ()
    
    if use_xy and xy.size > 0:
        # Discretize XY to grid cells (0.5 unit resolution)
        xy_discrete = (xy * 2).astype(int)
        xy_hash = tuple(map(tuple, xy_discrete.tolist()))
        return f"{action_hash}_{xy_hash}"
    else:
        return str(action_hash)


def analyze_diversity(
    routes_actions: np.ndarray,
    routes_xy: np.ndarray,
    routes_selected_ep: np.ndarray,
) -> Dict[str, Any]:
    """
    Analyze trajectory diversity.
    
    Returns dict with:
    - total_trajectories
    - unique_by_actions: unique action sequences
    - unique_by_xy: unique XY paths (discretized)
    - unique_combined: unique (actions + xy)
    - duplicate_ratio
    - diversity_ratio
    """
    kept_mask = routes_selected_ep >= 0
    n_kept = kept_mask.sum()
    
    if n_kept == 0:
        return {
            "total_trajectories": 0,
            "unique_by_actions": 0,
            "unique_by_xy": 0,
            "unique_combined": 0,
            "duplicate_ratio": 1.0,
            "diversity_ratio": 0.0,
        }
    
    # Extract kept trajectories
    actions_list = [routes_actions[i] for i in range(len(routes_actions)) if kept_mask[i]]
    xy_list = [routes_xy[i] for i in range(len(routes_xy)) if kept_mask[i]]
    
    # Compute hashes
    action_hashes = set()
    xy_hashes = set()
    combined_hashes = set()
    
    for i, (actions, xy) in enumerate(zip(actions_list, xy_list)):
        # Action hash
        if actions.size > 0:
            ah = tuple(actions.tolist())
            action_hashes.add(ah)
        
        # XY hash (discretized)
        if xy.size > 0:
            xy_discrete = (xy * 2).astype(int)
            xyh = tuple(map(tuple, xy_discrete.tolist()))
            xy_hashes.add(xyh)
        
        # Combined hash
        combined = compute_trajectory_hash(actions, xy, use_xy=True)
        combined_hashes.add(combined)
    
    n_unique_actions = len(action_hashes)
    n_unique_xy = len(xy_hashes)
    n_unique_combined = len(combined_hashes)
    
    return {
        "total_trajectories": int(n_kept),
        "unique_by_actions": n_unique_actions,
        "unique_by_xy": n_unique_xy,
        "unique_combined": n_unique_combined,
        "duplicate_ratio": 1.0 - (n_unique_combined / n_kept) if n_kept > 0 else 1.0,
        "diversity_ratio": n_unique_combined / n_kept if n_kept > 0 else 0.0,
    }


def analyze_action_distribution(routes_actions: np.ndarray, routes_selected_ep: np.ndarray) -> Dict[str, Any]:
    """Analyze action distribution across all trajectories."""
    kept_mask = routes_selected_ep >= 0
    
    all_actions = []
    for i in range(len(routes_actions)):
        if kept_mask[i] and routes_actions[i].size > 0:
            all_actions.extend(routes_actions[i].tolist())
    
    if len(all_actions) == 0:
        return {"action_counts": {}, "total_actions": 0}
    
    action_counts = Counter(all_actions)
    return {
        "action_counts": dict(action_counts),
        "total_actions": len(all_actions),
    }


def plot_length_distribution(
    routes_ep_len: np.ndarray,
    routes_selected_ep: np.ndarray,
    out_path: str,
):
    """Plot episode length distribution."""
    kept_mask = routes_selected_ep >= 0
    lengths = routes_ep_len[kept_mask]
    
    if len(lengths) == 0:
        print("[warn] No kept trajectories to plot length distribution")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram
    ax = axes[0]
    bins = np.arange(0, min(lengths.max() + 5, 105), 5)
    ax.hist(lengths, bins=bins, edgecolor='white', alpha=0.8, color='#2ecc71')
    ax.axvline(np.mean(lengths), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lengths):.1f}')
    ax.axvline(np.median(lengths), color='#3498db', linestyle='--', linewidth=2, label=f'Median: {np.median(lengths):.0f}')
    ax.set_xlabel('Episode Length')
    ax.set_ylabel('Count')
    ax.set_title('Episode Length Distribution')
    ax.legend()
    
    # CDF
    ax = axes[1]
    sorted_lens = np.sort(lengths)
    cdf = np.arange(1, len(sorted_lens) + 1) / len(sorted_lens)
    ax.plot(sorted_lens, cdf, linewidth=2, color='#9b59b6')
    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(np.median(lengths), color='#3498db', linestyle='--', alpha=0.7)
    ax.set_xlabel('Episode Length')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution of Episode Lengths')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[saved] {out_path}")


def plot_xy_trajectories(
    routes_xy: np.ndarray,
    routes_selected_ep: np.ndarray,
    routes_seed: np.ndarray,
    out_path: str,
    max_trajectories: int = 500,
    sample_random: bool = True,
):
    """Plot XY trajectories (sample if too many)."""
    kept_mask = routes_selected_ep >= 0
    kept_indices = np.where(kept_mask)[0]
    
    if len(kept_indices) == 0:
        print("[warn] No kept trajectories to plot")
        return
    
    # Sample if needed
    if len(kept_indices) > max_trajectories:
        if sample_random:
            np.random.seed(42)
            kept_indices = np.random.choice(kept_indices, max_trajectories, replace=False)
        else:
            kept_indices = kept_indices[:max_trajectories]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: All trajectories overlaid
    ax = axes[0]
    
    # Create color gradient based on trajectory index
    cmap = plt.cm.viridis
    colors = [cmap(i / len(kept_indices)) for i in range(len(kept_indices))]
    
    for i, idx in enumerate(kept_indices):
        xy = routes_xy[idx]
        if xy.size == 0:
            continue
        
        # Skip trajectories with NaN
        if np.any(np.isnan(xy)):
            continue
        
        ax.plot(xy[:, 0], xy[:, 1], alpha=0.3, linewidth=0.5, color=colors[i])
        # Mark start with small circle
        ax.scatter(xy[0, 0], xy[0, 1], s=5, c='green', alpha=0.5, zorder=5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'XY Trajectories (n={len(kept_indices)})')
    ax.set_aspect('equal')
    
    # Add start/end legend
    start_patch = mpatches.Patch(color='green', label='Start positions')
    ax.legend(handles=[start_patch], loc='upper right')
    
    # Plot 2: Heatmap of positions visited
    ax = axes[1]
    
    all_xy = []
    for idx in kept_indices:
        xy = routes_xy[idx]
        if xy.size > 0 and not np.any(np.isnan(xy)):
            all_xy.append(xy)
    
    if len(all_xy) > 0:
        all_xy = np.vstack(all_xy)
        
        # Create 2D histogram
        x_range = (np.nanmin(all_xy[:, 0]) - 1, np.nanmax(all_xy[:, 0]) + 1)
        y_range = (np.nanmin(all_xy[:, 1]) - 1, np.nanmax(all_xy[:, 1]) + 1)
        
        h, xedges, yedges = np.histogram2d(
            all_xy[:, 0], all_xy[:, 1],
            bins=50,
            range=[x_range, y_range]
        )
        
        im = ax.imshow(
            h.T, origin='lower', aspect='auto',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap='hot'
        )
        plt.colorbar(im, ax=ax, label='Visit count')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Position Heatmap')
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[saved] {out_path}")


def plot_seed_coverage(
    routes_seed: np.ndarray,
    routes_selected_ep: np.ndarray,
    out_path: str,
):
    """Plot seed coverage and distribution."""
    kept_mask = routes_selected_ep >= 0
    all_seeds = routes_seed
    kept_seeds = routes_seed[kept_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Seed frequency histogram
    ax = axes[0]
    seed_counts = Counter(all_seeds)
    count_values = list(seed_counts.values())
    
    ax.hist(count_values, bins=max(10, min(50, max(count_values) - min(count_values) + 1)),
            edgecolor='white', alpha=0.8, color='#3498db')
    ax.set_xlabel('Occurrences per Seed')
    ax.set_ylabel('Number of Seeds')
    ax.set_title('Seed Occurrence Distribution')
    
    # If all counts are 1, add annotation
    if all(c == 1 for c in count_values):
        ax.annotate('All seeds unique!', xy=(0.5, 0.5), xycoords='axes fraction',
                   ha='center', va='center', fontsize=14, color='#27ae60',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Seed range coverage
    ax = axes[1]
    unique_seeds = np.unique(all_seeds)
    min_seed, max_seed = unique_seeds.min(), unique_seeds.max()
    
    # Create coverage visualization
    seed_range = max_seed - min_seed + 1
    if seed_range <= 1000:
        # Can visualize each seed
        coverage = np.zeros(seed_range)
        for s in unique_seeds:
            coverage[s - min_seed] = 1
        
        ax.fill_between(range(seed_range), coverage, alpha=0.5, color='#2ecc71')
        ax.set_xlabel(f'Seed (offset from {min_seed})')
        ax.set_ylabel('Covered')
        ax.set_title('Seed Range Coverage')
        ax.set_ylim(-0.1, 1.1)
    else:
        # Too many seeds, show histogram
        ax.hist(unique_seeds, bins=100, edgecolor='white', alpha=0.8, color='#2ecc71')
        ax.set_xlabel('Seed Value')
        ax.set_ylabel('Count')
        ax.set_title('Seed Distribution')
    
    # Add statistics text
    stats_text = f"Total: {len(all_seeds)}\nUnique: {len(unique_seeds)}\nDiversity: {100*len(unique_seeds)/len(all_seeds):.1f}%"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[saved] {out_path}")


def plot_action_distribution(
    action_stats: Dict[str, Any],
    out_path: str,
):
    """Plot action distribution."""
    action_counts = action_stats["action_counts"]
    if not action_counts:
        print("[warn] No actions to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by action index
    actions = sorted(action_counts.keys())
    counts = [action_counts[a] for a in actions]
    
    # Action names for procgen maze
    action_names = {
        0: 'LEFT_DOWN', 1: 'LEFT', 2: 'LEFT_UP', 3: 'DOWN',
        4: 'NOOP', 5: 'UP', 6: 'RIGHT_DOWN', 7: 'RIGHT',
        8: 'RIGHT_UP', 9: 'UNUSED1', 10: 'UNUSED2', 11: 'UNUSED3',
        12: 'UNUSED4', 13: 'UNUSED5', 14: 'UNUSED6',
    }
    
    labels = [action_names.get(a, f'A{a}') for a in actions]
    
    bars = ax.bar(range(len(actions)), counts, color='#9b59b6', alpha=0.8, edgecolor='white')
    ax.set_xticks(range(len(actions)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel('Action')
    ax.set_ylabel('Count')
    ax.set_title('Action Distribution')
    
    # Add percentage labels
    total = sum(counts)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        pct = 100 * count / total
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[saved] {out_path}")


def plot_xy_jump_check(
    routes_xy: np.ndarray,
    routes_selected_ep: np.ndarray,
    out_path: str,
    max_samples: int = 20,
):
    """
    Plot XY trajectories to check for teleportation bugs.
    Shows start, end, and trajectory to verify no jumps to reset position.
    """
    kept_mask = routes_selected_ep >= 0
    kept_indices = np.where(kept_mask)[0]
    
    if len(kept_indices) == 0:
        print("[warn] No trajectories to check")
        return
    
    # Sample trajectories
    np.random.seed(42)
    sample_indices = np.random.choice(kept_indices, min(max_samples, len(kept_indices)), replace=False)
    
    # Create grid of plots
    n_cols = 5
    n_rows = (len(sample_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    for i, idx in enumerate(sample_indices):
        ax = axes[i]
        xy = routes_xy[idx]
        
        if xy.size == 0 or np.any(np.isnan(xy)):
            ax.text(0.5, 0.5, 'Invalid XY', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Seed {idx}')
            continue
        
        # Plot trajectory
        ax.plot(xy[:, 0], xy[:, 1], 'b-', alpha=0.5, linewidth=1)
        
        # Mark start (green), end (red), and intermediate points
        ax.scatter(xy[0, 0], xy[0, 1], c='green', s=50, marker='o', zorder=10, label='Start')
        ax.scatter(xy[-1, 0], xy[-1, 1], c='red', s=50, marker='X', zorder=10, label='End')
        
        # Check for large jumps (potential bug indicator)
        if len(xy) > 1:
            diffs = np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1))
            max_jump = np.max(diffs)
            if max_jump > 2.0:  # More than 2 units is suspicious
                ax.set_title(f'Idx {idx} (JUMP: {max_jump:.1f})', color='red')
            else:
                ax.set_title(f'Idx {idx}')
        else:
            ax.set_title(f'Idx {idx}')
        
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
    
    # Hide unused axes
    for i in range(len(sample_indices), len(axes)):
        axes[i].set_visible(False)
    
    # Add legend to first plot
    axes[0].legend(fontsize=6, loc='upper right')
    
    fig.suptitle('XY Trajectory Check (Red title = potential teleport bug)', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[saved] {out_path}")


def check_xy_jumps(routes_xy: np.ndarray, routes_selected_ep: np.ndarray) -> Dict[str, Any]:
    """Check for suspicious XY jumps that might indicate teleportation bugs."""
    kept_mask = routes_selected_ep >= 0
    
    jump_stats = {
        "trajectories_checked": 0,
        "trajectories_with_large_jumps": 0,
        "max_jump_overall": 0.0,
        "mean_max_jump": 0.0,
        "jump_threshold": 2.0,  # More than 2 units is suspicious for maze
    }
    
    max_jumps = []
    
    for i in range(len(routes_xy)):
        if not kept_mask[i]:
            continue
        
        xy = routes_xy[i]
        if xy.size == 0 or xy.shape[0] < 2:
            continue
        
        if np.any(np.isnan(xy)):
            continue
        
        jump_stats["trajectories_checked"] += 1
        
        # Compute step-to-step distances
        diffs = np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1))
        max_jump = np.max(diffs)
        max_jumps.append(max_jump)
        
        if max_jump > jump_stats["jump_threshold"]:
            jump_stats["trajectories_with_large_jumps"] += 1
    
    if max_jumps:
        jump_stats["max_jump_overall"] = float(np.max(max_jumps))
        jump_stats["mean_max_jump"] = float(np.mean(max_jumps))
    
    return jump_stats


def main():
    parser = argparse.ArgumentParser(description="Trajectory Statistics and Diversity Analysis")
    parser.add_argument("--routes_npz", type=str, required=True, help="Path to routes.npz file")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for figures (default: same as routes_npz)")
    parser.add_argument("--max_plot_trajectories", type=int, default=500, help="Max trajectories to plot")
    args = parser.parse_args()
    
    if args.out_dir is None:
        args.out_dir = os.path.dirname(args.routes_npz)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("="*60)
    print("TRAJECTORY STATISTICS AND DIVERSITY ANALYSIS")
    print("="*60)
    print(f"Input: {args.routes_npz}")
    print(f"Output dir: {args.out_dir}")
    print()
    
    # Load data
    print("[loading] Loading routes data...")
    data = load_routes(args.routes_npz)
    
    routes_seed = data["routes_seed"]
    routes_selected_ep = data["routes_selected_ep"]
    routes_actions = data["routes_actions"]
    routes_xy = data["routes_xy"]
    routes_ep_len = data["routes_ep_len"]
    routes_success = data["routes_success"]
    
    kept_mask = routes_selected_ep >= 0
    n_total = len(routes_seed)
    n_kept = kept_mask.sum()
    
    print(f"[data] Total tasks: {n_total}")
    print(f"[data] Kept trajectories: {n_kept} ({100*n_kept/n_total:.1f}%)")
    print()
    
    # ============ DIVERSITY ANALYSIS ============
    print("-"*40)
    print("DIVERSITY ANALYSIS")
    print("-"*40)
    
    diversity = analyze_diversity(routes_actions, routes_xy, routes_selected_ep)
    
    print(f"[Trajectory Diversity]")
    print(f"  Total trajectories:     {diversity['total_trajectories']}")
    print(f"  Unique (by actions):    {diversity['unique_by_actions']}")
    print(f"  Unique (by XY path):    {diversity['unique_by_xy']}")
    print(f"  Unique (combined):      {diversity['unique_combined']}")
    print(f"  Duplicate ratio:        {100*diversity['duplicate_ratio']:.1f}%")
    print(f"  Diversity ratio:        {100*diversity['diversity_ratio']:.1f}%")
    print()
    
    # Seed diversity
    unique_seeds = len(np.unique(routes_seed))
    print(f"[Seed Diversity]")
    print(f"  Total seeds:            {len(routes_seed)}")
    print(f"  Unique seeds:           {unique_seeds}")
    print(f"  Seed diversity:         {100*unique_seeds/len(routes_seed):.1f}%")
    print()
    
    # ============ XY JUMP CHECK ============
    print("-"*40)
    print("XY JUMP CHECK (Teleportation Bug Detection)")
    print("-"*40)
    
    jump_stats = check_xy_jumps(routes_xy, routes_selected_ep)
    print(f"[XY Jumps]")
    print(f"  Trajectories checked:   {jump_stats['trajectories_checked']}")
    print(f"  With large jumps (>{jump_stats['jump_threshold']}): {jump_stats['trajectories_with_large_jumps']}")
    print(f"  Max jump overall:       {jump_stats['max_jump_overall']:.2f}")
    print(f"  Mean max jump:          {jump_stats['mean_max_jump']:.2f}")
    
    if jump_stats['trajectories_with_large_jumps'] > 0:
        pct_bad = 100 * jump_stats['trajectories_with_large_jumps'] / max(1, jump_stats['trajectories_checked'])
        print(f"  [WARN] {pct_bad:.1f}% of trajectories have suspicious jumps!")
    else:
        print(f"  [OK] No teleportation issues detected")
    print()
    
    # ============ LENGTH STATISTICS ============
    print("-"*40)
    print("LENGTH STATISTICS")
    print("-"*40)
    
    kept_lens = routes_ep_len[kept_mask]
    if len(kept_lens) > 0:
        print(f"[Episode Lengths]")
        print(f"  Min:    {kept_lens.min()}")
        print(f"  Max:    {kept_lens.max()}")
        print(f"  Mean:   {kept_lens.mean():.1f}")
        print(f"  Median: {np.median(kept_lens):.0f}")
        print(f"  Std:    {kept_lens.std():.1f}")
    print()
    
    # ============ SUCCESS STATISTICS ============
    print("-"*40)
    print("SUCCESS STATISTICS")
    print("-"*40)
    
    kept_success = routes_success[kept_mask]
    if len(kept_success) > 0:
        success_rate = kept_success.sum() / len(kept_success)
        print(f"[Success]")
        print(f"  Success rate: {100*success_rate:.1f}%")
    print()
    
    # ============ ACTION STATISTICS ============
    print("-"*40)
    print("ACTION STATISTICS")
    print("-"*40)
    
    action_stats = analyze_action_distribution(routes_actions, routes_selected_ep)
    print(f"[Actions]")
    print(f"  Total actions: {action_stats['total_actions']}")
    if action_stats['action_counts']:
        top_actions = sorted(action_stats['action_counts'].items(), key=lambda x: -x[1])[:5]
        print(f"  Top 5 actions:")
        for action, count in top_actions:
            pct = 100 * count / action_stats['total_actions']
            print(f"    Action {action}: {count} ({pct:.1f}%)")
    print()
    
    # ============ GENERATE PLOTS ============
    print("-"*40)
    print("GENERATING PLOTS")
    print("-"*40)
    
    # Length distribution
    plot_length_distribution(
        routes_ep_len, routes_selected_ep,
        os.path.join(args.out_dir, "length_distribution.png")
    )
    
    # XY trajectories
    plot_xy_trajectories(
        routes_xy, routes_selected_ep, routes_seed,
        os.path.join(args.out_dir, "xy_trajectories.png"),
        max_trajectories=args.max_plot_trajectories
    )
    
    # Seed coverage
    plot_seed_coverage(
        routes_seed, routes_selected_ep,
        os.path.join(args.out_dir, "seed_coverage.png")
    )
    
    # Action distribution
    plot_action_distribution(
        action_stats,
        os.path.join(args.out_dir, "action_distribution.png")
    )
    
    # XY jump check
    plot_xy_jump_check(
        routes_xy, routes_selected_ep,
        os.path.join(args.out_dir, "xy_jump_check.png"),
        max_samples=20
    )
    
    print()
    print("="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Output directory: {args.out_dir}")
    print()
    
    # Summary verdict
    print("SUMMARY:")
    if diversity['diversity_ratio'] < 0.5:
        print(f"  [FAIL] Low diversity ({100*diversity['diversity_ratio']:.1f}%) - check seed assignment!")
    else:
        print(f"  [OK] Good diversity ({100*diversity['diversity_ratio']:.1f}%)")
    
    if jump_stats['trajectories_with_large_jumps'] > jump_stats['trajectories_checked'] * 0.01:
        print(f"  [FAIL] Many teleportation artifacts detected - check XY extraction!")
    else:
        print(f"  [OK] No teleportation issues")
    
    if unique_seeds < n_total * 0.9:
        print(f"  [WARN] Seed diversity is low ({100*unique_seeds/n_total:.1f}%) - duplicate seeds detected")
    else:
        print(f"  [OK] High seed diversity ({100*unique_seeds/n_total:.1f}%)")
    
    print()


if __name__ == "__main__":
    main()

