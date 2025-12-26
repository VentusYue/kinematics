"""
render_alignment_3d.py

Utility: render Plotly 3D alignment html from an existing `cca_results.npz`
without re-running the full CCA pipeline.

Typical use (Chaser example):
  python -W ignore analysis/render_alignment_3d.py \
    --cca_results_npz /root/backup/kinematics/experiments/run_20k_chaser_ckpt_optimized_analysis/figures/cca_results.npz \
    --out_dir /root/backup/kinematics/experiments/run_20k_chaser_ckpt_optimized_analysis/figures \
    --color_by Displacement
"""

import argparse
import os
import sys
import numpy as np

# Allow running this file from anywhere (match cca_alignment.py pattern)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.cca_alignment import plot_3d_interactive


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cca_results_npz", required=True, help="Path to cca_results.npz produced by cca_alignment.py")
    parser.add_argument("--out_dir", required=True, help="Directory to write html files into")
    parser.add_argument("--color_by", type=str, default="Displacement", choices=["Length", "Displacement", "Angle"])
    parser.add_argument("--outfile", type=str, default=None, help="Override output html filename (default depends on color_by)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    d = np.load(args.cca_results_npz, allow_pickle=True)
    if "U_means" not in d.files or "V_means" not in d.files:
        raise ValueError(f"Missing U_means/V_means in {args.cca_results_npz}. Keys: {d.files}")

    U = np.asarray(d["U_means"], dtype=np.float64)
    V = np.asarray(d["V_means"], dtype=np.float64)

    if U.ndim != 2 or V.ndim != 2 or U.shape != V.shape:
        raise ValueError(f"Unexpected U/V shapes: U={U.shape}, V={V.shape}")
    if U.shape[1] < 3:
        raise ValueError(f"Need at least 3 CCA modes for 3D plot; got {U.shape[1]}")

    # Metadata keys are what cca_alignment.py expects.
    meta = {}
    if "cycle_lengths" in d.files:
        meta["Length"] = np.asarray(d["cycle_lengths"], dtype=np.float64)
    if "cycle_disps" in d.files:
        meta["Displacement"] = np.asarray(d["cycle_disps"], dtype=np.float64)
    if "cycle_angles" in d.files:
        meta["Angle"] = np.asarray(d["cycle_angles"], dtype=np.float64)

    if not meta:
        raise ValueError(f"No metadata arrays found in {args.cca_results_npz} (cycle_lengths/cycle_disps/cycle_angles)")

    if args.color_by == "Displacement":
        colorscale = "Plasma"
        default_name = "alignment_3d_by_displacement.html"
    elif args.color_by == "Angle":
        colorscale = "Twilight"
        default_name = "alignment_3d_by_angle.html"
    else:
        colorscale = "Viridis"
        default_name = "alignment_3d.html"

    out_name = args.outfile if args.outfile is not None else default_name
    out_path = os.path.join(args.out_dir, out_name)

    plot_3d_interactive(
        U[:, :3],
        V[:, :3],
        meta,
        out_path,
        title=f"3D Alignment (colored by {args.color_by})",
        color_by=args.color_by,
        colorscale=colorscale,
    )


if __name__ == "__main__":
    main()


