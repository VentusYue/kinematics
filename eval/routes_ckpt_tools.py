#!/usr/bin/env python3
"""
Command-line tools for checkpoint management.

Usage:
    python eval/routes_ckpt_tools.py info --ckpt_dir <path>
    python eval/routes_ckpt_tools.py build --ckpt_dir <path> --out_npz <path> [--max_routes N] [--mode MODE] [--no_compress]
"""

import argparse
import os
import sys
from typing import Optional
import numpy as np

# Add project root to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from routes_ckpt_storage import CheckpointManifest, build_routes_npz_from_ckpt


def cmd_info(ckpt_dir: str):
    """Print checkpoint information."""
    manifest = CheckpointManifest.load(ckpt_dir)
    if manifest is None:
        print(f"[ERROR] No checkpoint found at {ckpt_dir}")
        print("  (manifest.json not found)")
        return 1
    
    print("="*60)
    print("CHECKPOINT INFO")
    print("="*60)
    print(f"Checkpoint directory: {ckpt_dir}")
    print(f"Storage format:       {manifest.storage_format}")
    print(f"Created:               {manifest.created_at:.0f}")
    print(f"Last updated:          {manifest.updated_at:.0f}")
    print()
    print("Progress:")
    print(f"  Collected:           {manifest.num_tasks_collected} / {manifest.num_tasks_target}")
    print(f"  Completion:          {100*manifest.num_tasks_collected/max(1,manifest.num_tasks_target):.1f}%")
    print(f"  Current seed:        {manifest.current_seed}")
    print(f"  Seeds attempted:     {manifest.seeds_attempted}")
    print(f"  Shards written:      {manifest.shards_written}")
    print()
    print("Configuration:")
    print(f"  Env:                 {manifest.env_name}")
    print(f"  Model:               {manifest.model_ckpt}")
    print(f"  Distribution:        {manifest.distribution_mode}")
    print(f"  Max steps:           {manifest.max_steps}")
    print(f"  Max ep len:          {manifest.max_ep_len}")
    print(f"  Require success:     {bool(manifest.require_success)}")
    print(f"  Adapt episodes:      {manifest.adapt_episodes}")
    print(f"  Record episodes:     {manifest.record_episodes}")
    print()
    
    # Count shard files
    shards_dir = os.path.join(ckpt_dir, "shards")
    if os.path.exists(shards_dir):
        shard_files = [
            f for f in os.listdir(shards_dir)
            if f.startswith("shard_") and f.endswith(".npz")
        ]
        print(f"Shard files:           {len(shard_files)}")
        
        # Try to load and show some stats from first shard
        if len(shard_files) > 0:
            try:
                first_shard = os.path.join(shards_dir, sorted(shard_files)[0])
                data = np.load(first_shard, allow_pickle=True)
                if "routes_ep_len" in data.files:
                    lens = data["routes_ep_len"]
                    print(f"  Sample ep lengths:   min={lens.min()}, max={lens.max()}, mean={lens.mean():.1f}")
            except Exception as e:
                print(f"  (Could not load shard stats: {e})")
    else:
        print("Shard files:           0 (shards/ directory not found)")
    
    print("="*60)
    return 0


def cmd_build(ckpt_dir: str, out_npz: str, max_routes: Optional[int] = None):
    """Build routes.npz from checkpoint shards."""
    print("="*60)
    print("BUILDING ROUTES.NPZ FROM CHECKPOINT")
    print("="*60)
    print(f"Checkpoint directory: {ckpt_dir}")
    print(f"Output file:          {out_npz}")
    if max_routes is not None:
        print(f"Max routes:            {max_routes}")
    print()
    
    try:
        n_routes = build_routes_npz_from_ckpt(ckpt_dir, out_npz, max_routes=max_routes)
        print()
        print("="*60)
        print("SUCCESS")
        print("="*60)
        print(f"Exported {n_routes} trajectories to {out_npz}")
        print()
        print("You can now use this file with:")
        print(f"  python analysis/pkd_cycle_sampler.py --routes_npz={out_npz} ...")
        print("="*60)
        return 0
    except Exception as e:
        print(f"[ERROR] Failed to build routes.npz: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint management tools for route collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show checkpoint info
  python eval/routes_ckpt_tools.py info --ckpt_dir experiments/my_exp/data/ckpt

  # Export all routes
  python eval/routes_ckpt_tools.py build --ckpt_dir experiments/my_exp/data/ckpt --out_npz routes_partial.npz

  # Export first 100 routes only
  python eval/routes_ckpt_tools.py build --ckpt_dir experiments/my_exp/data/ckpt --out_npz routes_100.npz --max_routes 100

  # Export a small file for CCA only (no obs/actions)
  python eval/routes_ckpt_tools.py build --ckpt_dir experiments/my_exp/data/ckpt --out_npz routes_cca.npz --mode cca

  # Export a PKD-only file (obs/actions only)
  python eval/routes_ckpt_tools.py build --ckpt_dir experiments/my_exp/data/ckpt --out_npz routes_pkd.npz --mode pkd
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show checkpoint information")
    info_parser.add_argument("--ckpt_dir", type=str, required=True,
                            help="Checkpoint directory path")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build routes.npz from checkpoint shards")
    build_parser.add_argument("--ckpt_dir", type=str, required=True,
                             help="Checkpoint directory path")
    build_parser.add_argument("--out_npz", type=str, required=True,
                             help="Output routes.npz file path")
    build_parser.add_argument("--max_routes", type=int, default=None,
                             help="Maximum number of routes to export (default: all)")
    build_parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "pkd", "cca", "pkd_cca", "analysis"],
        help=(
            "Which keys to export. "
            "'full' exports everything in the shards. "
            "'pkd' exports only what PKD needs (routes_obs/routes_actions + minimal metadata). "
            "'cca' exports only what CCA needs (routes_xy/routes_ep_len + minimal metadata). "
            "'pkd_cca' exports the minimal union needed by both pkd_cycle_sampler.py and cca_alignment.py. "
            "'analysis' is an alias for 'pkd_cca'."
        ),
    )
    build_parser.add_argument(
        "--no_compress",
        action="store_true",
        help="Write an uncompressed .npz (faster to build/load, but larger on disk).",
    )
    build_parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable tqdm progress bar while merging shards.",
    )
    build_parser.add_argument(
        "--no_atomic",
        action="store_true",
        help="Disable atomic write (no temp file). Use if disk space is very tight.",
    )
    build_parser.add_argument(
        "--tmp_dir",
        type=str,
        default=None,
        help="Directory for temp file during atomic write. Default: /tmp (to avoid filling output dir).",
    )
    
    args = parser.parse_args()
    
    if args.command == "info":
        return cmd_info(args.ckpt_dir)
    elif args.command == "build":
        # Decide which keys to export.
        mode = getattr(args, "mode", "full")
        keys = None
        if mode == "pkd":
            keys = [
                "routes_seed",
                "routes_selected_ep",
                "routes_obs",
                "routes_actions",
                "routes_ep_len",
                "routes_success",
            ]
        elif mode == "cca":
            keys = [
                "routes_seed",
                "routes_xy",
                "routes_ep_len",
                "routes_success",
            ]
        elif mode in ("analysis", "pkd_cca"):
            # Minimal union for: PKD (obs/actions) + CCA (xy/ep_len)
            keys = [
                "routes_seed",
                "routes_selected_ep",
                "routes_obs",
                "routes_actions",
                "routes_xy",
                "routes_ep_len",
                "routes_success",
            ]
        else:
            keys = None  # full

        compress = not bool(getattr(args, "no_compress", False))
        show_progress = not bool(getattr(args, "no_progress", False))
        atomic = not bool(getattr(args, "no_atomic", False))
        tmp_dir = getattr(args, "tmp_dir", None)
        try:
            print("="*60)
            print("BUILDING ROUTES.NPZ FROM CHECKPOINT")
            print("="*60)
            print(f"Checkpoint directory: {args.ckpt_dir}")
            print(f"Output file:          {args.out_npz}")
            print(f"Mode:                 {mode}")
            print(f"Compression:          {'off' if not compress else 'on'}")
            print(f"Atomic write:         {'off' if not atomic else 'on'}")
            if tmp_dir:
                print(f"Temp directory:       {tmp_dir}")
            if args.max_routes is not None:
                print(f"Max routes:           {args.max_routes}")
            print()

            n_routes = build_routes_npz_from_ckpt(
                args.ckpt_dir,
                args.out_npz,
                max_routes=args.max_routes,
                keys=keys,
                compress=compress,
                show_progress=show_progress,
                atomic=atomic,
                tmp_dir=tmp_dir,
            )
            print()
            print("="*60)
            print("SUCCESS")
            print("="*60)
            print(f"Exported {n_routes} trajectories to {args.out_npz}")
            print("="*60)
            return 0
        except Exception as e:
            print(f"[ERROR] Failed to build routes.npz: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

