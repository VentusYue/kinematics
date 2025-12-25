#!/usr/bin/env python3
"""
Checkpoint storage for route collection (ckpt_shards_v1 format).

Provides:
- CheckpointManifest: load/save manifest.json
- RoutesShardWriter: buffer routes and flush shards atomically
- build_routes_npz_from_ckpt: merge shards into standard routes.npz
"""

import os
import json
import time
import signal
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class CheckpointManifest:
    """Manifest metadata for checkpoint directory."""
    storage_format: str = "ckpt_shards_v1"
    created_at: float = 0.0
    updated_at: float = 0.0
    
    # Collection state
    num_tasks_target: int = 0
    num_tasks_collected: int = 0
    current_seed: int = 0
    seeds_attempted: int = 0
    shards_written: int = 0
    
    # Config (for resume validation)
    env_name: str = ""
    model_ckpt: str = ""
    distribution_mode: str = "easy"
    max_steps: int = 512
    max_ep_len: int = 0
    require_success: int = 1
    adapt_episodes: int = 5
    record_episodes: int = 2
    
    # Output path (for reference)
    out_npz: str = ""
    
    @classmethod
    def load(cls, ckpt_dir: str) -> Optional["CheckpointManifest"]:
        """Load manifest from checkpoint directory."""
        manifest_path = os.path.join(ckpt_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            return None
        
        try:
            with open(manifest_path, "r") as f:
                data = json.load(f)
            return cls(**data)
        except Exception as e:
            print(f"[WARNING] Failed to load manifest: {e}")
            return None
    
    def save(self, ckpt_dir: str):
        """Save manifest to checkpoint directory."""
        os.makedirs(ckpt_dir, exist_ok=True)
        manifest_path = os.path.join(ckpt_dir, "manifest.json")
        self.updated_at = time.time()
        
        # Write to temp file first, then atomic replace
        tmp_path = manifest_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        os.replace(tmp_path, manifest_path)
    
    def validate_resume(self, args) -> Tuple[bool, Optional[str]]:
        """
        Validate that current args match manifest config.
        Returns (is_valid, error_message).
        """
        checks = [
            ("env_name", self.env_name, args.env_name),
            ("model_ckpt", self.model_ckpt, args.model_ckpt),
            ("distribution_mode", self.distribution_mode, args.distribution_mode),
            ("max_steps", self.max_steps, args.max_steps),
            ("max_ep_len", self.max_ep_len, args.max_ep_len),
            ("require_success", self.require_success, args.require_success),
            ("adapt_episodes", self.adapt_episodes, args.adapt_episodes),
            ("record_episodes", self.record_episodes, args.record_episodes),
        ]
        
        mismatches = []
        for name, stored, current in checks:
            if stored != current:
                mismatches.append(f"{name}: stored={stored}, current={current}")
        
        if mismatches:
            msg = "Config mismatch:\n  " + "\n  ".join(mismatches)
            return False, msg
        
        return True, None


class RoutesShardWriter:
    """Buffer routes and flush to shard files atomically."""
    
    def __init__(
        self,
        ckpt_dir: str,
        shard_size: int = 25,
        flush_secs: float = 60.0,
        manifest: Optional[CheckpointManifest] = None,
    ):
        self.ckpt_dir = ckpt_dir
        self.shard_size = shard_size
        self.flush_secs = flush_secs
        self.last_flush_time = time.time()
        
        # Create directories
        os.makedirs(ckpt_dir, exist_ok=True)
        self.shards_dir = os.path.join(ckpt_dir, "shards")
        os.makedirs(self.shards_dir, exist_ok=True)
        
        # Buffer for current shard
        self.buffer: List[Dict[str, Any]] = []
        
        # Load or create manifest
        if manifest is None:
            manifest = CheckpointManifest.load(ckpt_dir)
            if manifest is None:
                manifest = CheckpointManifest()
                manifest.created_at = time.time()
        
        self.manifest = manifest
        
        # Register signal handlers for graceful shutdown
        self._original_handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._original_handlers[sig] = signal.signal(sig, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Flush buffer on interrupt."""
        print(f"\n[Signal {signum}] Flushing buffered routes before exit...")
        self.flush(force=True)
        # Restore original handler and re-raise
        signal.signal(signum, self._original_handlers[signum])
        signal.raise_signal(signum)
    
    def append_route(
        self,
        seed: int,
        selected_ep: int,
        obs: np.ndarray,
        actions: np.ndarray,
        xy: np.ndarray,
        player_v: np.ndarray,
        ents_count: np.ndarray,
        nearest_ents: np.ndarray,
        rewards: np.ndarray,
        ep_len: int,
        ep_return: float,
        success: bool,
        diag: Dict[str, Any],
        all_episodes: Optional[List[Dict[str, Any]]] = None,
    ):
        """Append a route to buffer and flush if needed."""
        route_data = {
            "seed": int(seed),
            "selected_ep": int(selected_ep),
            "obs": obs,
            "actions": actions,
            "xy": xy,
            "player_v": player_v,
            "ents_count": ents_count,
            "nearest_ents": nearest_ents,
            "rewards": rewards,
            "ep_len": int(ep_len),
            "ep_return": float(ep_return),
            "success": bool(success),
            "diag": diag,
        }
        if all_episodes is not None:
            route_data["all_episodes"] = all_episodes
        
        self.buffer.append(route_data)
        self.manifest.num_tasks_collected += 1
        
        # Check flush conditions
        should_flush = False
        if len(self.buffer) >= self.shard_size:
            should_flush = True
        elif time.time() - self.last_flush_time >= self.flush_secs:
            should_flush = True
        
        if should_flush:
            self.flush()
    
    def flush(self, force: bool = False):
        """Flush buffer to a shard file."""
        if len(self.buffer) == 0:
            return
        
        # Ensure shards directory exists
        os.makedirs(self.shards_dir, exist_ok=True)
        
        # Determine shard index
        shard_idx = self.manifest.shards_written
        
        # Build shard data (same format as final routes.npz)
        shard_data = {
            "routes_seed": np.asarray([r["seed"] for r in self.buffer], dtype=np.int64),
            "routes_selected_ep": np.asarray([r["selected_ep"] for r in self.buffer], dtype=np.int64),
            "routes_obs": np.asarray([r["obs"] for r in self.buffer], dtype=object),
            "routes_actions": np.asarray([r["actions"] for r in self.buffer], dtype=object),
            "routes_xy": np.asarray([r["xy"] for r in self.buffer], dtype=object),
            "routes_player_v": np.asarray([r["player_v"] for r in self.buffer], dtype=object),
            "routes_ents_count": np.asarray([r["ents_count"] for r in self.buffer], dtype=object),
            "routes_nearest_ents": np.asarray([r["nearest_ents"] for r in self.buffer], dtype=object),
            "routes_rewards": np.asarray([r["rewards"] for r in self.buffer], dtype=object),
            "routes_ep_len": np.asarray([r["ep_len"] for r in self.buffer], dtype=np.int64),
            "routes_return": np.asarray([r["ep_return"] for r in self.buffer], dtype=np.float32),
            "routes_success": np.asarray([r["success"] for r in self.buffer], dtype=np.bool_),
            "routes_diag": np.asarray([r["diag"] for r in self.buffer], dtype=object),
        }
        
        # Add all_episodes if present
        if "all_episodes" in self.buffer[0]:
            shard_data["episodes_all"] = np.asarray(
                [r.get("all_episodes", []) for r in self.buffer], dtype=object
            )
        
        # Write shard atomically using tempfile for safety
        shard_filename = f"shard_{shard_idx:06d}.npz"
        shard_path = os.path.join(self.shards_dir, shard_filename)
        
        # Use tempfile to ensure atomic write
        fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=self.shards_dir)
        try:
            os.close(fd)  # Close the fd, np.savez will open it again
            np.savez_compressed(tmp_path, **shard_data)
            os.replace(tmp_path, shard_path)
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
        
        # Update manifest
        self.manifest.shards_written += 1
        self.manifest.save(self.ckpt_dir)
        
        print(f"[ckpt] Flushed shard {shard_idx} ({len(self.buffer)} routes)")
        
        # Clear buffer
        self.buffer.clear()
        self.last_flush_time = time.time()
    
    def update_progress(self, current_seed: int, seeds_attempted: int):
        """Update progress counters in manifest."""
        self.manifest.current_seed = current_seed
        self.manifest.seeds_attempted = seeds_attempted
        self.manifest.save(self.ckpt_dir)
    
    def close(self):
        """Flush remaining buffer and restore signal handlers."""
        self.flush(force=True)
        # Restore original signal handlers
        for sig, handler in self._original_handlers.items():
            signal.signal(sig, handler)


def build_routes_npz_from_ckpt(
    ckpt_dir: str,
    out_npz: str,
    max_routes: Optional[int] = None,
    keys: Optional[List[str]] = None,
    compress: bool = True,
    show_progress: bool = True,
    atomic: bool = True,
    tmp_dir: Optional[str] = None,
) -> int:
    """
    Merge all shards from checkpoint directory into a single routes.npz.
    
    Args:
        ckpt_dir: Path to checkpoint directory containing manifest.json and shards/
        out_npz: Output file path
        max_routes: Maximum number of routes to export (None = all)
        keys: List of keys to export (None = all keys in shards)
        compress: Whether to use np.savez_compressed (smaller but slower)
        show_progress: Show tqdm progress bar
        atomic: Use atomic write (temp file + rename) to avoid corruption
        tmp_dir: Directory for temp file during atomic write (None = use /tmp or system temp)
    
    Returns:
        Number of trajectories in the merged file.
    """
    manifest = CheckpointManifest.load(ckpt_dir)
    if manifest is None:
        raise ValueError(f"No manifest found in {ckpt_dir}")
    
    shards_dir = os.path.join(ckpt_dir, "shards")
    if not os.path.exists(shards_dir):
        raise ValueError(f"Shards directory not found: {shards_dir}")
    
    # Collect all shard files
    shard_files = sorted([
        os.path.join(shards_dir, f)
        for f in os.listdir(shards_dir)
        if f.startswith("shard_") and f.endswith(".npz")
    ])
    
    if len(shard_files) == 0:
        raise ValueError(f"No shard files found in {shards_dir}")
    
    print(f"[merge] Found {len(shard_files)} shard files")
    if keys is not None:
        print(f"[merge] Key filter enabled: {keys}")
    print(f"[merge] Compression: {'on' if compress else 'off'}")
    
    # Load and concatenate all shards
    all_routes: Dict[str, List] = {}
    
    total_routes = 0
    it = shard_files
    if show_progress and tqdm is not None:
        it = tqdm(shard_files, desc="[merge] Reading shards", unit="shard")

    for shard_path in it:
        shard_data = np.load(shard_path, allow_pickle=True)
        
        n = len(shard_data["routes_seed"])
        if max_routes is not None and total_routes + n > max_routes:
            n = max_routes - total_routes
            if n <= 0:
                break
        
        # Initialize lists on first shard
        if len(all_routes) == 0:
            if keys is None:
                init_keys = [k for k in shard_data.files if k != "meta"]
            else:
                # Keep only keys present in shards (silently ignore missing requested keys)
                init_keys = [k for k in keys if k in shard_data.files]
            for key in init_keys:
                all_routes[key] = []
        
        # Append shard data
        for key in all_routes.keys():
            if key in shard_data.files:
                arr = shard_data[key]
                if n < len(arr):
                    arr = arr[:n]
                all_routes[key].extend(arr)
        
        total_routes += n
        if max_routes is not None and total_routes >= max_routes:
            break
    
    if total_routes == 0:
        raise ValueError("No routes found in shards")
    
    print(f"[merge] Loaded {total_routes} routes, converting to arrays...")
    
    # Convert lists to numpy arrays
    save_dict = {}
    for key, values in all_routes.items():
        if key.startswith("routes_"):
            if key in ("routes_seed", "routes_selected_ep", "routes_ep_len"):
                save_dict[key] = np.asarray(values, dtype=np.int64)
            elif key == "routes_return":
                save_dict[key] = np.asarray(values, dtype=np.float32)
            elif key == "routes_success":
                save_dict[key] = np.asarray(values, dtype=np.bool_)
            else:
                save_dict[key] = np.asarray(values, dtype=object)
        else:
            save_dict[key] = np.asarray(values, dtype=object)
    
    # Add meta from manifest
    save_dict["meta"] = {
        "env_name": manifest.env_name,
        "distribution_mode": manifest.distribution_mode,
        "num_tasks_target": manifest.num_tasks_target,
        "num_tasks_collected": total_routes,
        "seeds_attempted": manifest.seeds_attempted,
        "seed_offset": 0,  # Will be inferred from routes_seed
        "max_steps": manifest.max_steps,
        "max_ep_len": manifest.max_ep_len,
        "require_success": bool(manifest.require_success),
        "adapt_episodes": manifest.adapt_episodes,
        "record_episodes": manifest.record_episodes,
    }
    
    # Save merged file
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    
    # Write atomically to avoid partially-written/corrupted .npz if interrupted.
    # (This is especially important because .npz is a zip container with a central directory.)
    if atomic:
        # Use a temp directory with more space if possible (avoid filling up the output dir)
        # Priority: explicit tmp_dir > /tmp > system temp > output dir (last resort)
        actual_tmp_dir = None
        if tmp_dir is not None and os.path.isdir(tmp_dir):
            actual_tmp_dir = tmp_dir
        elif os.path.isdir("/tmp"):
            actual_tmp_dir = "/tmp"
        else:
            actual_tmp_dir = tempfile.gettempdir()
        
        print(f"[merge] Writing to temp file in: {actual_tmp_dir}")
        fd, tmp_path = tempfile.mkstemp(suffix=".npz", dir=actual_tmp_dir)
        os.close(fd)
        try:
            if compress:
                np.savez_compressed(tmp_path, **save_dict)
            else:
                np.savez(tmp_path, **save_dict)
            # Move/copy to final destination
            # os.replace works across filesystems in Python 3.3+, but may fall back to copy+delete
            os.replace(tmp_path, out_npz)
            print(f"[merge] Saved {total_routes} trajectories to {out_npz}")
        except OSError as e:
            # os.replace fails across filesystems on some systems; try shutil.move
            import shutil
            try:
                shutil.move(tmp_path, out_npz)
                print(f"[merge] Saved {total_routes} trajectories to {out_npz}")
            except Exception:
                # Clean up temp file on error
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise e
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
    else:
        if compress:
            np.savez_compressed(out_npz, **save_dict)
        else:
            np.savez(out_npz, **save_dict)
        print(f"[merge] Saved {total_routes} trajectories to {out_npz}")
    
    return total_routes

