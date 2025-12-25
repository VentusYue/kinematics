#!/usr/bin/env python3
"""
Meta-RL Route Collection (Optimized with Filtering)

Key features:
1. Only collects SUCCESSFUL trajectories (filters out failures)
2. Dynamic seed pool - keeps trying new seeds until target count reached
3. Batch-based with proper seed diversity
4. Live progress tracking with success rate, speed, ETA
5. Fixed XY teleportation bug (uses terminal_xy)
6. SPEED OPTIMIZATIONS:
   - Early batch completion (don't wait for slowest 10%)
   - Early failure detection (abort stuck tasks)
   - Per-task timeout
"""

import os
import sys

# Suppress warnings BEFORE importing gym/numpy (must be at top)
import warnings
import contextlib
import io
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*gym.*")
warnings.filterwarnings("ignore", message=".*gymnasium.*")
warnings.filterwarnings("ignore", message=".*minigrid.*")
warnings.filterwarnings("ignore", message=".*np.bool8.*")

os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")  # best-effort (Gym may ignore)
os.environ["GYM_LOGGER_LEVEL"] = "error"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"  # Suppress warnings in subprocesses too

import argparse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

# Performance: Set thread counts for optimal CPU utilization
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool  # for legacy codebases

import torch
# Performance optimizations for PyTorch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Suppress gym warnings + gym's noisy startup banner (printed to stderr in some versions)
_gym_import_stderr = io.StringIO()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        with contextlib.redirect_stderr(_gym_import_stderr):
            import gym
    except Exception:
        # If gym import genuinely fails, replay captured stderr so the error is visible.
        sys.stderr.write(_gym_import_stderr.getvalue())
        raise
    gym.logger.set_level(40)

from tqdm import tqdm
import time

# Add project root to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import checkpoint storage (optional)
try:
    from routes_ckpt_storage import (
        CheckpointManifest,
        RoutesShardWriter,
        build_routes_npz_from_ckpt,
    )
    CKPT_AVAILABLE = True
except ImportError:
    CKPT_AVAILABLE = False

from baselines.common.vec_env import SubprocVecEnv, VecExtractDictObs, VecMonitor
from level_replay.envs import VecNormalize, VecPyTorchProcgen
from level_replay.model import Policy, SimplePolicy

# =============================================================================
# Procgen state helpers (XY + extra state features)
#
# Why:
# - Procgen CoinRun 的 `info` 基本不提供 player 的 (x,y)/(vx,vy) 等可直接用作
#   behavior embedding 的状态量；但 procgen 的 gym3 环境支持 `callmethod("get_state")`
#   返回可序列化的 state bytes。
# - 这些 bytes 的格式在多个 procgen 游戏间是“相当通用”的：包含一个 entity 列表 `ents`，
#   其中 `ents[0]` 在 Maze/CoinRun 中都对应 agent/player（我们在实践中验证 CoinRun 可解析）。
#
# What we extract (per step):
# - **xy**: (x,y) from ents[0]  -> used as behavior trajectory for ridge embedding / CCA.
# - **player_v**: (vx,vy) from ents[0] (if present) -> 目前未被 CCA 使用，但可用于后续行为特征。
# - **nearest_ents**: K nearest entities around player as [dx, dy, type, image_type]
#   -> 目前未被 CCA 使用，但可用于后续行为特征（例如“离危险/奖励有多近”）。
#
# Where used in pipeline:
# - PKD sampler (`analysis/pkd_cycle_sampler.py`): uses ONLY routes_obs + routes_actions
#   to sample hidden-state limit cycles. It does NOT use routes_xy / extra state.
# - CCA alignment (`analysis/cca_alignment.py`): uses routes_xy (behavior) + cycles_hidden (neural).
# =============================================================================
PROCGEN_TOOLS_PATH = "/root/test/procgen-tools-main"
if PROCGEN_TOOLS_PATH not in sys.path:
    sys.path.append(PROCGEN_TOOLS_PATH)

try:
    import procgen
    if not hasattr(procgen, "ProcgenGym3Env"):
        class ProcgenGym3Env:
            pass
        procgen.ProcgenGym3Env = ProcgenGym3Env
except ImportError:
    pass

try:
    # NOTE: Despite the module name, `procgen_tools.maze.EnvState` parses the
    # common procgen `get_state()` byte format for multiple games (incl. coinrun).
    from procgen_tools import maze as procgen_state_tools
except Exception as e:
    print(f"Warning: Could not import procgen_tools.maze (state parser): {e}")
    procgen_state_tools = None


def _unwrap_to_gym3_like(env):
    """Unwrap until we find an object exposing `callmethod` (gym3 env wrapper)."""
    inner = env
    for _ in range(64):
        if hasattr(inner, "callmethod"):
            return inner
        if hasattr(inner, "env"):
            inner = inner.env
        else:
            break
    return inner


def extract_procgen_state_from_gym_env(env) -> Optional[Dict[str, Any]]:
    """
    Return parsed procgen `state_vals` dict from a gym-wrapped procgen env.
    Works for Maze and CoinRun (and typically other procgen envs) as long as
    `get_state()` is available and the procgen-tools parser is importable.
    """
    if procgen_state_tools is None:
        return None
    try:
        inner = _unwrap_to_gym3_like(env)
        if not hasattr(inner, "callmethod"):
            return None
        state_bytes_list = inner.callmethod("get_state")
        if not state_bytes_list:
            return None
        state = procgen_state_tools.EnvState(state_bytes_list[0])
        return state.state_vals
    except Exception:
        return None


def extract_xy_from_gym_env(env) -> Tuple[float, float]:
    """
    Extract (x, y) from a gym-wrapped procgen env (maze/coinrun/etc).

    Semantics:
    - Maze: ents[0] corresponds to the mouse/agent. xy is in procgen world coords.
    - CoinRun: ents[0] corresponds to the player. xy is in procgen world coords.

    This `xy` is what we save as routes_xy and later feed into ridge embedding.
    """
    vals = extract_procgen_state_from_gym_env(env)
    if vals is None:
        return (float("nan"), float("nan"))
    try:
        ents0 = vals["ents"][0]
        return (float(ents0["x"].val), float(ents0["y"].val))
    except Exception:
        return (float("nan"), float("nan"))


def extract_xy_from_state_vals(vals: Optional[Dict[str, Any]]) -> Tuple[float, float]:
    """Extract (x,y) directly from already-parsed `state_vals`."""
    if vals is None:
        return (float("nan"), float("nan"))
    try:
        ents0 = vals["ents"][0]
        return (float(ents0["x"].val), float(ents0["y"].val))
    except Exception:
        return (float("nan"), float("nan"))


def _extract_step_state_features(vals: Optional[Dict[str, Any]], k_nearest: int = 8) -> Dict[str, Any]:
    """
    Extract a small, fixed-shape set of state features from `state_vals`.

    We intentionally keep this generic (no hard-coded CoinRun entity-type mapping)
    so it remains usable across procgen games.

    Output keys (per-step):
    - player_v: float32 (2,) = [vx, vy] for ents[0] (nan if missing)
    - ents_count: int
    - nearest_ents: float32 (K,4) rows = [dx, dy, type, image_type]
    """
    out: Dict[str, Any] = {}

    if vals is None:
        out["player_v"] = np.asarray([np.nan, np.nan], dtype=np.float32)
        out["ents_count"] = int(0)
        out["nearest_ents"] = np.full((k_nearest, 4), np.nan, dtype=np.float32)
        return out

    try:
        ents = vals.get("ents", [])
        out["ents_count"] = int(len(ents))
        if len(ents) == 0:
            out["player_v"] = np.asarray([np.nan, np.nan], dtype=np.float32)
            out["nearest_ents"] = np.full((k_nearest, 4), np.nan, dtype=np.float32)
            return out

        e0 = ents[0]
        px = float(e0["x"].val)
        py = float(e0["y"].val)
        pvx = float(e0["vx"].val) if "vx" in e0 else float("nan")
        pvy = float(e0["vy"].val) if "vy" in e0 else float("nan")
        out["player_v"] = np.asarray([pvx, pvy], dtype=np.float32)

        # Nearest other entities by Euclidean distance.
        # Each row: [dx, dy, type, image_type]
        rows = []
        for j in range(1, len(ents)):
            ej = ents[j]
            try:
                ex = float(ej["x"].val)
                ey = float(ej["y"].val)
                dx = ex - px
                dy = ey - py
                typ = float(ej["type"].val) if "type" in ej else float("nan")
                img = float(ej["image_type"].val) if "image_type" in ej else float("nan")
                d2 = dx * dx + dy * dy
                rows.append((d2, dx, dy, typ, img))
            except Exception:
                continue
        rows.sort(key=lambda r: r[0])
        nearest = np.full((k_nearest, 4), np.nan, dtype=np.float32)
        for idx, r in enumerate(rows[:k_nearest]):
            _, dx, dy, typ, img = r
            nearest[idx, 0] = dx
            nearest[idx, 1] = dy
            nearest[idx, 2] = typ
            nearest[idx, 3] = img
        out["nearest_ents"] = nearest
        return out
    except Exception:
        out["player_v"] = np.asarray([np.nan, np.nan], dtype=np.float32)
        out["ents_count"] = int(0)
        out["nearest_ents"] = np.full((k_nearest, 4), np.nan, dtype=np.float32)
        return out


class XYInfoWrapper(gym.Wrapper):
    """
    Wrapper that extracts (x, y) coordinates on each step.
    Handles VecEnv auto-reset by storing terminal_xy separately.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._last_xy = (float("nan"), float("nan"))
        self._last_state_feats: Dict[str, Any] = {}
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        vals = extract_procgen_state_from_gym_env(self.env)
        self._last_xy = extract_xy_from_state_vals(vals)
        self._last_state_feats = _extract_step_state_features(vals)
        return obs
    
    def step(self, action):
        xy_before = self._last_xy
        feats_before = self._last_state_feats
        obs, reward, done, info = self.env.step(action)
        
        if info is None:
            info = {}
        
        if done:
            info["terminal_xy"] = xy_before
            info["terminal_player_v"] = feats_before.get("player_v", np.asarray([np.nan, np.nan], dtype=np.float32))
            info["terminal_ents_count"] = int(feats_before.get("ents_count", 0))
            info["terminal_nearest_ents"] = feats_before.get("nearest_ents", np.full((8, 4), np.nan, dtype=np.float32))
            vals_after_reset = extract_procgen_state_from_gym_env(self.env)
            xy_after_reset = extract_xy_from_state_vals(vals_after_reset)
            info["xy"] = xy_after_reset
            self._last_xy = xy_after_reset

            # after auto-reset, refresh state features to match new episode start
            feats_after_reset = _extract_step_state_features(vals_after_reset)
            info["player_v"] = feats_after_reset["player_v"]
            info["ents_count"] = feats_after_reset["ents_count"]
            info["nearest_ents"] = feats_after_reset["nearest_ents"]
            self._last_state_feats = feats_after_reset
        else:
            vals = extract_procgen_state_from_gym_env(self.env)
            feats = _extract_step_state_features(vals)
            xy = extract_xy_from_state_vals(vals)
            self._last_xy = xy
            info["xy"] = xy
            info["terminal_xy"] = None

            info["player_v"] = feats["player_v"]
            info["ents_count"] = feats["ents_count"]
            info["nearest_ents"] = feats["nearest_ents"]
            info["terminal_player_v"] = None
            info["terminal_ents_count"] = None
            info["terminal_nearest_ents"] = None
            self._last_state_feats = feats
        
        return obs, reward, done, info
    
    def get_xy(self):
        return self._last_xy


def _suppress_all_warnings():
    """Aggressively suppress all warnings - call at subprocess start."""
    import warnings
    import os
    import sys
    
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    warnings.warn = lambda *args, **kwargs: None
    
    if not sys.warnoptions:
        sys.warnoptions.append("ignore")


def make_env_fn(env_name: str, seed: int, distribution_mode: str):
    """Create a procgen env factory with XYInfoWrapper."""
    def _thunk():
        _suppress_all_warnings()
        
        import gym
        gym.logger.set_level(40)
        
        env = gym.make(
            f"procgen:procgen-{env_name}-v0",
            start_level=int(seed),
            num_levels=1,
            distribution_mode=distribution_mode,
        )
        env = XYInfoWrapper(env)
        return env
    return _thunk


@dataclass
class EpisodeBuffer:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    xy: List[np.ndarray] = field(default_factory=list)
    # Extra state features (primarily useful for CoinRun; optional for Maze)
    # NOTE: These are collected per-step and saved into the routes .npz, but are
    # NOT required by PKD sampler or the current CCA pipeline.
    player_v: List[np.ndarray] = field(default_factory=list)
    ents_count: List[int] = field(default_factory=list)
    nearest_ents: List[np.ndarray] = field(default_factory=list)  # (K,4)
    info_level_complete: bool = False
    ret: float = 0.0
    length: int = 0
    xy_missing_count: int = 0

    def finalize(self) -> Dict[str, Any]:
        xy_missing_frac = self.xy_missing_count / max(1, self.length)
        
        if self.xy:
            xy_arr = np.stack(self.xy, axis=0)
        else:
            xy_arr = np.empty((0, 2), dtype=np.float32)
        
        if self.obs:
            obs_arr = np.stack(self.obs, axis=0)
        else:
            obs_arr = np.empty((0, 3, 64, 64), dtype=np.float16)

        if self.player_v:
            player_v_arr = np.stack(self.player_v, axis=0).astype(np.float32)
        else:
            player_v_arr = np.empty((0, 2), dtype=np.float32)

        if self.ents_count:
            ents_count_arr = np.asarray(self.ents_count, dtype=np.int32)
        else:
            ents_count_arr = np.empty((0,), dtype=np.int32)

        if self.nearest_ents:
            nearest_ents_arr = np.stack(self.nearest_ents, axis=0).astype(np.float32)
        else:
            nearest_ents_arr = np.empty((0, 8, 4), dtype=np.float32)
        
        return {
            "obs": obs_arr,
            "actions": np.array(self.actions, dtype=np.int64),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "xy": xy_arr,
            # --- Optional extra state ---
            # Saved for future behavior features/debugging; not used by current PKD/CCA.
            "player_v": player_v_arr,
            "ents_count": ents_count_arr,
            "nearest_ents": nearest_ents_arr,
            "success": bool(self.info_level_complete or (self.ret > 0)),
            "level_complete": bool(self.info_level_complete),
            "return": float(self.ret),
            "len": int(self.length),
            "xy_missing_frac": float(xy_missing_frac),
        }


def select_best_episode(
    episodes: List[Dict[str, Any]],
    min_len: int = 5,
    max_len: int = 0,
    require_success: bool = True,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """Select the best episode from a task's episode list."""
    diag: Dict[str, Any] = {
        "num_episodes": len(episodes),
        "min_len": min_len,
        "max_len": max_len,
        "require_success": require_success,
        "selected_reason": None,
    }
    
    if len(episodes) == 0:
        diag["selected_reason"] = "no_episodes"
        return None, diag
    
    def _len_ok(ep_len: int) -> bool:
        if ep_len < min_len:
            return False
        if max_len > 0 and ep_len > max_len:
            return False
        return True
    
    valid_indices = []
    for idx in range(len(episodes)):
        ep = episodes[idx]
        ep_len = ep.get("len", 0)
        is_success = ep.get("success", False)
        
        if require_success and not is_success:
            continue
        if not _len_ok(ep_len):
            continue
        
        valid_indices.append(idx)
    
    if len(valid_indices) == 0:
        if require_success:
            diag["selected_reason"] = "no_successful_episodes"
        else:
            diag["selected_reason"] = "no_episodes_in_length_range"
        return None, diag
    
    best_idx = valid_indices[-1]
    diag["selected_reason"] = f"valid_ep_{best_idx}"
    return best_idx, diag


def build_model(model_ckpt: str, arch: str, hidden_size: int, device: torch.device, env_name: str, use_compile: bool = True):
    dummy_env = gym.make(f"procgen:procgen-{env_name}-v0", start_level=0, num_levels=1)
    obs_shape = dummy_env.observation_space.shape
    obs_shape = (3, obs_shape[0], obs_shape[1])
    num_actions = dummy_env.action_space.n
    dummy_env.close()

    if arch == "simple":
        actor_critic = SimplePolicy(obs_shape, num_actions)
    else:
        actor_critic = Policy(
            obs_shape,
            num_actions,
            arch=arch,
            base_kwargs={"recurrent": True, "hidden_size": hidden_size},
        )
    actor_critic.to(device)
    checkpoint = torch.load(model_ckpt, map_location=device, weights_only=False)
    if "state_dict" in checkpoint:
        actor_critic.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        actor_critic.load_state_dict(checkpoint["model_state_dict"])
    else:
        actor_critic.load_state_dict(checkpoint)
    actor_critic.eval()
    
    if use_compile and hasattr(torch, "compile") and device.type == "cuda":
        try:
            actor_critic = torch.compile(actor_critic, mode="reduce-overhead")
            print("[perf] Model compiled with torch.compile")
        except Exception as e:
            print(f"[perf] torch.compile failed: {e}")
    
    return actor_critic, obs_shape, num_actions


class FastInference:
    """Optimized inference wrapper with AMP support."""
    
    def __init__(self, actor_critic, batch_size: int, hidden_size: int, device: torch.device, use_amp: bool = True):
        self.actor_critic = actor_critic
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.batch_size = batch_size
        self.hidden_size = hidden_size
    
    @torch.inference_mode()
    def act_batched(
        self,
        obs: torch.Tensor,
        rnn_hxs: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute actions for all envs."""
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                _, actions, _, rnn_hxs_new = self.actor_critic.act(
                    obs, rnn_hxs, masks, deterministic=deterministic
                )
            rnn_hxs_new = rnn_hxs_new.float()
        else:
            _, actions, _, rnn_hxs_new = self.actor_critic.act(
                obs, rnn_hxs, masks, deterministic=deterministic
            )
        return actions, rnn_hxs_new


def create_vec_env(env_name: str, seeds: List[int], distribution_mode: str, device: torch.device):
    """Create vectorized environment with unique seeds."""
    env_fns = [make_env_fn(env_name, int(s), distribution_mode) for s in seeds]
    venv = SubprocVecEnv(env_fns)
    
    if isinstance(venv.observation_space, gym.spaces.Dict):
        venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv, ob=False, ret=False)
    envs = VecPyTorchProcgen(venv, device, meta_rl=False)
    return envs


def run_batch_collection(
    envs,
    actor_critic,
    batch_seeds: List[int],
    args,
    device: torch.device,
    obs_shape: Tuple[int, ...],
    fast_inf: FastInference,
) -> Tuple[List[Tuple[int, List[Dict], bool]], Dict]:
    """
    Run collection for a batch with SPEED OPTIMIZATIONS:
    1. Early batch completion - don't wait for slowest 10%
    2. Early failure detection - abort tasks with no success after adapt phase
    3. Per-task timeout - total steps limit per task
    """
    batch_size = len(batch_seeds)
    total_episodes = args.adapt_episodes + args.record_episodes
    
    # Speed params
    completion_threshold = getattr(args, 'batch_completion_threshold', 0.9)  # Complete when 90% done
    task_timeout_steps = getattr(args, 'task_timeout', args.max_steps * total_episodes)  # Total steps per task
    early_abort_no_success = getattr(args, 'early_abort', True)  # Abort if no success after adapt
    
    stats = {
        "time_inference": 0.0,
        "time_env_step": 0.0,
        "total_steps": 0,
        "successful_tasks": 0,
        "failed_tasks": 0,
        "aborted_early": 0,
        "timed_out": 0,
    }
    
    # Initialize RNN hidden state
    rnn_hxs = torch.zeros(batch_size, args.hidden_size, device=device)
    masks = torch.zeros(batch_size, 1, device=device)
    masks_ones = torch.ones(batch_size, 1, device=device)
    
    obs = envs.reset()
    
    # Per-env state
    ep_counts = [0] * batch_size
    ep_bufs = [EpisodeBuffer() for _ in range(batch_size)]
    ep_lists: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]
    env_done = [False] * batch_size
    env_has_success = [False] * batch_size
    task_total_steps = [0] * batch_size  # Total steps taken for this task
    
    # Record initial observation
    obs_np = obs.cpu().numpy().astype(np.float16)
    for i in range(batch_size):
        ep_bufs[i].obs.append(obs_np[i])
    
    step_count = 0
    max_batch_steps = args.max_steps * total_episodes * batch_size  # Reduced safety cap
    
    while True:
        step_count += 1
        
        # Check completion conditions
        done_count = sum(env_done)
        active_count = batch_size - done_count
        
        # OPTIMIZATION 1: Early batch completion
        # Don't wait for the slowest environments
        if done_count >= batch_size * completion_threshold:
            # Mark remaining as aborted
            for i in range(batch_size):
                if not env_done[i]:
                    env_done[i] = True
                    stats["aborted_early"] += 1
            break
        
        # Safety cap
        if step_count > max_batch_steps or active_count == 0:
            break
        
        # Determine deterministic flag
        in_record = all(ep_counts[i] >= args.adapt_episodes for i in range(batch_size) if not env_done[i])
        deterministic = bool(args.record_deterministic) if in_record else bool(args.adapt_deterministic)
        
        # Compute actions
        t0 = time.perf_counter()
        actions, rnn_hxs = fast_inf.act_batched(obs, rnn_hxs, masks, deterministic=deterministic)
        stats["time_inference"] += time.perf_counter() - t0
        
        # Step environment
        t0 = time.perf_counter()
        obs, reward, done, infos = envs.step(actions)
        stats["time_env_step"] += time.perf_counter() - t0
        
        masks = masks_ones.clone()
        
        # Convert tensors
        reward_np = reward.cpu().numpy().ravel() if isinstance(reward, torch.Tensor) else np.asarray(reward).ravel()
        done_np = done.cpu().numpy().astype(bool).ravel() if isinstance(done, torch.Tensor) else np.asarray(done, dtype=bool).ravel()
        action_np = actions.cpu().numpy().ravel()
        obs_np = obs.cpu().numpy().astype(np.float16)
        
        # Update buffers
        for i in range(batch_size):
            if env_done[i]:
                continue
            
            task_total_steps[i] += 1
            ep_bufs[i].actions.append(int(action_np[i]))
            ep_bufs[i].rewards.append(float(reward_np[i]))
            ep_bufs[i].ret += float(reward_np[i])
            ep_bufs[i].length += 1
            stats["total_steps"] += 1
            
            # Extract XY
            info_i = infos[i] if i < len(infos) else {}
            
            if isinstance(info_i, dict) and info_i.get("level_complete", False):
                ep_bufs[i].info_level_complete = True
            
            if done_np[i]:
                if isinstance(info_i, dict):
                    terminal_xy = info_i.get("terminal_xy", None)
                    xy = terminal_xy if terminal_xy is not None else (float("nan"), float("nan"))
                else:
                    xy = (float("nan"), float("nan"))
            else:
                xy = info_i.get("xy", (float("nan"), float("nan"))) if isinstance(info_i, dict) else (float("nan"), float("nan"))
            
            if np.isnan(xy[0]) or np.isnan(xy[1]):
                ep_bufs[i].xy_missing_count += 1
            
            ep_bufs[i].xy.append(np.asarray([xy[0], xy[1]], dtype=np.float32))

            # Extra procgen state features (generic across games; used esp. for CoinRun)
            if isinstance(info_i, dict):
                if done_np[i]:
                    pv = info_i.get("terminal_player_v", None)
                    ec = info_i.get("terminal_ents_count", None)
                    ne = info_i.get("terminal_nearest_ents", None)
                else:
                    pv = info_i.get("player_v", None)
                    ec = info_i.get("ents_count", None)
                    ne = info_i.get("nearest_ents", None)
            else:
                pv, ec, ne = None, None, None

            if pv is None:
                pv_arr = np.asarray([np.nan, np.nan], dtype=np.float32)
            else:
                pv_arr = np.asarray(pv, dtype=np.float32).reshape(2,)
            if ec is None:
                ec_i = 0
            else:
                try:
                    ec_i = int(ec)
                except Exception:
                    ec_i = 0
            if ne is None:
                ne_arr = np.full((8, 4), np.nan, dtype=np.float32)
            else:
                ne_arr = np.asarray(ne, dtype=np.float32).reshape(8, 4)

            ep_bufs[i].player_v.append(pv_arr)
            ep_bufs[i].ents_count.append(ec_i)
            ep_bufs[i].nearest_ents.append(ne_arr)
            
            # OPTIMIZATION 2: Per-task timeout
            if task_total_steps[i] >= task_timeout_steps:
                env_done[i] = True
                stats["timed_out"] += 1
                # Finalize current episode if any
                if ep_bufs[i].length > 0:
                    ep_data = ep_bufs[i].finalize()
                    ep_lists[i].append(ep_data)
                    if ep_data["success"]:
                        env_has_success[i] = True
                continue
            
            # Handle episode done
            if done_np[i]:
                ep_data = ep_bufs[i].finalize()
                ep_lists[i].append(ep_data)
                ep_counts[i] += 1
                
                if ep_data["success"]:
                    env_has_success[i] = True
                
                # OPTIMIZATION 3: Early failure detection
                # If no success after adaptation phase, abort this task
                if early_abort_no_success and ep_counts[i] >= args.adapt_episodes and not env_has_success[i]:
                    env_done[i] = True
                    stats["aborted_early"] += 1
                    continue
                
                if ep_counts[i] >= total_episodes:
                    env_done[i] = True
                else:
                    ep_bufs[i] = EpisodeBuffer()
                    ep_bufs[i].obs.append(obs_np[i])
            
            elif ep_bufs[i].length >= args.max_steps:
                ep_data = ep_bufs[i].finalize()
                ep_lists[i].append(ep_data)
                ep_counts[i] += 1
                
                if ep_data["success"]:
                    env_has_success[i] = True
                
                # Early failure detection
                if early_abort_no_success and ep_counts[i] >= args.adapt_episodes and not env_has_success[i]:
                    env_done[i] = True
                    stats["aborted_early"] += 1
                    continue
                
                if ep_counts[i] >= total_episodes:
                    env_done[i] = True
                else:
                    ep_bufs[i] = EpisodeBuffer()
                    ep_bufs[i].obs.append(obs_np[i])
            else:
                ep_bufs[i].obs.append(obs_np[i])
    
    # Count success/failure
    for i in range(batch_size):
        if env_has_success[i]:
            stats["successful_tasks"] += 1
        else:
            stats["failed_tasks"] += 1
    
    results = [(batch_seeds[i], ep_lists[i], env_has_success[i]) for i in range(batch_size)]
    return results, stats


class LiveStats:
    """Track and display live statistics."""
    
    def __init__(self, target_count: int):
        self.target = target_count
        self.collected = 0
        self.attempted = 0
        self.total_steps = 0
        self.start_time = time.perf_counter()
        self.batch_times = []
        self.aborted = 0
        self.timed_out = 0
    
    def update(self, successful: int, attempted: int, steps: int, batch_time: float, 
               aborted: int = 0, timed_out: int = 0):
        self.collected += successful
        self.attempted += attempted
        self.total_steps += steps
        self.batch_times.append(batch_time)
        self.aborted += aborted
        self.timed_out += timed_out
    
    @property
    def success_rate(self) -> float:
        return self.collected / max(1, self.attempted)
    
    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time
    
    @property
    def speed(self) -> float:
        """Trajectories per second."""
        return self.collected / max(0.01, self.elapsed)
    
    @property
    def steps_per_sec(self) -> float:
        return self.total_steps / max(0.01, self.elapsed)
    
    @property
    def eta_seconds(self) -> float:
        """Estimated time remaining."""
        if self.collected == 0:
            return float('inf')
        remaining = self.target - self.collected
        return remaining / self.speed
    
    def format_time(self, seconds: float) -> str:
        if seconds == float('inf'):
            return "??:??"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
    
    def get_desc(self) -> str:
        return (
            f"✓{self.collected}/{self.target} "
            f"({100*self.success_rate:.0f}%ok) "
            f"{self.speed:.1f}/s "
            f"{self.steps_per_sec/1000:.1f}k stp/s "
            f"ETA {self.format_time(self.eta_seconds)}"
        )


def main():
    p = argparse.ArgumentParser("meta_collect_routes (Optimized with Filtering)")
    # Core
    p.add_argument("--model_ckpt", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--env_name", type=str, default="maze")
    p.add_argument("--distribution_mode", type=str, default="easy")

    # Task sampling
    p.add_argument("--num_tasks", type=int, default=200, help="Target number of SUCCESSFUL trajectories")
    p.add_argument("--seed_offset", type=int, default=0)
    p.add_argument("--num_processes", type=int, default=64)
    p.add_argument("--max_seed_attempts", type=int, default=0, help="Max seeds to try (0 = 10x num_tasks)")

    # Policy / device
    p.add_argument("--arch", type=str, default="large")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")

    # Meta-RL schedule
    p.add_argument("--adapt_episodes", type=int, default=5)
    p.add_argument("--record_episodes", type=int, default=2)
    p.add_argument("--adapt_deterministic", type=int, default=0)
    p.add_argument("--record_deterministic", type=int, default=1)

    # Legacy compat
    p.add_argument("--trials_per_task", type=int, default=None)
    p.add_argument("--deterministic", type=int, default=None)

    # Episode caps and filtering
    p.add_argument("--max_steps", type=int, default=512, help="Max steps per episode")
    p.add_argument("--min_len", type=int, default=5, help="Min episode length")
    p.add_argument("--max_ep_len", type=int, default=0, help="Max episode length (0=no limit)")
    p.add_argument("--require_success", type=int, default=1, help="1 = only keep successful trajectories")

    # SPEED OPTIMIZATION params
    p.add_argument("--batch_completion_threshold", type=float, default=0.9, 
                   help="Complete batch when this fraction done (0.9 = don't wait for slowest 10%)")
    p.add_argument("--task_timeout", type=int, default=0, 
                   help="Max total steps per task (0 = max_steps * total_episodes)")
    p.add_argument("--early_abort", type=int, default=1,
                   help="1 = abort tasks with no success after adapt phase")

    # XY handling
    p.add_argument("--xy_fail_policy", type=str, default="warn_only", choices=["drop_task", "warn_only"])
    p.add_argument("--xy_fail_threshold", type=float, default=0.5)
    p.add_argument("--save_all_episodes", type=int, default=0)

    # Performance
    p.add_argument("--use_compile", type=int, default=1)
    p.add_argument("--use_amp", type=int, default=1)

    # Checkpointing (optional)
    p.add_argument("--ckpt_dir", type=str, default=None,
                   help="Enable checkpointing: directory to store shards (default: disabled)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from checkpoint if manifest exists")
    p.add_argument("--resume_force", action="store_true",
                   help="Resume even if config differs (use with caution)")
    p.add_argument("--ckpt_shard_size", type=int, default=25,
                   help="Flush shard when buffer reaches this many routes")
    p.add_argument("--ckpt_flush_secs", type=float, default=60.0,
                   help="Flush shard after this many seconds even if buffer not full")

    args = p.parse_args()

    # Legacy mapping
    if args.trials_per_task is not None:
        args.adapt_episodes = max(int(args.trials_per_task) - 1, 0)
        args.record_episodes = 1

    if args.deterministic is not None:
        args.adapt_deterministic = int(args.deterministic)
        args.record_deterministic = int(args.deterministic)

    # Set defaults
    if args.max_seed_attempts <= 0:
        args.max_seed_attempts = args.num_tasks * 10
    
    if args.task_timeout <= 0:
        args.task_timeout = args.max_steps * (args.adapt_episodes + args.record_episodes)

    # Checkpoint setup
    use_ckpt = args.ckpt_dir is not None
    if use_ckpt and not CKPT_AVAILABLE:
        raise RuntimeError("Checkpointing requested but routes_ckpt_storage not available")
    
    ckpt_writer = None
    manifest = None
    if use_ckpt:
        # Load or create manifest
        manifest = CheckpointManifest.load(args.ckpt_dir)
        if manifest is None:
            # New checkpoint
            manifest = CheckpointManifest()
            manifest.created_at = time.time()
            manifest.num_tasks_target = args.num_tasks
            manifest.env_name = args.env_name
            manifest.model_ckpt = args.model_ckpt
            manifest.distribution_mode = args.distribution_mode
            manifest.max_steps = args.max_steps
            manifest.max_ep_len = args.max_ep_len
            manifest.require_success = args.require_success
            manifest.adapt_episodes = args.adapt_episodes
            manifest.record_episodes = args.record_episodes
            manifest.out_npz = args.out_npz
            manifest.current_seed = args.seed_offset
            manifest.save(args.ckpt_dir)
        elif args.resume:
            # Resume: validate config
            is_valid, error_msg = manifest.validate_resume(args)
            if not is_valid:
                if args.resume_force:
                    print(f"[WARNING] Config mismatch (--resume_force):\n{error_msg}")
                else:
                    raise ValueError(
                        f"Cannot resume: {error_msg}\n"
                        "Use --resume_force to override (may cause issues)"
                    )
            # Update target if changed
            manifest.num_tasks_target = args.num_tasks
            manifest.out_npz = args.out_npz
            manifest.save(args.ckpt_dir)
        else:
            raise ValueError(
                f"Checkpoint directory exists but --resume not specified: {args.ckpt_dir}\n"
                "Use --resume to continue, or delete/rename the directory to start fresh"
            )
        
        # Create shard writer
        ckpt_writer = RoutesShardWriter(
            args.ckpt_dir,
            shard_size=args.ckpt_shard_size,
            flush_secs=args.ckpt_flush_secs,
            manifest=manifest,
        )
        
        if args.resume:
            print(f"[resume] Continuing from checkpoint:")
            print(f"  Collected: {manifest.num_tasks_collected}/{manifest.num_tasks_target}")
            print(f"  Current seed: {manifest.current_seed}")
            print(f"  Seeds attempted: {manifest.seeds_attempted}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print("="*60)
    print("META-RL ROUTE COLLECTION (Speed Optimized)")
    print("="*60)
    print(f"Target: {args.num_tasks} successful trajectories")
    print(f"Device: {device}")
    print(f"Batch size: {args.num_processes}")
    print(f"Require success: {bool(args.require_success)}")
    print(f"Max seed attempts: {args.max_seed_attempts}")
    print(f"\nSpeed optimizations:")
    print(f"  Batch completion threshold: {args.batch_completion_threshold:.0%}")
    print(f"  Task timeout: {args.task_timeout} steps")
    print(f"  Early abort (no success): {bool(args.early_abort)}")
    print("="*60)

    actor_critic, obs_shape, num_actions = build_model(
        args.model_ckpt, args.arch, args.hidden_size, device, args.env_name,
        use_compile=bool(args.use_compile)
    )

    # Warmup
    if args.use_compile and hasattr(torch, "compile") and device.type == "cuda":
        print("[warmup] Running torch.compile warmup...")
        with torch.inference_mode():
            dummy_obs = torch.zeros(1, *obs_shape, device=device)
            dummy_hxs = torch.zeros(1, args.hidden_size, device=device)
            dummy_masks = torch.ones(1, 1, device=device)
            for _ in range(3):
                actor_critic.act(dummy_obs, dummy_hxs, dummy_masks, deterministic=True)
            torch.cuda.synchronize()

    # Output containers
    #
    # routes_*.npz contract (high-level):
    # - routes_obs/actions: used by PKD sampler to compute hidden-state cycles.
    # - routes_xy: used by ridge embedding / CCA alignment as "behavior trajectory".
    # - routes_player_v/routes_ents_count/routes_nearest_ents: optional extra state
    #   (currently NOT used by PKD sampler nor cca_alignment.py).
    routes_obs: List[np.ndarray] = []
    routes_actions: List[np.ndarray] = []
    routes_xy: List[np.ndarray] = []
    routes_player_v: List[np.ndarray] = []
    routes_ents_count: List[np.ndarray] = []
    routes_nearest_ents: List[np.ndarray] = []
    routes_rewards: List[np.ndarray] = []
    routes_ep_len: List[int] = []
    routes_return: List[float] = []
    routes_success: List[bool] = []
    routes_seed: List[int] = []
    routes_selected_ep: List[int] = []
    routes_diag: List[Dict[str, Any]] = []
    all_episodes: List[List[Dict[str, Any]]] = []

    # Seed generator (resume from checkpoint if available)
    if use_ckpt and manifest and args.resume:
        current_seed = manifest.current_seed
        seeds_tried = manifest.seeds_attempted
        # Adjust progress bar
        initial_collected = manifest.num_tasks_collected
    else:
        current_seed = args.seed_offset
        seeds_tried = 0
        initial_collected = 0
    
    # Live stats
    live_stats = LiveStats(args.num_tasks)
    
    # Progress bar
    pbar = tqdm(
        total=args.num_tasks,
        initial=initial_collected,
        desc=live_stats.get_desc(),
        unit="traj",
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        file=sys.stdout,  # Write to stdout so it's not suppressed
        dynamic_ncols=True,
    )
    
    # Collection loop
    while True:
        # Check completion conditions
        if use_ckpt:
            current_collected = manifest.num_tasks_collected
        else:
            current_collected = len(routes_seed)
        
        if current_collected >= args.num_tasks:
            break
        if seeds_tried >= args.max_seed_attempts:
            break
        # Generate batch of seeds
        batch_size = min(args.num_processes, args.max_seed_attempts - seeds_tried)
        batch_seeds = list(range(current_seed, current_seed + batch_size))
        current_seed += batch_size
        seeds_tried += batch_size
        
        if batch_size == 0:
            break
        
        batch_start = time.perf_counter()
        batch_num = seeds_tried // args.num_processes
        
        # Show batch progress
        print(f"\r[Batch {batch_num}] Creating {batch_size} envs (seeds {batch_seeds[0]}-{batch_seeds[-1]})...", end="", flush=True)
        
        # Create environments
        envs = create_vec_env(args.env_name, batch_seeds, args.distribution_mode, device)
        print(f" running...", end="", flush=True)
        
        # Create inference wrapper
        fast_inf = FastInference(
            actor_critic, batch_size, args.hidden_size, device,
            use_amp=bool(args.use_amp)
        )
        
        # Run collection
        results, stats = run_batch_collection(
            envs, actor_critic, batch_seeds, args, device, obs_shape, fast_inf
        )
        
        envs.close()
        
        batch_time = time.perf_counter() - batch_start
        
        # Process results
        batch_successful = 0
        for seed, episodes, has_success in results:
            if not has_success and args.require_success:
                continue
            
            sel_idx, diag = select_best_episode(
                episodes,
                min_len=args.min_len,
                max_len=args.max_ep_len,
                require_success=bool(args.require_success),
            )
            
            if sel_idx is None:
                continue
            
            ep = episodes[sel_idx]
            
            xy_missing_frac = ep.get("xy_missing_frac", 1.0)
            if args.xy_fail_policy == "drop_task" and xy_missing_frac > args.xy_fail_threshold:
                continue
            
            # Save successful trajectory
            if use_ckpt:
                # Write to checkpoint shard
                ckpt_writer.append_route(
                    seed=int(seed),
                    selected_ep=int(sel_idx),
                    obs=ep["obs"],
                    actions=ep["actions"],
                    xy=ep["xy"],
                    player_v=ep.get("player_v", np.empty((0, 2), dtype=np.float32)),
                    ents_count=ep.get("ents_count", np.empty((0,), dtype=np.int32)),
                    nearest_ents=ep.get("nearest_ents", np.empty((0, 8, 4), dtype=np.float32)),
                    rewards=ep["rewards"],
                    ep_len=int(ep["len"]),
                    ep_return=float(ep["return"]),
                    success=bool(ep["success"]),
                    diag=diag,
                    all_episodes=episodes if args.save_all_episodes else None,
                )
                # Update progress in manifest
                ckpt_writer.update_progress(current_seed, seeds_tried)
            else:
                # Original in-memory storage
                routes_seed.append(int(seed))
                routes_selected_ep.append(int(sel_idx))
                routes_obs.append(ep["obs"])
                routes_actions.append(ep["actions"])
                routes_xy.append(ep["xy"])
                routes_player_v.append(ep.get("player_v", np.empty((0, 2), dtype=np.float32)))
                routes_ents_count.append(ep.get("ents_count", np.empty((0,), dtype=np.int32)))
                routes_nearest_ents.append(ep.get("nearest_ents", np.empty((0, 8, 4), dtype=np.float32)))
                routes_rewards.append(ep["rewards"])
                routes_ep_len.append(int(ep["len"]))
                routes_return.append(float(ep["return"]))
                routes_success.append(bool(ep["success"]))
                routes_diag.append(diag)
                
                if args.save_all_episodes:
                    all_episodes.append(episodes)
            
            batch_successful += 1
            pbar.update(1)
            
            # Check if we've reached target
            if use_ckpt:
                if manifest.num_tasks_collected >= args.num_tasks:
                    break
            else:
                if len(routes_seed) >= args.num_tasks:
                    break
        
        # Update live stats
        live_stats.update(
            batch_successful, batch_size, stats["total_steps"], batch_time,
            aborted=stats.get("aborted_early", 0),
            timed_out=stats.get("timed_out", 0)
        )
        pbar.set_description(live_stats.get_desc())
        
        # Print batch summary
        print(f" done! +{batch_successful} traj ({batch_time:.1f}s)", flush=True)
        
        # Sync CUDA periodically
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    pbar.close()
    
    # Flush any remaining buffered routes
    if use_ckpt:
        ckpt_writer.close()
        final_collected = manifest.num_tasks_collected
    else:
        final_collected = len(routes_seed)
    
    # Final stats
    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)
    print(f"Target:            {args.num_tasks}")
    print(f"Collected:         {final_collected}")
    print(f"Seeds attempted:   {seeds_tried}")
    print(f"Success rate:      {100*live_stats.success_rate:.1f}%")
    print(f"Total time:        {live_stats.format_time(live_stats.elapsed)}")
    print(f"Speed:             {live_stats.speed:.1f} traj/s, {live_stats.steps_per_sec:.0f} steps/s")
    print(f"\nOptimization stats:")
    print(f"  Early aborted:   {live_stats.aborted}")
    print(f"  Timed out:       {live_stats.timed_out}")
    
    # Diversity and length stats (only if not using ckpt, or load from merged file)
    if not use_ckpt:
        unique_seeds = len(set(routes_seed))
        print(f"\n[Diversity]")
        print(f"  Unique seeds:    {unique_seeds} / {len(routes_seed)} ({100*unique_seeds/max(1,len(routes_seed)):.1f}%)")
        
        # Length stats
        if routes_ep_len:
            lens = np.array(routes_ep_len)
            print(f"\n[Episode Lengths]")
            print(f"  Min: {lens.min()}, Max: {lens.max()}, Mean: {lens.mean():.1f}, Median: {np.median(lens):.0f}")
    else:
        print(f"\n[Note] Use 'python eval/routes_ckpt_tools.py info --ckpt_dir {args.ckpt_dir}' for detailed stats")
    
    print("="*60)

    # Save final output
    if use_ckpt:
        # Merge shards into final routes.npz
        print(f"\n[ckpt] Merging shards into {args.out_npz}...")
        n_merged = build_routes_npz_from_ckpt(args.ckpt_dir, args.out_npz)
        print(f"[saved] {args.out_npz} ({n_merged} trajectories)")
    else:
        # Original save path
        os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)

        save_dict = dict(
            routes_seed=np.asarray(routes_seed, dtype=np.int64),
            routes_selected_ep=np.asarray(routes_selected_ep, dtype=np.int64),
            routes_obs=np.asarray(routes_obs, dtype=object),
            routes_actions=np.asarray(routes_actions, dtype=object),
            routes_xy=np.asarray(routes_xy, dtype=object),
            routes_player_v=np.asarray(routes_player_v, dtype=object),
            routes_ents_count=np.asarray(routes_ents_count, dtype=object),
            routes_nearest_ents=np.asarray(routes_nearest_ents, dtype=object),
            routes_rewards=np.asarray(routes_rewards, dtype=object),
            routes_ep_len=np.asarray(routes_ep_len, dtype=np.int64),
            routes_return=np.asarray(routes_return, dtype=np.float32),
            routes_success=np.asarray(routes_success, dtype=np.bool_),
            routes_diag=np.asarray(routes_diag, dtype=object),
            meta=dict(
                env_name=args.env_name,
                distribution_mode=args.distribution_mode,
                num_tasks_target=args.num_tasks,
                num_tasks_collected=len(routes_seed),
                seeds_attempted=seeds_tried,
                success_rate=float(live_stats.success_rate),
                seed_offset=args.seed_offset,
                num_processes=args.num_processes,
                arch=args.arch,
                hidden_size=args.hidden_size,
                adapt_episodes=args.adapt_episodes,
                record_episodes=args.record_episodes,
                min_len=args.min_len,
                max_ep_len=args.max_ep_len,
                max_steps=args.max_steps,
                require_success=bool(args.require_success),
                # Speed optimization params
                batch_completion_threshold=args.batch_completion_threshold,
                task_timeout=args.task_timeout,
                early_abort=bool(args.early_abort),
            ),
        )

        if args.save_all_episodes:
            save_dict["episodes_all"] = np.asarray(all_episodes, dtype=object)

        np.savez_compressed(args.out_npz, **save_dict)
        print(f"\n[saved] {args.out_npz}")


if __name__ == "__main__":
    main()
