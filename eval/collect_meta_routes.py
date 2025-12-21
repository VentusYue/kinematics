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
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*gym.*")
warnings.filterwarnings("ignore", message=".*gymnasium.*")
warnings.filterwarnings("ignore", message=".*minigrid.*")
warnings.filterwarnings("ignore", message=".*np.bool8.*")

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

# Suppress gym warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import gym
    gym.logger.set_level(40)

from tqdm import tqdm
import time

# Add project root to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.common.vec_env import SubprocVecEnv, VecExtractDictObs, VecMonitor
from level_replay.envs import VecNormalize, VecPyTorchProcgen
from level_replay.model import Policy, SimplePolicy

# XY helpers
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
    from procgen_tools import maze as maze_tools
except Exception as e:
    print(f"Warning: Could not import procgen_tools.maze: {e}")
    maze_tools = None


def extract_xy_from_gym_env(env) -> Tuple[float, float]:
    """Extract (x, y) from a gym-wrapped procgen maze env."""
    if maze_tools is None:
        return (float("nan"), float("nan"))
    
    try:
        inner = env
        while hasattr(inner, "env"):
            if hasattr(inner, "callmethod"):
                break
            inner = inner.env
        
        if hasattr(inner, "callmethod"):
            state_bytes_list = inner.callmethod("get_state")
            state = maze_tools.EnvState(state_bytes_list[0])
            vals = state.state_vals
            ents = vals["ents"][0]
            return (float(ents['x'].val), float(ents['y'].val))
        
        return (float("nan"), float("nan"))
    except Exception:
        return (float("nan"), float("nan"))


class XYInfoWrapper(gym.Wrapper):
    """
    Wrapper that extracts (x, y) coordinates on each step.
    Handles VecEnv auto-reset by storing terminal_xy separately.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._last_xy = (float("nan"), float("nan"))
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._last_xy = extract_xy_from_gym_env(self.env)
        return obs
    
    def step(self, action):
        xy_before = self._last_xy
        obs, reward, done, info = self.env.step(action)
        
        if info is None:
            info = {}
        
        if done:
            info["terminal_xy"] = xy_before
            xy_after_reset = extract_xy_from_gym_env(self.env)
            info["xy"] = xy_after_reset
            self._last_xy = xy_after_reset
        else:
            xy = extract_xy_from_gym_env(self.env)
            self._last_xy = xy
            info["xy"] = xy
            info["terminal_xy"] = None
        
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
        
        return {
            "obs": obs_arr,
            "actions": np.array(self.actions, dtype=np.int64),
            "rewards": np.array(self.rewards, dtype=np.float32),
            "xy": xy_arr,
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
    routes_obs: List[np.ndarray] = []
    routes_actions: List[np.ndarray] = []
    routes_xy: List[np.ndarray] = []
    routes_rewards: List[np.ndarray] = []
    routes_ep_len: List[int] = []
    routes_return: List[float] = []
    routes_success: List[bool] = []
    routes_seed: List[int] = []
    routes_selected_ep: List[int] = []
    routes_diag: List[Dict[str, Any]] = []
    all_episodes: List[List[Dict[str, Any]]] = []

    # Seed generator
    current_seed = args.seed_offset
    seeds_tried = 0
    
    # Live stats
    live_stats = LiveStats(args.num_tasks)
    
    # Progress bar
    pbar = tqdm(
        total=args.num_tasks,
        desc=live_stats.get_desc(),
        unit="traj",
        ncols=120,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        file=sys.stdout,  # Write to stdout so it's not suppressed
        dynamic_ncols=True,
    )
    
    # Collection loop
    while len(routes_seed) < args.num_tasks and seeds_tried < args.max_seed_attempts:
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
            routes_seed.append(int(seed))
            routes_selected_ep.append(int(sel_idx))
            routes_obs.append(ep["obs"])
            routes_actions.append(ep["actions"])
            routes_xy.append(ep["xy"])
            routes_rewards.append(ep["rewards"])
            routes_ep_len.append(int(ep["len"]))
            routes_return.append(float(ep["return"]))
            routes_success.append(bool(ep["success"]))
            routes_diag.append(diag)
            
            if args.save_all_episodes:
                all_episodes.append(episodes)
            
            batch_successful += 1
            pbar.update(1)
            
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
    
    # Final stats
    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)
    print(f"Target:            {args.num_tasks}")
    print(f"Collected:         {len(routes_seed)}")
    print(f"Seeds attempted:   {seeds_tried}")
    print(f"Success rate:      {100*live_stats.success_rate:.1f}%")
    print(f"Total time:        {live_stats.format_time(live_stats.elapsed)}")
    print(f"Speed:             {live_stats.speed:.1f} traj/s, {live_stats.steps_per_sec:.0f} steps/s")
    print(f"\nOptimization stats:")
    print(f"  Early aborted:   {live_stats.aborted}")
    print(f"  Timed out:       {live_stats.timed_out}")
    
    # Diversity check
    unique_seeds = len(set(routes_seed))
    print(f"\n[Diversity]")
    print(f"  Unique seeds:    {unique_seeds} / {len(routes_seed)} ({100*unique_seeds/max(1,len(routes_seed)):.1f}%)")
    
    # Length stats
    if routes_ep_len:
        lens = np.array(routes_ep_len)
        print(f"\n[Episode Lengths]")
        print(f"  Min: {lens.min()}, Max: {lens.max()}, Mean: {lens.mean():.1f}, Median: {np.median(lens):.0f}")
    
    print("="*60)

    # Save
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)

    save_dict = dict(
        routes_seed=np.asarray(routes_seed, dtype=np.int64),
        routes_selected_ep=np.asarray(routes_selected_ep, dtype=np.int64),
        routes_obs=np.asarray(routes_obs, dtype=object),
        routes_actions=np.asarray(routes_actions, dtype=object),
        routes_xy=np.asarray(routes_xy, dtype=object),
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
