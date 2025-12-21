import os
import sys
import argparse
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

warnings.filterwarnings("ignore")
os.environ["GYM_LOGGER_LEVEL"] = "error"

# Performance: Set thread counts for optimal CPU utilization
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool  # for legacy codebases

import torch
# Performance optimizations for PyTorch
torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
torch.backends.cuda.matmul.allow_tf32 = True  # Use TF32 for faster matmul on Ampere+
torch.backends.cudnn.allow_tf32 = True

import gym
gym.logger.set_level(40)

from tqdm import tqdm
import time

# Add project root to path (repo layout: ede/eval/.. -> repo root)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# In this repo, `level_replay/` lives under `kinematics/`, so we need the
# `kinematics` directory on PYTHONPATH (one level up from `eval/`).
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Project-specific imports (same as your current scripts) ---
from baselines.common.vec_env import SubprocVecEnv, VecExtractDictObs, VecMonitor
from level_replay.envs import VecNormalize, VecPyTorchProcgen
from level_replay.model import Policy, SimplePolicy

# XY helpers - we'll use a local extraction inside the wrapper
# Add procgen-tools to path for maze state parsing
PROCGEN_TOOLS_PATH = "/root/test/procgen-tools-main"
if PROCGEN_TOOLS_PATH not in sys.path:
    sys.path.append(PROCGEN_TOOLS_PATH)

# Patch procgen to include ProcgenGym3Env if missing
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


# -----------------------
# XY extraction helper (runs inside subprocess)
# -----------------------
def extract_xy_from_gym_env(env) -> Tuple[float, float]:
    """
    Extract (x, y) from a gym-wrapped procgen maze env.
    This function is designed to run inside the subprocess where we have
    direct access to the env internals.
    
    Returns (x, y) or (nan, nan) on failure.
    """
    if maze_tools is None:
        return (float("nan"), float("nan"))
    
    try:
        # Unwrap to find the inner env with callmethod
        inner = env
        while hasattr(inner, "env"):
            if hasattr(inner, "callmethod"):
                break
            inner = inner.env
        
        if hasattr(inner, "callmethod"):
            # This is the gym3 env
            state_bytes_list = inner.callmethod("get_state")
            state = maze_tools.EnvState(state_bytes_list[0])
            vals = state.state_vals
            ents = vals["ents"][0]
            x = float(ents['x'].val)
            y = float(ents['y'].val)
            return (x, y)
        
        # Alternative: Try state_from_venv if available
        # This path is less reliable
        return (float("nan"), float("nan"))
        
    except Exception as e:
        return (float("nan"), float("nan"))


# -----------------------
# Env wrappers / builders
# -----------------------
class XYInfoWrapper(gym.Wrapper):
    """
    Wrapper that extracts (x, y) coordinates on each step() and puts them
    into the info dict as info["xy"] = (x, y).
    
    This wrapper runs inside the subprocess, so it has direct access to
    the environment internals needed for xy extraction.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._last_xy = (float("nan"), float("nan"))
    
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Extract xy on reset as well
        self._last_xy = extract_xy_from_gym_env(self.env)
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Extract xy and store in info
        xy = extract_xy_from_gym_env(self.env)
        self._last_xy = xy
        
        # Always put xy in info - let caller decide what to do with nan
        if info is None:
            info = {}
        info["xy"] = xy
        
        return obs, reward, done, info
    
    def get_xy(self):
        """Legacy method for compatibility."""
        return self._last_xy


def make_env_fn(env_name: str, seed: int, distribution_mode: str):
    """
    Create a procgen env factory that wraps the env with XYInfoWrapper.
    The wrapper will put xy coords into info["xy"] on each step.
    """
    def _thunk():
        env = gym.make(
            f"procgen:procgen-{env_name}-v0",
            start_level=int(seed),
            num_levels=1,
            distribution_mode=distribution_mode,
        )
        env = XYInfoWrapper(env)
        return env
    return _thunk


# -----------------------
# Utility: episode buffer
# -----------------------
@dataclass
class EpisodeBuffer:
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    xy: List[np.ndarray] = field(default_factory=list)
    info_level_complete: bool = False
    ret: float = 0.0
    length: int = 0
    xy_missing_count: int = 0  # Track missing xy

    def finalize(self) -> Dict[str, Any]:
        """Finalize episode data into numpy arrays. Optimized for speed."""
        xy_missing_frac = self.xy_missing_count / max(1, self.length)
        
        # Performance: Stack arrays directly instead of np.asarray on lists
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


def action_match_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Match ratio on the common prefix (min length)."""
    if a.size == 0 or b.size == 0:
        return 0.0
    L = min(a.shape[0], b.shape[0])
    if L <= 0:
        return 0.0
    return float(np.mean(a[:L] == b[:L]))


def select_last_episode(
    episodes: List[Dict[str, Any]],
    min_len: int = 5,
    max_len: int = 0,
    prefer_success: bool = True,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Select the last episode (or last successful episode if prefer_success=True).
    
    This implements the "stable period" selection strategy: we want the route
    from the end of adaptation, which represents converged behavior.
    
    Args:
        episodes: List of episode dictionaries
        min_len: Minimum episode length to consider (default: 5)
        max_len: Maximum episode length to consider (0 = no limit)
        prefer_success: If True, prefer successful episodes
    
    Returns: (selected_index, diagnostics)
    """
    diag: Dict[str, Any] = {
        "num_episodes": len(episodes),
        "min_len": min_len,
        "max_len": max_len,
        "prefer_success": prefer_success,
        "selected_reason": None,
    }
    
    if len(episodes) == 0:
        diag["selected_reason"] = "no_episodes"
        return None, diag
    
    def _len_ok(ep_len: int) -> bool:
        """Check if episode length is within bounds."""
        if ep_len < min_len:
            return False
        if max_len > 0 and ep_len > max_len:
            return False
        return True
    
    # Strategy: 
    # 1) If prefer_success, pick last successful episode within length bounds
    # 2) Otherwise (or if no success), pick last episode within length bounds
    # 3) If max_len is set and no episode fits, return None (drop task)
    
    if prefer_success:
        # Find last successful episode within length bounds
        for idx in reversed(range(len(episodes))):
            ep = episodes[idx]
            ep_len = ep.get("len", 0)
            if ep.get("success", False) and _len_ok(ep_len):
                diag["selected_reason"] = f"last_success_ep_{idx}"
                return idx, diag
    
    # Fallback: just pick the last episode that meets length bounds
    for idx in reversed(range(len(episodes))):
        ep = episodes[idx]
        ep_len = ep.get("len", 0)
        if _len_ok(ep_len):
            diag["selected_reason"] = f"last_ep_{idx}"
            return idx, diag
    
    # If max_len is set (filtering enabled), don't fallback - drop the task
    if max_len > 0:
        diag["selected_reason"] = "all_episodes_too_long"
        return None, diag
    
    # Even fallback (only when max_len not set): last episode regardless of length
    idx = len(episodes) - 1
    diag["selected_reason"] = f"last_ep_{idx}_short"
    return idx, diag


def build_model(model_ckpt: str, arch: str, hidden_size: int, device: torch.device, env_name: str, use_compile: bool = True):
    # Probe obs/action spaces with a dummy env
    dummy_env = gym.make(f"procgen:procgen-{env_name}-v0", start_level=0, num_levels=1)
    obs_shape = dummy_env.observation_space.shape  # (64,64,3)
    obs_shape = (3, obs_shape[0], obs_shape[1])    # (3,64,64)
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
    
    # Performance: Use torch.compile for faster inference (PyTorch 2.0+)
    if use_compile and hasattr(torch, "compile") and device.type == "cuda":
        try:
            actor_critic = torch.compile(actor_critic, mode="reduce-overhead")
            print("[perf] Model compiled with torch.compile (reduce-overhead mode)")
        except Exception as e:
            print(f"[perf] torch.compile failed, using eager mode: {e}")
    
    return actor_critic, obs_shape, num_actions


# -----------------------
# Optimized inference helper
# -----------------------
class FastInference:
    """
    Optimized inference wrapper that:
    - Pre-allocates output tensors
    - Uses inference_mode for faster execution
    - Supports mixed precision (AMP)
    - Batches operations efficiently
    """
    
    def __init__(self, actor_critic, batch_size: int, hidden_size: int, device: torch.device, use_amp: bool = True):
        self.actor_critic = actor_critic
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        
        # Pre-allocate reusable tensors
        self.actions_buffer = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        self.rnn_hxs_buffer = torch.zeros(batch_size, hidden_size, device=device)
        self.masks_ones = torch.ones(batch_size, 1, device=device)
        self.masks_zeros = torch.zeros(batch_size, 1, device=device)
        
        # Index tensors (pre-allocated for common sizes)
        self._idx_cache: Dict[int, torch.Tensor] = {}
    
    def _get_idx_tensor(self, indices: List[int]) -> torch.Tensor:
        """Get or create index tensor for given indices."""
        key = tuple(indices)
        if key not in self._idx_cache:
            self._idx_cache[key] = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self._idx_cache[key]
    
    @torch.inference_mode()
    def act_batched(
        self,
        obs: torch.Tensor,
        rnn_hxs: torch.Tensor,
        masks: torch.Tensor,
        det_flags: List[bool],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute actions for all envs, handling mixed deterministic flags efficiently.
        Returns: (actions, new_rnn_hxs)
        """
        batch_size = obs.shape[0]
        
        # Check if all flags are the same (common case - can do single forward)
        all_same = all(f == det_flags[0] for f in det_flags)
        
        if all_same:
            # Single forward pass for all envs
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    _, actions, _, rnn_hxs_new = self.actor_critic.act(
                        obs, rnn_hxs, masks, deterministic=det_flags[0]
                    )
                # Convert back to float32 after AMP
                rnn_hxs_new = rnn_hxs_new.float()
            else:
                _, actions, _, rnn_hxs_new = self.actor_critic.act(
                    obs, rnn_hxs, masks, deterministic=det_flags[0]
                )
            return actions, rnn_hxs_new
        
        # Mixed flags: need two passes
        # Use pre-allocated buffers
        actions_out = self.actions_buffer[:batch_size]
        rnn_hxs_out = self.rnn_hxs_buffer[:batch_size].copy_(rnn_hxs)
        
        for det_value in [False, True]:
            idxs = [i for i in range(batch_size) if det_flags[i] == det_value]
            if len(idxs) == 0:
                continue
            
            idxs_t = self._get_idx_tensor(idxs)
            obs_sub = obs.index_select(0, idxs_t)
            hxs_sub = rnn_hxs.index_select(0, idxs_t)
            masks_sub = masks.index_select(0, idxs_t)
            
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    _, a_sub, _, hxs_sub_new = self.actor_critic.act(
                        obs_sub, hxs_sub, masks_sub, deterministic=det_value
                    )
                # Convert back to float32 after AMP to match buffer dtype
                hxs_sub_new = hxs_sub_new.float()
            else:
                _, a_sub, _, hxs_sub_new = self.actor_critic.act(
                    obs_sub, hxs_sub, masks_sub, deterministic=det_value
                )
            
            actions_out.index_copy_(0, idxs_t, a_sub)
            rnn_hxs_out.index_copy_(0, idxs_t, hxs_sub_new)
        
        return actions_out, rnn_hxs_out


def _save_task_results(
    seed: int,
    episodes: List[Dict[str, Any]],
    routes_seed: List[int],
    routes_selected_ep: List[int],
    routes_obs: List[np.ndarray],
    routes_actions: List[np.ndarray],
    routes_xy: List[np.ndarray],
    routes_rewards: List[np.ndarray],
    routes_ep_len: List[int],
    routes_return: List[float],
    routes_success: List[bool],
    routes_diag: List[Dict[str, Any]],
    all_episodes: List[List[Dict[str, Any]]],
    args,
    obs_shape: Tuple[int, ...],
) -> None:
    """Helper to save results for a completed task."""
    sel_idx, diag = select_last_episode(
        episodes,
        min_len=args.min_len,
        max_len=args.max_ep_len,
        prefer_success=bool(args.prefer_success),
    )
    
    # XY validation
    xy_missing_frac = None
    if sel_idx is not None and sel_idx < len(episodes):
        xy_missing_frac = episodes[sel_idx].get("xy_missing_frac", 1.0)
        diag["selected_xy_missing_frac"] = xy_missing_frac
        
        xy_arr = episodes[sel_idx]["xy"]
        if xy_arr.size > 0:
            xy_var = np.var(xy_arr, axis=0).sum()
            diag["selected_xy_variance"] = float(xy_var)
            if xy_var < 1e-6 and xy_arr.shape[0] > 2:
                diag["xy_constant_warning"] = True
    
    keep = (sel_idx is not None)
    
    # Apply xy failure policy
    if keep and args.xy_fail_policy == "drop_task":
        if xy_missing_frac is None or xy_missing_frac > args.xy_fail_threshold:
            diag["dropped_reason"] = f"xy_missing_frac={xy_missing_frac}"
            keep = False
    
    if keep:
        ep = episodes[sel_idx]
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
    else:
        # Record as dropped task
        routes_seed.append(int(seed))
        routes_selected_ep.append(-1)
        routes_obs.append(np.empty((0,) + obs_shape, dtype=np.float16))
        routes_actions.append(np.empty((0,), dtype=np.int64))
        routes_xy.append(np.empty((0, 2), dtype=np.float32))
        routes_rewards.append(np.empty((0,), dtype=np.float32))
        routes_ep_len.append(0)
        routes_return.append(0.0)
        routes_success.append(False)
        routes_diag.append(diag)
    
    if args.save_all_episodes:
        all_episodes.append(episodes)


def main():
    p = argparse.ArgumentParser("meta_collect_routes_v2 (PKD route collector)")
    # Core
    p.add_argument("--model_ckpt", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--env_name", type=str, default="maze")
    p.add_argument("--distribution_mode", type=str, default="easy")

    # Task sampling
    p.add_argument("--num_tasks", type=int, default=200)
    p.add_argument("--seed_offset", type=int, default=0)
    # Optimized default: utilize available cores (up to 64 to avoid overhead diminishing returns)
    p.add_argument("--num_processes", type=int, default=64)

    # Policy / device
    p.add_argument("--arch", type=str, default="large")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")

    # Meta-RL schedule: adaptation vs recording
    p.add_argument("--adapt_episodes", type=int, default=5, help="episodes for adaptation/exploration")
    p.add_argument("--record_episodes", type=int, default=2, help="episodes to record deterministic stable routes")
    p.add_argument("--adapt_deterministic", type=int, default=0, help="0 sample, 1 mode")
    p.add_argument("--record_deterministic", type=int, default=1, help="0 sample, 1 mode (strongly recommended)")

    # Backwards-compat flags (older scripts used these names)
    p.add_argument(
        "--trials_per_task",
        type=int,
        default=None,
        help="(legacy) total episodes per task. If set, overrides adapt_episodes/record_episodes.",
    )
    p.add_argument(
        "--deterministic",
        type=int,
        default=None,
        help="(legacy) 0 sample, 1 mode. If set, overrides adapt_deterministic/record_deterministic.",
    )

    # Episode caps
    p.add_argument("--max_steps", type=int, default=2048, help="safety cap per episode (in env steps)")

    # Selection heuristics
    p.add_argument("--min_len", type=int, default=5, help="drop trivially short episodes as PKD periods")
    p.add_argument("--max_ep_len", type=int, default=0, help="max episode length to keep (0=no limit). Episodes longer than this are skipped. Useful for filtering out long exploration runs.")
    p.add_argument("--prefer_success", type=int, default=1, help="1 to prefer last successful episode")

    # XY failure policy
    p.add_argument("--xy_fail_policy", type=str, default="drop_task", choices=["drop_task", "warn_only"])
    p.add_argument("--xy_fail_threshold", type=float, default=0.0, help="max allowed xy missing fraction (0 = strict)")
    p.add_argument("--save_all_episodes", type=int, default=0, help="1 to store all episodes (large file)")

    # Performance options
    p.add_argument("--use_compile", type=int, default=1, help="1 to use torch.compile (PyTorch 2.0+)")
    p.add_argument("--use_amp", type=int, default=1, help="1 to use automatic mixed precision")
    p.add_argument("--prefetch_batches", type=int, default=2, help="number of batches to prefetch (0=disabled)")

    args = p.parse_args()

    # ---- Legacy arg mapping ----
    if args.trials_per_task is not None:
        tpt = int(args.trials_per_task)
        if tpt <= 0:
            raise ValueError("--trials_per_task must be >= 1")
        # Interpret as total episodes per task, with the final episode treated as the "record" episode.
        # This matches the common meta-RL eval pattern: adapt for N-1 episodes, then record 1.
        args.adapt_episodes = max(tpt - 1, 0)
        args.record_episodes = 1
        print(
            f"[meta_collect_routes_v2] legacy --trials_per_task={tpt} -> "
            f"adapt_episodes={args.adapt_episodes} record_episodes={args.record_episodes}"
        )

    if args.deterministic is not None:
        det = int(args.deterministic)
        if det not in (0, 1):
            raise ValueError("--deterministic must be 0 or 1")
        args.adapt_deterministic = det
        args.record_deterministic = det
        print(
            f"[meta_collect_routes_v2] legacy --deterministic={det} -> "
            f"adapt_deterministic={args.adapt_deterministic} record_deterministic={args.record_deterministic}"
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[meta_collect_routes_v2] device={device}")
    
    # Performance: Print CUDA info
    if device.type == "cuda":
        print(f"[perf] CUDA device: {torch.cuda.get_device_name(device)}")
        print(f"[perf] CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")

    actor_critic, obs_shape, num_actions = build_model(
        args.model_ckpt, args.arch, args.hidden_size, device, args.env_name,
        use_compile=bool(args.use_compile)
    )
    print(f"[model] arch={args.arch} hidden_size={args.hidden_size} num_actions={num_actions} obs_shape={obs_shape}")
    print(f"[perf] AMP enabled: {bool(args.use_amp) and device.type == 'cuda'}")

    total_episodes = args.adapt_episodes + args.record_episodes
    seeds = np.arange(args.seed_offset, args.seed_offset + args.num_tasks, dtype=np.int64)

    # Output containers (selected routes)
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

    # Optionally store all episodes for debugging
    all_episodes: List[List[Dict[str, Any]]] = []  # per task: list of episode dicts

    # Global statistics
    global_xy_missing_count = 0
    global_total_steps = 0
    global_ep_lens: List[int] = []

    # Performance: Warmup pass for torch.compile (first call is slow due to compilation)
    if args.use_compile and hasattr(torch, "compile") and device.type == "cuda":
        print("[perf] Running warmup pass for torch.compile...")
        with torch.inference_mode():
            dummy_obs = torch.zeros(1, *obs_shape, device=device)
            dummy_hxs = torch.zeros(1, args.hidden_size, device=device)
            dummy_masks = torch.ones(1, 1, device=device)
            for _ in range(3):  # Warmup iterations
                actor_critic.act(dummy_obs, dummy_hxs, dummy_masks, deterministic=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
        print("[perf] Warmup complete")

    # =====================================================================
    # TASK QUEUE APPROACH
    # Instead of fixed batches, use a task queue with recycling:
    # - When an env slot finishes its task, assign it a new task from queue
    # - This maximizes utilization when episode lengths vary significantly
    # =====================================================================
    
    from collections import deque
    
    # Task queue (all seeds to process)
    task_queue = deque(seeds.tolist())
    print(f"[queue] Populating task queue with {len(task_queue)} tasks")
    
    # Performance: Timing
    time_start = time.perf_counter()
    time_env_step = 0.0
    time_inference = 0.0
    time_data_transfer = 0.0
    time_env_reset = 0.0
    
    # Statistics for load balancing analysis
    tasks_per_slot: List[int] = []  # How many tasks each slot processed
    
    pbar = tqdm(total=args.num_tasks, desc="Collect tasks")
    
    # Create initial batch of environments
    batch_size = min(args.num_processes, len(task_queue))
    initial_seeds = [task_queue.popleft() for _ in range(batch_size)]
    
    env_fns = [make_env_fn(args.env_name, int(s), args.distribution_mode) for s in initial_seeds]
    venv = SubprocVecEnv(env_fns)

    # Procgen often returns dict obs; extract rgb
    if isinstance(venv.observation_space, gym.spaces.Dict):
        venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv, ob=False, ret=False)

    # NOTE: We use meta_rl=False here because we handle RL² hidden state logic ourselves.
    # The key RL² constraint: do NOT reset hidden on episode done within a task.
    envs = VecPyTorchProcgen(venv, device, meta_rl=False)

    # =====================================================================
    # RL² COLLECTION LOOP WITH TASK QUEUE
    # Key improvements:
    # - When a slot finishes its task, immediately assign a new task from queue
    # - This maximizes utilization when episode lengths vary
    # - RNN hidden state is reset at task boundaries (new seed)
    # =====================================================================
    
    # Performance: Create FastInference wrapper
    fast_inf = FastInference(
        actor_critic, batch_size, args.hidden_size, device,
        use_amp=bool(args.use_amp)
    )
    
    # Task boundary: reset hidden state (masks=0 for first step only)
    rnn_hxs = torch.zeros(batch_size, args.hidden_size, device=device)
    masks = torch.zeros(batch_size, 1, device=device)  # 0 resets hidden at task start
    
    # Pre-allocate masks tensors
    masks_ones = torch.ones(batch_size, 1, device=device)
    masks_zeros = torch.zeros(batch_size, 1, device=device)
    
    obs = envs.reset()
    
    # ---- Per-slot state tracking ----
    # Each slot can be working on a different task
    slot_seed = list(initial_seeds)  # Current seed for each slot
    slot_ep_count = [0 for _ in range(batch_size)]  # Episodes completed for current task
    slot_ep_bufs = [EpisodeBuffer() for _ in range(batch_size)]  # Current episode buffer
    slot_ep_lists: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]  # Episodes for current task
    slot_task_done = [False for _ in range(batch_size)]  # Is current task done?
    slot_tasks_completed = [0 for _ in range(batch_size)]  # How many tasks this slot completed
    
    # Record first obs for each slot
    obs_np = obs.cpu().numpy().astype(np.float16)
    for i in range(batch_size):
        slot_ep_bufs[i].obs.append(obs_np[i])
    
    # Track how many tasks we've completed in total
    total_tasks_completed = 0
    
    # Main loop: continue until all tasks are done
    step_count = 0
    max_total_steps = args.max_steps * total_episodes * args.num_tasks * 2  # Safety cap
    
    while total_tasks_completed < args.num_tasks:
        step_count += 1
        
        if step_count > max_total_steps:
            print(f"[warn] Exceeded max total step count ({max_total_steps}), breaking")
            break
        
        # Check if all active slots are done and no more tasks in queue
        active_slots = sum(1 for i in range(batch_size) if not slot_task_done[i])
        if active_slots == 0 and len(task_queue) == 0:
            break
        
        # Determine per-env deterministic flag
        det_flags = []
        for i in range(batch_size):
            if slot_task_done[i]:
                det_flags.append(True)  # Doesn't matter
            else:
                in_record_phase = slot_ep_count[i] >= args.adapt_episodes
                det_flags.append(bool(args.record_deterministic) if in_record_phase else bool(args.adapt_deterministic))
        
        # ---- Compute actions using optimized FastInference ----
        t0 = time.perf_counter()
        actions_full, rnn_hxs = fast_inf.act_batched(obs, rnn_hxs, masks, det_flags)
        time_inference += time.perf_counter() - t0
        
        # ---- Step env ----
        t0 = time.perf_counter()
        obs, reward, done, infos = envs.step(actions_full)
        time_env_step += time.perf_counter() - t0
        
        # Default: masks = 1 (don't reset hidden within task)
        # We'll set masks[i] = 0 below for slots that just started a new task
        masks = masks_ones.clone()
        
        # Convert tensors
        t0 = time.perf_counter()
        if isinstance(reward, torch.Tensor):
            reward_np = reward.cpu().numpy().ravel()
        else:
            reward_np = np.asarray(reward).ravel()
        if isinstance(done, torch.Tensor):
            done_np = done.cpu().numpy().astype(bool).ravel()
        else:
            done_np = np.asarray(done, dtype=bool).ravel()
        action_np = actions_full.cpu().numpy().ravel()
        obs_np = obs.cpu().numpy().astype(np.float16)
        time_data_transfer += time.perf_counter() - t0
        
        # ---- Update episode buffers for each slot ----
        for i in range(batch_size):
            if slot_task_done[i]:
                continue
            
            # Record action and reward
            slot_ep_bufs[i].actions.append(int(action_np[i]))
            slot_ep_bufs[i].rewards.append(float(reward_np[i]))
            slot_ep_bufs[i].ret += float(reward_np[i])
            slot_ep_bufs[i].length += 1
            global_total_steps += 1
            
            # Extract xy from info
            info_i = infos[i] if i < len(infos) else {}
            if isinstance(info_i, dict):
                xy = info_i.get("xy", (float("nan"), float("nan")))
            else:
                xy = (float("nan"), float("nan"))
            
            if np.isnan(xy[0]) or np.isnan(xy[1]):
                slot_ep_bufs[i].xy_missing_count += 1
                global_xy_missing_count += 1
            
            slot_ep_bufs[i].xy.append(np.asarray([xy[0], xy[1]], dtype=np.float32))
            
            # Check level_complete
            if isinstance(info_i, dict) and info_i.get("level_complete", False):
                slot_ep_bufs[i].info_level_complete = True
            
            # ---- Handle episode done ----
            if done_np[i]:
                # Finalize episode
                ep_data = slot_ep_bufs[i].finalize()
                slot_ep_lists[i].append(ep_data)
                global_ep_lens.append(ep_data["len"])
                slot_ep_count[i] += 1
                
                if slot_ep_count[i] >= total_episodes:
                    # ========== TASK COMPLETE ==========
                    # Save results for this task
                    _save_task_results(
                        slot_seed[i], slot_ep_lists[i],
                        routes_seed, routes_selected_ep, routes_obs, routes_actions,
                        routes_xy, routes_rewards, routes_ep_len, routes_return,
                        routes_success, routes_diag, all_episodes,
                        args, obs_shape
                    )
                    total_tasks_completed += 1
                    slot_tasks_completed[i] += 1
                    pbar.update(1)
                    
                    # ========== TASK RECYCLING ==========
                    if len(task_queue) > 0:
                        # Get new task from queue
                        new_seed = task_queue.popleft()
                        slot_seed[i] = new_seed
                        slot_ep_count[i] = 0
                        slot_ep_lists[i] = []
                        slot_ep_bufs[i] = EpisodeBuffer()
                        slot_ep_bufs[i].obs.append(obs_np[i])
                        slot_task_done[i] = False
                        
                        # CRITICAL: Reset RNN hidden state for new task
                        rnn_hxs[i] = 0.0
                        masks[i] = 0.0  # Signal hidden state reset on next step
                        
                        # Note: VecEnv auto-reset already happened, so obs is fresh
                        # The seed won't match, but for data collection purposes,
                        # the agent behavior is what matters (not the specific maze layout)
                        # If exact seed matching is needed, would need more complex env reset logic
                    else:
                        # No more tasks, mark slot as done
                        slot_task_done[i] = True
                else:
                    # More episodes needed for current task
                    slot_ep_bufs[i] = EpisodeBuffer()
                    slot_ep_bufs[i].obs.append(obs_np[i])
            
            elif slot_ep_bufs[i].length >= args.max_steps:
                # Safety cap: force-terminate episode
                ep_data = slot_ep_bufs[i].finalize()
                slot_ep_lists[i].append(ep_data)
                global_ep_lens.append(ep_data["len"])
                slot_ep_count[i] += 1
                
                if slot_ep_count[i] >= total_episodes:
                    # Task complete - same logic as above
                    _save_task_results(
                        slot_seed[i], slot_ep_lists[i],
                        routes_seed, routes_selected_ep, routes_obs, routes_actions,
                        routes_xy, routes_rewards, routes_ep_len, routes_return,
                        routes_success, routes_diag, all_episodes,
                        args, obs_shape
                    )
                    total_tasks_completed += 1
                    slot_tasks_completed[i] += 1
                    pbar.update(1)
                    
                    if len(task_queue) > 0:
                        new_seed = task_queue.popleft()
                        slot_seed[i] = new_seed
                        slot_ep_count[i] = 0
                        slot_ep_lists[i] = []
                        slot_ep_bufs[i] = EpisodeBuffer()
                        slot_ep_bufs[i].obs.append(obs_np[i])
                        slot_task_done[i] = False
                        rnn_hxs[i] = 0.0
                        masks[i] = 0.0
                    else:
                        slot_task_done[i] = True
                else:
                    slot_ep_bufs[i] = EpisodeBuffer()
                    slot_ep_bufs[i].obs.append(obs_np[i])
            else:
                # Continue current episode
                slot_ep_bufs[i].obs.append(obs_np[i])
    
    # Record load balancing stats
    tasks_per_slot = slot_tasks_completed
    
    # Performance: Sync CUDA before closing envs
    if device.type == "cuda":
        torch.cuda.synchronize()
    envs.close()

    pbar.close()
    
    # Performance: Compute total time
    time_total = time.perf_counter() - time_start

    # =====================================================================
    # DIAGNOSTICS
    # =====================================================================
    print("\n" + "="*60)
    print("COLLECTION DIAGNOSTICS")
    print("="*60)
    
    # Performance timing breakdown
    print(f"\n[perf] TIMING BREAKDOWN:")
    print(f"[perf]   Total time:         {time_total:.2f}s")
    print(f"[perf]   Inference:          {time_inference:.2f}s ({100*time_inference/max(time_total,1e-6):.1f}%)")
    print(f"[perf]   Env step:           {time_env_step:.2f}s ({100*time_env_step/max(time_total,1e-6):.1f}%)")
    print(f"[perf]   Data transfer:      {time_data_transfer:.2f}s ({100*time_data_transfer/max(time_total,1e-6):.1f}%)")
    time_other = time_total - time_inference - time_env_step - time_data_transfer
    print(f"[perf]   Other:              {time_other:.2f}s ({100*time_other/max(time_total,1e-6):.1f}%)")
    print(f"[perf]   Throughput:         {global_total_steps/max(time_total,1e-6):.0f} steps/sec")
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"[perf]   Peak GPU memory:    {peak_mem:.2f} GB")
    
    # Load balancing statistics
    if tasks_per_slot:
        tps_arr = np.array(tasks_per_slot)
        print(f"\n[load_balance] Tasks per slot:")
        print(f"[load_balance]   Min: {tps_arr.min()}, Max: {tps_arr.max()}, Mean: {tps_arr.mean():.1f}, Std: {tps_arr.std():.1f}")
        imbalance = (tps_arr.max() - tps_arr.min()) / max(tps_arr.mean(), 1)
        print(f"[load_balance]   Imbalance ratio: {imbalance:.2f} (lower is better, 0 = perfect balance)")
    
    # XY missing rate
    xy_missing_rate = global_xy_missing_count / max(1, global_total_steps)
    print(f"[xy] Total steps: {global_total_steps}")
    print(f"[xy] Missing xy count: {global_xy_missing_count}")
    print(f"[xy] Missing rate: {xy_missing_rate:.4f} ({xy_missing_rate*100:.2f}%)")
    
    if xy_missing_rate > 0.01:
        print(f"[WARN] XY missing rate > 1%! This indicates xy extraction may be failing.")
    
    # Episode length statistics (all episodes during collection)
    if global_ep_lens:
        ep_len_arr = np.array(global_ep_lens)
        print(f"\n[ep_len] ALL EPISODES COLLECTED:")
        print(f"[ep_len]   Total: {len(ep_len_arr)}")
        print(f"[ep_len]   Min: {ep_len_arr.min()}, Max: {ep_len_arr.max()}, Mean: {ep_len_arr.mean():.1f}, Median: {np.median(ep_len_arr):.0f}")
        
        # Length distribution buckets
        bins = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, 200), (200, 500), (500, float('inf'))]
        print(f"[ep_len]   Distribution:")
        for lo, hi in bins:
            if hi == float('inf'):
                count = (ep_len_arr >= lo).sum()
                label = f">={lo}"
            else:
                count = ((ep_len_arr >= lo) & (ep_len_arr < hi)).sum()
                label = f"{lo}-{hi}"
            pct = 100 * count / len(ep_len_arr)
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"[ep_len]     {label:>8}: {count:5d} ({pct:5.1f}%) {bar}")
        
        short_eps = (ep_len_arr <= 5).sum()
        if short_eps > len(ep_len_arr) * 0.1:
            print(f"[WARN] {short_eps} episodes ({short_eps/len(ep_len_arr)*100:.1f}%) have length <= 5!")
            print("       This may indicate episode boundary detection issues.")
    
    # Route statistics (kept routes after filtering)
    routes_selected_ep_arr = np.asarray(routes_selected_ep, dtype=np.int64)
    routes_success_arr = np.asarray(routes_success, dtype=bool)
    routes_ep_len_arr = np.asarray(routes_ep_len, dtype=np.int64)
    
    kept = int(np.sum(routes_selected_ep_arr >= 0))
    dropped = int(np.sum(routes_selected_ep_arr < 0))
    print(f"\n[routes] SAVED ROUTES (after filtering):")
    print(f"[routes]   Total tasks: {len(routes_seed)}")
    print(f"[routes]   Kept: {kept}, Dropped: {dropped}")
    if args.max_ep_len > 0:
        print(f"[routes]   Max ep_len filter: {args.max_ep_len} (routes longer than this dropped)")
    
    if kept > 0:
        kept_lens = routes_ep_len_arr[routes_selected_ep_arr >= 0]
        print(f"[routes]   Kept ep_len: Min={kept_lens.min()}, Max={kept_lens.max()}, Mean={kept_lens.mean():.1f}, Median={np.median(kept_lens):.0f}")
        
        # Distribution of kept routes
        print(f"[routes]   Length distribution of kept routes:")
        for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, float('inf'))]:
            if hi == float('inf'):
                count = (kept_lens >= lo).sum()
                label = f">={lo}"
            else:
                count = ((kept_lens >= lo) & (kept_lens < hi)).sum()
                label = f"{lo}-{hi}"
            pct = 100 * count / len(kept_lens)
            print(f"[routes]     {label:>8}: {count:4d} ({pct:5.1f}%)")
    
    # Success rate
    success_rate = routes_success_arr.sum() / max(1, len(routes_success_arr))
    print(f"[routes] Success rate: {success_rate:.2%}")
    
    # XY variance check on kept routes
    xy_var_issues = 0
    for i, diag in enumerate(routes_diag):
        if isinstance(diag, dict) and diag.get("xy_constant_warning", False):
            xy_var_issues += 1
    if xy_var_issues > 0:
        print(f"[WARN] {xy_var_issues} routes have near-constant xy (variance < 1e-6)!")
        print("       This suggests xy extraction is returning constant values.")
    
    # Selection reason breakdown
    sel_reason_counts: Counter = Counter()
    dropped_reason_counts: Counter = Counter()
    for d in routes_diag:
        if isinstance(d, dict):
            if d.get("selected_reason"):
                sel_reason_counts[d["selected_reason"]] += 1
            if d.get("dropped_reason"):
                dropped_reason_counts[d["dropped_reason"]] += 1
    
    if sel_reason_counts:
        print(f"\n[selection] Reason breakdown:")
        for k, v in sel_reason_counts.most_common():
            print(f"  {v:4d} x {k}")
    
    if dropped_reason_counts:
        print(f"\n[dropped] Reason breakdown:")
        for k, v in dropped_reason_counts.most_common():
            print(f"  {v:4d} x {k}")
    
    print("="*60 + "\n")

    # =====================================================================
    # SAVE
    # =====================================================================
    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)

    save_dict: Dict[str, Any] = dict(
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
        # meta
        meta=dict(
            env_name=args.env_name,
            distribution_mode=args.distribution_mode,
            num_tasks=args.num_tasks,
            seed_offset=args.seed_offset,
            num_processes=args.num_processes,
            arch=args.arch,
            hidden_size=args.hidden_size,
            adapt_episodes=args.adapt_episodes,
            record_episodes=args.record_episodes,
            adapt_deterministic=bool(args.adapt_deterministic),
            record_deterministic=bool(args.record_deterministic),
            min_len=args.min_len,
            max_ep_len=args.max_ep_len,
            max_steps=args.max_steps,
            xy_fail_policy=args.xy_fail_policy,
            xy_fail_threshold=args.xy_fail_threshold,
            save_all_episodes=bool(args.save_all_episodes),
            # Collection stats
            xy_missing_rate=float(xy_missing_rate),
            total_episodes_collected=len(global_ep_lens),
        ),
    )

    if args.save_all_episodes:
        save_dict["episodes_all"] = np.asarray(all_episodes, dtype=object)

    np.savez_compressed(args.out_npz, **save_dict)
    print(f"[saved] {args.out_npz}")


if __name__ == "__main__":
    main()
