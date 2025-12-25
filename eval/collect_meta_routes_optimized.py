#!/usr/bin/env python3
"""
Faster Meta-RL Route Collection (Persistent Envs + Opt Data Path)
-----------------------------------------------------------------

Improvements over `collect_meta_routes_fast.py`:
1. Persistent SubprocVecEnv:
   - Reuses worker processes across batches (avoids fork/spawn overhead).
   - Uses `reset_task(seed)` to switch levels in-place.
2. Optimized Data Path:
   - Adaptation episodes: Obs go uint8->GPU->float directly (skips CPU float16/storage copies).
   - Record episodes: Only then do we convert to float16 and store in RAM.
   - XY/Info: Strictly computed only during record episodes.
3. Custom FasterSubprocVecEnv:
   - Implements `reset_task` command for workers.
   - Handles dict observations correctly (via _flatten_obs).

"""

import os
import sys
import time
import struct
import argparse
import warnings
import contextlib
import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool

import torch

# Multiprocessing imports for Custom VecEnv
from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
os.environ.setdefault("GYM_LOGGER_LEVEL", "error")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

_gym_import_stderr = io.StringIO()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        with contextlib.redirect_stderr(_gym_import_stderr):
            import gym
    except Exception:
        sys.stderr.write(_gym_import_stderr.getvalue())
        raise
    gym.logger.set_level(40)

from tqdm import tqdm

# Add project root
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.common.vec_env import VecExtractDictObs, VecMonitor
from level_replay.envs import VecNormalize
from level_replay.model import Policy, SimplePolicy

try:
    from routes_ckpt_storage import (
        CheckpointManifest,
        RoutesShardWriter,
        build_routes_npz_from_ckpt,
    )
    CKPT_AVAILABLE = True
except ImportError:
    CKPT_AVAILABLE = False

PROCGEN_TOOLS_PATH = "/root/test/procgen-tools-main"
if PROCGEN_TOOLS_PATH not in sys.path:
    sys.path.append(PROCGEN_TOOLS_PATH)

try:
    import procgen  # noqa: F401
except Exception:
    pass

try:
    from procgen_tools import maze as procgen_state_tools
except Exception as e:
    print(f"Warning: could not import procgen_tools.maze: {e}")
    procgen_state_tools = None


# -----------------------------------------------------------------------------
# Fast XY Extraction (Shared with fast.py)
# -----------------------------------------------------------------------------

def _unwrap_to_gym3_like(env):
    inner = env
    for _ in range(64):
        if hasattr(inner, "callmethod"):
            return inner
        if hasattr(inner, "env"):
            inner = inner.env
        else:
            break
    return inner

def _fast_xy_from_state_bytes(state_bytes: bytes) -> Tuple[float, float]:
    if procgen_state_tools is None:
        return (float("nan"), float("nan"))
    try:
        tmpl = procgen_state_tools.MAZE_STATE_DICT_TEMPLATE
    except Exception:
        return (float("nan"), float("nan"))

    try:
        sb = state_bytes
        if isinstance(sb, memoryview):
            sb = sb.tobytes()
        idx = 0
        unpack_i = struct.unpack_from
        unpack_f = struct.unpack_from
        ents_size = None

        for val_def in tmpl:
            typ = val_def[0]
            name = val_def[1]

            if typ == "int":
                v = unpack_i("@i", sb, idx)[0]
                idx += 4
                if name == "ents.size":
                    ents_size = int(v)
            elif typ == "float":
                idx += 4
            elif typ == "string":
                sz = unpack_i("@i", sb, idx)[0]
                idx += 4 + int(sz)
            elif typ == "loop":
                if name == "ents":
                    if ents_size is None or ents_size <= 0:
                        return (float("nan"), float("nan"))
                    x = float(unpack_f("@f", sb, idx)[0])
                    y = float(unpack_f("@f", sb, idx + 4)[0])
                    return (x, y)
                return (float("nan"), float("nan"))
        return (float("nan"), float("nan"))
    except Exception:
        return (float("nan"), float("nan"))

def _extract_xy_from_env(env) -> Tuple[float, float]:
    # This wrapper logic is slightly duplicated but keeps file standalone
    if procgen_state_tools is None:
        return (float("nan"), float("nan"))
    try:
        inner = _unwrap_to_gym3_like(env)
        if not hasattr(inner, "callmethod"):
            return (float("nan"), float("nan"))
        state_bytes_list = inner.callmethod("get_state")
        if not state_bytes_list:
            return (float("nan"), float("nan"))
        return _fast_xy_from_state_bytes(state_bytes_list[0])
    except Exception:
        return (float("nan"), float("nan"))


# -----------------------------------------------------------------------------
# Persistent Env Wrapper & Worker
# -----------------------------------------------------------------------------

class PersistentTaskWrapper(gym.Wrapper):
    """
    Wraps a Procgen env but allows switching tasks (levels) in-place
    without destroying the process.
    Also handles XY extraction.
    """
    def __init__(self, env_name: str, distribution_mode: str, adapt_episodes: int):
        self.env_name = env_name
        self.distribution_mode = distribution_mode
        self.adapt_episodes = adapt_episodes
        self.current_seed = -1
        
        # Create initial dummy env to satisfy gym.Wrapper
        dummy = self._create_inner(0)
        super().__init__(dummy)
        
        self._last_xy = (float("nan"), float("nan"))
        self._episode_idx = 0

    def _create_inner(self, seed: int):
        # Create the actual gym env
        env = gym.make(
            f"procgen:procgen-{self.env_name}-v0",
            start_level=int(seed),
            num_levels=1,
            distribution_mode=self.distribution_mode,
        )
        return env

    def reset_task(self, seed: int):
        """Called via env_method to switch task."""
        if hasattr(self.env, "close"):
            self.env.close()
        
        self.env = self._create_inner(seed)
        self.current_seed = seed
        self._episode_idx = 0
        self._last_xy = (float("nan"), float("nan"))
        
        # Reset new env
        obs = self.env.reset()
        
        # Init XY if needed (usually not needed at step 0 of adapt)
        if self._tracking_enabled():
            self._last_xy = _extract_xy_from_env(self.env)
            
        return obs

    def _tracking_enabled(self) -> bool:
        return self._episode_idx >= self.adapt_episodes

    def reset(self, **kwargs):
        # Normal reset (e.g. auto-reset by VecEnv or manual)
        obs = self.env.reset(**kwargs)
        if self._tracking_enabled():
            self._last_xy = _extract_xy_from_env(self.env)
        else:
            self._last_xy = (float("nan"), float("nan"))
        return obs

    def step(self, action):
        tracking_now = self._tracking_enabled()
        xy_before = self._last_xy
        
        obs, reward, done, info = self.env.step(action)
        if info is None:
            info = {}
            
        if done:
            info["terminal_xy"] = xy_before
            self._episode_idx += 1
            
            if self._tracking_enabled():
                xy_after = _extract_xy_from_env(self.env)
                info["xy"] = xy_after
                self._last_xy = xy_after
            else:
                info["xy"] = (float("nan"), float("nan"))
                self._last_xy = (float("nan"), float("nan"))
        else:
            if tracking_now:
                xy = _extract_xy_from_env(self.env)
                info["xy"] = xy
                self._last_xy = xy
            else:
                info["xy"] = (float("nan"), float("nan"))
                self._last_xy = (float("nan"), float("nan"))
            info["terminal_xy"] = None
            
        return obs, reward, done, info

    def get_xy(self):
        return self._last_xy


def make_persistent_env_fn(env_name: str, distribution_mode: str, adapt_episodes: int):
    def _thunk():
        _suppress_all_warnings()
        import gym as _gym
        _gym.logger.set_level(40)
        return PersistentTaskWrapper(env_name, distribution_mode, adapt_episodes)
    return _thunk


# -----------------------------------------------------------------------------
# Custom SubprocVecEnv
# -----------------------------------------------------------------------------

def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0
    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'reset':
                ob = env.reset()
                remote.send(ob)
            elif cmd == 'reset_task':
                ob = env.reset_task(data)
                remote.send(ob)
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError(f"Unknown command {cmd}")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"FasterSubprocVecEnv worker error: {e}")
    finally:
        env.close()

class FasterSubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True 
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, nenvs, observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        return _flatten_obs(results)

    def reset_task_batch(self, seeds: List[int]):
        # Only reset the first len(seeds) envs
        for i, seed in enumerate(seeds):
            if i >= len(self.remotes): break
            self.remotes[i].send(('reset_task', seed))
        
        # We don't necessarily need to return the initial obs here if we call reset() later
        # but the protocol waits for response.
        results = []
        for i in range(len(seeds)):
            if i >= len(self.remotes): break
            results.append(self.remotes[i].recv())
        return results # List of obs, not stacked

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


def _suppress_all_warnings():
    import warnings as _warnings
    _warnings.filterwarnings("ignore")
    _warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    _warnings.warn = lambda *args, **kwargs: None


def create_persistent_vec_env(env_name: str, num_processes: int, distribution_mode: str, adapt_episodes: int):
    # Create env fns
    env_fns = [make_persistent_env_fn(env_name, distribution_mode, adapt_episodes) for _ in range(num_processes)]
    
    # Use FasterSubprocVecEnv
    venv = FasterSubprocVecEnv(env_fns)
    
    if isinstance(venv.observation_space, gym.spaces.Dict):
        venv = VecExtractDictObs(venv, "rgb")
    
    venv = VecMonitor(venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv, ob=False, ret=False)
    return venv


# -----------------------------------------------------------------------------
# Data & Inference Helpers
# -----------------------------------------------------------------------------

def _obs_to_chw_uint8(obs: np.ndarray) -> np.ndarray:
    if obs.ndim != 4:
        raise ValueError(f"Unexpected obs shape: {obs.shape}")
    if obs.shape[1] == 3:
        return obs.astype(np.uint8, copy=False)
    if obs.shape[-1] == 3:
        return obs.transpose(0, 3, 1, 2).astype(np.uint8, copy=False)
    raise ValueError(f"Unexpected obs shape: {obs.shape}")

def _prep_obs_for_storage(obs_chw_u8: np.ndarray) -> np.ndarray:
    # CPU: uint8 -> float32 -> float16 (0-1)
    x = obs_chw_u8.astype(np.float32) / 255.0
    return x.astype(np.float16, copy=False)

def _prep_obs_for_model(obs_f16: np.ndarray, device: torch.device) -> torch.Tensor:
    # CPU float16 -> GPU float16/32
    if device.type == "cuda":
        return torch.from_numpy(obs_f16).to(device, non_blocking=True)
    return torch.from_numpy(obs_f16.astype(np.float32)).to(device)

def _prep_obs_direct_to_gpu(obs_chw_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    # FAST PATH: CPU uint8 -> GPU uint8 -> GPU float (0-1)
    # Skips CPU float conversion and host-device float transfer
    t = torch.from_numpy(obs_chw_u8).to(device, non_blocking=True) # Transfer uint8
    return t.float().div_(255.0) # Convert to float on GPU

@dataclass
class EpisodeBuffer:
    # Optional storage - only populated if needed
    obs: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    xy: List[np.ndarray] = field(default_factory=list)
    
    info_level_complete: bool = False
    # Track whether env ever provided a `level_complete` key (even if False).
    # Useful for envs where positive reward can occur without true completion.
    level_complete_key_seen: bool = False
    ret: float = 0.0
    length: int = 0
    xy_missing_count: int = 0
    
    def finalize(self, success_policy: str = "legacy") -> Dict[str, Any]:
        xy_missing_frac = self.xy_missing_count / max(1, self.length)
        
        # If lists are empty, return empty arrays
        obs_arr = np.stack(self.obs, axis=0) if self.obs else np.empty((0, 3, 64, 64), dtype=np.float16)
        xy_arr = np.stack(self.xy, axis=0) if self.xy else np.empty((0, 2), dtype=np.float32)
        actions_arr = np.asarray(self.actions, dtype=np.int64) if self.actions else np.empty((0,), dtype=np.int64)
        rewards_arr = np.asarray(self.rewards, dtype=np.float32) if self.rewards else np.empty((0,), dtype=np.float32)

        # Success semantics:
        # - legacy: success := level_complete OR (return > 0)
        # - prefer_level_complete: if env provides `level_complete` key, use it exclusively;
        #   otherwise fall back to (return > 0).
        if success_policy == "prefer_level_complete":
            success = bool(self.info_level_complete) if self.level_complete_key_seen else bool(self.ret > 0)
        else:
            success = bool(self.info_level_complete or (self.ret > 0))
        
        return {
            "obs": obs_arr,
            "actions": actions_arr,
            "rewards": rewards_arr,
            "xy": xy_arr,
            "success": bool(success),
            "level_complete": bool(self.info_level_complete),
            "return": float(self.ret),
            "len": int(self.length),
            "xy_missing_frac": float(xy_missing_frac),
        }


def select_best_episode(episodes, min_len=5, max_len=0, require_success=True):
    diag = {"num_episodes": len(episodes), "selected_reason": None}
    if not episodes:
        diag["selected_reason"] = "no_episodes"
        return None, diag
    
    valid = []
    for i, ep in enumerate(episodes):
        if require_success and not ep.get("success", False): continue
        l = int(ep.get("len", 0))
        if l < min_len: continue
        if max_len > 0 and l > max_len: continue
        valid.append(i)
        
    if not valid:
        diag["selected_reason"] = "none_valid"
        return None, diag
    
    best_idx = valid[-1]
    diag["selected_reason"] = f"valid_ep_{best_idx}"
    return best_idx, diag

def build_model(model_ckpt, arch, hidden_size, device, env_name, use_compile):
    dummy_env = gym.make(f"procgen:procgen-{env_name}-v0", start_level=0, num_levels=1)
    obs_shape = dummy_env.observation_space.shape
    obs_shape = (3, obs_shape[0], obs_shape[1])
    num_actions = dummy_env.action_space.n
    dummy_env.close()

    if arch == "simple":
        ac = SimplePolicy(obs_shape, num_actions)
    else:
        ac = Policy(obs_shape, num_actions, arch=arch, base_kwargs={"recurrent": True, "hidden_size": hidden_size})
    
    ac.to(device)
    ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
    if "state_dict" in ckpt: ac.load_state_dict(ckpt["state_dict"])
    elif "model_state_dict" in ckpt: ac.load_state_dict(ckpt["model_state_dict"])
    else: ac.load_state_dict(ckpt)
    ac.eval()
    
    if use_compile and hasattr(torch, "compile") and device.type == "cuda":
        try:
            ac = torch.compile(ac, mode="reduce-overhead")
            print("[perf] Model compiled")
        except:
            pass
    return ac, obs_shape, num_actions

class FastInference:
    def __init__(self, actor_critic, device, use_amp):
        self.ac = actor_critic
        self.device = device
        self.use_amp = bool(use_amp) and device.type == "cuda"

    @torch.inference_mode()
    def act_batched(self, obs, rnn_hxs, masks, deterministic):
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                _, actions, _, rnn_hxs_new = self.ac.act(obs, rnn_hxs, masks, deterministic=deterministic)
            return actions, rnn_hxs_new.float()
        _, actions, _, rnn_hxs_new = self.ac.act(obs, rnn_hxs, masks, deterministic=deterministic)
        return actions, rnn_hxs_new

class LiveStats:
    def __init__(self, target):
        self.target = target
        self.collected = 0
        self.start = time.perf_counter()
        self.total_steps = 0
        self.aborted = 0
        self.timed_out = 0
    
    def update(self, n, steps, aborted=0, timed_out=0):
        self.collected += n
        self.total_steps += steps
        self.aborted += aborted
        self.timed_out += timed_out
        
    @property
    def elapsed(self):
        return time.perf_counter() - self.start

    @property
    def speed(self):
        return self.collected / max(0.01, self.elapsed)
    
    @property
    def steps_per_sec(self):
        return self.total_steps / max(0.01, time.perf_counter() - self.start)
        
    def get_desc(self):
        return f"{self.collected}/{self.target} {self.speed:.1f}/s {self.steps_per_sec/1000:.1f}k stp/s"
    
    @staticmethod
    def format_time(s):
        m, s = divmod(int(s), 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"


# -----------------------------------------------------------------------------
# Main Collection Loop
# -----------------------------------------------------------------------------

def run_batch_collection_v2(envs, actor_critic, batch_seeds, args, device, fast_inf, num_processes):
    batch_size = len(batch_seeds)
    total_episodes = args.adapt_episodes + args.record_episodes
    completion_threshold = float(getattr(args, "batch_completion_threshold", 0.9))
    task_timeout_steps = int(getattr(args, "task_timeout", args.max_steps * total_episodes))
    early_abort = bool(int(getattr(args, "early_abort", 1)))

    stats = {
        "time_inference": 0.0, "time_env_step": 0.0, "time_obs_prep": 0.0,
        "total_steps": 0, "aborted_early": 0, "timed_out": 0,
        "successful_tasks": 0, "failed_tasks": 0,
    }

    rnn_hxs = torch.zeros(num_processes, args.hidden_size, device=device)
    masks = torch.zeros(num_processes, 1, device=device)
    masks_ones = torch.ones(num_processes, 1, device=device)

    # Force a reset to ensure all wrappers are aligned for new tasks
    # This also returns the initial obs for the new tasks.
    # Note: reset_task_batch already seeded the inner envs.
    obs = envs.reset()
    
    t0 = time.perf_counter()
    obs_chw_u8 = _obs_to_chw_uint8(obs)
    stats["time_obs_prep"] += time.perf_counter() - t0

    ep_counts = [0] * batch_size
    ep_bufs = [EpisodeBuffer() for _ in range(batch_size)]
    ep_lists = [[] for _ in range(batch_size)]
    env_done = [False] * batch_size
    env_has_success = [False] * batch_size # Tracks success across ALL episodes (adapt+record)
    task_total_steps = [0] * batch_size

    # Handle Initial Frame Storage (only if recording immediately)
    for i in range(batch_size):
        if args.adapt_episodes == 0:
            # We are recording immediately
            # Need float16
            f16 = _prep_obs_for_storage(obs_chw_u8[i:i+1])
            ep_bufs[i].obs.append(f16[0])

    step_count = 0
    max_batch_steps = args.max_steps * total_episodes * batch_size

    while True:
        step_count += 1
        done_count = sum(env_done)
        active_count = batch_size - done_count
        if done_count >= batch_size * completion_threshold:
            for i in range(batch_size):
                if not env_done[i]:
                    env_done[i] = True
                    stats["aborted_early"] += 1
            break
        if step_count > max_batch_steps or active_count == 0:
            break

        in_record_all = all(ep_counts[i] >= args.adapt_episodes for i in range(batch_size) if not env_done[i])
        deterministic = bool(args.record_deterministic) if in_record_all else bool(args.adapt_deterministic)

        # 1. Obs -> GPU
        t0 = time.perf_counter()
        obs_t = _prep_obs_direct_to_gpu(obs_chw_u8, device)
        stats["time_obs_prep"] += time.perf_counter() - t0

        # 2. Infer
        t0 = time.perf_counter()
        actions_t, rnn_hxs = fast_inf.act_batched(obs_t, rnn_hxs, masks, deterministic=deterministic)
        stats["time_inference"] += time.perf_counter() - t0

        # 3. Step
        t0 = time.perf_counter()
        actions_np = actions_t.squeeze(1).detach().cpu().numpy()
        obs, reward, done, infos = envs.step(actions_np)
        stats["time_env_step"] += time.perf_counter() - t0

        masks = masks_ones.clone()
        reward_np = np.asarray(reward).ravel()
        done_np = np.asarray(done, dtype=bool).ravel()

        # 4. Next Obs Prep (for storage + next infer)
        t0 = time.perf_counter()
        obs_chw_u8 = _obs_to_chw_uint8(obs)
        stats["time_obs_prep"] += time.perf_counter() - t0

        # 5. Process Results
        for i in range(batch_size):
            if env_done[i]: continue
            
            task_total_steps[i] += 1
            is_recording = (ep_counts[i] >= args.adapt_episodes)
            
            # --- TRACKING ---
            # We accumulate return/length even in adapt to check success
            ep_bufs[i].ret += float(reward_np[i])
            ep_bufs[i].length += 1
            
            info_i = infos[i] if i < len(infos) else {}
            if isinstance(info_i, dict):
                if "level_complete" in info_i:
                    ep_bufs[i].level_complete_key_seen = True
                if info_i.get("level_complete", False):
                    ep_bufs[i].info_level_complete = True

            # Storage (Obs/Act/Rew/XY) - ONLY if recording
            if is_recording:
                ep_bufs[i].actions.append(int(actions_np[i]))
                ep_bufs[i].rewards.append(float(reward_np[i]))
                
                # XY
                if done_np[i]:
                    term_xy = info_i.get("terminal_xy", None) if isinstance(info_i, dict) else None
                    xy = term_xy if term_xy is not None else (float("nan"), float("nan"))
                else:
                    xy = info_i.get("xy", (float("nan"), float("nan"))) if isinstance(info_i, dict) else (float("nan"), float("nan"))
                
                if np.isnan(xy[0]) or np.isnan(xy[1]):
                    ep_bufs[i].xy_missing_count += 1
                ep_bufs[i].xy.append(np.asarray([xy[0], xy[1]], dtype=np.float32))

            # Check Termination
            is_done = done_np[i]
            timed_out = task_total_steps[i] >= task_timeout_steps
            max_len_reached = ep_bufs[i].length >= args.max_steps
            
            # Count steps for stats
            stats["total_steps"] += 1
            
            should_finalize = is_done or timed_out or max_len_reached
            
            if should_finalize:
                # Check success (valid in adapt too). Some envs (e.g. caveflyer) can
                # yield positive reward without reaching the exit, so optionally
                # prefer the `level_complete` signal if it exists.
                if getattr(args, "success_policy", "legacy") == "prefer_level_complete":
                    success = bool(ep_bufs[i].info_level_complete) if ep_bufs[i].level_complete_key_seen else bool(ep_bufs[i].ret > 0)
                else:
                    success = bool(ep_bufs[i].info_level_complete or (ep_bufs[i].ret > 0))
                if success:
                    env_has_success[i] = True
                
                if is_recording:
                    # Finalize and store
                    ep_data = ep_bufs[i].finalize(success_policy=str(getattr(args, "success_policy", "legacy")))
                    # Add missing fields if not stored? finalize handles empty lists.
                    ep_lists[i].append(ep_data)
                
                ep_counts[i] += 1
                
                # Check Task Termination
                if ep_counts[i] >= total_episodes:
                    env_done[i] = True
                elif timed_out:
                    env_done[i] = True
                    stats["timed_out"] += 1
                elif early_abort and ep_counts[i] >= args.adapt_episodes and not env_has_success[i]:
                    env_done[i] = True
                    stats["aborted_early"] += 1
                
                # Reset Buffer
                ep_bufs[i] = EpisodeBuffer()
                
                # Store Next Frame (Initial Obs of Next Ep) - ONLY if recording next ep
                if not env_done[i]:
                    next_is_recording = (ep_counts[i] >= args.adapt_episodes)
                    if next_is_recording:
                        f16 = _prep_obs_for_storage(obs_chw_u8[i:i+1])
                        ep_bufs[i].obs.append(f16[0])
            
            else:
                # Continue Episode
                if is_recording:
                    # Store current frame
                    f16 = _prep_obs_for_storage(obs_chw_u8[i:i+1])
                    ep_bufs[i].obs.append(f16[0])

    # Count
    for i in range(batch_size):
        if env_has_success[i]: stats["successful_tasks"] += 1
        else: stats["failed_tasks"] += 1

    results = [(batch_seeds[i], ep_lists[i], env_has_success[i]) for i in range(batch_size)]
    return results, stats


def main():
    p = argparse.ArgumentParser("collect_meta_routes_faster")
    # Copied args from fast.py
    p.add_argument("--model_ckpt", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--env_name", type=str, default="maze")
    p.add_argument("--distribution_mode", type=str, default="easy")
    p.add_argument("--num_tasks", type=int, default=200)
    p.add_argument("--seed_offset", type=int, default=0)
    p.add_argument("--num_processes", type=int, default=64)
    p.add_argument("--max_seed_attempts", type=int, default=0)
    p.add_argument("--arch", type=str, default="large")
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--adapt_episodes", type=int, default=5)
    p.add_argument("--record_episodes", type=int, default=2)
    p.add_argument("--adapt_deterministic", type=int, default=0)
    p.add_argument("--record_deterministic", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=512)
    p.add_argument("--min_len", type=int, default=5)
    p.add_argument("--max_ep_len", type=int, default=0)
    p.add_argument("--require_success", type=int, default=1)
    p.add_argument("--batch_completion_threshold", type=float, default=0.9)
    p.add_argument("--task_timeout", type=int, default=0)
    p.add_argument("--early_abort", type=int, default=1)
    p.add_argument("--xy_fail_policy", type=str, default="warn_only")
    p.add_argument("--xy_fail_threshold", type=float, default=0.5)
    p.add_argument(
        "--success_policy",
        type=str,
        default="legacy",
        choices=["legacy", "prefer_level_complete"],
        help=(
            "How to define per-episode success used for filtering/selection. "
            "'legacy' = level_complete OR (return > 0). "
            "'prefer_level_complete' = if env provides `level_complete` key, use it exclusively; "
            "otherwise fall back to (return > 0). "
            "Recommended for procgen envs where positive reward can occur without finishing "
            "the level (e.g., caveflyer with target rewards)."
        ),
    )
    p.add_argument("--save_all_episodes", type=int, default=0)
    p.add_argument("--use_compile", type=int, default=1)
    p.add_argument("--use_amp", type=int, default=1)
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume_force", action="store_true")
    p.add_argument("--ckpt_shard_size", type=int, default=25)
    p.add_argument("--ckpt_flush_secs", type=float, default=60.0)

    args = p.parse_args()
    
    # Defaults
    if args.max_seed_attempts <= 0: args.max_seed_attempts = args.num_tasks * 10
    if args.task_timeout <= 0: args.task_timeout = args.max_steps * (args.adapt_episodes + args.record_episodes)

    # -------------------------------------------------------------------------
    # Checkpointing (match the contract/metadata of collect_meta_routes.py)
    # -------------------------------------------------------------------------
    use_ckpt = args.ckpt_dir is not None
    if use_ckpt and not CKPT_AVAILABLE:
        raise RuntimeError("Checkpointing requested but routes_ckpt_storage not available")

    ckpt_writer, manifest = None, None
    if use_ckpt:
        manifest = CheckpointManifest.load(args.ckpt_dir)
        if manifest is None:
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
            ok, msg = manifest.validate_resume(args)
            if not ok:
                if args.resume_force:
                    print(f"[WARNING] Config mismatch (--resume_force):\n{msg}")
                else:
                    raise ValueError(f"Cannot resume: {msg}\nUse --resume_force to override")
            manifest.num_tasks_target = args.num_tasks
            manifest.out_npz = args.out_npz
            manifest.save(args.ckpt_dir)
        else:
            raise ValueError(
                f"Checkpoint directory exists but --resume not specified: {args.ckpt_dir}"
            )

        ckpt_writer = RoutesShardWriter(
            args.ckpt_dir,
            shard_size=args.ckpt_shard_size,
            flush_secs=args.ckpt_flush_secs,
            manifest=manifest,
        )

        if args.resume:
            print("[resume] Continuing from checkpoint:")
            print(f"  Collected: {manifest.num_tasks_collected}/{manifest.num_tasks_target}")
            print(f"  Current seed: {manifest.current_seed}")
            print(f"  Seeds attempted: {manifest.seeds_attempted}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    print("=" * 60)
    print("META-RL ROUTE COLLECTION (FASTER: Persistent + Opt Data)")
    print("=" * 60)
    print(f"Device: {device}")
    
    # Model
    actor_critic, obs_shape, num_actions = build_model(
        args.model_ckpt, args.arch, args.hidden_size, device, args.env_name, bool(args.use_compile)
    )

    # PERSISTENT ENV CREATION
    # We create them ONCE.
    print(f"Creating {args.num_processes} persistent env processes...", end="", flush=True)
    envs = create_persistent_vec_env(args.env_name, args.num_processes, args.distribution_mode, args.adapt_episodes)
    print(" done.")

    # Loop state
    if use_ckpt and manifest and args.resume:
        current_seed = manifest.current_seed
        seeds_tried = manifest.seeds_attempted
        initial_collected = manifest.num_tasks_collected
    else:
        current_seed = args.seed_offset
        seeds_tried = 0
        initial_collected = 0

    # Output lists
    routes_seed, routes_obs, routes_xy, routes_actions = [], [], [], []
    routes_rewards, routes_ep_len, routes_return, routes_success = [], [], [], []
    routes_diag, routes_selected_ep = [], []
    
    live = LiveStats(args.num_tasks)
    pbar = tqdm(total=args.num_tasks, initial=initial_collected, desc=live.get_desc(), unit="traj", ncols=120)

    try:
        while True:
            cur_coll = manifest.num_tasks_collected if use_ckpt else len(routes_seed)
            if cur_coll >= args.num_tasks or seeds_tried >= args.max_seed_attempts:
                break
            
            batch_size = min(args.num_processes, args.max_seed_attempts - seeds_tried)
            if batch_size == 0: break
            
            batch_seeds = list(range(current_seed, current_seed + batch_size))
            current_seed += batch_size
            seeds_tried += batch_size
            
            batch_start = time.perf_counter()
            
            # RECONFIGURE ENVS
            # This is the "Persistent" magic. We reset tasks in existing workers.
            # We only reset the first 'batch_size' workers.
            # print(f"\rConfiguring batch {batch_seeds[0]}...", end="", flush=True)
            envs.reset_task_batch(batch_seeds)
            
            # RUN
            fast_inf = FastInference(actor_critic, device=device, use_amp=args.use_amp)
            results, st = run_batch_collection_v2(
                envs, actor_critic, batch_seeds, args, device, fast_inf, args.num_processes
            )
            
            batch_time = time.perf_counter() - batch_start
            
            # STORE RESULTS
            batch_suc = 0
            for seed, episodes, has_success in results:
                if not has_success and args.require_success: continue
                
                sel_idx, diag = select_best_episode(episodes, args.min_len, args.max_ep_len, bool(args.require_success))
                if sel_idx is None: continue
                
                ep = episodes[sel_idx]
                
                # Verify XY presence
                xy_frac = float(ep.get("xy_missing_frac", 1.0))
                if args.xy_fail_policy == "drop_task" and xy_frac > args.xy_fail_threshold: continue

                if use_ckpt:
                    ckpt_writer.append_route(
                        seed=int(seed), selected_ep=int(sel_idx),
                        obs=ep["obs"], actions=ep["actions"], xy=ep["xy"],
                        rewards=ep["rewards"], ep_len=int(ep["len"]), ep_return=float(ep["return"]),
                        success=bool(ep["success"]), diag=diag,
                        # Fill optional args with empty
                        player_v=np.empty((0,2)), ents_count=np.empty((0,)), nearest_ents=np.empty((0,8,4)),
                        all_episodes=None
                    )
                    ckpt_writer.update_progress(current_seed, seeds_tried)
                else:
                    routes_seed.append(int(seed))
                    routes_obs.append(ep["obs"])
                    routes_actions.append(ep["actions"])
                    routes_xy.append(ep["xy"])
                    routes_rewards.append(ep["rewards"])
                    routes_ep_len.append(int(ep["len"]))
                    routes_return.append(float(ep["return"]))
                    routes_success.append(bool(ep["success"]))
                    routes_diag.append(diag)
                    routes_selected_ep.append(int(sel_idx))
                
                batch_suc += 1
                pbar.update(1)
                
                if (use_ckpt and manifest.num_tasks_collected >= args.num_tasks) or \
                   (not use_ckpt and len(routes_seed) >= args.num_tasks):
                    break
            
            live.update(batch_suc, int(st.get("total_steps", 0)), 
                        aborted=int(st.get("aborted_early", 0)), 
                        timed_out=int(st.get("timed_out", 0)))
            pbar.set_description(live.get_desc())
            
            # Log Perf
            ti, te, to = st["time_inference"], st["time_env_step"], st["time_obs_prep"]
            print(f" done! +{batch_suc} ({batch_time:.1f}s) | env {te:.2f}s, inf {ti:.2f}s, obs {to:.2f}s", flush=True)

    finally:
        print("\nClosing envs...")
        envs.close()
        if use_ckpt and ckpt_writer:
            ckpt_writer.close()
    
    print(f"Total time:        {LiveStats.format_time(live.elapsed)}")
    print(f"Speed:             {live.speed:.2f} traj/s, {live.steps_per_sec:.0f} steps/s")
    print(f"Early aborted:     {live.aborted}")
    print(f"Timed out:         {live.timed_out}")
    print("=" * 60)

    # Save final NPZ (omitted detailed save logic for brevity, uses ckpt or simple save)
    if use_ckpt:
        print(f"Merging to {args.out_npz}...")
        build_routes_npz_from_ckpt(args.ckpt_dir, args.out_npz)
    else:
        # Minimal save
        np.savez_compressed(
            args.out_npz,
            routes_seed=routes_seed, routes_obs=routes_obs, routes_actions=routes_actions,
            routes_xy=routes_xy, routes_rewards=routes_rewards, routes_success=routes_success
        )
    print(f"Done. Saved to {args.out_npz}")

if __name__ == "__main__":
    main()
