#!/usr/bin/env python3
"""
Fast Meta-RL Route Collection (Procgen / CoinRun)
-------------------------------------------------

This script is intentionally a "new layer" on top of the existing
`eval/collect_meta_routes.py` (which we DO NOT modify). It keeps the same output
contract (routes.npz / checkpoint shards) but optimizes the hot path.

Key optimizations vs `collect_meta_routes.py`:
  1) Avoid per-step GPU->CPU observation copies:
     - We keep the VecEnv on CPU (numpy obs), normalize/transpose on CPU once,
       then move ONLY the normalized tensor to GPU for inference.
     - This removes the old ping-pong: env->GPU (VecPyTorchProcgen) then GPU->CPU
       (collector storing obs).

  2) Reduce `info` payload size across SubprocVecEnv:
     - We only attach XY (and terminal XY) to `info`.
     - We do NOT attach extra per-step state arrays (nearest_ents, etc.).

Notes:
  - We still need per-step XY (for CCA ridge embedding), so we still call procgen
    `get_state()` inside workers, but we keep the payload minimal.
  - Output `routes_obs` remains (T, C, H, W) float16 in [0,1].
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
    np.bool = bool  # legacy compat

import torch

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Suppress noisy warnings before importing gym
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

# Add project root to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from multiprocessing import Process, Pipe
from baselines.common.vec_env import VecEnv, CloudpickleWrapper, VecExtractDictObs, VecMonitor
from level_replay.envs import VecNormalize
from level_replay.model import Policy, SimplePolicy

# Optional checkpointing (reused from old collector)
try:
    from routes_ckpt_storage import (
        CheckpointManifest,
        RoutesShardWriter,
        build_routes_npz_from_ckpt,
    )
    CKPT_AVAILABLE = True
except ImportError:
    CKPT_AVAILABLE = False

# Procgen state parser (same approach as old collector)
PROCGEN_TOOLS_PATH = "/root/test/procgen-tools-main"
if PROCGEN_TOOLS_PATH not in sys.path:
    sys.path.append(PROCGEN_TOOLS_PATH)

try:
    import procgen  # noqa: F401
except Exception:
    pass

try:
    # Despite the module name, parser works across procgen games incl coinrun.
    from procgen_tools import maze as procgen_state_tools
except Exception as e:
    print(f"Warning: could not import procgen_tools.maze (state parser): {e}")
    procgen_state_tools = None


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
    """
    Fast-path XY extraction from procgen `get_state()` bytes.

    Why:
      `procgen_tools.maze.EnvState(...).state_vals` does a full recursive parse and
      deep-copies results (see procgen-tools). That is extremely slow per step.

    Strategy:
      Walk `procgen_tools.maze.MAZE_STATE_DICT_TEMPLATE` until we reach the `ents`
      loop, then decode ONLY the first entity's x/y (ents[0].x/y) and return.

    Falls back to NaNs if procgen-tools isn't available or parsing fails.
    """
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

        # Native (C) layout: procgen-tools uses struct formats "@i" and "@f".
        unpack_i = struct.unpack_from  # type: ignore[name-defined]
        unpack_f = struct.unpack_from  # type: ignore[name-defined]

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
                loop_name = name
                if loop_name == "ents":
                    if ents_size is None:
                        return (float("nan"), float("nan"))
                    if ents_size <= 0:
                        return (float("nan"), float("nan"))

                    # Entity template begins with ["float","x"], ["float","y"], ...
                    x = float(unpack_f("@f", sb, idx)[0])
                    y = float(unpack_f("@f", sb, idx + 4)[0])
                    return (x, y)

                # We never expect to hit other loops before `ents`, but keep
                # conservative behaviour if template changes.
                return (float("nan"), float("nan"))

        return (float("nan"), float("nan"))
    except Exception:
        return (float("nan"), float("nan"))


def _extract_xy_from_env(env) -> Tuple[float, float]:
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


class XYInfoWrapperLite(gym.Wrapper):
    """
    Lightweight XY wrapper:
      - Adds info["xy"] each step (x,y)
      - Adds info["terminal_xy"] on done (x,y) before auto-reset
    Does NOT add extra arrays (player_v/nearest_ents/etc) to keep IPC small.
    """

    def __init__(self, env, adapt_episodes: int = 0):
        super().__init__(env)
        self._last_xy = (float("nan"), float("nan"))
        self._adapt_episodes = int(adapt_episodes)
        self._episode_idx = 0  # increments when an episode ends (done=True)

    def _tracking_enabled(self) -> bool:
        # Only compute XY during record episodes (i.e., after adaptation episodes)
        return self._episode_idx >= self._adapt_episodes

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # Only pay get_state() cost at episode boundaries when we will track XY.
        if self._tracking_enabled():
            self._last_xy = _extract_xy_from_env(self.env)
        else:
            self._last_xy = (float("nan"), float("nan"))
        return obs

    def step(self, action):
        # IMPORTANT: action must remain a scalar for procgen/gym3.
        tracking_now = self._tracking_enabled()

        xy_before = self._last_xy
        obs, reward, done, info = self.env.step(action)
        if info is None:
            info = {}

        if done:
            # terminal_xy should correspond to the terminal state of this episode.
            # If we didn't compute XY during the episode, keep NaNs (fine for adapt eps).
            info["terminal_xy"] = xy_before

            # Move to next episode; procgen often auto-resets immediately.
            self._episode_idx += 1
            # For collectors we only need terminal_xy on done; avoid an extra get_state().
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


def _suppress_all_warnings():
    import warnings as _warnings
    _warnings.filterwarnings("ignore")
    _warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    _warnings.warn = lambda *args, **kwargs: None


# -----------------------------------------------------------------------------
# Persistent VecEnv (aggressive perf): reuse worker processes across batches.
# -----------------------------------------------------------------------------

class PersistentTaskWrapper(gym.Wrapper):
    """
    One worker process holds one Gym procgen env, but can switch tasks by recreating
    the underlying env in-place (no respawn of the worker process).
    """

    def __init__(self, env_name: str, distribution_mode: str, adapt_episodes: int):
        self._env_name = env_name
        self._distribution_mode = distribution_mode
        self._adapt_episodes = int(adapt_episodes)
        env = self._make_env(seed=0)
        super().__init__(env)

    def _make_env(self, seed: int):
        import gym as _gym
        _gym.logger.set_level(40)
        env = _gym.make(
            f"procgen:procgen-{self._env_name}-v0",
            start_level=int(seed),
            num_levels=1,
            distribution_mode=self._distribution_mode,
        )
        env = XYInfoWrapperLite(env, adapt_episodes=self._adapt_episodes)
        return env

    def reset_task(self, seed: int):
        try:
            if hasattr(self.env, "close"):
                self.env.close()
        except Exception:
            pass
        self.env = self._make_env(int(seed))
        return self.env.reset()


def make_persistent_env_fn(env_name: str, distribution_mode: str, adapt_episodes: int):
    def _thunk():
        _suppress_all_warnings()
        return PersistentTaskWrapper(env_name, distribution_mode, adapt_episodes=adapt_episodes)
    return _thunk


def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple)) and len(obs) > 0
    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    return np.stack(obs)


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == "reset":
                ob = env.reset()
                remote.send(ob)
            elif cmd == "reset_task":
                ob = env.reset_task(data)
                remote.send(ob)
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError(f"Unknown command {cmd}")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass


class FasterSubprocVecEnv(VecEnv):
    """
    Minimal SubprocVecEnv variant that supports reset_task(seed) without respawning workers.
    """

    def __init__(self, env_fns, spaces=None):
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, nenvs, observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        return _flatten_obs(results)

    def reset_task_batch(self, seeds: List[int]):
        for i, seed in enumerate(seeds):
            if i >= len(self.remotes):
                break
            self.remotes[i].send(("reset_task", int(seed)))
        # Must recv to keep pipes in sync (obs values are not used by wrappers).
        for i in range(min(len(seeds), len(self.remotes))):
            self.remotes[i].recv()

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True


def _find_venv_with_attr(venv, attr: str):
    cur = venv
    for _ in range(64):
        if hasattr(cur, attr):
            return cur
        if hasattr(cur, "venv"):
            cur = cur.venv
        else:
            break
    return None


def create_persistent_vec_env_cpu(env_name: str, num_processes: int, distribution_mode: str, adapt_episodes: int):
    env_fns = [make_persistent_env_fn(env_name, distribution_mode, adapt_episodes=adapt_episodes) for _ in range(num_processes)]
    venv = FasterSubprocVecEnv(env_fns)
    if isinstance(venv.observation_space, gym.spaces.Dict):
        venv = VecExtractDictObs(venv, "rgb")
    venv = VecMonitor(venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv, ob=False, ret=False)
    return venv


def _obs_to_chw_uint8(obs: np.ndarray) -> np.ndarray:
    """
    Normalize obs shape to (N, 3, 64, 64) uint8 on CPU.
    Supports (N, 64, 64, 3) or (N, 3, 64, 64).
    """
    if obs.ndim != 4:
        raise ValueError(f"Unexpected obs shape: {obs.shape}")
    if obs.shape[1] == 3:
        # (N, 3, H, W)
        return obs.astype(np.uint8, copy=False)
    if obs.shape[-1] == 3:
        # (N, H, W, 3) -> (N, 3, H, W)
        return obs.transpose(0, 3, 1, 2).astype(np.uint8, copy=False)
    raise ValueError(f"Unexpected obs shape: {obs.shape}")


def _prep_obs_u8_for_model(obs_chw_u8: np.ndarray, device: torch.device, fp16_on_cuda: bool = True) -> torch.Tensor:
    """
    Fast path: keep CPU obs as uint8, move to device, and normalize on device.
    Input:  (N,3,64,64) uint8 (CPU)
    Output: (N,3,64,64) float16/float32 on device in [0,1]
    """
    if device.type == "cuda":
        dtype = torch.float16 if fp16_on_cuda else torch.float32
        t = torch.from_numpy(obs_chw_u8).to(device=device, dtype=dtype, non_blocking=True)
        return t.div_(255.0)
    # CPU: convert to float32 and normalize
    return torch.from_numpy(obs_chw_u8.astype(np.float32)).to(device).div_(255.0)


@dataclass
class EpisodeBuffer:
    # Store CHW frames as uint8 to minimize CPU conversion work; convert once at finalize.
    obs_u8: List[np.ndarray] = field(default_factory=list)   # each element: (3,64,64) uint8 (view)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    xy: List[np.ndarray] = field(default_factory=list)       # each element: (2,) float32
    info_level_complete: bool = False
    ret: float = 0.0
    length: int = 0
    xy_missing_count: int = 0

    def finalize(self) -> Dict[str, Any]:
        xy_missing_frac = self.xy_missing_count / max(1, self.length)
        if self.obs_u8:
            obs_u8_arr = np.stack(self.obs_u8, axis=0)  # (T,3,64,64) uint8
            obs_f16 = (obs_u8_arr.astype(np.float32) / 255.0).astype(np.float16)
        else:
            obs_f16 = np.empty((0, 3, 64, 64), dtype=np.float16)
        xy_arr = np.stack(self.xy, axis=0) if self.xy else np.empty((0, 2), dtype=np.float32)
        return {
            "obs": obs_f16,
            "actions": np.asarray(self.actions, dtype=np.int64),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
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

    valid = []
    for i, ep in enumerate(episodes):
        if require_success and not ep.get("success", False):
            continue
        if not _len_ok(int(ep.get("len", 0))):
            continue
        valid.append(i)

    if not valid:
        diag["selected_reason"] = "no_successful_episodes" if require_success else "no_episodes_in_length_range"
        return None, diag

    best_idx = valid[-1]
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
    def __init__(self, actor_critic, device: torch.device, use_amp: bool = True):
        self.actor_critic = actor_critic
        self.device = device
        self.use_amp = bool(use_amp) and device.type == "cuda"

    @torch.inference_mode()
    def act_batched(self, obs: torch.Tensor, rnn_hxs: torch.Tensor, masks: torch.Tensor, deterministic: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                _, actions, _, rnn_hxs_new = self.actor_critic.act(obs, rnn_hxs, masks, deterministic=deterministic)
            return actions, rnn_hxs_new.float()
        _, actions, _, rnn_hxs_new = self.actor_critic.act(obs, rnn_hxs, masks, deterministic=deterministic)
        return actions, rnn_hxs_new


class LiveStats:
    def __init__(self, target_count: int):
        self.target = target_count
        self.collected = 0
        self.attempted = 0
        self.total_steps = 0
        self.start_time = time.perf_counter()
        self.aborted = 0
        self.timed_out = 0

    def update(self, successful: int, attempted: int, steps: int, aborted: int = 0, timed_out: int = 0):
        self.collected += successful
        self.attempted += attempted
        self.total_steps += steps
        self.aborted += aborted
        self.timed_out += timed_out

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time

    @property
    def success_rate(self) -> float:
        return self.collected / max(1, self.attempted)

    @property
    def speed(self) -> float:
        return self.collected / max(0.01, self.elapsed)

    @property
    def steps_per_sec(self) -> float:
        return self.total_steps / max(0.01, self.elapsed)

    @property
    def eta_seconds(self) -> float:
        if self.collected == 0:
            return float("inf")
        remaining = self.target - self.collected
        return remaining / max(1e-9, self.speed)

    @staticmethod
    def format_time(seconds: float) -> str:
        if seconds == float("inf"):
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


def run_batch_collection(
    envs,
    actor_critic,
    batch_seeds: List[int],
    args,
    device: torch.device,
    fast_inf: FastInference,
) -> Tuple[List[Tuple[int, List[Dict[str, Any]], bool]], Dict[str, Any]]:
    batch_size = len(batch_seeds)
    num_envs = int(getattr(envs, "num_envs", batch_size))
    total_episodes = args.adapt_episodes + args.record_episodes

    completion_threshold = float(getattr(args, "batch_completion_threshold", 0.9))
    task_timeout_steps = int(getattr(args, "task_timeout", args.max_steps * total_episodes))
    early_abort_no_success = bool(int(getattr(args, "early_abort", 1)))

    stats: Dict[str, Any] = {
        "time_inference": 0.0,
        "time_env_step": 0.0,
        "time_obs_prep": 0.0,
        "total_steps": 0,
        "successful_tasks": 0,
        "failed_tasks": 0,
        "aborted_early": 0,
        "timed_out": 0,
    }

    rnn_hxs = torch.zeros(num_envs, args.hidden_size, device=device)
    masks = torch.zeros(num_envs, 1, device=device)
    masks_ones = torch.ones(num_envs, 1, device=device)

    obs = envs.reset()  # numpy

    ep_counts = [0] * num_envs
    ep_bufs = [EpisodeBuffer() for _ in range(num_envs)]
    ep_lists: List[List[Dict[str, Any]]] = [[] for _ in range(num_envs)]
    env_done = [False] * num_envs
    env_has_success = [False] * num_envs
    task_total_steps = [0] * num_envs
    # Mark inactive envs (beyond batch_size) as done so they don't affect stopping logic.
    for i in range(batch_size, num_envs):
        env_done[i] = True

    # Initial obs record (first frame of episode)
    t0 = time.perf_counter()
    obs_chw_u8 = _obs_to_chw_uint8(obs)
    stats["time_obs_prep"] += time.perf_counter() - t0
    for i in range(batch_size):
        ep_bufs[i].obs_u8.append(obs_chw_u8[i])

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

        in_record = all(ep_counts[i] >= args.adapt_episodes for i in range(batch_size) if not env_done[i])
        deterministic = bool(args.record_deterministic) if in_record else bool(args.adapt_deterministic)

        # Prepare obs for model
        t0 = time.perf_counter()
        obs_t = _prep_obs_u8_for_model(obs_chw_u8, device, fp16_on_cuda=bool(args.use_amp))
        stats["time_obs_prep"] += time.perf_counter() - t0

        # Inference
        t0 = time.perf_counter()
        actions_t, rnn_hxs = fast_inf.act_batched(obs_t, rnn_hxs, masks, deterministic=deterministic)
        stats["time_inference"] += time.perf_counter() - t0

        # Step env (CPU). XY work is skipped internally by env wrapper during adapt episodes.
        t0 = time.perf_counter()
        actions_np = actions_t.squeeze(1).detach().cpu().numpy().astype(np.int32, copy=False)
        obs, reward, done, infos = envs.step(actions_np)
        stats["time_env_step"] += time.perf_counter() - t0

        masks = masks_ones.clone()

        reward_np = np.asarray(reward).ravel()
        done_np = np.asarray(done, dtype=bool).ravel()

        # Prepare obs for next step (keep as uint8)
        t0 = time.perf_counter()
        obs_chw_u8 = _obs_to_chw_uint8(obs)
        stats["time_obs_prep"] += time.perf_counter() - t0

        for i in range(batch_size):
            if env_done[i]:
                continue

            task_total_steps[i] += 1
            ep_bufs[i].actions.append(int(actions_np[i]))
            ep_bufs[i].rewards.append(float(reward_np[i]))
            ep_bufs[i].ret += float(reward_np[i])
            ep_bufs[i].length += 1
            stats["total_steps"] += 1

            info_i = infos[i] if i < len(infos) else {}
            if isinstance(info_i, dict) and info_i.get("level_complete", False):
                ep_bufs[i].info_level_complete = True

            # XY extraction from info (terminal_xy on done)
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

            # Timeout per task
            if task_total_steps[i] >= task_timeout_steps:
                env_done[i] = True
                stats["timed_out"] += 1
                if ep_bufs[i].length > 0:
                    ep_data = ep_bufs[i].finalize()
                    ep_lists[i].append(ep_data)
                    if ep_data["success"]:
                        env_has_success[i] = True
                continue

            # Handle episode termination or cap
            if done_np[i] or ep_bufs[i].length >= args.max_steps:
                ep_data = ep_bufs[i].finalize()
                ep_lists[i].append(ep_data)
                ep_counts[i] += 1
                if ep_data["success"]:
                    env_has_success[i] = True

                if early_abort_no_success and ep_counts[i] >= args.adapt_episodes and not env_has_success[i]:
                    env_done[i] = True
                    stats["aborted_early"] += 1
                    continue

                if ep_counts[i] >= total_episodes:
                    env_done[i] = True
                else:
                    ep_bufs[i] = EpisodeBuffer()
                    # New episode starts immediately (procgen often auto-resets).
                    # Treat current obs as episode-start frame, matching old collector.
                    ep_bufs[i].obs_u8.append(obs_chw_u8[i])
            else:
                # Episode continues: append next frame so obs length matches actions length at finalize.
                ep_bufs[i].obs_u8.append(obs_chw_u8[i])

    for i in range(batch_size):
        if env_has_success[i]:
            stats["successful_tasks"] += 1
        else:
            stats["failed_tasks"] += 1

    results = [(batch_seeds[i], ep_lists[i], env_has_success[i]) for i in range(batch_size)]
    return results, stats


def main():
    p = argparse.ArgumentParser("collect_meta_routes_fast")

    p.add_argument("--model_ckpt", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--env_name", type=str, default="maze")
    p.add_argument("--distribution_mode", type=str, default="easy")

    p.add_argument("--num_tasks", type=int, default=200)
    p.add_argument("--seed_offset", type=int, default=0)
    p.add_argument("--num_processes", type=int, default=64)
    p.add_argument("--max_seed_attempts", type=int, default=0, help="0 => 10x num_tasks")

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

    p.add_argument("--xy_fail_policy", type=str, default="warn_only", choices=["drop_task", "warn_only"])
    p.add_argument("--xy_fail_threshold", type=float, default=0.5)
    p.add_argument("--save_all_episodes", type=int, default=0)

    p.add_argument("--use_compile", type=int, default=1)
    p.add_argument("--use_amp", type=int, default=1)

    # Checkpointing
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--resume_force", action="store_true")
    p.add_argument("--ckpt_shard_size", type=int, default=25)
    p.add_argument("--ckpt_flush_secs", type=float, default=60.0)

    args = p.parse_args()

    if args.max_seed_attempts <= 0:
        args.max_seed_attempts = args.num_tasks * 10
    if args.task_timeout <= 0:
        args.task_timeout = args.max_steps * (args.adapt_episodes + args.record_episodes)

    use_ckpt = args.ckpt_dir is not None
    if use_ckpt and not CKPT_AVAILABLE:
        raise RuntimeError("Checkpointing requested but routes_ckpt_storage not available")

    ckpt_writer = None
    manifest = None
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
    print("META-RL ROUTE COLLECTION (FAST)")
    print("=" * 60)
    print(f"Target: {args.num_tasks} successful trajectories")
    print(f"Device: {device}")
    print(f"Batch size: {args.num_processes}")
    print(f"Require success: {bool(args.require_success)}")
    print(f"Max seed attempts: {args.max_seed_attempts}")
    print("=" * 60)

    actor_critic, obs_shape, num_actions = build_model(
        args.model_ckpt,
        args.arch,
        args.hidden_size,
        device,
        args.env_name,
        use_compile=bool(args.use_compile),
    )

    if args.use_compile and hasattr(torch, "compile") and device.type == "cuda":
        print("[warmup] torch.compile warmup...")
        with torch.inference_mode():
            dummy = torch.zeros(1, *obs_shape, device=device, dtype=torch.float16)
            dummy_h = torch.zeros(1, args.hidden_size, device=device)
            dummy_m = torch.ones(1, 1, device=device)
            for _ in range(3):
                actor_critic.act(dummy, dummy_h, dummy_m, deterministic=True)
            torch.cuda.synchronize()

    # In-memory storage (only used when ckpt disabled)
    routes_seed: List[int] = []
    routes_actions: List[np.ndarray] = []
    routes_obs: List[np.ndarray] = []
    routes_xy: List[np.ndarray] = []
    routes_rewards: List[np.ndarray] = []
    routes_ep_len: List[int] = []
    routes_return: List[float] = []
    routes_success: List[bool] = []
    routes_selected_ep: List[int] = []
    routes_diag: List[Dict[str, Any]] = []
    all_episodes: List[List[Dict[str, Any]]] = []

    if use_ckpt and manifest and args.resume:
        current_seed = manifest.current_seed
        seeds_tried = manifest.seeds_attempted
        initial_collected = manifest.num_tasks_collected
    else:
        current_seed = args.seed_offset
        seeds_tried = 0
        initial_collected = 0

    live = LiveStats(args.num_tasks)
    pbar = tqdm(
        total=args.num_tasks,
        initial=initial_collected,
        desc=live.get_desc(),
        unit="traj",
        ncols=120,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        file=sys.stdout,
        dynamic_ncols=True,
    )

    # Persistent envs: create ONCE (big win vs per-batch SubprocVecEnv spawn)
    print(f"[perf] Creating {args.num_processes} persistent env workers...", end="", flush=True)
    envs = create_persistent_vec_env_cpu(args.env_name, args.num_processes, args.distribution_mode, adapt_episodes=args.adapt_episodes)
    reset_venv = _find_venv_with_attr(envs, "reset_task_batch")
    if reset_venv is None:
        raise RuntimeError("Could not find reset_task_batch on VecEnv chain")
    print(" done.")

    try:
        while True:
            current_collected = manifest.num_tasks_collected if use_ckpt else len(routes_seed)
            if current_collected >= args.num_tasks:
                break
            if seeds_tried >= args.max_seed_attempts:
                break

            batch_size = min(args.num_processes, args.max_seed_attempts - seeds_tried)
            batch_seeds = list(range(current_seed, current_seed + batch_size))
            current_seed += batch_size
            seeds_tried += batch_size
            if batch_size == 0:
                break

            batch_start = time.perf_counter()
            batch_num = max(1, seeds_tried // max(1, args.num_processes))
            print(f"\r[Batch {batch_num}] Resetting tasks for {batch_size} envs (seeds {batch_seeds[0]}-{batch_seeds[-1]})...", end="", flush=True)
            reset_venv.reset_task_batch(batch_seeds)
            print(" running...", end="", flush=True)

            fast_inf = FastInference(actor_critic, device=device, use_amp=bool(args.use_amp))
            results, st = run_batch_collection(envs, actor_critic, batch_seeds, args, device, fast_inf)

            batch_time = time.perf_counter() - batch_start

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
                xy_missing_frac = float(ep.get("xy_missing_frac", 1.0))
                if args.xy_fail_policy == "drop_task" and xy_missing_frac > args.xy_fail_threshold:
                    continue

                if use_ckpt:
                    assert ckpt_writer is not None
                    ckpt_writer.append_route(
                        seed=int(seed),
                        selected_ep=int(sel_idx),
                        obs=ep["obs"],
                        actions=ep["actions"],
                        xy=ep["xy"],
                        player_v=np.empty((0, 2), dtype=np.float32),
                        ents_count=np.empty((0,), dtype=np.int32),
                        nearest_ents=np.empty((0, 8, 4), dtype=np.float32),
                        rewards=ep["rewards"],
                        ep_len=int(ep["len"]),
                        ep_return=float(ep["return"]),
                        success=bool(ep["success"]),
                        diag=diag,
                        all_episodes=episodes if args.save_all_episodes else None,
                    )
                    ckpt_writer.update_progress(current_seed, seeds_tried)
                else:
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

                if use_ckpt:
                    if manifest.num_tasks_collected >= args.num_tasks:
                        break
                else:
                    if len(routes_seed) >= args.num_tasks:
                        break

            live.update(
                batch_successful,
                batch_size,
                int(st.get("total_steps", 0)),
                aborted=int(st.get("aborted_early", 0)),
                timed_out=int(st.get("timed_out", 0)),
            )
            pbar.set_description(live.get_desc())

            # Print one-line perf breakdown (helps diagnosing bottlenecks quickly)
            ti = float(st.get("time_inference", 0.0))
            te = float(st.get("time_env_step", 0.0))
            to = float(st.get("time_obs_prep", 0.0))
            print(
                f" done! +{batch_successful} traj ({batch_time:.1f}s) | "
                f"env {te:.2f}s, infer {ti:.2f}s, obs_prep {to:.2f}s",
                flush=True,
            )

            if device.type == "cuda":
                torch.cuda.synchronize()
    finally:
        try:
            envs.close()
        except Exception:
            pass

    pbar.close()

    if use_ckpt:
        assert ckpt_writer is not None
        ckpt_writer.close()
        final_collected = manifest.num_tasks_collected
    else:
        final_collected = len(routes_seed)

    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE (FAST)")
    print("=" * 60)
    print(f"Target:            {args.num_tasks}")
    print(f"Collected:         {final_collected}")
    print(f"Seeds attempted:   {seeds_tried}")
    print(f"Success rate:      {100*live.success_rate:.1f}%")
    print(f"Total time:        {LiveStats.format_time(live.elapsed)}")
    print(f"Speed:             {live.speed:.2f} traj/s, {live.steps_per_sec:.0f} steps/s")
    print(f"Early aborted:     {live.aborted}")
    print(f"Timed out:         {live.timed_out}")
    print("=" * 60)

    if use_ckpt:
        print(f"\n[ckpt] Merging shards into {args.out_npz}...")
        n_merged = build_routes_npz_from_ckpt(args.ckpt_dir, args.out_npz)
        print(f"[saved] {args.out_npz} ({n_merged} trajectories)")
        return

    os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
    save_dict = dict(
        routes_seed=np.asarray(routes_seed, dtype=np.int64),
        routes_selected_ep=np.asarray(routes_selected_ep, dtype=np.int64),
        routes_obs=np.asarray(routes_obs, dtype=object),
        routes_actions=np.asarray(routes_actions, dtype=object),
        routes_xy=np.asarray(routes_xy, dtype=object),
        routes_player_v=np.asarray([np.empty((0, 2), dtype=np.float32) for _ in routes_seed], dtype=object),
        routes_ents_count=np.asarray([np.empty((0,), dtype=np.int32) for _ in routes_seed], dtype=object),
        routes_nearest_ents=np.asarray([np.empty((0, 8, 4), dtype=np.float32) for _ in routes_seed], dtype=object),
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
            success_rate=float(live.success_rate),
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
            batch_completion_threshold=args.batch_completion_threshold,
            task_timeout=args.task_timeout,
            early_abort=bool(args.early_abort),
            fast_impl=True,
        ),
    )
    if args.save_all_episodes:
        save_dict["episodes_all"] = np.asarray(all_episodes, dtype=object)
    np.savez_compressed(args.out_npz, **save_dict)
    print(f"\n[saved] {args.out_npz}")


if __name__ == "__main__":
    main()


