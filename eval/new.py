import os
import sys
import argparse
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
os.environ["GYM_LOGGER_LEVEL"] = "error"

import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool  # for legacy codebases

import torch
import gym
gym.logger.set_level(40)

from tqdm import tqdm

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

# XY helpers (your repo already has these)
from analysis.procgen_xy import get_xy_from_venv, get_xy_from_gym_env


# -----------------------
# Env wrappers / builders
# -----------------------
class XYWrapper(gym.Wrapper):
    """Expose a get_xy() method for procgen envs."""
    def get_xy(self):
        return get_xy_from_gym_env(self.env)

def make_env_fn(env_name: str, seed: int, distribution_mode: str):
    def _thunk():
        env = gym.make(
            f"procgen:procgen-{env_name}-v0",
            start_level=int(seed),
            num_levels=1,
            distribution_mode=distribution_mode,
        )
        env = XYWrapper(env)
        return env
    return _thunk


# -----------------------
# Utility: episode buffer
# -----------------------
@dataclass
class EpisodeBuffer:
    obs: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    xy: List[np.ndarray]
    info_level_complete: bool
    ret: float
    length: int

    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.xy = []
        self.info_level_complete = False
        self.ret = 0.0
        self.length = 0

    def finalize(self) -> Dict[str, Any]:
        return {
            "obs": np.asarray(self.obs, dtype=np.float16),           # (T, C, H, W)
            "actions": np.asarray(self.actions, dtype=np.int64),     # (T,)
            "rewards": np.asarray(self.rewards, dtype=np.float32),   # (T,)
            "xy": np.asarray(self.xy, dtype=np.float32),             # (T, 2)  (position aligned to obs_t)
            "success": bool(self.info_level_complete or (self.ret > 0)),
            "level_complete": bool(self.info_level_complete),
            "return": float(self.ret),
            "len": int(self.length),
        }


def action_match_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Match ratio on the common prefix (min length)."""
    if a.size == 0 or b.size == 0:
        return 0.0
    L = min(a.shape[0], b.shape[0])
    if L <= 0:
        return 0.0
    return float(np.mean(a[:L] == b[:L]))


def select_stable_recorded_episode(
    episodes: List[Dict[str, Any]],
    adapt_episodes: int,
    record_episodes: int,
    stable_match_thresh: float = 0.95,
    min_len: int = 5,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    Choose 1 episode to be the "one-period solved route" used for PKD.
    We prefer a *record phase* episode that:
      - succeeded
      - is not trivially short
      - matches previous record episode (action consistency)
    Fallback: last successful record episode, else last successful overall, else None.
    Returns: (selected_index, diagnostics)
    """
    diag: Dict[str, Any] = {
        "num_episodes": len(episodes),
        "adapt_episodes": adapt_episodes,
        "record_episodes": record_episodes,
        "stable_match_thresh": stable_match_thresh,
        "min_len": min_len,
        "record_candidates": [],
        "selected_reason": None,
    }
    if len(episodes) == 0:
        return None, diag

    total = adapt_episodes + record_episodes
    # the script *attempts* to collect total episodes; but be robust
    start_record = min(adapt_episodes, len(episodes))
    record_idxs = list(range(start_record, min(start_record + record_episodes, len(episodes))))

    # Build candidate list with pairwise match to previous episode
    for idx in record_idxs:
        ep = episodes[idx]
        if not ep.get("success", False):
            continue
        if ep.get("len", 0) < min_len:
            continue
        prev_match = None
        if idx - 1 >= 0:
            prev_match = action_match_ratio(ep["actions"], episodes[idx - 1]["actions"])
        diag["record_candidates"].append(
            {
                "idx": idx,
                "len": int(ep.get("len", 0)),
                "return": float(ep.get("return", 0.0)),
                "prev_match": None if prev_match is None else float(prev_match),
            }
        )

    # Prefer last record episode with high prev_match
    for idx in reversed(record_idxs):
        ep = episodes[idx]
        if not ep.get("success", False):
            continue
        if ep.get("len", 0) < min_len:
            continue
        if idx - 1 >= 0:
            pm = action_match_ratio(ep["actions"], episodes[idx - 1]["actions"])
            if pm >= stable_match_thresh:
                diag["selected_reason"] = f"record_ep_{idx}_stable_vs_prev(pm={pm:.3f})"
                return idx, diag

    # Fallback 1: last successful record episode
    for idx in reversed(record_idxs):
        ep = episodes[idx]
        if ep.get("success", False):
            diag["selected_reason"] = f"record_ep_{idx}_success_fallback"
            return idx, diag

    # Fallback 2: last successful overall
    for idx in reversed(range(len(episodes))):
        ep = episodes[idx]
        if ep.get("success", False):
            diag["selected_reason"] = f"any_ep_{idx}_success_fallback"
            return idx, diag

    diag["selected_reason"] = "no_success"
    return None, diag


def build_model(model_ckpt: str, arch: str, hidden_size: int, device: torch.device, env_name: str):
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
    checkpoint = torch.load(model_ckpt, map_location=device)
    if "state_dict" in checkpoint:
        actor_critic.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        actor_critic.load_state_dict(checkpoint["model_state_dict"])
    else:
        actor_critic.load_state_dict(checkpoint)
    actor_critic.eval()
    return actor_critic, obs_shape, num_actions


def main():
    p = argparse.ArgumentParser("meta_collect_routes_v2 (PKD route collector)")
    # Core
    p.add_argument("--model_ckpt", type=str, required=True)
    p.add_argument("--out_npz", type=str, required=True)
    p.add_argument("--env_name", type=str, default="maze")
    p.add_argument("--distribution_mode", type=str, default="hard")

    # Task sampling
    p.add_argument("--num_tasks", type=int, default=200)
    p.add_argument("--seed_offset", type=int, default=0)
    p.add_argument("--num_processes", type=int, default=8)

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
    p.add_argument("--stable_match_thresh", type=float, default=0.95)
    p.add_argument("--min_len", type=int, default=5, help="drop trivially short episodes as PKD periods")

    # XY failure policy
    p.add_argument("--xy_fail_policy", type=str, default="drop_task", choices=["drop_task", "nan_fill"])
    p.add_argument("--save_all_episodes", type=int, default=0, help="1 to store all episodes (large file)")

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

    actor_critic, obs_shape, num_actions = build_model(
        args.model_ckpt, args.arch, args.hidden_size, device, args.env_name
    )
    print(f"[model] arch={args.arch} hidden_size={args.hidden_size} num_actions={num_actions} obs_shape={obs_shape}")

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

    # Process in batches for parallelism
    num_batches = int(np.ceil(args.num_tasks / args.num_processes))
    pbar = tqdm(total=args.num_tasks, desc="Collect tasks")

    for b in range(num_batches):
        batch_seeds = seeds[b * args.num_processes : (b + 1) * args.num_processes]
        if batch_seeds.size == 0:
            continue
        batch_size = int(batch_seeds.size)

        env_fns = [make_env_fn(args.env_name, int(s), args.distribution_mode) for s in batch_seeds]
        venv = SubprocVecEnv(env_fns)

        # Procgen often returns dict obs; extract rgb
        if isinstance(venv.observation_space, gym.spaces.Dict):
            venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv, filename=None, keep_buf=100)
        venv = VecNormalize(venv, ob=False, ret=False)

        # NOTE: `VecPyTorchProcgen(meta_rl=True)` is intended for the repo's `ProcgenEnv`-based
        # pipeline (see `level_replay.envs.make_lr_venv`) and requires `meta_train_seeds`.
        # Here we explicitly construct per-task Gym procgen envs with fixed `start_level`,
        # so we keep `meta_rl=False` and handle RL² hidden-state logic ourselves.
        envs = VecPyTorchProcgen(venv, device, meta_rl=False)

        # Task boundary: reset hidden state
        rnn_hxs = torch.zeros(batch_size, args.hidden_size, device=device)
        masks = torch.zeros(batch_size, 1, device=device)  # 0 only at task start
        obs = envs.reset()

        # Per-env episode state
        ep_idx = [0 for _ in range(batch_size)]
        ep_bufs = [EpisodeBuffer() for _ in range(batch_size)]
        ep_lists: List[List[Dict[str, Any]]] = [[] for _ in range(batch_size)]
        # For tasks that finish total_episodes, we keep them "inactive"
        task_done = [False for _ in range(batch_size)]

        # Run until all envs collected total_episodes
        while not all(task_done):
            # Determine per-env deterministic flag (adapt vs record phase)
            det_flags = [None] * batch_size
            for i in range(batch_size):
                if task_done[i]:
                    det_flags[i] = True  # irrelevant
                else:
                    det_flags[i] = bool(args.adapt_deterministic) if ep_idx[i] < args.adapt_episodes else bool(args.record_deterministic)

            # Record current obs + xy for active envs (obs_t)
            obs_np = obs.detach().cpu().numpy().astype(np.float16)
            for i in range(batch_size):
                if task_done[i]:
                    continue
                # safety cap: if episode is too long, force-cut it (treat as done)
                if ep_bufs[i].length >= args.max_steps:
                    # finalize as failure
                    ep_lists[i].append(ep_bufs[i].finalize())
                    ep_idx[i] += 1
                    ep_bufs[i] = EpisodeBuffer()
                    if ep_idx[i] >= total_episodes:
                        task_done[i] = True
                    continue

                ep_bufs[i].obs.append(obs_np[i])

                # XY (do NOT silently set to (0,0))
                try:
                    x, y, _ = get_xy_from_venv(envs, i)
                    ep_bufs[i].xy.append(np.asarray([x, y], dtype=np.float32))
                except Exception as e:
                    if args.xy_fail_policy == "nan_fill":
                        ep_bufs[i].xy.append(np.asarray([np.nan, np.nan], dtype=np.float32))
                    else:
                        # drop this whole task later by marking special flag
                        ep_bufs[i].xy.append(np.asarray([np.nan, np.nan], dtype=np.float32))
                        # also stash an indicator in buffer (using info_level_complete as a hack)
                        # we'll detect nan later and drop.
                        pass

            # ---- Compute actions with possibly different deterministic flags per env ----
            # We do two forward passes: one for det=False subset and one for det=True subset.
            actions_full = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            rnn_hxs_new = rnn_hxs.clone()

            for det_value in [False, True]:
                idxs = [i for i in range(batch_size) if (not task_done[i]) and (det_flags[i] == det_value)]
                if len(idxs) == 0:
                    continue
                idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
                obs_sub = obs.index_select(0, idxs_t)
                hxs_sub = rnn_hxs.index_select(0, idxs_t)
                masks_sub = masks.index_select(0, idxs_t)

                with torch.no_grad():
                    _, a_sub, _, hxs_sub_new = actor_critic.act(obs_sub, hxs_sub, masks_sub, deterministic=det_value)

                actions_full.index_copy_(0, idxs_t, a_sub)
                rnn_hxs_new.index_copy_(0, idxs_t, hxs_sub_new)

            rnn_hxs = rnn_hxs_new

            # Step env
            obs, reward, done, infos = envs.step(actions_full)

            # After first step, masks must be 1 for RL² (never reset on done)
            masks = torch.ones(batch_size, 1, device=device)

            # Convert tensors
            if isinstance(reward, torch.Tensor):
                reward_np = reward.detach().cpu().numpy().reshape(-1)
            else:
                reward_np = np.asarray(reward).reshape(-1)
            if isinstance(done, torch.Tensor):
                done_np = done.detach().cpu().numpy().astype(bool).reshape(-1)
            else:
                done_np = np.asarray(done).astype(bool).reshape(-1)
            action_np = actions_full.detach().cpu().numpy().reshape(-1)

            # Update episode buffers
            for i in range(batch_size):
                if task_done[i]:
                    continue

                ep_bufs[i].actions.append(int(action_np[i]))
                ep_bufs[i].rewards.append(float(reward_np[i]))
                ep_bufs[i].ret += float(reward_np[i])
                ep_bufs[i].length += 1

                # success detection from info
                try:
                    info_i = infos[i]
                except Exception:
                    info_i = {}
                if isinstance(info_i, dict) and info_i.get("level_complete", False):
                    ep_bufs[i].info_level_complete = True

                if done_np[i]:
                    # finalize episode i
                    ep_lists[i].append(ep_bufs[i].finalize())
                    ep_idx[i] += 1
                    ep_bufs[i] = EpisodeBuffer()

                    if ep_idx[i] >= total_episodes:
                        task_done[i] = True

        # ---- Task selection / save batch ----
        for i in range(batch_size):
            episodes_i = ep_lists[i]
            sel_idx, diag = select_stable_recorded_episode(
                episodes_i,
                adapt_episodes=args.adapt_episodes,
                record_episodes=args.record_episodes,
                stable_match_thresh=args.stable_match_thresh,
                min_len=args.min_len,
            )

            # XY validation: if too many NaNs, drop (likely get_xy broken)
            xy_nan_frac = None
            if sel_idx is not None:
                xy = episodes_i[sel_idx]["xy"]
                if xy.size > 0:
                    xy_nan_frac = float(np.mean(np.isnan(xy).any(axis=1)))
                else:
                    xy_nan_frac = 1.0
                diag["selected_xy_nan_frac"] = xy_nan_frac

            keep = (sel_idx is not None)
            if keep and args.xy_fail_policy == "drop_task":
                if xy_nan_frac is None or xy_nan_frac > 0.0:
                    # drop if ANY NaNs for now (strict); you can relax to >0.1 later
                    diag["dropped_reason"] = f"xy_nan_frac={xy_nan_frac}"
                    keep = False

            if keep:
                ep = episodes_i[sel_idx]
                routes_seed.append(int(batch_seeds[i]))
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
                # still record as failure task (for accounting)
                routes_seed.append(int(batch_seeds[i]))
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
                all_episodes.append(episodes_i)

        envs.close()
        pbar.update(batch_size)

    pbar.close()

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)

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
            stable_match_thresh=args.stable_match_thresh,
            min_len=args.min_len,
            max_steps=args.max_steps,
            xy_fail_policy=args.xy_fail_policy,
            save_all_episodes=bool(args.save_all_episodes),
        ),
    )

    if args.save_all_episodes:
        save_dict["episodes_all"] = np.asarray(all_episodes, dtype=object)

    np.savez_compressed(args.out_npz, **save_dict)
    print(f"[saved] {args.out_npz}")
    kept = int(np.sum(np.asarray(routes_selected_ep) >= 0))
    print(f"[summary] tasks={len(routes_seed)} kept_selected={kept} (selected_ep>=0)")

if __name__ == "__main__":
    main()