# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import partial
from typing import Callable, List, Tuple, Union

import gym
import numpy as np
import torch
import wandb
from baselines.common.vec_env import SubprocVecEnv, VecEnvWrapper, VecExtractDictObs, VecMonitor, VecNormalize
# from custom_envs import ObstructedMazeGamut  # noqa: F401
from gym.spaces.box import Box
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
try:
    from procgen import ProcgenEnv
    import procgen
except ModuleNotFoundError:
    # Fallback to local bundled procgen if not installed in the environment
    import sys
    _fallback_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "_procgen_local")
    )
    if os.path.isdir(os.path.join(_fallback_root, "procgen")):
        sys.path.insert(0, _fallback_root)
        from procgen import ProcgenEnv  # type: ignore
        import procgen  # type: ignore
    else:
        raise

from level_replay.level_sampler import LevelSampler, DQNLevelSampler

CUBE_ROOTS = {1, 8, 27, 64, 125, 216, 343, 512, 729, 1000}


def cubes_idx_checker(idx: int) -> bool:
    return idx % 1000 == 0 or idx in CUBE_ROOTS


def all_idx_checker(_: int) -> bool:
    return True


def _resolve_procgen_assets_root() -> str:
    """
    Resolve a valid Procgen assets root directory and return it with a trailing os.sep.
    Precedence:
      1) ENV PROCGEN_ASSETS_ROOT (if set and valid)
      2) Assets co-located with the imported procgen package (site-packages)
      3) Local workspace fallback: /root/workspace/_procgen_local/procgen/data/assets
    """
    env_override = os.environ.get("PROCGEN_ASSETS_ROOT")
    if env_override:
        candidate = os.path.abspath(env_override)
        if os.path.isdir(candidate):
            return candidate if candidate.endswith(os.sep) else candidate + os.sep

    # Assets next to the imported 'procgen' package
    pkg_dir = os.path.dirname(os.path.abspath(procgen.__file__))
    site_assets = os.path.join(pkg_dir, "data", "assets")
    if os.path.isdir(site_assets):
        return site_assets + os.sep

    # Local workspace fallback (present in this repo)
    local_assets = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "_procgen_local", "procgen", "data", "assets")
    )
    if os.path.isdir(local_assets):
        return local_assets + os.sep

    # As a last resort, return the package default (BaseProcgenEnv will assert if invalid)
    return os.path.join(pkg_dir, "data", "assets") + os.sep


class VideoWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        log_videos: bool = True,
        fps: int = 16,
        idx_checker: Callable[[int], bool] = all_idx_checker,
    ) -> None:
        self.env = env
        self.fps = fps
        self.frames: List[np.ndarray] = []
        self.episode_idx = 0
        self.idx_checker = idx_checker
        self.log_videos = log_videos
        self._reward = 0
        self.videos: List[wandb.Video]
        super().__init__(env)

    def _log_video(self, info: dict) -> None:
        self.episode_idx += 1

        caption_dict = {"reward": self._reward, "length": len(self.frames)}
        caption_dict.update(**info)
        caption = ", ".join([f"{key}: {value}" for key, value in caption_dict.items()])

        if self.idx_checker(self.episode_idx) and self.frames:
            video = wandb.Video(np.stack(self.frames), caption=caption, fps=self.fps, format="gif")
            if self.log_videos:
                wandb.log({"video": video})
            else:
                self.videos.append(video)

        self._reward = 0
        self.frames = []

    def step(self, action) -> Tuple[np.ndarray, Union[np.number, int], Union[np.bool_, bool], dict]:
        obs, reward, done, info = super().step(action)
        self._reward += reward
        self.frames.append(obs)
        if done:
            self._log_video(info)
        return obs, reward, done, info

    def get_videos(self) -> List[wandb.Video]:
        tmp_videos = self.videos
        self.videos = []
        return tmp_videos


class VecVideoWrapper(VecEnvWrapper):
    def __init__(
        self,
        venv: gym.Env,
        log_videos: bool = True,
        fps: int = 16,
        idx_checker: Callable[[int], bool] = all_idx_checker,
    ) -> None:
        super().__init__(venv)
        self.frames: List[List[np.ndarray]] = [[] for i in range(self.num_envs)]
        self.vid_rewards = np.zeros(self.num_envs)
        self.videos: List[wandb.Video] = []

        self.fps = fps
        self.episode_idx = 0
        self.idx_checker = idx_checker
        self.log_videos = log_videos

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        self.frames = [[] for i in range(self.num_envs)]
        self.vid_rewards = np.zeros(self.num_envs)
        return obs

    def _log_video(self, idx: int, info: dict) -> None:
        self.episode_idx += 1
        frames = self.frames[idx]

        caption_dict = info
        # Unneeded as reward and length info are in info dict from VecMonitor
        # {"reward": reward, "length": len(frames)}
        # caption_dict.update(**info)
        caption = ", ".join([f"{key}: {value}" for key, value in caption_dict.items()])

        if self.idx_checker(self.episode_idx) and frames:
            np_frames = np.stack(frames).transpose(0, 3, 1, 2)  # type: ignore
            video = wandb.Video(np_frames, caption=caption, fps=self.fps, format="gif")
            if self.log_videos:
                wandb.log({"video": video})
            else:
                self.videos.append(video)

        self.vid_rewards[idx] = 0
        self.frames[idx] = []

    def step_wait(self) -> Tuple[np.ndarray, Union[np.number, int], Union[np.bool_, bool], dict]:
        obs, rewards, dones, infos = self.venv.step_wait()
        self.vid_rewards += rewards
        for i in range(len(obs)):
            self.frames[i].append(obs[i])
        for i in range(len(dones)):
            if dones[i]:
                self._log_video(i, infos[i])
        return obs, rewards, dones, infos

    def get_videos(self) -> List[wandb.Video]:
        tmp_videos = self.videos
        self.videos = []
        return tmp_videos


# Lightweight replacement for baselines' VecNormalize that avoids TensorFlow
# Only supports reward/return normalization (ob=False expected in this codebase)
class VecNormalize(VecEnvWrapper):  # type: ignore[no-redef]
    def __init__(
        self,
        venv: gym.Env,
        ob: bool = False,  # unused; kept for API compatibility
        ret: bool = True,
        cliprew: float = 10.0,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(venv)
        self.normalize_returns = ret
        self.clip_reward = cliprew
        self.gamma = gamma
        self.epsilon = epsilon

        self.returns = np.zeros(self.num_envs, dtype=np.float32)
        # Running mean/variance of returns (scalar) using parallel-batch updates
        self.ret_mean = 0.0
        self.ret_var = 1.0
        self.ret_count = epsilon  # non-zero to avoid div-by-zero

    def reset(self):
        self.returns[...] = 0.0
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        if self.normalize_returns:
            # Update discounted returns and running stats
            self.returns = self.returns * self.gamma + rewards
            if isinstance(dones, np.ndarray):
                self.returns[dones.astype(bool)] = 0.0
            elif np.isscalar(dones):
                if bool(dones):
                    self.returns[...] = 0.0

            # Batch update of running mean/var (parallel combine formula)
            batch = self.returns.astype(np.float64)
            batch_mean = float(batch.mean())
            batch_var = float(batch.var())
            batch_count = float(batch.size)

            delta = batch_mean - self.ret_mean
            total_count = self.ret_count + batch_count
            new_mean = self.ret_mean + delta * (batch_count / total_count)

            m_a = self.ret_var * self.ret_count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta * delta * (self.ret_count * batch_count / total_count)
            new_var = M2 / total_count if total_count > 0.0 else 1.0

            self.ret_mean = new_mean
            self.ret_var = max(new_var, self.epsilon)
            self.ret_count = total_count

            # Normalize rewards using return variance (as in baselines)
            rewards = rewards / np.sqrt(self.ret_var + self.epsilon)
            if self.clip_reward is not None:
                rewards = np.clip(rewards, -self.clip_reward, self.clip_reward)

        return obs, rewards, dones, infos


class VecSuccessSpeedReward(VecEnvWrapper):
    """
    Optional reward shaping to make sparse-success tasks (e.g., Procgen Maze) more informative.

    Modes:
      - "env": pass through environment rewards unchanged.
      - "success_speed": add a small per-step penalty and a terminal success bonus that
        is larger when the episode finishes in fewer steps (encourages shorter paths and
        more successes per fixed step budget).

    Notes:
      - Does NOT require access to goal position or coordinates.
      - Designed to sit above VecMonitor (so episode stats are still env-return) and
        below VecNormalize (so normalization/clipping can still be applied if enabled).
    """

    SUPPORTED_MODES = ("env", "success_speed")

    def __init__(
        self,
        venv: gym.Env,
        reward_mode: str = "env",
        step_penalty: float = 0.0,
        success_speed_bonus: float = 0.0,
        success_speed_max_steps: int = 500,
        success_threshold: float = 0.0,
    ) -> None:
        super().__init__(venv)
        if reward_mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported reward_mode={reward_mode!r}. Supported: {self.SUPPORTED_MODES}"
            )

        self.reward_mode = reward_mode
        self.step_penalty = float(step_penalty)
        self.success_speed_bonus = float(success_speed_bonus)
        self.success_speed_max_steps = int(success_speed_max_steps)
        self.success_threshold = float(success_threshold)

        self._episode_steps = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self):
        self._episode_steps[...] = 0
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()

        # No-op path: preserve existing behaviour exactly.
        if self.reward_mode == "env":
            # Still maintain step counters for potential downstream use.
            if isinstance(dones, np.ndarray):
                self._episode_steps += 1
                self._episode_steps[dones.astype(bool)] = 0
            elif np.isscalar(dones):
                self._episode_steps[...] = 0 if bool(dones) else (self._episode_steps + 1)
            return obs, rewards, dones, infos

        # "success_speed" shaping.
        raw = np.asarray(rewards, dtype=np.float32)
        shaped = raw.copy()

        if self.step_penalty != 0.0:
            shaped -= self.step_penalty

        max_steps = max(1, int(self.success_speed_max_steps))

        if isinstance(dones, np.ndarray):
            # Update step counters (episode length includes current step).
            self._episode_steps += 1

            done_idx = dones.nonzero()[0]
            for e in done_idx:
                ep_len = int(self._episode_steps[e])
                success = bool(raw[e] > self.success_threshold)

                if success and self.success_speed_bonus != 0.0:
                    frac = max(0.0, float(max_steps - ep_len) / float(max_steps))
                    bonus = float(self.success_speed_bonus * frac)
                    shaped[e] += bonus
                    infos[e]["shaping_speed_bonus"] = bonus

                infos[e]["shaping_episode_steps"] = ep_len
                infos[e]["shaping_success"] = success

                # New episode starts after this step.
                self._episode_steps[e] = 0

        elif np.isscalar(dones):
            self._episode_steps += 1
            if bool(dones):
                ep_len = int(self._episode_steps[0])
                success = bool(float(raw) > self.success_threshold)
                if success and self.success_speed_bonus != 0.0:
                    frac = max(0.0, float(max_steps - ep_len) / float(max_steps))
                    shaped = shaped + float(self.success_speed_bonus * frac)
                    infos[0]["shaping_speed_bonus"] = float(self.success_speed_bonus * frac)
                infos[0]["shaping_episode_steps"] = ep_len
                infos[0]["shaping_success"] = success
                self._episode_steps[...] = 0

        return obs, shaped, dones, infos


class SeededSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns):
        super(SubprocVecEnv, self).__init__(
            env_fns,
        )

    def seed_async(self, seed, index):
        self._assert_not_closed()
        self.remotes[index].send(("seed", seed))
        self.waiting = True

    def seed_wait(self, index):
        self._assert_not_closed()
        obs = self.remotes[index].recv()
        self.waiting = False
        return obs

    def seed(self, seed, index):
        self.seed_async(seed, index)
        return self.seed_wait(index)

    def observe_async(self, index):
        self._assert_not_closed()
        self.remotes[index].send(("observe", None))
        self.waiting = True

    def observe_wait(self, index):
        self._assert_not_closed()
        obs = self.remotes[index].recv()
        self.waiting = False
        return obs

    def observe(self, index):
        self.observe_async(index)
        return self.observe_wait(index)

    def level_seed_async(self, index):
        self._assert_not_closed()
        self.remotes[index].send(("level_seed", None))
        self.waiting = True

    def level_seed_wait(self, index):
        self._assert_not_closed()
        level_seed = self.remotes[index].recv()
        self.waiting = False
        return level_seed

    def level_seed(self, index):
        self.level_seed_async(index)
        return self.level_seed_wait(index)


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImageProcgen(TransposeObs):
    def __init__(self, env=None, op=[0, 3, 2, 1]):  # noqa: B006
        """
        Transpose observation space for images
        """
        super(TransposeImageProcgen, self).__init__(env)
        self.observation_space: gym.Space
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, ob):
        if ob.shape[0] == 1:
            ob = ob[0]
        return ob.transpose(self.op[0], self.op[1], self.op[2], self.op[3])


class VecPyTorchProcgen(VecEnvWrapper):
    def __init__(
        self,
        venv,
        device,
        attach_task_id=False,
        level_sampler=None,
        meta_rl=False,
        meta_train_seeds=None,
        meta_trials_per_task=1,
    ):
        """
        Environment wrapper that returns tensors (for obs and reward)
        """
        super(VecPyTorchProcgen, self).__init__(venv)
        self.device = device

        self.level_sampler = level_sampler

        # Meta-RL config
        self.meta_rl = meta_rl
        self.meta_trials_per_task = meta_trials_per_task
        if self.meta_rl:
            import numpy as np
            assert meta_train_seeds is not None, "meta_rl=True requires meta_train_seeds"
            self.meta_train_seeds = np.array(meta_train_seeds, dtype=np.int64)
        else:
            self.meta_train_seeds = None

        if self.meta_rl:
            n_envs = self.venv.num_envs
            self.current_task_seeds = np.zeros(n_envs, dtype=np.int64)
            self.current_trial_ids = np.zeros(n_envs, dtype=np.int32)
            self.trials_remaining = np.zeros(n_envs, dtype=np.int32)

        channel = 4 if attach_task_id else 3
        # channel = 3
        self.attach_task_id = attach_task_id
        self.seeds = torch.zeros(self.venv.num_envs, dtype=torch.int)

        self.observation_space: gym.Space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [channel, 64, 64],
            dtype=self.observation_space.dtype,
        )

    def _sample_meta_seed(self) -> int:
        assert self.meta_train_seeds is not None
        idx = np.random.randint(0, len(self.meta_train_seeds))
        return int(self.meta_train_seeds[idx])

    @property
    def raw_venv(self):
        rvenv = self.venv
        while hasattr(rvenv, "venv"):
            rvenv = rvenv.venv
        return rvenv

    def reset(self):
        if self.level_sampler:
            seeds = torch.zeros(self.venv.num_envs, dtype=torch.int)
            for e in range(self.venv.num_envs):
                seed = self.level_sampler.sample("sequential")
                seeds[e] = seed
                self.venv.seed(seed, e)
            self.seeds = seeds
        elif self.meta_rl:
            n_envs = self.venv.num_envs
            seeds = torch.zeros(n_envs, dtype=torch.int)
            for e in range(n_envs):
                seed = self._sample_meta_seed()
                self.current_task_seeds[e] = seed
                self.current_trial_ids[e] = 0
                self.trials_remaining[e] = self.meta_trials_per_task
                self.venv.seed(seed, e)
                seeds[e] = seed
            self.seeds = seeds

        obs = self.venv.reset()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.0

        if self.attach_task_id:
            obs = self._attach_seeds(obs)

        if self.level_sampler or self.meta_rl:
            return obs, self.seeds
        else:
            return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor) or len(actions.shape) > 1:
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, infos = self.venv.step_wait()
        # print(f"stepping {info[0]['level_seed']}, done: {done}")

        if self.level_sampler:
            # reset environment here
            for e in done.nonzero()[0]:
                seed = self.level_sampler.sample()
                self.venv.seed(seed, e)  # seed resets the corresponding level
                self.seeds[e] = seed

            # NB: This reset call propagates upwards through all VecEnvWrappers
            obs = self.raw_venv.observe()[
                "rgb"
            ]  # Note reset does not reset game instances, but only returns latest observations
        elif self.meta_rl:
            done_indices = done.nonzero()[0]

            for e in range(self.venv.num_envs):
                infos[e]["level_seed"] = int(self.current_task_seeds[e])
                infos[e]["trial_id"] = int(self.current_trial_ids[e])
                infos[e]["task_switch"] = False

            for e in done_indices:
                self.trials_remaining[e] -= 1

                if self.trials_remaining[e] > 0:
                    self.current_trial_ids[e] += 1
                    seed = int(self.current_task_seeds[e])
                else:
                    seed = self._sample_meta_seed()
                    self.current_task_seeds[e] = seed
                    self.current_trial_ids[e] = 0
                    self.trials_remaining[e] = self.meta_trials_per_task
                    infos[e]["task_switch"] = True

                self.venv.seed(seed, e)
                self.seeds[e] = seed

            if len(done_indices) > 0:
                obs = self.raw_venv.observe()["rgb"]

        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device) / 255.0
        if self.attach_task_id:
            obs = self._attach_seeds(obs)
        # torch.from_numpy(reward).unsqueeze(dim=1).float()
        reward = np.expand_dims(reward.astype(float), 1)

        return obs, reward, done, infos

    def _attach_seeds(self, obs):
        expand_seeds = torch.reshape(self.seeds, shape=(-1, 1, 1, 1))
        expand_seeds = expand_seeds.expand_as(obs)[:, :1, :, :].float().to(obs.device)
        obs = torch.cat([expand_seeds, obs], axis=1)
        return obs


class VecMinigrid(SeededSubprocVecEnv):
    def __init__(self, num_envs, env_name, seeds=None):
        if seeds is None:
            seeds = [int.from_bytes(os.urandom(4), byteorder="little") for _ in range(num_envs)]
        else:
            seeds = [int(s) for s in np.random.choice(seeds, num_envs)]

        env_fn = [partial(self._make_minigrid_env, env_name, seeds[i]) for i in range(num_envs)]

        super(SeededSubprocVecEnv, self).__init__(env_fn)

    @staticmethod
    def _make_minigrid_env(env_name, seed):
        env = gym.make(env_name)
        env.seed(seed)
        env = FullyObsWrapper(env)
        env = ImgObsWrapper(env)
        return env


class VecPyTorchMinigrid(VecEnvWrapper):
    def __init__(self, venv, device, level_sampler=None):
        """
        Environment wrapper that returns tensors (for obs and reward)
        """
        super(VecPyTorchMinigrid, self).__init__(venv)
        self.device = device
        self.is_first_step = False
        self.observation_space: gym.Space

        self.level_sampler = level_sampler

        m, n, c = venv.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [c, m, n],
            dtype=self.observation_space.dtype,
        )

    @property
    def raw_venv(self):
        rvenv = self.venv
        while hasattr(rvenv, "venv"):
            rvenv = rvenv.venv
        return rvenv

    def reset(self):
        if self.level_sampler:
            seeds = torch.zeros(self.venv.num_envs, dtype=torch.int)
            for e in range(self.venv.num_envs):
                seed = self.level_sampler.sample("sequential")
                seeds[e] = seed
                self.venv.seed(seed, e)

        obs = self.venv.reset()
        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device)
        # obs = torch.from_numpy(obs).float().to(self.device) / 255.

        if self.level_sampler:
            return obs, seeds
        else:
            return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor) or len(actions.shape) > 1:
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()

        # reset environment here
        for e in done.nonzero()[0]:
            if self.level_sampler:
                seed = self.level_sampler.sample()
            else:
                # seed = int.from_bytes(os.urandom(4), byteorder="little")
                seed = np.random.randint(1, 1e12)  # type: ignore
            obs[e] = self.venv.seed(seed, e)  # seed resets the corresponding level

        if obs.shape[1] != 3:
            obs = obs.transpose(0, 3, 1, 2)
        obs = torch.from_numpy(obs).float().to(self.device)
        # obs = torch.from_numpy(obs).float().to(self.device) / 255.
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()

        return obs, reward, done, info


PROCGEN_ENVS = {
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
}


# Makes a vector environment
def make_lr_venv(num_envs, env_name, seeds, device, **kwargs):
    level_sampler = kwargs.get("level_sampler")
    level_sampler_args = kwargs.get("level_sampler_args")

    ret_normalization = not kwargs.get("no_ret_normalization", False)
    record_runs = kwargs.get("record_runs", False)

    # Optional reward shaping
    reward_mode = kwargs.get("reward_mode", "env")
    if reward_mode not in VecSuccessSpeedReward.SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported reward_mode={reward_mode!r}. Supported: {VecSuccessSpeedReward.SUPPORTED_MODES}"
        )

    # Meta-RL arguments
    meta_rl = kwargs.get("meta_rl", False)
    meta_train_seeds = kwargs.get("meta_train_seeds", None)
    meta_trials_per_task = kwargs.get("meta_trials_per_task", 1)
    meta_attach_task_id = kwargs.get("meta_attach_task_id", False)

    if env_name in PROCGEN_ENVS:
        num_levels = kwargs.get("num_levels", 1)
        start_level = kwargs.get("start_level", 0)
        distribution_mode = kwargs.get("distribution_mode", "easy")
        paint_vel_info = kwargs.get("paint_vel_info", False)
        use_sequential_levels = kwargs.get("use_sequential_levels", False)

        venv = ProcgenEnv(
            num_envs=num_envs,
            env_name=env_name,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode=distribution_mode,
            paint_vel_info=paint_vel_info,
            use_sequential_levels=use_sequential_levels,
            resource_root=_resolve_procgen_assets_root(),
        )
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        if record_runs:
            venv = VecVideoWrapper(venv=venv, log_videos=False)

        # Reward shaping sits above VecMonitor (so episode returns remain env-returns)
        # and below VecNormalize (so normalization/clipping still works if enabled).
        if reward_mode != "env":
            step_penalty = kwargs.get("reward_step_penalty", 0.01)
            success_speed_bonus = kwargs.get("reward_success_speed_bonus", 0.0)
            # Procgen Maze uses timeout=500; keep a sensible default.
            default_max_steps = 500 if env_name == "maze" else 1000
            success_speed_max_steps = kwargs.get("reward_success_speed_max_steps", default_max_steps)
            success_threshold = kwargs.get("reward_success_threshold", 0.0)
            venv = VecSuccessSpeedReward(
                venv=venv,
                reward_mode=reward_mode,
                step_penalty=step_penalty,
                success_speed_bonus=success_speed_bonus,
                success_speed_max_steps=success_speed_max_steps,
                success_threshold=success_threshold,
            )

        venv = VecNormalize(venv=venv, ob=False, ret=ret_normalization)

        if level_sampler_args:
            level_sampler = LevelSampler(
                seeds, venv.observation_space, venv.action_space, **level_sampler_args
            )

        attach_task_id = kwargs.get("attach_task_id", False) or meta_attach_task_id

        envs = VecPyTorchProcgen(
            venv,
            device,
            attach_task_id=attach_task_id,
            level_sampler=level_sampler,
            meta_rl=meta_rl,
            meta_train_seeds=meta_train_seeds,
            meta_trials_per_task=meta_trials_per_task,
        )

    elif env_name.startswith("MiniGrid"):
        venv = VecMinigrid(num_envs=num_envs, env_name=env_name, seeds=seeds)
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        venv = VecNormalize(venv=venv, ob=False, ret=ret_normalization)

        if level_sampler_args:
            level_sampler = LevelSampler(
                seeds, venv.observation_space, venv.action_space, **level_sampler_args
            )

        elif seeds:
            level_sampler = LevelSampler(
                seeds,
                venv.observation_space,
                venv.action_space,
                strategy="random",
            )

        envs = VecPyTorchMinigrid(venv, device, level_sampler=level_sampler)

    else:
        raise ValueError(f"Unsupported env {env_name}")

    return envs, level_sampler


# Makes a vector environment for DQN
def make_dqn_lr_venv(num_envs, env_name, seeds, device, **kwargs):
    level_sampler = kwargs.get("level_sampler")
    level_sampler_args = kwargs.get("level_sampler_args")

    ret_normalization = not kwargs.get("no_ret_normalization", False)
    record_runs = kwargs.get("record_runs", False)

    if env_name in PROCGEN_ENVS:
        num_levels = kwargs.get("num_levels", 1)
        start_level = kwargs.get("start_level", 0)
        distribution_mode = kwargs.get("distribution_mode", "easy")
        paint_vel_info = kwargs.get("paint_vel_info", False)
        use_sequential_levels = kwargs.get("use_sequential_levels", False)
        attach_task_id = kwargs.get("attach_task_id", False)

        venv = ProcgenEnv(
            num_envs=num_envs,
            env_name=env_name,
            num_levels=num_levels,
            start_level=start_level,
            distribution_mode=distribution_mode,
            paint_vel_info=paint_vel_info,
            use_sequential_levels=use_sequential_levels,
            resource_root=_resolve_procgen_assets_root(),
        )
        venv = VecExtractDictObs(venv, "rgb")
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        if record_runs:
            venv = VecVideoWrapper(venv=venv, log_videos=False)
        venv = VecNormalize(venv=venv, ob=False, ret=ret_normalization)

        if level_sampler_args:
            level_sampler = DQNLevelSampler(
                seeds, venv.observation_space, venv.action_space, **level_sampler_args
            )

        envs = VecPyTorchProcgen(
            venv,
            device,
            level_sampler=level_sampler,
            attach_task_id=attach_task_id
        )

    elif env_name.startswith("MiniGrid"):
        venv = VecMinigrid(num_envs=num_envs, env_name=env_name, seeds=seeds)
        venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
        venv = VecNormalize(venv=venv, ob=False, ret=ret_normalization)

        if level_sampler_args:
            level_sampler = DQNLevelSampler(
                seeds, venv.observation_space, venv.action_space, **level_sampler_args
            )

        elif seeds:
            level_sampler = DQNLevelSampler(
                seeds,
                venv.observation_space,
                venv.action_space,
                strategy="random",
            )

        envs = VecPyTorchMinigrid(venv, device, level_sampler=level_sampler)

    else:
        raise ValueError(f"Unsupported env {env_name}")

    return envs, level_sampler
