import numpy as np
import torch
from collections import defaultdict
import os

from level_replay.envs import make_lr_venv
from level_replay.logging_utils import log_videos_to_wandb


def shaped_return_from_episode_stats(
    args,
    episode_return: float,
    episode_length: int,
) -> float:
    """
    Compute the *training reward* return for an episode from (env_return, ep_length),
    matching the shaping logic used in VecSuccessSpeedReward (but without VecNormalize).

    This provides a stable metric to log during evaluation that differs from the
    environment return metrics (which are based on Procgen's sparse reward).
    """
    reward_mode = getattr(args, "reward_mode", "env")
    if reward_mode != "success_speed":
        return float(episode_return)

    step_penalty = float(getattr(args, "reward_step_penalty", 0.0))
    success_bonus = float(getattr(args, "reward_success_speed_bonus", 0.0))
    max_steps = int(getattr(args, "reward_success_speed_max_steps", 500))
    success_threshold = float(getattr(args, "reward_success_threshold", 0.0))

    shaped = float(episode_return) - step_penalty * float(episode_length)

    # Success detection is based on the raw env terminal reward (Maze: +10 on success, 0 on timeout)
    if float(episode_return) > success_threshold and success_bonus != 0.0:
        max_steps = max(1, max_steps)
        frac = max(0.0, float(max_steps - int(episode_length)) / float(max_steps))
        shaped += success_bonus * frac

    return shaped


def evaluate(
    args,
    actor_critic,
    num_episodes,
    device,
    num_processes=1,
    deterministic=False,
    start_level=0,
    num_levels=0,
    seeds=None,
    level_sampler=None,
    progressbar=None,
    record=False,
    video_tag="evaluation_behaviour",
    step=None,
    return_episode_info=False,
):
    """
    PPO evaluation helper used by train_ppo and standalone eval scripts.
    """
    actor_critic.eval()

    if level_sampler:
        start_level = level_sampler.seed_range()[0]
        num_levels = 1

    eval_envs, level_sampler = make_lr_venv(
        num_envs=num_processes,
        env_name=args.env_name,
        seeds=seeds,
        device=device,
        num_levels=num_levels,
        start_level=start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        level_sampler=level_sampler,
        record_runs=record,
        reward_mode=getattr(args, "reward_mode", "env"),
        reward_step_penalty=getattr(args, "reward_step_penalty", 0.01),
        reward_success_speed_bonus=getattr(args, "reward_success_speed_bonus", 0.0),
        reward_success_speed_max_steps=getattr(args, "reward_success_speed_max_steps", 500),
        reward_success_threshold=getattr(args, "reward_success_threshold", 0.0),
    )

    eval_episode_rewards = []
    eval_episode_lengths = []

    if level_sampler:
        obs, _ = eval_envs.reset()
    else:
        obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes,
        actor_critic.recurrent_hidden_state_size,
        device=device,
    )
    eval_masks = torch.ones(num_processes, 1, device=device)

    # Track per-env episode length as a robust fallback (VecMonitor also provides "episode"["l"]).
    ep_steps = np.zeros(num_processes, dtype=np.int32)

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=deterministic,
            )

        obs, _, done, infos = eval_envs.step(action)

        # Update episode step counters.
        ep_steps += 1

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device,
        )

        for i, info in enumerate(infos):
            if "episode" in info.keys():
                eval_episode_rewards.append(info["episode"]["r"])
                # Prefer VecMonitor length, fallback to local counter.
                ep_len = info["episode"].get("l", int(ep_steps[i]))
                eval_episode_lengths.append(int(ep_len))
                if progressbar:
                    progressbar.update(1)
                ep_steps[i] = 0

    if record:
        log_videos_to_wandb(eval_envs.get_videos(), video_tag, step=step)

    eval_envs.close()
    if progressbar:
        progressbar.close()

    if getattr(args, "verbose", False):
        print(
            "Last {} test episodes: mean/median reward {:.1f}/{:.1f}\n".format(
                len(eval_episode_rewards),
                np.mean(eval_episode_rewards),
                np.median(eval_episode_rewards),
            )
        )

    if return_episode_info:
        return eval_episode_rewards, eval_episode_lengths
    return eval_episode_rewards


def evaluate_meta(
    args,
    actor_critic,
    device,
    test_seeds,
    num_trials_per_task=None,
    num_processes=1,
    deterministic=True,
    record_traces=False,
    trace_dir="meta_eval_traces",
    return_episode_info=False,
):
    """
    Meta-RL evaluation:
      - task = level seed
      - For each test seed, run K trials consecutively.
      - RNN hidden state is preserved between trials, reset only when switching seeds.
    """
    actor_critic.eval()

    if num_trials_per_task is None:
        num_trials_per_task = (
            getattr(args, "meta_eval_trials_per_task", None)
            or getattr(args, "meta_trials_per_task", 1)
        )

    if record_traces:
        os.makedirs(trace_dir, exist_ok=True)

    # Use make_lr_venv to open a meta-RL env
    # Note: we reuse existing make_lr_venv but pass meta_rl=True
    eval_envs, _ = make_lr_venv(
        num_envs=num_processes,
        env_name=args.env_name,
        seeds=test_seeds,
        device=device,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        meta_rl=True,
        meta_train_seeds=test_seeds,
        meta_trials_per_task=num_trials_per_task,
        meta_attach_task_id=getattr(args, "meta_attach_task_id", False),
        reward_mode=getattr(args, "reward_mode", "env"),
        reward_step_penalty=getattr(args, "reward_step_penalty", 0.01),
        reward_success_speed_bonus=getattr(args, "reward_success_speed_bonus", 0.0),
        reward_success_speed_max_steps=getattr(args, "reward_success_speed_max_steps", 500),
        reward_success_threshold=getattr(args, "reward_success_threshold", 0.0),
    )

    eval_recurrent_hidden_states = torch.zeros(
        num_processes,
        actor_critic.recurrent_hidden_state_size,
        device=device,
    )
    # Initial task boundaries (all tasks freshly started)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    # Records: {seed: [(trial_id, return), ...]}
    returns_per_seed_and_trial = defaultdict(list)

    total_episodes_needed = len(test_seeds) * num_trials_per_task
    target_trials = total_episodes_needed * 2

    trials_completed = 0

    obs, _ = eval_envs.reset()
    ep_steps = np.zeros(num_processes, dtype=np.int32)

    while trials_completed < target_trials:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=deterministic,
            )

        obs, _, done, infos = eval_envs.step(action)
        ep_steps += 1

        # Meta-RL: masks are 0 if task_switch is True, else 1
        eval_masks = torch.FloatTensor(
            [[0.0] if info.get("task_switch", False) else [1.0] for info in infos]
        ).to(device)

        for i, info in enumerate(infos):
            if "episode" in info:
                # This corresponds to a trial finishing
                r = info["episode"]["r"]
                l = info["episode"].get("l", int(ep_steps[i]))
                seed = info.get("level_seed", -1)
                trial_id = info.get("trial_id", -1)
                if return_episode_info:
                    returns_per_seed_and_trial[seed].append((trial_id, r, int(l)))
                else:
                    returns_per_seed_and_trial[seed].append((trial_id, r))
                trials_completed += 1
                ep_steps[i] = 0

    eval_envs.close()

    return returns_per_seed_and_trial
