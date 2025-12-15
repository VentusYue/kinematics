# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections import deque

import numpy as np
import torch
import wandb
from baselines import logger

from level_replay import algo, utils
from level_replay.arguments import parser
from level_replay.envs import make_lr_venv
from level_replay.model import model_for_env_name
from level_replay.storage import RolloutStorage
from level_replay.utils import ppo_normalise_reward, min_max_normalise_reward
from level_replay.logging_utils import (
    log_train_episode_returns,
    log_eval_returns,
)
from level_replay.ppo_eval import evaluate, shaped_return_from_episode_stats

os.environ["OMP_NUM_THREADS"] = "1"
# Align with train_rainbow: do not override the user's WANDB credentials here.
# This lets `wandb login` and env vars from the shell control authentication.
# os.environ["WANDB_API_KEY"] = "anon"



def eval_and_record_ppo(args, actor_critic, device, env_names, num_episodes, step):
    original_env_name = args.env_name
    for env_name in env_names:
        args.env_name = env_name
        seeds = [np.random.randint(0, 1000000) for _ in range(num_episodes)]
        
        rewards, lengths = evaluate(
            args, 
            actor_critic, 
            num_episodes, 
            device,
            seeds=seeds,
            record=True,
            video_tag=f"videos/{env_name}",
            step=step,
            return_episode_info=True,
        )
        wandb.log({
            f"Test / {env_name} Return": np.mean(rewards)
        }, step=step)
        shaped = [shaped_return_from_episode_stats(args, r, l) for r, l in zip(rewards, lengths)]
        wandb.log({f"Test / {env_name} ShapedReturn": float(np.mean(shaped))}, step=step)
    args.env_name = original_env_name


def _resolve_eval_video_env_names(args) -> list:
    """
    Resolve which env(s) to record evaluation videos for.

    - If args.eval_video_envs is empty or 'auto', record only the current training env.
    - Otherwise, interpret it as a comma-separated list.
    """
    spec = getattr(args, "eval_video_envs", "auto")
    if spec is None:
        return [args.env_name]
    spec = str(spec).strip()
    if spec == "" or spec.lower() in {"auto", "current", "train", "env"}:
        return [args.env_name]
    envs = [e.strip() for e in spec.split(",") if e.strip()]
    # Deduplicate while preserving order
    seen = set()
    out = []
    for e in envs:
        if e not in seen:
            out.append(e)
            seen.add(e)
    return out


def train(args, seeds):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if "cuda" in device.type:
        print("Using CUDA\n")

    # Give PPO runs a unique default checkpoint path to avoid clashes with DQN.
    # Only override if the user kept the shared default from the CLI/args file.
    if getattr(args, "model_path", "models/model.tar") == "models/model.tar":
        run_name = f"ppo-{args.env_name}-{args.num_train_seeds}levels-seed{args.seed}"
        args.model_path = os.path.join("models", f"{run_name}.tar")

    torch.set_num_threads(1)
    start_time = time.time()
    last_checkpoint_time = start_time

    # Match the behaviour of train_rainbow: respect the current WandB login
    # and project, without forcing a specific entity or API key.
    settings = wandb.Settings()
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        tags=["ppo", "procgen"] + (args.wandb_tags.split(",") if args.wandb_tags else []),
        group=(args.wandb_group if args.wandb_group else None),
        settings=settings,
    )
    wandb.run.name = f"ppo-{args.env_name}-{args.num_train_seeds}-levels"

    utils.seed(args.seed)

    # Configure actor envs
    start_level = 0
    if args.full_train_distribution:
        num_levels = 0
        level_sampler_args = None
        seeds = None
    else:
        num_levels = 1
        level_sampler_args = dict(
            num_actors=args.num_processes,
            strategy=args.level_replay_strategy,
            replay_schedule=args.level_replay_schedule,
            score_transform=args.level_replay_score_transform,
            temperature=args.level_replay_temperature,
            eps=args.level_replay_eps,
            rho=args.level_replay_rho,
            nu=args.level_replay_nu,
            alpha=args.level_replay_alpha,
            staleness_coef=args.staleness_coef,
            staleness_transform=args.staleness_transform,
            staleness_temperature=args.staleness_temperature,
        )
    envs, level_sampler = make_lr_venv(
        num_envs=args.num_processes,
        env_name=args.env_name,
        seeds=seeds,
        device=device,
        num_levels=num_levels,
        start_level=start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        use_sequential_levels=args.use_sequential_levels,
        level_sampler_args=level_sampler_args,
        reward_mode=getattr(args, "reward_mode", "env"),
        reward_step_penalty=getattr(args, "reward_step_penalty", 0.01),
        reward_success_speed_bonus=getattr(args, "reward_success_speed_bonus", 0.0),
        reward_success_speed_max_steps=getattr(args, "reward_success_speed_max_steps", 500),
        reward_success_threshold=getattr(args, "reward_success_threshold", 0.0),
    )

    # is_minigrid = args.env_name.startswith("MiniGrid")

    actor_critic = model_for_env_name(args, envs)
    actor_critic.to(device)

    rollouts = RolloutStorage(
        args.num_steps,
        args.num_processes,
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size,
    )

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        env_name=args.env_name,
    )

    level_seeds = torch.zeros(args.num_processes)
    if level_sampler:
        obs, level_seeds = envs.reset()
    else:
        obs = envs.reset()
    level_seeds = level_seeds.unsqueeze(-1)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards: deque = deque(maxlen=10)
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    count = 0
    for j in range(num_updates):
        actor_critic.train()
        for step in range(args.num_steps):
            count += 1
            # Sample actions
            with torch.no_grad():
                obs_id = rollouts.obs[step]
                value, action, action_log_dist, recurrent_hidden_states = actor_critic.act(
                    obs_id, rollouts.recurrent_hidden_states[step], rollouts.masks[step]
                )
                action_log_prob = action_log_dist.gather(-1, action)

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)
            reward = torch.from_numpy(reward)

            # Reset all done levels by sampling from level sampler
            for i, info in enumerate(infos):
                if "episode" in info.keys():
                    episode_reward = info["episode"]["r"]
                    episode_rewards.append(episode_reward)
                    log_train_episode_returns(
                        episode_reward,
                        args.env_name,
                        step=count * args.num_processes,
                    )
                    if "l" in info["episode"]:
                        wandb.log(
                            {"Train Episode Length": info["episode"]["l"]},
                            step=count * args.num_processes,
                        )
                    if args.log_per_seed_stats:
                        plot_level_returns(level_seeds, episode_reward, i, step=count * args.num_processes)
                if level_sampler:
                    level_seed = info["level_seed"]
                    if level_seeds[i][0] != level_seed:
                        level_seeds[i][0] = level_seed
                        if args.log_per_seed_stats:
                            new_episode(value, level_seed, i, step=count * args.num_processes)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )

            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                action_log_dist,
                value,
                reward,
                masks,
                bad_masks,
                level_seeds,
            )

        with torch.no_grad():
            obs_id = rollouts.obs[-1]
            next_value = actor_critic.get_value(
                obs_id, rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]
            ).detach()

        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        # Update level sampler
        if level_sampler:
            level_sampler.update_with_rollouts(rollouts)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        wandb.log({"Value Loss": value_loss}, step=count * args.num_processes)
        rollouts.after_update()
        if level_sampler:
            level_sampler.after_update()

        # Log stats every log_interval updates or if it is the last update
        if (j % args.log_interval == 0 and len(episode_rewards) > 1) or j == num_updates - 1:
            test_returns, test_lengths = evaluate(
                args, actor_critic, args.num_test_seeds, device, return_episode_info=True
            )
            mean_eval_rewards = float(np.mean(test_returns)) if test_returns else 0.0
            test_shaped = [
                shaped_return_from_episode_stats(args, r, l)
                for r, l in zip(test_returns, test_lengths)
            ]
            mean_eval_shaped_rewards = float(np.mean(test_shaped)) if test_shaped else 0.0

            train_returns, train_lengths = evaluate(
                args,
                actor_critic,
                args.num_test_seeds,
                device,
                start_level=0,
                num_levels=args.num_train_seeds,
                seeds=seeds,
                return_episode_info=True,
            )
            mean_train_rewards = float(np.mean(train_returns)) if train_returns else 0.0
            train_shaped = [
                shaped_return_from_episode_stats(args, r, l)
                for r, l in zip(train_returns, train_lengths)
            ]
            mean_train_shaped_rewards = float(np.mean(train_shaped)) if train_shaped else 0.0

            wandb.log(
                {
                    "Test Evaluation Returns (shaped)": mean_eval_shaped_rewards,
                    "Train Evaluation Returns (shaped)": mean_train_shaped_rewards,
                },
                step=count * args.num_processes,
            )
            log_eval_returns(
                mean_test_rewards=mean_eval_rewards,
                mean_train_rewards=mean_train_rewards,
                env_name=args.env_name,
                step=count * args.num_processes,
                style="plain",
            )

            # Console logging similar to train_rainbow.py
            total_num_steps = count * args.num_processes
            elapsed = time.time() - start_time
            if total_num_steps > 0:
                logger.logkv("minutes elapse", elapsed / 60.0)
                logger.logkv("time / step", elapsed / total_num_steps)
                logger.logkv("train / total_num_steps", total_num_steps)
                logger.logkv("train / mean_episode_reward", mean_train_rewards)
                logger.logkv("test / mean_episode_reward", mean_eval_rewards)
                logger.dumpkvs()

            if j == num_updates - 1:
                print(f"\nLast update: Evaluating on {args.final_num_test_seeds} test levels...\n  ")
                final_eval_episode_rewards, final_eval_episode_lengths = evaluate(
                    args, actor_critic, args.final_num_test_seeds, device, return_episode_info=True
                )

                mean_final_eval_episode_rewards = np.mean(final_eval_episode_rewards)
                median_final_eval_episide_rewards = np.median(final_eval_episode_rewards)
                final_shaped = [
                    shaped_return_from_episode_stats(args, r, l)
                    for r, l in zip(final_eval_episode_rewards, final_eval_episode_lengths)
                ]
                mean_final_eval_shaped_rewards = float(np.mean(final_shaped)) if final_shaped else 0.0

                print("Mean Final Evaluation Rewards: ", mean_final_eval_episode_rewards)
                print("Median Final Evaluation Rewards: ", median_final_eval_episide_rewards)
                print("Mean Final Evaluation Rewards (shaped): ", mean_final_eval_shaped_rewards)

                wandb.log(
                    {
                        "Mean Final Evaluation Rewards": mean_final_eval_episode_rewards,
                        "Median Final Evaluation Rewards": median_final_eval_episide_rewards,
                        "Mean Final Evaluation Rewards (shaped)": mean_final_eval_shaped_rewards,
                    }
                )
        
        # Video logging
        # eval_video_freq is treated as per-process steps to match train_rainbow consistency
        # We update every num_steps (default 256).
        updates_per_eval = max(1, int(args.eval_video_freq / args.num_steps))
        if args.record_eval_video and j % updates_per_eval == 0:
            env_names = _resolve_eval_video_env_names(args)
            eval_and_record_ppo(
                args, 
                actor_critic, 
                device, 
                env_names, 
                args.eval_video_episodes, 
                count * args.num_processes
            )

        # Periodic checkpointing based on wall-clock time
        if (not getattr(args, "disable_checkpoint", False)) and getattr(args, "save_interval", 0) > 0:
            elapsed_minutes = (time.time() - last_checkpoint_time) / 60.0
            if elapsed_minutes >= args.save_interval:
                total_num_steps = count * args.num_processes
                root, ext = os.path.splitext(args.model_path)
                ckpt_path = f"{root}_step_{total_num_steps}{ext}"
                ckpt_dir = os.path.dirname(ckpt_path)
                if ckpt_dir and not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": actor_critic.state_dict(),
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"Checkpoint saved to {ckpt_path} at step {total_num_steps}")
                last_checkpoint_time = time.time()

    # After training, save only the final (latest) model checkpoint if enabled.
    if args.save_model:
        print(f"Saving model to {args.model_path}")
        model_dir = os.path.dirname(args.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        torch.save(
            {
                "model_state_dict": actor_critic.state_dict(),
                "args": vars(args),
            },
            args.model_path,
        )
        wandb.save(args.model_path)


def generate_seeds(num_seeds, base_seed=0):
    return [base_seed + i for i in range(num_seeds)]


def load_seeds(seed_path):
    seed_path = os.path.expandvars(os.path.expanduser(seed_path))
    seeds = open(seed_path).readlines()
    return [int(s) for s in seeds]


def new_episode(value, level_seed, i, step):
    wandb.log({f"Start State Value Estimate for Level {level_seed}": value[i].item()}, step=step)


def plot_level_returns(level_seeds, episode_reward, i, step):
    seed = level_seeds[i][0].item()
    wandb.log({f"Empirical Return for Level {seed}": episode_reward}, step=step)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.seed_path:
        train_seeds = load_seeds(args.seed_path)
    else:
        train_seeds = generate_seeds(args.num_train_seeds)

    train(args, train_seeds)
