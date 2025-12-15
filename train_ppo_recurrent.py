import os
import time
from collections import deque

import numpy as np
import torch
import wandb
from baselines import logger

from level_replay import algo, utils
from level_replay.arguments import parser
from level_replay.envs import make_lr_venv, PROCGEN_ENVS
from level_replay.model import Policy, SimplePolicy
from level_replay.storage import RolloutStorage
from level_replay.utils import ppo_normalise_reward, min_max_normalise_reward
from level_replay.logging_utils import (
    log_train_episode_returns,
    log_eval_returns,
)
from level_replay.ppo_eval import evaluate, evaluate_meta, shaped_return_from_episode_stats

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
        wandb.log({f"Test / {env_name} Return": np.mean(rewards)}, step=step)
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


def build_recurrent_policy(args, envs, device):
    """
    Construct a recurrent PPO policy for Procgen environments.

    For Procgen envs and non-`simple` architectures we use a `Policy` whose base
    network is recurrent (GRU on top of the conv encoder). For `arch == "simple"`
    we fall back to the non-recurrent `SimplePolicy` for interpretability.
    """
    if args.env_name not in PROCGEN_ENVS:
        raise ValueError(
            f"train_ppo_recurrent.py is intended for Procgen envs; got env_name={args.env_name!r}"
        )

    obs_shape = envs.observation_space.shape
    num_actions = envs.action_space.n

    if args.arch == "simple":
        # Preserve the existing simple, non-recurrent architecture.
        actor_critic = SimplePolicy(obs_shape, num_actions)
    else:
        actor_critic = Policy(
            obs_shape,
            num_actions,
            arch=args.arch,
            base_kwargs={"recurrent": True, "hidden_size": args.hidden_size},
        )

    actor_critic.to(device)
    return actor_critic


def log_rnn_stats_if_enabled(args, rollouts, actor_critic, step):
    """
    Log simple GRU hidden-state statistics to WandB if enabled.
    """
    if not getattr(args, "log_rnn_stats", False):
        return
    if not getattr(actor_critic, "is_recurrent", False):
        return
    if not actor_critic.is_recurrent:
        return

    # Exclude the final bootstrap state.
    h = rollouts.recurrent_hidden_states[:-1]  # [T, N, H]
    # Flatten over (T, N).
    h_flat = h.view(-1, h.size(-1))

    with torch.no_grad():
        norms = torch.linalg.norm(h_flat, dim=-1)
        wandb.log(
            {
                "RNN/hidden_norm_mean": norms.mean().item(),
                "RNN/hidden_norm_std": norms.std().item(),
                "RNN/hidden_norm_max": norms.max().item(),
            },
            step=step,
        )


def maybe_dump_rnn_traces(args, rollouts, step):
    """
    Optionally dump RNN traces for a small subset of environments.

    We store raw rollout sequences (obs, action, reward, done, h) so that
    episodes can be segmented and analysed offline.
    """
    if not getattr(args, "log_rnn_traces", False):
        return

    # Treat log_rnn_every as per-process environment steps, mirroring
    # how eval_video_freq is interpreted in train_ppo.py.
    updates_per_dump = max(1, int(args.log_rnn_every / args.num_steps))
    # We call this once per PPO update; j is implicit, so use step to gate.
    # step here is total_num_steps (across all envs). Convert back to
    # per-process steps to compare fairly.
    per_process_steps = step // args.num_processes
    if per_process_steps // args.num_steps % updates_per_dump != 0:
        return

    trace_dir = getattr(args, "rnn_trace_dir", "rnn_traces")
    if not trace_dir:
        return

    os.makedirs(trace_dir, exist_ok=True)

    # Limit to a small number of envs to keep files manageable.
    num_envs_to_log = min(4, rollouts.rewards.size(1))
    T = rollouts.num_steps

    # Shape conventions:
    #   obs: [T+1, N, ...]
    #   actions: [T, N, action_dim]
    #   rewards: [T, N, 1]
    #   masks: [T+1, N, 1]
    #   recurrent_hidden_states: [T+1, N, H]
    obs_traces = rollouts.obs[:T, :num_envs_to_log].cpu().numpy()
    actions_traces = rollouts.actions[:, :num_envs_to_log].cpu().numpy()
    rewards_traces = rollouts.rewards[:, :num_envs_to_log].cpu().numpy()
    masks_traces = rollouts.masks[1 : T + 1, :num_envs_to_log].cpu().numpy()
    h_traces = rollouts.recurrent_hidden_states[:T, :num_envs_to_log].cpu().numpy()
    level_seeds_traces = rollouts.level_seeds[:, :num_envs_to_log].cpu().numpy()

    if hasattr(rollouts, "episode_dones"):
        dones_traces = rollouts.episode_dones[:T, :num_envs_to_log].cpu().numpy()
    else:
        dones_traces = 1.0 - masks_traces

    trial_ids_traces = None
    if hasattr(rollouts, "trial_ids"):
        trial_ids_traces = rollouts.trial_ids[:T, :num_envs_to_log].cpu().numpy()

    trace_path = os.path.join(
        trace_dir,
        f"rnn_traces_env-{args.env_name}_steps-{step}.npz",
    )
    np.savez_compressed(
        trace_path,
        obs=obs_traces,
        actions=actions_traces,
        rewards=rewards_traces,
        dones=dones_traces,
        hidden=h_traces,
        level_seeds=level_seeds_traces,
        trial_ids=trial_ids_traces,
        env_name=args.env_name,
        step=step,
        num_train_seeds=args.num_train_seeds,
        meta_rl=getattr(args, "meta_rl", False),
        meta_trials_per_task=getattr(args, "meta_trials_per_task", 1),
    )
    wandb.log({"RNN/trace_path": trace_path}, step=step)


def train(args, seeds):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if "cuda" in device.type:
        print("Using CUDA\n")

    # Consolidate all outputs under one experiment folder (log_dir/xpid)
    # This ensures model checkpoints, traces, and logs are grouped together.
    exp_dir = os.path.join(os.path.expanduser(args.log_dir), args.xpid)
    os.makedirs(exp_dir, exist_ok=True)

    # Update model_path to be inside exp_dir if it's the default
    if getattr(args, "model_path", "models/model.tar") == "models/model.tar":
        args.model_path = os.path.join(exp_dir, "model.tar")
    
    # Update rnn_trace_dir to be inside exp_dir
    args.rnn_trace_dir = os.path.join(exp_dir, "rnn_traces")

    torch.set_num_threads(1)
    start_time = time.time()
    last_checkpoint_time = start_time

    # Match the behaviour of train_rainbow: respect the current WandB login
    # and project, without forcing a specific entity or API key.
    settings = wandb.Settings()
    wandb.init(
        project=args.wandb_project,
        dir=exp_dir,  # Set wandb dir to exp_dir
        config=vars(args),
        tags=["ppo", "procgen", "rnn"] + (args.wandb_tags.split(",") if args.wandb_tags else []),
        group=(args.wandb_group if args.wandb_group else None),
        settings=settings,
    )
    wandb.run.name = f"{args.xpid}-{args.env_name}"

    utils.seed(args.seed)

    # Configure actor envs
    start_level = 0
    
    meta_kwargs = {}
    if getattr(args, "meta_rl", False):
        meta_kwargs = dict(
            meta_rl=True,
            meta_train_seeds=seeds,
            meta_trials_per_task=args.meta_trials_per_task,
            meta_attach_task_id=args.meta_attach_task_id,
        )

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
        if getattr(args, "meta_rl", False):
            level_sampler_args = None

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
        reward_success_speed_bonus=getattr(args, "reward_success_speed_bonus", 1.0),
        reward_success_speed_max_steps=getattr(args, "reward_success_speed_max_steps", 1000),
        reward_success_threshold=getattr(args, "reward_success_threshold", 0.0),
        **meta_kwargs,
    )

    actor_critic = build_recurrent_policy(args, envs, device)

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
    elif getattr(args, "meta_rl", False):
        obs, level_seeds = envs.reset()
    else:
        obs = envs.reset()
    level_seeds = level_seeds.unsqueeze(-1)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards: deque = deque(maxlen=100)
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
                    # Useful for Maze: shorter episode length ~= closer to optimal path.
                    if "l" in info["episode"]:
                        wandb.log(
                            {"Train Episode Length": info["episode"]["l"]},
                            step=count * args.num_processes,
                        )
                    if args.log_per_seed_stats:
                        plot_level_returns(level_seeds, episode_reward, i, step=count * args.num_processes)
                
                # Update level seed from info
                if "level_seed" in info:
                    level_seed = info["level_seed"]
                    if level_seeds[i][0] != level_seed:
                        level_seeds[i][0] = level_seed
                        if args.log_per_seed_stats and level_sampler:
                             new_episode(value, level_seed, i, step=count * args.num_processes)
                elif level_sampler:
                    # Fallback to old behavior if needed
                    level_seed = info["level_seed"]
                    if level_seeds[i][0] != level_seed:
                        level_seeds[i][0] = level_seed
                        if args.log_per_seed_stats:
                            new_episode(value, level_seed, i, step=count * args.num_processes)

            # If done then clean the history of observations.
            if getattr(args, "meta_rl", False):
                 masks = torch.FloatTensor(
                    [[0.0] if info.get("task_switch", False) else [1.0] for info in infos]
                )
            else:
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )

            if getattr(args, "meta_rl", False):
                episode_dones = torch.FloatTensor([[1.0] if d else [0.0] for d in done])
                trial_ids = torch.FloatTensor([[info.get("trial_id", 0)] for info in infos])
            else:
                episode_dones = None
                trial_ids = None

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
                episode_dones=episode_dones,
                trial_ids=trial_ids,
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

        total_num_steps = count * args.num_processes
        wandb.log({"Value Loss": value_loss}, step=total_num_steps)

        # RNN-specific logging (stats + optional traces).
        log_rnn_stats_if_enabled(args, rollouts, actor_critic, step=total_num_steps)
        maybe_dump_rnn_traces(args, rollouts, step=total_num_steps)

        rollouts.after_update()
        if level_sampler:
            level_sampler.after_update()

        # Log stats every log_interval updates or if it is the last update
        if (j % args.log_interval == 0 and len(episode_rewards) > 1) or j == num_updates - 1:
            if getattr(args, "meta_rl", False):
                test_seeds = generate_seeds(args.num_test_seeds, base_seed=args.num_train_seeds)
                meta_stats = evaluate_meta(
                    args,
                    actor_critic,
                    device,
                    test_seeds=test_seeds,
                    num_trials_per_task=args.meta_eval_trials_per_task,
                    num_processes=args.num_processes,
                    deterministic=True,
                    record_traces=args.record_eval_video,
                    return_episode_info=True,
                )

                # Log per-trial returns
                from collections import defaultdict
                trial_returns_map = defaultdict(list)
                trial_shaped_returns_map = defaultdict(list)
                for seed, rs in meta_stats.items():
                    for (tid, r, l) in rs:
                        trial_returns_map[tid].append(r)
                        trial_shaped_returns_map[tid].append(shaped_return_from_episode_stats(args, r, l))

                for trial_id in sorted(trial_returns_map.keys()):
                    avg_ret = np.mean(trial_returns_map[trial_id])
                    wandb.log(
                        {f"MetaEval/trial_{trial_id}_mean_return": avg_ret},
                        step=total_num_steps,
                    )
                    wandb.log(
                        {f"MetaEval/trial_{trial_id}_mean_shaped_return": float(np.mean(trial_shaped_returns_map[trial_id]))},
                        step=total_num_steps,
                    )
                
                # Log first and last trial returns for adaptation visualization
                max_trial = max(trial_returns_map.keys()) if trial_returns_map else 0
                if 0 in trial_returns_map:
                    wandb.log({"MetaEval/first_trial_mean_return": np.mean(trial_returns_map[0])}, step=total_num_steps)
                if max_trial in trial_returns_map:
                    wandb.log({"MetaEval/last_trial_mean_return": np.mean(trial_returns_map[max_trial])}, step=total_num_steps)
                if 0 in trial_shaped_returns_map:
                    wandb.log(
                        {"MetaEval/first_trial_mean_shaped_return": float(np.mean(trial_shaped_returns_map[0]))},
                        step=total_num_steps,
                    )
                if max_trial in trial_shaped_returns_map:
                    wandb.log(
                        {"MetaEval/last_trial_mean_shaped_return": float(np.mean(trial_shaped_returns_map[max_trial]))},
                        step=total_num_steps,
                    )

                all_returns = [r for seed, rs in meta_stats.items() for (_, r, _) in rs]
                all_shaped_returns = [
                    shaped_return_from_episode_stats(args, r, l)
                    for seed, rs in meta_stats.items()
                    for (_, r, l) in rs
                ]
                mean_eval_rewards = np.mean(all_returns) if all_returns else 0.0
                mean_eval_shaped_rewards = float(np.mean(all_shaped_returns)) if all_shaped_returns else 0.0
                mean_train_rewards = np.mean(episode_rewards) if episode_rewards else 0.0
                # Training shaped return is not directly available from env-return logs; log eval shaped only.
                wandb.log(
                    {"Test Evaluation Returns (shaped)": mean_eval_shaped_rewards},
                    step=total_num_steps,
                )
            else:
                test_returns, test_lengths = evaluate(
                    args,
                    actor_critic,
                    args.num_test_seeds,
                    device,
                    return_episode_info=True,
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
                    step=total_num_steps,
                )

            log_eval_returns(
                mean_test_rewards=mean_eval_rewards,
                mean_train_rewards=mean_train_rewards,
                env_name=args.env_name,
                step=total_num_steps,
                style="plain",
            )

            # Console logging similar to train_rainbow.py
            elapsed = time.time() - start_time
            if total_num_steps > 0:
                logger.logkv("minutes elapse", elapsed / 60.0)
                logger.logkv("time / step", elapsed / total_num_steps)
                logger.logkv("train / total_num_steps", total_num_steps)
                logger.logkv("train / mean_episode_reward", mean_train_rewards)
                logger.logkv("test / mean_episode_reward", mean_eval_rewards)
                logger.logkv("train / value_loss", value_loss)
                logger.logkv("train / dist_entropy", dist_entropy)
                if getattr(args, "meta_rl", False):
                    if 0 in trial_returns_map:
                        logger.logkv("test / first_trial_mean_return", np.mean(trial_returns_map[0]))
                    if max_trial in trial_returns_map:
                        logger.logkv("test / last_trial_mean_return", np.mean(trial_returns_map[max_trial]))
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
                total_num_steps,
            )

        # Periodic checkpointing based on wall-clock time
        if (not getattr(args, "disable_checkpoint", False)) and getattr(args, "save_interval", 0) > 0:
            elapsed_minutes = (time.time() - last_checkpoint_time) / 60.0
            if elapsed_minutes >= args.save_interval:
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


