# Copyright (c) 2017 Ilya Kostrikov
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/rraileanu/auto-drac/blob/master/ucb_rl2_meta/arguments.py
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser(description="RL")

# PPO Arguments.
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--eps", type=float, default=1e-5, help="RMSprop optimizer epsilon")
parser.add_argument("--alpha", type=float, default=0.99, help="RMSprop optimizer apha")
parser.add_argument("--gamma", type=float, default=0.999, help="discount factor for rewards")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="gae lambda parameter")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="entropy term coefficient")
parser.add_argument(
    "--value_loss_coef", type=float, default=0.5, help="value loss coefficient (default: 0.5)"
)
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max norm of gradients)")
parser.add_argument("--no_ret_normalization", action="store_true", help="Whether to use unnormalized returns")
parser.add_argument("--seed", type=int, default=1, help="random seed")
parser.add_argument("--num_processes", type=int, default=64, help="how many training CPU processes to use")
parser.add_argument("--num_steps", type=int, default=256, help="number of forward steps in A2C")
parser.add_argument("--ppo_epoch", type=int, default=3, help="number of ppo epochs")
parser.add_argument("--num_mini_batch", type=int, default=8, help="number of batches for ppo")
parser.add_argument("--clip_param", type=float, default=0.2, help="ppo clip parameter")
parser.add_argument("--num_env_steps", type=int, default=25e6, help="number of environment steps to train")
parser.add_argument("--env_name", type=str, default="bigfish", help="environment to train on")
parser.add_argument("--xpid", default="latest", help="name for the run - prefix to log files")
parser.add_argument("--log_dir", default="~/logs/ppo/", help="directory to save agent logs")
parser.add_argument("--no_cuda", action="store_true", default=False, help="disables CUDA training")
parser.add_argument("--hidden_size", type=int, default=256, help="state embedding dimension")
parser.add_argument(
    "--arch", type=str, default="large", choices=["small", "large", "simple"], help="agent architecture"
)
parser.add_argument("--num_workers", type=int, default=1)

# Procgen arguments.
parser.add_argument("--distribution_mode", default="easy", help="distribution of envs for procgen")
parser.add_argument("--paint_vel_info", action="store_true", help="Paint velocity vector at top of frames.")
parser.add_argument(
    "--num_train_seeds", type=int, default=200, help="number of Procgen levels to use for training"
)
parser.add_argument("--use_sequential_levels", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--start_level", type=int, default=0, help="start level id for sampling Procgen levels")
parser.add_argument("--num_test_seeds", type=int, default=10, help="Number of test seeds")
parser.add_argument("--final_num_test_seeds", type=int, default=1000, help="Number of test seeds")
parser.add_argument(
    "--seed_path", type=str, default=None, help="Path to file containing specific training seeds"
)
parser.add_argument("--full_train_distribution", action="store_true", help="Train on the full distribution")

# Level Replay arguments.
parser.add_argument(
    "--level_replay_score_transform",
    type=str,
    default="rank",
    choices=["constant", "max", "eps_greedy", "rank", "power", "softmax"],
    help="Level replay scoring strategy",
)
parser.add_argument(
    "--level_replay_temperature", type=float, default=0.1, help="Level replay scoring strategy"
)
parser.add_argument(
    "--level_replay_strategy",
    type=str,
    default="random",
    choices=[
        "off",
        "random",
        "sequential",
        "policy_entropy",
        "least_confidence",
        "min_margin",
        "gae",
        "value_l1",
        "one_step_td_error",
    ],
    help="Level replay scoring strategy",
)
parser.add_argument(
    "--level_replay_eps", type=float, default=0.05, help="Level replay epsilon for eps-greedy sampling"
)
parser.add_argument(
    "--level_replay_schedule",
    type=str,
    default="proportionate",
    help="Level replay schedule for sampling seen levels",
)
parser.add_argument(
    "--level_replay_rho",
    type=float,
    default=1.0,
    help="Minimum size of replay set relative to total number of levels before sampling replays.",
)
parser.add_argument(
    "--level_replay_nu",
    type=float,
    default=0.5,
    help="Probability of sampling a new level instead of a replay level.",
)
parser.add_argument("--level_replay_alpha", type=float, default=1.0, help="Level score EWA smoothing factor")
parser.add_argument("--staleness_coef", type=float, default=0.1, help="Staleness weighing")
parser.add_argument(
    "--staleness_transform",
    type=str,
    default="power",
    choices=["max", "eps_greedy", "rank", "power", "softmax"],
    help="Staleness normalization transform",
)
parser.add_argument(
    "--staleness_temperature", type=float, default=1.0, help="Staleness normalization temperature"
)

# Logging arguments
parser.add_argument("--verbose", action="store_true", help="Whether to print logs")
parser.add_argument("--log_interval", type=int, default=1, help="log interval, one log per n updates")
parser.add_argument("--save_interval", type=int, default=60, help="Save model every this many minutes.")
parser.add_argument(
    "--weight_log_interval", type=int, default=1, help="Save level weights every this many updates"
)
parser.add_argument("--disable_checkpoint", action="store_true", help="Disable saving checkpoint.")
parser.add_argument(
    "--save_model",
    type=lambda x: bool(strtobool(x)),
    default=True,
    help="Whether to save the final model checkpoint (default: True)",
)
parser.add_argument(
    "--model_path",
    default="models/model.tar",
    help="Path to save the final model checkpoint",
)

parser.add_argument(
    "--wandb_project",
    type=str,
    default="off-policy-procgen",
    choices=["off-policy-procgen", "thesis-experiments"],
)
parser.add_argument(
    "--wandb_tags",
    type=str,
    default="",
    help="Additional tags for this wandb run",
)
parser.add_argument(
    "--wandb_group",
    type=str,
    default="",
    help="Wandb group for this run",
)
parser.add_argument(
    "--record_eval_video",
    type=lambda x: bool(strtobool(x)),
    default=False,
    help="Whether to record videos during evaluation",
)
parser.add_argument(
    "--eval_video_freq",
    type=int,
    default=1000,
    help="Frequency of video evaluation in environment steps",
)
parser.add_argument(
    "--eval_video_envs",
    type=str,
    default="auto",
    help=(
        "Comma separated list of environments to record videos for. "
        "Use 'auto' (default) to record videos only for the current training env (--env_name)."
    ),
)
parser.add_argument(
    "--eval_video_episodes",
    type=int,
    default=3,
    help="Number of episodes to record per environment",
)
parser.add_argument("--log_per_seed_stats", type=lambda x: bool(strtobool(x)), default=False)

# Meta-RL arguments
parser.add_argument(
    "--meta_rl",
    action="store_true",
    default=True,
    help="Enable RL^2-style meta-RL: task=level seed, multi-trial meta-episodes.",
)
parser.add_argument(
    "--meta_trials_per_task",
    type=int,
    default=20,
    help="Number of trials per task (seed) within a meta-episode.",
)
parser.add_argument(
    "--meta_eval_trials_per_task",
    type=int,
    default=None,
    help="Number of trials per task during meta-eval (defaults to meta_trials_per_task).",
)
parser.add_argument(
    "--meta_attach_task_id",
    action="store_true",
    help="Attach current task id (level seed) as an extra observation channel.",
)

# RNN / recurrent policy logging arguments.
parser.add_argument(
    "--log_rnn_stats",
    type=lambda x: bool(strtobool(x)),
    default=False,
    help="Whether to log GRU hidden-state statistics for recurrent policies",
)
parser.add_argument(
    "--log_rnn_traces",
    type=lambda x: bool(strtobool(x)),
    default=False,
    help="Whether to periodically dump GRU hidden-state trajectories to disk",
)
parser.add_argument(
    "--log_rnn_every",
    type=int,
    default=2048,
    help="Interval (in per-process environment steps) between RNN trace dumps",
)
parser.add_argument(
    "--rnn_trace_dir",
    type=str,
    default="rnn_traces",
    help="Directory to store RNN trace `.npz` files when `log_rnn_traces` is enabled",
)

# Reward shaping arguments (optional)
parser.add_argument(
    "--reward_mode",
    type=str,
    default="env",
    choices=["env", "success_speed"],
    help=(
        "Reward mode. "
        "'env' uses the environment reward as-is. "
        "'success_speed' adds a small per-step penalty and a terminal bonus that is larger "
        "when the episode finishes in fewer steps (no xy/goal needed; helps learn shorter paths)."
    ),
)
parser.add_argument(
    "--reward_step_penalty",
    type=float,
    default=0.01,
    help="Per-step penalty applied when reward_mode=success_speed (dense signal to encourage shorter solutions).",
)
parser.add_argument(
    "--reward_success_speed_bonus",
    type=float,
    default=0.0,
    help="Extra terminal bonus on success when reward_mode=success_speed (scaled by how fast the episode finished).",
)
parser.add_argument(
    "--reward_success_speed_max_steps",
    type=int,
    default=500,
    help="Max steps used to normalise the speed bonus when reward_mode=success_speed (Procgen Maze timeout=500).",
)
parser.add_argument(
    "--reward_success_threshold",
    type=float,
    default=0.0,
    help="Success detection threshold on the *raw* per-step env reward at terminal steps (Maze: success reward=10, timeout reward=0).",
)
