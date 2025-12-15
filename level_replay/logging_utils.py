import wandb

from level_replay.utils import ppo_normalise_reward, min_max_normalise_reward


def log_train_episode_returns(episode_reward, env_name, step):
    """
    Log per-episode training returns with multiple normalisations.
    Shared between PPO and DQN trainers.
    """
    ppo_norm = ppo_normalise_reward(episode_reward, env_name)
    min_max_norm = min_max_normalise_reward(episode_reward, env_name)

    wandb.log(
        {
            "Train Episode Returns": episode_reward,
            "Train Episode Returns (normalised)": ppo_norm,
            "Train Episode Returns (ppo normalised)": ppo_norm,
            "Train Episode Returns (min-max normalised)": min_max_norm,
        },
        step=step,
    )


def log_eval_returns(
    mean_test_rewards,
    mean_train_rewards,
    env_name,
    step,
    style="slash",
):
    """
    Log test/train evaluation returns and their normalised variants.

    style:
        - \"slash\": labels like \"Test / Evaluation Returns\" (used by DQN)
        - \"plain\": labels like \"Test Evaluation Returns\" (used by PPO)
    """
    if style == "plain":
        test_prefix = "Test "
        train_prefix = "Train "
    else:
        test_prefix = "Test / "
        train_prefix = "Train / "

    test_ppo_norm = ppo_normalise_reward(mean_test_rewards, env_name)
    train_ppo_norm = ppo_normalise_reward(mean_train_rewards, env_name)
    test_min_max_norm = min_max_normalise_reward(mean_test_rewards, env_name)
    train_min_max_norm = min_max_normalise_reward(mean_train_rewards, env_name)

    wandb.log(
        {
            f"{test_prefix}Evaluation Returns": mean_test_rewards,
            f"{train_prefix}Evaluation Returns": mean_train_rewards,
            "Generalization Gap:": mean_train_rewards - mean_test_rewards,
            f"{test_prefix}Evaluation Returns (normalised)": test_ppo_norm,
            f"{train_prefix}Evaluation Returns (normalised)": train_ppo_norm,
            f"{test_prefix}Evaluation Returns (ppo normalised)": test_ppo_norm,
            f"{train_prefix}Evaluation Returns (ppo normalised)": train_ppo_norm,
            f"{test_prefix}Evaluation Returns (min-max normalised)": test_min_max_norm,
            f"{train_prefix}Evaluation Returns (min-max normalised)": train_min_max_norm,
        },
        step=step,
    )


def log_videos_to_wandb(videos, video_tag, step=None):
    """
    Log a sequence of videos to WandB under a given tag.
    """
    for video in videos:
        if step is None:
            wandb.log({video_tag: video})
        else:
            wandb.log({video_tag: video}, step=step)


