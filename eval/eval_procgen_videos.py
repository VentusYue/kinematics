import sys
import os
import argparse
import torch
import numpy as np
import wandb
from distutils.util import strtobool

# Add root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from level_replay.algo.policy import DQNAgent, ATCAgent
from level_replay.model import model_for_env_name
from level_replay.envs import make_dqn_lr_venv, make_lr_venv
from train_rainbow import eval_policy as eval_policy_dqn
from level_replay.ppo_eval import evaluate as evaluate_ppo

def construct_class_from_dict(d):
    class Args:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)
    return Args(d)

def main():
    parser = argparse.ArgumentParser(description="Eval Procgen Videos")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--envs", type=str, default="maze,chaser,heist", help="Comma separated envs")
    parser.add_argument("--num_episodes", type=int, default=3, help="Episodes per env")
    parser.add_argument("--wandb_project", type=str, default="procgen-videos")
    parser.add_argument("--wandb_group", type=str, default="eval-videos")
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument("--no_cuda", type=lambda x: bool(strtobool(x)), default=False)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    saved_args_dict = checkpoint["args"]
    saved_args = construct_class_from_dict(saved_args_dict)
    
    # Detect algorithm
    is_dqn = False
    if hasattr(saved_args, "algo") and saved_args.algo in ["rainbow", "dqn"]:
        is_dqn = True
    
    # Override device in saved args
    saved_args.device = device
    saved_args.cuda = not args.no_cuda and torch.cuda.is_available()
    saved_args.no_cuda = args.no_cuda # Ensure consistency
    
    # Initialize WandB
    wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        tags=args.wandb_tags.split(",") if args.wandb_tags else [],
        config=vars(saved_args)
    )
    
    env_names = args.envs.split(",")
    
    if is_dqn:
        print("Detected DQN/Rainbow model.")
        # Initialize agent
        # We need dummy env for init
        dummy_env, _ = make_dqn_lr_venv(1, env_names[0], None, device)
        
        if getattr(saved_args, "atc", False):
             agent = ATCAgent(saved_args, dummy_env)
        else:
             agent = DQNAgent(saved_args, dummy_env)
             
        agent.Q.load_state_dict(checkpoint["model_state_dict"])
        dummy_env.close()
        
        for env_name in env_names:
            print(f"Evaluating {env_name}...")
            saved_args.env_name = env_name
            seeds = [np.random.randint(0, 1000000) for _ in range(args.num_episodes)]
            rewards = eval_policy_dqn(
                saved_args,
                agent,
                args.num_episodes,
                seeds=seeds,
                record=True,
                print_score=True,
                video_tag=f"videos/{env_name}"
            )
            wandb.log({f"Eval/{env_name}_return": np.mean(rewards)})
            
    else:
        print("Detected PPO model (assumed).")
        # Initialize actor_critic
        dummy_env, _ = make_lr_venv(1, env_names[0], None, device)
        actor_critic = model_for_env_name(saved_args, dummy_env)
        actor_critic.to(device)
        actor_critic.load_state_dict(checkpoint["model_state_dict"])
        dummy_env.close()
        
        for env_name in env_names:
            print(f"Evaluating {env_name}...")
            saved_args.env_name = env_name
            seeds = [np.random.randint(0, 1000000) for _ in range(args.num_episodes)]
            rewards = evaluate_ppo(
                saved_args,
                actor_critic,
                args.num_episodes,
                device,
                seeds=seeds,
                record=True,
                video_tag=f"videos/{env_name}"
            )
            wandb.log({f"Eval/{env_name}_return": np.mean(rewards)})
            
    print("Done!")

if __name__ == "__main__":
    main()

