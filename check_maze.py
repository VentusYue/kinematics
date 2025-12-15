
import gym
import numpy as np
from procgen import ProcgenEnv

def check_maze_difficulty():
    num_envs = 64
    env_name = "maze"
    distribution_mode = "easy"
    
    venv = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        num_levels=0, # unlimited levels available, but we will seed manually to check
        start_level=0,
        distribution_mode=distribution_mode,
    )
    
    # Check seeds 0 to 63
    seeds = np.arange(num_envs, dtype=np.int32)
    
    # We need to see if a random agent can solve these
    # ProcgenEnv doesn't support manual seeding in the constructor easily for per-env seed unless we use start_level/num_levels.
    # But let's try to just run it with default start_level=0, num_levels=200.
    
    venv = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        num_levels=200,
        start_level=0,
        distribution_mode=distribution_mode,
    )
    
    obs = venv.reset()
    total_rewards = np.zeros(num_envs)
    dones_count = np.zeros(num_envs)
    
    # Run for 256 steps
    for _ in range(256):
        # Random actions
        action = np.random.randint(0, venv.action_space.n, size=num_envs)
        obs, rew, done, info = venv.step(action)
        total_rewards += rew
        dones_count += done.astype(int)
        
    print(f"Mean reward of random agent over 256 steps: {np.mean(total_rewards)}")
    print(f"Max reward: {np.max(total_rewards)}")
    print(f"Num solved (reward >= 10): {np.sum(total_rewards >= 10)}")
    print(f"Num episodes done: {np.sum(dones_count)}")

if __name__ == "__main__":
    check_maze_difficulty()
