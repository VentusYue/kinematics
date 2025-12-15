
from procgen import ProcgenEnv
import numpy as np

def probe_sequential():
    # Try with default
    venv = ProcgenEnv(num_envs=2, env_name="maze", num_levels=1, start_level=100, distribution_mode="easy")
    obs = venv.reset()
    print(f"Default: Obs identical? {np.allclose(obs['rgb'][0], obs['rgb'][1])}")
    
    # Try with use_sequential_levels=True
    venv2 = ProcgenEnv(num_envs=2, env_name="maze", num_levels=1, start_level=100, distribution_mode="easy", use_sequential_levels=True)
    obs2 = venv2.reset()
    print(f"Sequential: Obs identical? {np.allclose(obs2['rgb'][0], obs2['rgb'][1])}")

if __name__ == "__main__":
    probe_sequential()

