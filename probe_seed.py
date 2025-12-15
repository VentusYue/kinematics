
import gym
import procgen
from procgen import ProcgenEnv
import numpy as np

def probe():
    venv = ProcgenEnv(num_envs=2, env_name="maze", num_levels=0, start_level=0, distribution_mode="easy")
    print(f"Venv type: {type(venv)}")
    print(f"Dir venv: {dir(venv)}")
    
    # Try to seed
    try:
        venv.seed(123, 0)
        print("venv.seed works")
    except Exception as e:
        print(f"venv.seed failed: {e}")
        
    # Unwrap to gym3
    if hasattr(venv, 'env'):
        gym3_env = venv.env
        print(f"Gym3 env type: {type(gym3_env)}")
        print(f"Gym3 env dir: {dir(gym3_env)}")
        
        # Try callmethod
        try:
            # Procgen specific methods?
            # 'get_state' works.
            # 'set_state' works?
            state = gym3_env.callmethod("get_state")[0]
            print("get_state works")
            
            # Can we set state?
            gym3_env.callmethod("set_state", [state, state])
            print("set_state works")
            
        except Exception as e:
            print(f"callmethod failed: {e}")

if __name__ == "__main__":
    probe()

