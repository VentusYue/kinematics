
from procgen import ProcgenEnv

def probe_set_level_seed():
    venv = ProcgenEnv(num_envs=2, env_name="maze", num_levels=0, start_level=0)
    try:
        # Try callmethod on gym3 env
        venv.env.callmethod("set_level_seed", 123)
        print("set_level_seed works!")
    except Exception as e:
        print(f"set_level_seed failed: {e}")

if __name__ == "__main__":
    probe_set_level_seed()

