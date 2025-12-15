
import sys
import os
import traceback

# Add procgen-tools to path
PROCGEN_TOOLS_PATH = "/root/test/procgen-tools-main"
if PROCGEN_TOOLS_PATH not in sys.path:
    sys.path.append(PROCGEN_TOOLS_PATH)

# Patch procgen to include ProcgenGym3Env if missing
try:
    import procgen
    if not hasattr(procgen, "ProcgenGym3Env"):
        class ProcgenGym3Env:
            pass
        procgen.ProcgenGym3Env = ProcgenGym3Env
except ImportError:
    pass

try:
    from procgen_tools import maze
except Exception as e:
    print(f"Warning: Could not import procgen_tools from {PROCGEN_TOOLS_PATH}")
    traceback.print_exc()
    maze = None

def get_xy_from_venv(venv, env_idx=0):
    """
    Extract (x, y) and grid_step from a procgen maze venv.
    """
    if maze is None:
        return 0.0, 0.0, 0.0
    
    # Handle SubprocVecEnv (special case for retrieving from remote)
    # venv might be VecPyTorchProcgen wrapping SubprocVecEnv
    if hasattr(venv, "venv"):
        if hasattr(venv.venv, "env_method"): # SubprocVecEnv has env_method
            try:
                # Call get_xy on the specific worker
                # We assume the worker env has a get_xy method (via wrapper)
                res = venv.venv.env_method("get_xy", indices=env_idx)
                if res and len(res) > 0:
                    return res[0]
            except Exception as e:
                pass
    
    # Standard logic for local ProcgenEnv
    if hasattr(venv, "raw_venv"):
        raw_venv = venv.raw_venv
    else:
        raw_venv = venv
    
    try:
        if not hasattr(raw_venv, "env") and hasattr(raw_venv, "callmethod"):
            state_bytes_list = raw_venv.callmethod("get_state")
            state = maze.EnvState(state_bytes_list[env_idx])
        else:
            state = maze.state_from_venv(raw_venv, env_idx)
            
        return _parse_state(state)
        
    except Exception as e:
        return 0.0, 0.0, 0.0

def get_xy_from_gym_env(env):
    """
    Extract (x, y) from a single Gym env (Procgen wrapped)
    """
    if maze is None:
        return 0.0, 0.0, 0.0
    
    try:
        # Unwrap until we find something with callmethod (Gym3 env)
        inner = env
        while hasattr(inner, "env"):
            if hasattr(inner, "callmethod"):
                break
            inner = inner.env
            
        if hasattr(inner, "callmethod"):
            state_bytes_list = inner.callmethod("get_state")
            # For single env, list has 1 element
            state = maze.EnvState(state_bytes_list[0])
            return _parse_state(state)
            
    except Exception as e:
        pass
        
    return 0.0, 0.0, 0.0

def _parse_state(state):
    vals = state.state_vals
    ents = vals["ents"][0]
    x = ents['x'].val
    y = ents['y'].val
    grid_step = vals['grid_step'].val
    return float(x), float(y), float(grid_step)
