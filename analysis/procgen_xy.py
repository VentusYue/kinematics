
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


def _nan_triplet():
    return float("nan"), float("nan"), float("nan")


def _find_env_method_venv(venv):
    """
    Walk through `.venv` wrappers to find the first object exposing `env_method`
    (e.g. a `SubprocVecEnv`). Returns None if not found.
    """
    cur = venv
    for _ in range(64):
        if hasattr(cur, "env_method"):
            return cur
        if hasattr(cur, "venv"):
            cur = cur.venv
        else:
            break
    return None

def get_xy_from_venv(venv, env_idx=0):
    """
    Extract (x, y) and grid_step from a procgen maze venv.
    """
    if maze is None:
        return _nan_triplet()
    
    # Handle SubprocVecEnv (special case for retrieving from remote)
    # venv might be VecPyTorchProcgen wrapping SubprocVecEnv
    env_method_venv = _find_env_method_venv(venv)
    if env_method_venv is not None:
        # Call get_xy on the specific worker; we assume the worker env has a
        # get_xy() method (via a wrapper like XYWrapper).
        try:
            res = env_method_venv.env_method("get_xy", indices=env_idx)
            if res and len(res) > 0:
                return res[0]
        except Exception:
            # Some VecEnv APIs expect a list of indices.
            try:
                res = env_method_venv.env_method("get_xy", indices=[env_idx])
                if res and len(res) > 0:
                    return res[0]
            except Exception:
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
        return _nan_triplet()

def get_xy_batch(venv):
    """
    Batch version of get_xy_from_venv for all envs.
    Returns a list of (x, y, grid_step) tuples, one for each env.
    """
    if maze is None:
        return [_nan_triplet()] * venv.num_envs
    
    env_method_venv = _find_env_method_venv(venv)
    if env_method_venv is not None:
        try:
            # Call get_xy on all workers at once
            results = env_method_venv.env_method("get_xy")
            if results and len(results) == venv.num_envs:
                return results
        except Exception:
            pass
            
    # Fallback to serial
    return [get_xy_from_venv(venv, i) for i in range(venv.num_envs)]

def get_xy_from_gym_env(env):
    """
    Extract (x, y) from a single Gym env (Procgen wrapped)
    """
    if maze is None:
        return _nan_triplet()
    
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
        return _nan_triplet()
        
    return _nan_triplet()

def _parse_state(state):
    vals = state.state_vals
    ents = vals["ents"][0]
    x = ents['x'].val
    y = ents['y'].val
    grid_step = vals['grid_step'].val
    return float(x), float(y), float(grid_step)
