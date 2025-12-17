import os
import sys
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
import torch
import argparse
import gym
import warnings
from tqdm import tqdm
from collections import deque

# Suppress warnings
warnings.filterwarnings("ignore")
try:
    # Filter specific Gym warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="gym")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym.envs.registration")
except:
    pass
os.environ['GYM_LOGGER_LEVEL'] = 'error'
gym.logger.set_level(40)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import multiprocessing
from baselines.common.vec_env import SubprocVecEnv, VecExtractDictObs, VecMonitor
from level_replay.envs import VecPyTorchProcgen
from level_replay.arguments import parser
from level_replay.model import Policy, SimplePolicy
from analysis.procgen_xy import get_xy_from_gym_env

def get_args():
    args, unknown = parser.parse_known_args()
    return args

class InfoXYWrapper(gym.Wrapper):
    """
    Wrapper that injects (x, y) coordinates into the info dict at every step.
    This runs inside the worker process, where we have direct access to the environment memory.
    """
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        try:
            x, y, grid_step = get_xy_from_gym_env(self.env)
            # Check for silent failure (0,0) if that's the failure mode, 
            # but get_xy_from_gym_env returns 0.0, 0.0, 0.0 on exception.
            # We assume (0,0) is invalid for maze usually? 
            # Actually (0,0) might be bottom-left wall.
            # But get_xy_from_gym_env returns floats.
            info['agent_x'] = x
            info['agent_y'] = y
            info['grid_step'] = grid_step
            info['xy_valid'] = True
        except Exception as e:
            info['agent_x'] = 0.0
            info['agent_y'] = 0.0
            info['xy_valid'] = False
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        # We can't return info in standard gym reset.
        return obs
        
    def get_xy(self):
        return get_xy_from_gym_env(self.env)

def make_env_fn(env_name, seed, distribution_mode):
    def _thunk():
        env = gym.make(f"procgen:procgen-{env_name}-v0", 
                       start_level=int(seed), 
                       num_levels=1, 
                       distribution_mode=distribution_mode)
        env = InfoXYWrapper(env)
        return env
    return _thunk

def collect_routes():
    local_parser = argparse.ArgumentParser(description="Collect Meta-RL Routes")
    local_parser.add_argument("--model_ckpt", type=str, required=True, help="Path to model checkpoint")
    local_parser.add_argument("--out_npz", type=str, required=True, help="Output .npz file")
    local_parser.add_argument("--num_tasks", type=int, default=200, help="Number of tasks (seeds) to eval")
    local_parser.add_argument("--trials_per_task", type=int, default=1, help="Trials per task (K)")
    local_parser.add_argument("--deterministic", type=int, default=1, help="Use deterministic policy")
    local_parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    local_parser.add_argument("--env_name", type=str, default="maze", help="Env name")
    local_parser.add_argument("--save_all", type=int, default=1, help="Save all trials")
    local_parser.add_argument("--num_processes", type=int, default=0, help="Number of processes (0 = auto-detect)")
    local_parser.add_argument("--discard_xy_missing", action='store_true', help="Discard tasks with missing XY")

    local_args, remaining = local_parser.parse_known_args()
    args = get_args()
    args.env_name = local_args.env_name
    
    device = torch.device(local_args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_processes = local_args.num_processes
    if num_processes == 0:
        # Auto-detect number of processes
        try:
            cpu_count = multiprocessing.cpu_count()
            # Heuristic: Procgen is CPU heavy, leave some headroom or cap if needed.
            # But user wants speed.
            num_processes = cpu_count
        except:
            num_processes = 1
            
    # Clamp to num_tasks to avoid empty processes
    num_processes = min(num_processes, local_args.num_tasks)
    
    print(f"Parallelizing collection with {num_processes} processes (SubprocVecEnv)")
    
    # Load Model (dummy env for shape)
    dummy_env = gym.make(f"procgen:procgen-{args.env_name}-v0", start_level=0, num_levels=1)
    obs_shape = dummy_env.observation_space.shape
    obs_shape = (3, obs_shape[0], obs_shape[1])
    num_actions = dummy_env.action_space.n
    dummy_env.close()
    
    print(f"Initializing Policy with arch={args.arch}, hidden_size={args.hidden_size}")
    if args.arch == "simple":
        actor_critic = SimplePolicy(obs_shape, num_actions)
    else:
        actor_critic = Policy(
            obs_shape,
            num_actions,
            arch=args.arch,
            base_kwargs={"recurrent": True, "hidden_size": args.hidden_size},
        )
    actor_critic.to(device)
    
    print(f"Loading checkpoint from {local_args.model_ckpt}")
    checkpoint = torch.load(local_args.model_ckpt, map_location=device)
    if 'state_dict' in checkpoint:
        actor_critic.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        actor_critic.load_state_dict(checkpoint['model_state_dict'])
    else:
        actor_critic.load_state_dict(checkpoint)
    actor_critic.eval()
    
    all_routes = {
        'obs': [], 'actions': [], 'xy': [], 'hiddens': [], 
        'seed': [], 'trial_id': [], 'len': [], 'return': [], 'success': []
    }
    
    # Global stats
    total_episodes_collected = 0
    total_xy_missing_steps = 0
    total_steps_collected = 0
    task_success_count = 0
    
    all_seeds = np.arange(local_args.num_tasks)
    num_batches = int(np.ceil(local_args.num_tasks / num_processes))
    
    pbar = tqdm(total=local_args.num_tasks)
    
    for b in range(num_batches):
        batch_start = b * num_processes
        batch_end = min((b + 1) * num_processes, local_args.num_tasks)
        batch_seeds = all_seeds[batch_start:batch_end]
        current_batch_size = len(batch_seeds)
        
        env_fns = [make_env_fn(args.env_name, s, args.distribution_mode) for s in batch_seeds]
        
        try:
            venv = SubprocVecEnv(env_fns)
            is_dict_obs = isinstance(venv.observation_space, gym.spaces.Dict)
            if is_dict_obs:
                venv = VecExtractDictObs(venv, "rgb")
            
            venv = VecMonitor(venv, filename=None, keep_buf=100)
            envs = VecPyTorchProcgen(venv, device, meta_rl=False)
            
            # --- Initialize Batch State ---
            rnn_hxs = torch.zeros(current_batch_size, args.hidden_size).to(device)
            # Masks: 0 for the first step of the task, then 1 forever (RL^2 within task)
            masks = torch.zeros(current_batch_size, 1).to(device)
            
            # Reset environments to start the tasks
            obs = envs.reset()
            
            # Fetch initial XYs (t=0)
            initial_xys = []
            try:
                subproc = envs.raw_venv
                if hasattr(subproc, "env_method"):
                    initial_xys = subproc.env_method("get_xy")
                else:
                    initial_xys = [(0.0, 0.0, 0.0)] * current_batch_size
            except Exception:
                initial_xys = [(0.0, 0.0, 0.0)] * current_batch_size

            # Data structures for collection
            # task_episodes[i] will contain a list of K episode dicts
            task_episodes = [[] for _ in range(current_batch_size)]
            
            # Current episode buffer for each env
            current_buffers = []
            if isinstance(obs, torch.Tensor):
                obs_np = obs.cpu().numpy().astype(np.float16)
            else:
                obs_np = obs.astype(np.float16)
            
            for i in range(current_batch_size):
                # Init first episode buffer
                buf = {
                    'obs': [obs_np[i]], 
                    'actions': [], 
                    'xy': [], 
                    'hiddens': [rnn_hxs[i].cpu().numpy()], 
                    'return': 0.0, 
                    'len': 0, 
                    'success': False,
                    'xy_missing': 0,
                    'trial_idx': 0
                }
                # Init XY
                if i < len(initial_xys) and initial_xys[i]:
                    x, y, _ = initial_xys[i]
                    buf['xy'].append([x, y])
                else:
                    buf['xy'].append([0.0, 0.0]) # Will flag missing later if needed
                    buf['xy_missing'] += 1
                
                current_buffers.append(buf)

            # Track how many episodes collected for each env
            ep_counts = np.zeros(current_batch_size, dtype=int)
            # Active mask for collection (stop collecting for an env if it reaches K)
            collecting_mask = np.ones(current_batch_size, dtype=bool)
            
            while np.any(collecting_mask):
                with torch.no_grad():
                    value, action, _, rnn_hxs = actor_critic.act(
                        obs, rnn_hxs, masks, deterministic=bool(local_args.deterministic)
                    )
                
                # Step
                obs, reward, done, infos = envs.step(action)
                
                # Update masks: 
                # RL^2: masks should ALWAYS be 1 after the very first step of the task, 
                # even if done=True (because we don't reset hidden state between trials).
                # The only time mask=0 was at init.
                masks = torch.ones(current_batch_size, 1).to(device)

                # Process results
                action_np = action.cpu().numpy()

                if isinstance(reward, torch.Tensor):
                    reward_np = reward.cpu().numpy().squeeze()
                else:
                    reward_np = reward.squeeze()
                
                if isinstance(done, torch.Tensor):
                    done_np = done.cpu().numpy()
                else:
                    done_np = done
                
                if isinstance(obs, torch.Tensor):
                    obs_np = obs.cpu().numpy().astype(np.float16)
                else:
                    obs_np = obs.astype(np.float16)

                hiddens_np = rnn_hxs.cpu().numpy()
                
                if reward_np.ndim == 0: reward_np = np.array([reward_np]) # Handle single env case
                
                for i in range(current_batch_size):
                    if not collecting_mask[i]:
                        continue
                    
                    # Append step data to current buffer
                    buf = current_buffers[i]
                    buf['actions'].append(action_np[i])
                    buf['return'] += reward_np[i]
                    buf['len'] += 1
                    
                    # Handle XY from info (step t)
                    info = infos[i]
                    valid_xy = info.get('xy_valid', False)
                    if valid_xy:
                        buf['xy'].append([info['agent_x'], info['agent_y']])
                    elif 'agent_x' in info: # Fallback if xy_valid not set but keys exist
                        buf['xy'].append([info['agent_x'], info['agent_y']])
                    else:
                        buf['xy'].append([0.0, 0.0])
                        buf['xy_missing'] += 1
                        
                    # Check for Success
                    if 'level_complete' in info and info['level_complete']:
                        buf['success'] = True
                    elif reward_np[i] > 0: # Backup check
                        buf['success'] = True

                    if done_np[i]:
                        # Episode finished
                        # Store buffer
                        task_episodes[i].append(buf)
                        ep_counts[i] += 1
                        
                        if ep_counts[i] < local_args.trials_per_task:
                            # Prepare next episode
                            # Important: 'obs' now contains the reset observation for the new episode
                            # We need to get the XY for this new start state.
                            # Query specific env for XY
                            try:
                                # This might be slow if many envs done same time, but robust
                                res = envs.raw_venv.env_method("get_xy", indices=[i])
                                if res:
                                    start_x, start_y, _ = res[0]
                                    start_xy_valid = True
                                else:
                                    start_x, start_y = 0.0, 0.0
                                    start_xy_valid = False
                            except:
                                start_x, start_y = 0.0, 0.0
                                start_xy_valid = False

                            new_buf = {
                                'obs': [obs_np[i]],
                                'actions': [],
                                'xy': [[start_x, start_y]],
                                'hiddens': [hiddens_np[i]], # Current hidden (preserved)
                                'return': 0.0,
                                'len': 0,
                                'success': False,
                                'xy_missing': 0 if start_xy_valid else 1,
                                'trial_idx': ep_counts[i]
                            }
                            current_buffers[i] = new_buf
                        else:
                            # Finished K trials for this task
                            collecting_mask[i] = False
                            pbar.update(1)
                    else:
                        # Episode continues
                        buf['obs'].append(obs_np[i])
                        buf['hiddens'].append(hiddens_np[i])
                        # XY already appended above

            envs.close()
            
            # Post-process batch
            for i in range(current_batch_size):
                episodes = task_episodes[i]
                if not episodes: continue
                
                total_episodes_collected += len(episodes)
                
                # Check for XY validity
                # If xy_missing rate is high in the selected route, warn or discard
                # Rule: Select Last Episode (or Last Success)
                
                # Find last successful episode
                successful_episodes = [ep for ep in episodes if ep['success']]
                
                selected_ep = None
                if successful_episodes:
                    selected_ep = successful_episodes[-1]
                else:
                    # Fallback to absolute last episode
                    if episodes:
                        selected_ep = episodes[-1]
                
                if selected_ep:
                    # Filter short episodes
                    if selected_ep['len'] < 5:
                        # Too short to be useful for geometry analysis
                        continue

                    # Check XY quality
                    missing_rate = selected_ep['xy_missing'] / max(1, selected_ep['len'])
                    total_xy_missing_steps += selected_ep['xy_missing']
                    total_steps_collected += selected_ep['len']
                    
                    if missing_rate > 0.05 and local_args.discard_xy_missing: # Strict threshold if requested
                         print(f"Discarding task {batch_seeds[i]} due to XY missing rate {missing_rate:.2f}")
                         continue
                    
                    # Also check for static XY (all 0 or constant)
                    xy_arr = np.array(selected_ep['xy'])
                    if len(xy_arr) > 5:
                        xy_std = np.std(xy_arr, axis=0)
                        if np.sum(xy_std) < 1e-3:
                             # print(f"Warning: Task {batch_seeds[i]} has static XY. Discarding.")
                             if local_args.discard_xy_missing:
                                 continue

                    all_routes['obs'].append(np.array(selected_ep['obs']))
                    all_routes['actions'].append(np.array(selected_ep['actions']))
                    all_routes['xy'].append(xy_arr)
                    all_routes['hiddens'].append(np.array(selected_ep['hiddens']))
                    all_routes['seed'].append(batch_seeds[i])
                    # Identify which trial it was
                    all_routes['trial_id'].append(selected_ep['trial_idx'])
                    all_routes['len'].append(selected_ep['len'])
                    all_routes['return'].append(selected_ep['return'])
                    all_routes['success'].append(selected_ep['success'])
                    
                    if selected_ep['success']:
                        task_success_count += 1
                        
        except Exception as e:
            print(f"Batch failed: {e}")
            import traceback
            traceback.print_exc()

    pbar.close()
    
    # Final Stats
    total_saved = len(all_routes['seed'])
    print(f"\nCollection Complete.")
    print(f"Total Tasks: {local_args.num_tasks}")
    print(f"Saved Routes: {total_saved}")
    print(f"Task Success Rate: {task_success_count}/{local_args.num_tasks} ({task_success_count/local_args.num_tasks*100:.1f}%)")
    
    if total_steps_collected > 0:
        print(f"Overall XY Missing Rate: {total_xy_missing_steps}/{total_steps_collected} ({total_xy_missing_steps/total_steps_collected*100:.2f}%)")
    
    # Calculate stats for lengths before filtering (we need to track skipped ones too, but for now we only have saved ones)
    # We can use lens of saved routes
    lens = np.array(all_routes['len']) if total_saved > 0 else np.array([])
    
    # Estimate discarded due to length < 5
    # We didn't track count of discarded short episodes explicitly in a var, but we can infer from logs if needed.
    # But better to just log what we have.
    
    if len(lens) > 0:
        print(f"Episode Lengths: min={np.min(lens)}, mean={np.mean(lens):.1f}, max={np.max(lens)}")
        
        # Hist bins
        # < 5 (should be 0 now), 5-10, 10-50, 50-100, >100
        bins = [0, 5, 10, 50, 100, 1000]
        hist, _ = np.histogram(lens, bins=bins)
        print(f"Length Distribution: <5: {hist[0]}, 5-10: {hist[1]}, 10-50: {hist[2]}, 50-100: {hist[3]}, >100: {hist[4]}")

    # Save debug info
    debug_log_path = os.path.join(os.path.dirname(local_args.out_npz), "collect_debug.txt")
    with open(debug_log_path, "w") as f:
        f.write(f"Collection Debug Stats\n")
        f.write(f"======================\n")
        f.write(f"Total Tasks: {local_args.num_tasks}\n")
        f.write(f"Saved Routes: {total_saved}\n")
        f.write(f"Task Success Rate: {task_success_count}/{local_args.num_tasks} ({task_success_count/local_args.num_tasks*100:.1f}%)\n")
        if total_steps_collected > 0:
            f.write(f"Overall XY Missing Rate: {total_xy_missing_steps}/{total_steps_collected} ({total_xy_missing_steps/total_steps_collected*100:.2f}%)\n")
        
        if len(lens) > 0:
            f.write(f"Episode Lengths: min={np.min(lens)}, mean={np.mean(lens):.1f}, max={np.max(lens)}\n")
            f.write(f"Length Distribution:\n")
            f.write(f"  <5: {hist[0]}\n")
            f.write(f"  5-10: {hist[1]}\n")
            f.write(f"  10-50: {hist[2]}\n")
            f.write(f"  50-100: {hist[3]}\n")
            f.write(f"  >100: {hist[4]}\n")
            
            # XY stats
            if len(all_routes['xy']) > 0:
                all_xy = np.concatenate(all_routes['xy'], axis=0)
                f.write(f"XY Stats:\n")
                f.write(f"  Min: {np.min(all_xy, axis=0)}\n")
                f.write(f"  Max: {np.max(all_xy, axis=0)}\n")
                f.write(f"  Mean: {np.mean(all_xy, axis=0)}\n")
                f.write(f"  Std: {np.std(all_xy, axis=0)}\n")
    
    print(f"Debug log saved to {debug_log_path}")

    print(f"Saving to {local_args.out_npz}")
    os.makedirs(os.path.dirname(local_args.out_npz), exist_ok=True)
    np.savez_compressed(
        local_args.out_npz,
        routes_obs=np.array(all_routes['obs'], dtype=object),
        routes_actions=np.array(all_routes['actions'], dtype=object),
        routes_xy=np.array(all_routes['xy'], dtype=object),
        routes_hiddens=np.array(all_routes['hiddens'], dtype=object),
        routes_seed=np.array(all_routes['seed']),
        routes_trial_id=np.array(all_routes['trial_id']),
        routes_ep_len=np.array(all_routes['len']),
        routes_return=np.array(all_routes['return']),
        routes_success=np.array(all_routes['success'])
    )
    print(f"Done!")

if __name__ == "__main__":
    collect_routes()
