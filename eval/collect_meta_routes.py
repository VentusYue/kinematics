
import os
import sys
import warnings

# Suppress warnings to improve terminal output
warnings.filterwarnings("ignore")
os.environ['GYM_LOGGER_LEVEL'] = 'error'

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
import torch
import argparse
import gym
gym.logger.set_level(40)  # Set gym logger to ERROR only
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from baselines.common.vec_env import SubprocVecEnv, VecExtractDictObs, VecMonitor
from level_replay.envs import VecNormalize, VecPyTorchProcgen
from level_replay.arguments import parser
from level_replay.model import Policy, SimplePolicy
from analysis.procgen_xy import get_xy_from_venv

def get_args():
    args, unknown = parser.parse_known_args()
    return args

class XYWrapper(gym.Wrapper):
    def get_xy(self):
        from analysis.procgen_xy import get_xy_from_gym_env
        return get_xy_from_gym_env(self.env)

def make_env_fn(env_name, seed, distribution_mode):
    def _thunk():
        # Create single env with specific seed using start_level
        env = gym.make(f"procgen:procgen-{env_name}-v0", 
                       start_level=int(seed), 
                       num_levels=1, 
                       distribution_mode=distribution_mode)
        env = XYWrapper(env)
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
    local_parser.add_argument("--num_processes", type=int, default=1, help="Number of processes")

    local_args, remaining = local_parser.parse_known_args()
    args = get_args()
    args.env_name = local_args.env_name
    
    device = torch.device(local_args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_processes = local_args.num_processes
    print(f"Parallelizing collection with {num_processes} processes (SubprocVecEnv)")
    
    # Load Model (once)
    # We need to know obs shape first.
    # Create dummy env to get shape
    dummy_env = gym.make(f"procgen:procgen-{args.env_name}-v0", start_level=0, num_levels=1)
    obs_shape = dummy_env.observation_space.shape # (64, 64, 3)
    # Transpose for PyTorch: (3, 64, 64)
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
    
    all_seeds = np.arange(local_args.num_tasks)
    num_batches = int(np.ceil(local_args.num_tasks / num_processes))
    
    pbar = tqdm(total=local_args.num_tasks * local_args.trials_per_task)
    
    for b in range(num_batches):
        batch_start = b * num_processes
        batch_end = min((b + 1) * num_processes, local_args.num_tasks)
        batch_seeds = all_seeds[batch_start:batch_end]
        current_batch_size = len(batch_seeds)
        
        # Create SubprocVecEnv with specific seeds
        env_fns = [make_env_fn(args.env_name, s, args.distribution_mode) for s in batch_seeds]
        
        try:
            venv = SubprocVecEnv(env_fns)
            
            # Wrappers
            # Check if dict
            is_dict_obs = isinstance(venv.observation_space, gym.spaces.Dict)
            if is_dict_obs:
                venv = VecExtractDictObs(venv, "rgb")
            
            venv = VecMonitor(venv, filename=None, keep_buf=100)
            venv = VecNormalize(venv, ob=False, ret=False)
            
            # ToPyTorch
            envs = VecPyTorchProcgen(venv, device, meta_rl=False)
            
            rnn_hxs = torch.zeros(current_batch_size, args.hidden_size).to(device)
            masks = torch.zeros(current_batch_size, 1).to(device) # Mask=0 for start
            
            batch_data = [[{} for _ in range(local_args.trials_per_task)] for _ in range(current_batch_size)]
            
            for trial_idx in range(local_args.trials_per_task):
                
                obs = envs.reset()
                
                if trial_idx == 0:
                    masks = torch.zeros(current_batch_size, 1).to(device)
                else:
                    masks = torch.ones(current_batch_size, 1).to(device)
                
                trial_buffers = [{
                    'obs': [], 'actions': [], 'xy': [], 'hiddens': [], 
                    'return': 0, 'len': 0, 'success': False
                } for _ in range(current_batch_size)]
                
                active_mask = [True] * current_batch_size
                
                obs_np = obs.cpu().numpy().astype(np.float16)
                hiddens_np = rnn_hxs.cpu().numpy()
                
                for i in range(current_batch_size):
                    trial_buffers[i]['obs'].append(obs_np[i])
                    trial_buffers[i]['hiddens'].append(hiddens_np[i])
                    try:
                        x, y, _ = get_xy_from_venv(envs, i)
                        trial_buffers[i]['xy'].append([x, y])
                    except:
                        trial_buffers[i]['xy'].append([0.0, 0.0])
                
                while any(active_mask):
                    with torch.no_grad():
                        value, action, _, rnn_hxs = actor_critic.act(
                            obs, rnn_hxs, masks, deterministic=bool(local_args.deterministic)
                        )
                    
                    obs, reward, done, infos = envs.step(action)
                    
                    action_np = action.cpu().numpy()
                    
                    if isinstance(reward, torch.Tensor):
                        reward_np = reward.cpu().numpy().squeeze()
                    else:
                        reward_np = reward.squeeze()
                        
                    if reward_np.ndim == 0: reward_np = np.array([reward_np])
                    
                    if isinstance(done, torch.Tensor):
                        done_np = done.cpu().numpy()
                    else:
                        done_np = done
                        
                    obs_np = obs.cpu().numpy().astype(np.float16)
                    hiddens_np = rnn_hxs.cpu().numpy()
                    
                    for i in range(current_batch_size):
                        if not active_mask[i]:
                            continue
                        
                        trial_buffers[i]['actions'].append(action_np[i])
                        trial_buffers[i]['return'] += reward_np[i]
                        trial_buffers[i]['len'] += 1
                        
                        if done_np[i]:
                            active_mask[i] = False
                            info = infos[i]
                            if 'level_complete' in info:
                                if info['level_complete']:
                                    trial_buffers[i]['success'] = True
                            elif reward_np[i] > 0:
                                trial_buffers[i]['success'] = True
                        else:
                            trial_buffers[i]['obs'].append(obs_np[i])
                            trial_buffers[i]['hiddens'].append(hiddens_np[i])
                            try:
                                x, y, _ = get_xy_from_venv(envs, i)
                                trial_buffers[i]['xy'].append([x, y])
                            except:
                                trial_buffers[i]['xy'].append([0.0, 0.0])
                                
                    masks = torch.ones(current_batch_size, 1).to(device)
                
                pbar.update(current_batch_size)
                
                for i in range(current_batch_size):
                    batch_data[i][trial_idx] = trial_buffers[i]
            
            envs.close()
            
            for i in range(current_batch_size):
                trials = batch_data[i]
                successful_trials = [t for t in trials if t['success']]
                
                if successful_trials:
                    successful_trials.sort(key=lambda x: x['len'])
                    to_save = successful_trials[0]
                elif local_args.save_all:
                    to_save = trials[-1]
                else:
                    to_save = None
                    
                if to_save:
                    all_routes['obs'].append(np.array(to_save['obs']))
                    all_routes['actions'].append(np.array(to_save['actions']))
                    all_routes['xy'].append(np.array(to_save['xy']))
                    all_routes['hiddens'].append(np.array(to_save['hiddens']))
                    all_routes['seed'].append(batch_seeds[i])
                    all_routes['trial_id'].append(0)
                    all_routes['len'].append(to_save['len'])
                    all_routes['return'].append(to_save['return'])
                    all_routes['success'].append(to_save['success'])
                    
        except Exception as e:
            print(f"Batch failed: {e}")
            import traceback
            traceback.print_exc()

    pbar.close()
    
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
    print(f"Done! Saved {len(all_routes['seed'])} routes.")

if __name__ == "__main__":
    collect_routes()
