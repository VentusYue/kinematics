
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

# Suppress warnings to improve terminal output
warnings.filterwarnings("ignore")
os.environ['GYM_LOGGER_LEVEL'] = 'error'
gym.logger.set_level(40)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from level_replay.arguments import parser
from level_replay.model import Policy, SimplePolicy

def get_args():
    args, unknown = parser.parse_known_args()
    return args

def load_model(args, device, obs_shape, num_actions):
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
    
    print(f"Loading checkpoint from {args.model_ckpt}")
    checkpoint = torch.load(args.model_ckpt, map_location=device)
    if 'state_dict' in checkpoint:
        actor_critic.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        actor_critic.load_state_dict(checkpoint['model_state_dict'])
    else:
        actor_critic.load_state_dict(checkpoint)
    
    actor_critic.eval()
    return actor_critic

def main():
    local_parser = argparse.ArgumentParser(description="PKD Cycle Sampler")
    local_parser.add_argument("--model_ckpt", type=str, required=True, help="Path to model checkpoint")
    local_parser.add_argument("--routes_npz", type=str, required=True, help="Path to collected routes npz")
    local_parser.add_argument("--out_npz", type=str, required=True, help="Output npz file")
    local_parser.add_argument("--num_h0", type=int, default=20, help="Number of initial hidden states to sample per route")
    local_parser.add_argument("--warmup_periods", type=int, default=8, help="Number of periods to warmup")
    local_parser.add_argument("--sample_periods", type=int, default=2, help="Number of periods to check convergence")
    local_parser.add_argument("--ac_match_thresh", type=float, default=0.70, help="Action consistency threshold")
    local_parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    local_parser.add_argument("--max_routes", type=int, default=None, help="Max routes to process (for debugging)")
    
    local_args, _ = local_parser.parse_known_args()
    args = get_args()
    args.model_ckpt = local_args.model_ckpt
    
    device = torch.device(local_args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading routes from {local_args.routes_npz}")
    data = np.load(local_args.routes_npz, allow_pickle=True)
    routes_obs = data['routes_obs']
    routes_actions = data['routes_actions']
    
    num_routes = len(routes_obs)
    print(f"Found {num_routes} routes.")
    
    if local_args.max_routes is not None:
        num_routes = min(num_routes, local_args.max_routes)
        print(f"Processing first {num_routes} routes.")
        
    sample_obs = routes_obs[0]
    obs_shape = sample_obs.shape[1:]
    num_actions = 15 
    
    model = load_model(args, device, obs_shape, num_actions)
    hidden_size = args.hidden_size
    
    cycles_hidden = []
    cycles_route_id = []
    cycles_match_ratio = []
    cycles_converged = []
    
    # Statistics for debugging
    stat_match_ratios = []
    stat_diffs = []
    
    # Loop over routes
    for r_idx in tqdm(range(num_routes), desc="Sampling Cycles"):
        obs_seq = routes_obs[r_idx] # (T, 3, 64, 64)
        target_actions = routes_actions[r_idx] # (T,) or (T, 1)
        if target_actions.ndim > 1:
            target_actions = target_actions.flatten()
            
        T = obs_seq.shape[0]
        
        # We want to process `num_h0` samples in parallel
        N = local_args.num_h0
        total_periods = local_args.warmup_periods + local_args.sample_periods
        L = T * total_periods
        
        # Prepare inputs: (L, N, C, H, W)
        # obs_seq repeated `total_periods` times
        obs_long = np.tile(obs_seq, (total_periods, 1, 1, 1)) # (L, 3, 64, 64)
        obs_long_torch = torch.from_numpy(obs_long.astype(np.float32)).to(device)
        
        # Expand for batch size N
        # (L, C, H, W) -> (L, 1, C, H, W) -> (L, N, C, H, W)
        inputs = obs_long_torch.unsqueeze(1).expand(-1, N, -1, -1, -1)
        # Flatten to (L*N, C, H, W) for model
        inputs_flat = inputs.reshape(L * N, *obs_shape)
        
        # Prepare initial h0: (N, H)
        h0 = torch.randn(N, hidden_size).to(device)
        
        # Prepare masks: (L*N, 1) - all ones (no resets)
        masks = torch.ones(L * N, 1).to(device)
        
        # Run model
        with torch.no_grad():
            # model.base returns (critic_val, actor_features, final_h)
            # actor_features is the sequence of hidden states: (L*N, H)
            _, h_seq_flat, _ = model.base(inputs_flat, h0, masks)
            
            # Reshape back to (L, N, H)
            h_seq = h_seq_flat.view(L, N, hidden_size)
            
            # Extract relevant parts
            # Last period: indices [-T:]
            # Previous period: indices [-2T:-T]
            
            last_period_h = h_seq[-T:] # (T, N, H)
            prev_period_h = h_seq[-2*T:-T] # (T, N, H)
            
            # Check convergence per sample
            # diff: (N,)
            diffs = torch.mean(torch.norm(last_period_h - prev_period_h, dim=2), dim=0).cpu().numpy()
            
            # Store stats for all samples
            stat_diffs.extend(diffs.flatten().tolist())
            
            # Compute actions for last period
            # Flatten last period h for distribution: (T*N, H)
            last_period_h_flat = last_period_h.view(T * N, hidden_size)
            dist = model.dist(last_period_h_flat)
            modes = dist.mode() # (T*N,)
            modes = modes.view(T, N).cpu().numpy() # (T, N)
            
            # Compare with target actions: (T,)
            # Broadcast target: (T, 1)
            target = target_actions[:, np.newaxis]
            
            # matches: (T, N)
            matches = (modes == target)
            match_ratios = np.mean(matches, axis=0) # (N,)
            
            # Store stats
            stat_match_ratios.extend(match_ratios.flatten().tolist())
            
            # Select best candidate
            # Filter by match threshold
            valid_indices = np.where(match_ratios >= local_args.ac_match_thresh)[0]
            
            if len(valid_indices) > 0:
                # Pick the one with best match ratio, or just the first valid one?
                # Let's pick best match ratio, then lowest convergence diff
                # Sort indices by match ratio (desc), then diff (asc)
                
                # Zip and sort
                candidates = []
                for idx in valid_indices:
                    # diffs[idx] might be a 0-d array, match_ratios[idx] too
                    
                    mr = match_ratios[idx]
                    df = diffs[idx]
                    
                    # Robust scalar extraction
                    if hasattr(mr, 'item'): 
                        try:
                            mr = mr.item()
                        except ValueError: # size > 1?
                            mr = mr.mean().item()
                    
                    if hasattr(df, 'item'):
                        try:
                            df = df.item()
                        except ValueError:
                            df = df.mean().item()
                    
                    candidates.append((idx, float(mr), float(df)))
                
                # Sort: primary key -match_ratio, secondary key diff
                candidates.sort(key=lambda x: (-x[1], x[2]))
                
                best_idx = candidates[0][0]
                
                # Save
                # We need numpy (T, H)
                best_h_cycle = last_period_h[:, best_idx, :].cpu().numpy() # (T, H)
                
                cycles_hidden.append(best_h_cycle)
                cycles_route_id.append(r_idx)
                cycles_match_ratio.append(match_ratios[best_idx])
                cycles_converged.append(diffs[best_idx] < 0.1)
        
    print(f"Collected {len(cycles_hidden)} cycles from {num_routes} routes.")
    
    # Debug logging
    debug_log_path = os.path.join(os.path.dirname(local_args.out_npz), "pkd_debug.txt")
    with open(debug_log_path, "w") as f:
        f.write(f"PKD Cycle Sampler Debug Stats\n")
        f.write(f"=============================\n")
        f.write(f"Total Routes Processed: {num_routes}\n")
        f.write(f"Cycles Collected: {len(cycles_hidden)}\n")
        f.write(f"Parameters:\n")
        f.write(f"  Match Threshold: {local_args.ac_match_thresh}\n")
        f.write(f"  Num H0: {local_args.num_h0}\n")
        
        if len(stat_match_ratios) > 0:
            mr = np.array(stat_match_ratios)
            f.write(f"Global Match Ratio Stats:\n")
            f.write(f"  Mean: {np.mean(mr):.4f}\n")
            f.write(f"  Median: {np.median(mr):.4f}\n")
            f.write(f"  Min: {np.min(mr):.4f}\n")
            f.write(f"  Max: {np.max(mr):.4f}\n")
            f.write(f"  >0.95: {np.sum(mr >= 0.95)} / {len(mr)}\n")
            f.write(f"  >0.90: {np.sum(mr >= 0.90)} / {len(mr)}\n")
            f.write(f"  >0.80: {np.sum(mr >= 0.80)} / {len(mr)}\n")
            f.write(f"  >0.70: {np.sum(mr >= 0.70)} / {len(mr)}\n")
            
        if len(stat_diffs) > 0:
            df = np.array(stat_diffs)
            f.write(f"Global Convergence Diff Stats:\n")
            f.write(f"  Mean: {np.mean(df):.4f}\n")
            f.write(f"  Median: {np.median(df):.4f}\n")
            f.write(f"  Min: {np.min(df):.4f}\n")
            f.write(f"  Max: {np.max(df):.4f}\n")

        if len(cycles_match_ratio) > 0:
            f.write(f"Collected Cycles Match Ratio Stats: min={np.min(cycles_match_ratio):.4f}, mean={np.mean(cycles_match_ratio):.4f}, max={np.max(cycles_match_ratio):.4f}\n")
            f.write(f"Converged Fraction: {np.mean(cycles_converged):.2f}\n")
        else:
            f.write("No cycles collected.\n")
    print(f"Debug log saved to {debug_log_path}")

    os.makedirs(os.path.dirname(local_args.out_npz), exist_ok=True)
    np.savez_compressed(
        local_args.out_npz,
        cycles_hidden=np.array(cycles_hidden, dtype=object),
        cycles_route_id=np.array(cycles_route_id),
        cycles_match_ratio=np.array(cycles_match_ratio),
        cycles_converged=np.array(cycles_converged)
    )
    print(f"Saved to {local_args.out_npz}")

if __name__ == "__main__":
    main()
