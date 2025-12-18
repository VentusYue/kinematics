
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
from collections import Counter, defaultdict

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


def process_route_batch(model, obs_batch, actions_batch, route_indices, 
                        num_h0, warmup_periods, sample_periods, 
                        hidden_size, device, ac_match_thresh, use_amp=True):
    """
    Process a batch of routes with the same sequence length T in parallel.
    
    Args:
        obs_batch: list of (T, C, H, W) numpy arrays
        actions_batch: list of (T,) numpy arrays
        route_indices: list of original route indices
        
    Returns:
        List of (route_idx, best_h_cycle, match_ratio, converged) tuples for valid cycles
    """
    M = len(obs_batch)  # Number of routes in this batch
    T = obs_batch[0].shape[0]
    total_periods = warmup_periods + sample_periods
    N = num_h0
    B = M * N  # Total batch size: routes × h0 samples
    
    # Stack observations: (M, T, C, H, W) -> (T, M, C, H, W)
    obs_stacked = np.stack(obs_batch, axis=0)  # (M, T, C, H, W)
    obs_tensor = torch.from_numpy(obs_stacked.astype(np.float32)).to(device)
    obs_tensor = obs_tensor.permute(1, 0, 2, 3, 4)  # (T, M, C, H, W)
    
    # Stack target actions: (M, T)
    actions_stacked = np.stack(actions_batch, axis=0)  # (M, T)
    
    # Initialize h0 for all routes × h0 samples: (M*N, H)
    # Reshape later to (M, N, H) for per-route analysis
    h_current = torch.randn(B, hidden_size, device=device)
    
    # Pre-allocate masks
    masks = torch.ones(B, 1, device=device)
    
    results = []
    batch_stats = {
        "total_candidates_tested": B,
        "candidates_passed_thresh": 0,
        "match_ratios": [],
        "convergence_diffs": [],
    }
    
    amp_dtype = torch.float16 if device.type == 'cuda' else torch.float32
    
    with torch.no_grad():
        # Storage for hidden states from last two periods
        prev_period_h_list = []
        last_period_h_list = []
        
        # Process each period
        for period_idx in range(total_periods):
            for t in range(T):
                # obs_tensor[t]: (M, C, H, W)
                # Expand to (M, N, C, H, W) then flatten to (M*N, C, H, W)
                obs_t = obs_tensor[t].unsqueeze(1).expand(-1, N, -1, -1, -1)  # (M, N, C, H, W)
                obs_t = obs_t.reshape(B, *obs_t.shape[2:])  # (M*N, C, H, W)
                
                # Forward pass with optional AMP
                if use_amp and device.type == 'cuda':
                    with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                        _, h_current, _ = model.base(obs_t, h_current.float(), masks)
                else:
                    _, h_current, _ = model.base(obs_t, h_current, masks)
                
                # Store hidden states for last two periods
                if period_idx == total_periods - 2:
                    prev_period_h_list.append(h_current.clone())
                elif period_idx == total_periods - 1:
                    last_period_h_list.append(h_current.clone())
        
        # Stack: (T, M*N, H) -> reshape to (T, M, N, H)
        prev_period_h = torch.stack(prev_period_h_list, dim=0).view(T, M, N, hidden_size)
        last_period_h = torch.stack(last_period_h_list, dim=0).view(T, M, N, hidden_size)
        
        # Convergence diff per route per h0 sample: (M, N)
        diffs = torch.mean(torch.norm(last_period_h - prev_period_h, dim=3), dim=0).cpu().numpy()
        
        # Compute actions: reshape last_period_h to (T*M*N, H)
        last_h_flat = last_period_h.permute(1, 2, 0, 3).reshape(M * N * T, hidden_size)  # (M, N, T, H) -> (M*N*T, H)
        dist = model.dist(last_h_flat.float())
        modes = dist.mode().view(M, N, T).cpu().numpy()  # (M, N, T)
        
        # Compare with targets: (M, T) -> broadcast to (M, N, T)
        targets = actions_stacked[:, np.newaxis, :]  # (M, 1, T)
        matches = (modes == targets)  # (M, N, T)
        match_ratios = np.mean(matches, axis=2)  # (M, N)
        
        # Process each route
        for m_idx in range(M):
            r_idx = route_indices[m_idx]
            mr_route = match_ratios[m_idx]  # (N,)
            df_route = diffs[m_idx]  # (N,)
            
            batch_stats["match_ratios"].extend(mr_route.tolist())
            batch_stats["convergence_diffs"].extend(df_route.tolist())
            
            # Filter by threshold
            valid_cands = np.where(mr_route >= ac_match_thresh)[0]
            batch_stats["candidates_passed_thresh"] += len(valid_cands)
            
            if len(valid_cands) > 0:
                candidates = []
                for idx in valid_cands:
                    mr = float(mr_route[idx])
                    df = float(df_route[idx])
                    candidates.append((idx, mr, df))
                
                candidates.sort(key=lambda x: (-x[1], x[2]))
                best_idx = candidates[0][0]
                
                # Extract best hidden cycle: (T, H)
                best_h_cycle = last_period_h[:, m_idx, best_idx, :].cpu().numpy()
                
                results.append((
                    r_idx,
                    best_h_cycle,
                    mr_route[best_idx],
                    df_route[best_idx] < 0.1,
                    T
                ))
    
    return results, batch_stats

def main():
    local_parser = argparse.ArgumentParser(description="PKD Cycle Sampler")
    local_parser.add_argument("--model_ckpt", type=str, required=True, help="Path to model checkpoint")
    local_parser.add_argument("--routes_npz", type=str, required=True, help="Path to collected routes npz")
    local_parser.add_argument("--out_npz", type=str, required=True, help="Output npz file")
    local_parser.add_argument("--num_h0", type=int, default=20, help="Number of initial hidden states to sample per route")
    local_parser.add_argument("--warmup_periods", type=int, default=8, help="Number of periods to warmup")
    local_parser.add_argument("--sample_periods", type=int, default=2, help="Number of periods to check convergence")
    local_parser.add_argument("--ac_match_thresh", type=float, default=0.95, help="Action consistency threshold")
    local_parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    local_parser.add_argument("--max_routes", type=int, default=None, help="Max routes to process (for debugging)")
    local_parser.add_argument("--batch_size", type=int, default=32, help="Number of routes to batch together (same length)")
    local_parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    local_parser.add_argument("--min_length", type=int, default=None, help="Minimum sequence length to include (default: no limit)")
    local_parser.add_argument("--max_length", type=int, default=None, help="Maximum sequence length to include (default: no limit)")
    local_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    local_args, _ = local_parser.parse_known_args()
    args = get_args()
    args.model_ckpt = local_args.model_ckpt
    
    # Set seeds for reproducibility
    if local_args.seed is not None:
        print(f"Setting random seed to {local_args.seed}")
        np.random.seed(local_args.seed)
        torch.manual_seed(local_args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(local_args.seed)
            
    device = torch.device(local_args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable cudnn benchmark for faster convolutions
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    print(f"Loading routes from {local_args.routes_npz}")
    data = np.load(local_args.routes_npz, allow_pickle=True)
    routes_obs = data["routes_obs"]
    routes_actions = data["routes_actions"]

    num_routes_total = len(routes_obs)
    print(f"Found {num_routes_total} routes.")

    # Optionally filter to routes that actually have a selected episode
    if "routes_selected_ep" in data.files:
        routes_selected_ep = data["routes_selected_ep"]
        candidate_indices = [i for i in range(num_routes_total) if routes_selected_ep[i] >= 0]
        print(f"{len(candidate_indices)} routes have selected episodes (selected_ep>=0).")
    else:
        candidate_indices = list(range(num_routes_total))

    # Drop any routes with empty observation sequences (T == 0)
    valid_indices = []
    empty_count = 0
    for i in candidate_indices:
        obs_i = routes_obs[i]
        if obs_i is None:
            empty_count += 1
            continue
        if hasattr(obs_i, "shape") and obs_i.shape[0] == 0:
            empty_count += 1
            continue
        valid_indices.append(i)
    
    if empty_count > 0:
        print(f"Dropped {empty_count} routes with empty observations.")

    if local_args.max_routes is not None:
        valid_indices = valid_indices[: local_args.max_routes]
        print(f"Processing first {len(valid_indices)} valid routes due to --max_routes.")

    num_routes = len(valid_indices)
    if num_routes == 0:
        print("No non-empty selected routes found; saving empty cycles file and exiting.")
        os.makedirs(os.path.dirname(local_args.out_npz) or ".", exist_ok=True)
        np.savez_compressed(
            local_args.out_npz,
            cycles_hidden=np.array([], dtype=object),
            cycles_route_id=np.array([], dtype=np.int64),
            cycles_match_ratio=np.array([], dtype=np.float32),
            cycles_converged=np.array([], dtype=np.bool_),
        )
        print(f"Saved empty cycles file to {local_args.out_npz}")
        return

    # Use the first valid route to infer obs_shape
    sample_obs = routes_obs[valid_indices[0]]
    obs_shape = sample_obs.shape[1:]
    num_actions = 15 
    
    model = load_model(args, device, obs_shape, num_actions)
    hidden_size = args.hidden_size
    
    # Statistics tracking
    cycles_hidden = []
    cycles_route_id = []
    cycles_match_ratio = []
    cycles_converged = []
    
    # Debug statistics
    stats = {
        "routes_processed": 0,
        "routes_with_cycles": 0,
        "routes_no_candidates": 0,
        "total_candidates_tested": 0,
        "candidates_passed_thresh": 0,
        "match_ratios": [],
        "convergence_diffs": [],
        "cycle_lengths": [],
    }
    
    # =========================================================================
    # OPTIMIZATION: Group routes by sequence length for batched processing
    # =========================================================================
    print(f"\nGrouping routes by sequence length...")
    
    # Length filtering info
    min_len = local_args.min_length if local_args.min_length is not None else 2
    max_len = local_args.max_length if local_args.max_length is not None else float('inf')
    if local_args.min_length is not None or local_args.max_length is not None:
        print(f"Length filter: [{local_args.min_length or 'None'}, {local_args.max_length or 'None'}]")
    
    # Group routes by their sequence length T
    length_to_routes = defaultdict(list)
    filtered_by_length = 0
    for r_idx in valid_indices:
        obs_seq = routes_obs[r_idx]
        T = obs_seq.shape[0]
        if T >= max(2, min_len) and T <= max_len:  # Apply length filter
            length_to_routes[T].append(r_idx)
        else:
            if T < 2:
                stats["routes_no_candidates"] += 1
            else:
                filtered_by_length += 1
            stats["routes_processed"] += 1
    
    if filtered_by_length > 0:
        print(f"Filtered out {filtered_by_length} routes by length constraint.")
    
    # Sort by length for more predictable processing
    sorted_lengths = sorted(length_to_routes.keys())
    total_groups = sum(
        (len(length_to_routes[T]) + local_args.batch_size - 1) // local_args.batch_size 
        for T in sorted_lengths
    )
    
    print(f"Found {len(sorted_lengths)} unique sequence lengths, {total_groups} batches total")
    print(f"Batch size: {local_args.batch_size} routes, AMP: {not local_args.no_amp}")
    print(f"Threshold: ac_match_thresh={local_args.ac_match_thresh}")
    print()
    
    use_amp = not local_args.no_amp
    
    # Process batches grouped by length
    with tqdm(total=total_groups, desc="Processing batches") as pbar:
        for T in sorted_lengths:
            route_indices_for_T = length_to_routes[T]
            
            # Process in batches of batch_size routes
            for batch_start in range(0, len(route_indices_for_T), local_args.batch_size):
                batch_end = min(batch_start + local_args.batch_size, len(route_indices_for_T))
                batch_route_indices = route_indices_for_T[batch_start:batch_end]
                
                # Gather observations and actions for this batch
                obs_batch = [routes_obs[r_idx] for r_idx in batch_route_indices]
                actions_batch = [routes_actions[r_idx] for r_idx in batch_route_indices]
                
                stats["routes_processed"] += len(batch_route_indices)
                
                # Process the batch
                results, batch_stats = process_route_batch(
                    model=model,
                    obs_batch=obs_batch,
                    actions_batch=actions_batch,
                    route_indices=batch_route_indices,
                    num_h0=local_args.num_h0,
                    warmup_periods=local_args.warmup_periods,
                    sample_periods=local_args.sample_periods,
                    hidden_size=hidden_size,
                    device=device,
                    ac_match_thresh=local_args.ac_match_thresh,
                    use_amp=use_amp,
                )
                
                # Aggregate results
                for r_idx, best_h_cycle, match_ratio, converged, cycle_len in results:
                    cycles_hidden.append(best_h_cycle)
                    cycles_route_id.append(r_idx)
                    cycles_match_ratio.append(match_ratio)
                    cycles_converged.append(converged)
                    stats["routes_with_cycles"] += 1
                    stats["cycle_lengths"].append(cycle_len)
                
                # Routes without valid cycles
                stats["routes_no_candidates"] += len(batch_route_indices) - len(results)
                
                # Aggregate stats
                stats["total_candidates_tested"] += batch_stats["total_candidates_tested"]
                stats["candidates_passed_thresh"] += batch_stats["candidates_passed_thresh"]
                stats["match_ratios"].extend(batch_stats["match_ratios"])
                stats["convergence_diffs"].extend(batch_stats["convergence_diffs"])
                
                pbar.update(1)
                pbar.set_postfix({
                    'T': T, 
                    'cycles': stats["routes_with_cycles"],
                    'rate': f"{100*stats['routes_with_cycles']/max(1,stats['routes_processed']):.1f}%"
                })

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================
    print("\n" + "="*60)
    print("PKD CYCLE SAMPLER DIAGNOSTICS")
    print("="*60)
    
    print(f"\n[Routes]")
    print(f"  Total processed: {stats['routes_processed']}")
    print(f"  With valid cycles: {stats['routes_with_cycles']} ({100*stats['routes_with_cycles']/max(1,stats['routes_processed']):.1f}%)")
    print(f"  No candidates passed: {stats['routes_no_candidates']}")
    
    print(f"\n[Candidates]")
    print(f"  Total tested: {stats['total_candidates_tested']}")
    print(f"  Passed threshold ({local_args.ac_match_thresh}): {stats['candidates_passed_thresh']} ({100*stats['candidates_passed_thresh']/max(1,stats['total_candidates_tested']):.1f}%)")
    
    if stats["match_ratios"]:
        mr_arr = np.array(stats["match_ratios"])
        print(f"\n[Match Ratios]")
        print(f"  Min: {mr_arr.min():.3f}")
        print(f"  Max: {mr_arr.max():.3f}")
        print(f"  Mean: {mr_arr.mean():.3f}")
        print(f"  Median: {np.median(mr_arr):.3f}")
        
        # Distribution
        bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
        hist, _ = np.histogram(mr_arr, bins=bins)
        print(f"  Distribution:")
        for i in range(len(bins)-1):
            print(f"    [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]}")
    
    if stats["convergence_diffs"]:
        cd_arr = np.array(stats["convergence_diffs"])
        print(f"\n[Convergence Diffs]")
        print(f"  Min: {cd_arr.min():.4f}")
        print(f"  Max: {cd_arr.max():.4f}")
        print(f"  Mean: {cd_arr.mean():.4f}")
        converged_count = (cd_arr < 0.1).sum()
        print(f"  Converged (diff < 0.1): {converged_count} ({100*converged_count/len(cd_arr):.1f}%)")
    
    if stats["cycle_lengths"]:
        cl_arr = np.array(stats["cycle_lengths"])
        print(f"\n[Cycle Lengths]")
        print(f"  Min: {cl_arr.min()}")
        print(f"  Max: {cl_arr.max()}")
        print(f"  Mean: {cl_arr.mean():.1f}")
    
    print("="*60 + "\n")
    
    # =========================================================================
    # SAVE
    # =========================================================================
    print(f"Collected {len(cycles_hidden)} cycles from {num_routes} routes.")
    
    os.makedirs(os.path.dirname(local_args.out_npz) or ".", exist_ok=True)
    np.savez_compressed(
        local_args.out_npz,
        cycles_hidden=np.array(cycles_hidden, dtype=object),
        cycles_route_id=np.array(cycles_route_id),
        cycles_match_ratio=np.array(cycles_match_ratio),
        cycles_converged=np.array(cycles_converged),
        # Meta info
        meta=dict(
            num_h0=local_args.num_h0,
            warmup_periods=local_args.warmup_periods,
            sample_periods=local_args.sample_periods,
            ac_match_thresh=local_args.ac_match_thresh,
            routes_processed=stats["routes_processed"],
            routes_with_cycles=stats["routes_with_cycles"],
            candidates_passed_rate=stats["candidates_passed_thresh"]/max(1,stats["total_candidates_tested"]),
        ),
    )
    print(f"Saved to {local_args.out_npz}")

if __name__ == "__main__":
    main()
