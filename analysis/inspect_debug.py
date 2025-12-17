
import numpy as np
import sys

def inspect(routes_path, cycles_path=None):
    print(f"--- Inspecting {routes_path} ---")
    try:
        data = np.load(routes_path, allow_pickle=True)
        print(f"Keys: {list(data.keys())}")
        
        lens = data['routes_ep_len']
        print(f"Episode lengths stats: Mean={np.mean(lens):.2f}, Min={np.min(lens)}, Max={np.max(lens)}")
        print(f"Lengths histogram: {np.histogram(lens, bins=10)[0]}")
        
        success = data['routes_success']
        print(f"Success rate: {np.mean(success):.2f}")
        
        # Check obs shape
        obs0 = data['routes_obs'][0]
        print(f"Sample obs shape: {obs0.shape}")
        
        # Check XY
        xy0 = data['routes_xy'][0]
        print(f"Sample XY shape: {xy0.shape}")
        print(f"Sample XY data:\n{xy0}")
        
        # Check range of XY across all
        all_xy = np.concatenate(data['routes_xy'])
        print(f"XY range: X [{np.min(all_xy[:,0]):.2f}, {np.max(all_xy[:,0]):.2f}], Y [{np.min(all_xy[:,1]):.2f}, {np.max(all_xy[:,1]):.2f}]")
        
        # Check step sizes
        if len(xy0) > 1:
            diffs = np.linalg.norm(xy0[1:] - xy0[:-1], axis=1)
            print(f"Sample step sizes: {diffs}")
            print(f"Median step: {np.median(diffs)}")
            
    except Exception as e:
        print(f"Error loading routes: {e}")

    if cycles_path:
        print(f"\n--- Inspecting {cycles_path} ---")
        try:
            cdata = np.load(cycles_path, allow_pickle=True)
            print(f"Keys: {list(cdata.keys())}")
            
            hiddens = cdata['cycles_hidden']
            print(f"Number of cycles: {len(hiddens)}")
            
            # Check shapes
            shapes = [h.shape for h in hiddens]
            lengths = [h.shape[0] for h in hiddens]
            print(f"Cycle lengths stats: Mean={np.mean(lengths):.2f}, Min={np.min(lengths)}, Max={np.max(lengths)}")
            
            # Check if they match routes
            r_ids = cdata['cycles_route_id']
            print(f"Route IDs: {r_ids[:10]}...")
            
        except Exception as e:
            print(f"Error loading cycles: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_data.py <routes.npz> [cycles.npz]")
    else:
        inspect(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

