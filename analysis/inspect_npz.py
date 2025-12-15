
import numpy as np
import sys

def inspect_npz(path):
    print(f"Loading {path}")
    data = np.load(path, allow_pickle=True)
    
    routes_xy = data["routes_xy"]
    routes_success = data["routes_success"]
    
    print(f"Successes: {routes_success}")
    
    for i in range(len(routes_xy)):
        if routes_success[i]:
            print(f"\nRoute {i} (SUCCESS):")
            xy = routes_xy[i]
            print(f"  Shape: {xy.shape}")
            change = np.sum(np.abs(np.diff(xy, axis=0)))
            print(f"  Total XY change: {change}")
            if change > 0:
                print("  Agent moved!")
            else:
                print("  Agent did NOT move (despite success flag??)")
        else:
            # Check if failed routes moved
            xy = routes_xy[i]
            change = np.sum(np.abs(np.diff(xy, axis=0)))
            if change > 0:
                 # print(f"Route {i} (FAIL) moved.")
                 pass

if __name__ == "__main__":
    inspect_npz(sys.argv[1])
