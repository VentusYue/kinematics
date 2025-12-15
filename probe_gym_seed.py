
import gym
import procgen
try:
    env = gym.make("procgen:procgen-maze-v0", start_level=0, num_levels=1)
    print("Gym make success")
    
    obs = env.reset()
    print("Reset success")
    
    try:
        env.seed(123)
        print("env.seed works")
        obs2 = env.reset()
        # Check if obs changes with new seed
    except Exception as e:
        print(f"env.seed failed: {e}")
        
except Exception as e:
    print(f"Gym make failed: {e}")

