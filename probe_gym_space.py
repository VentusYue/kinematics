
import gym
import procgen

def probe_gym_space():
    env = gym.make("procgen:procgen-maze-v0", start_level=0, num_levels=1)
    print(f"Obs space type: {type(env.observation_space)}")
    print(f"Obs space: {env.observation_space}")
    env.close()

if __name__ == "__main__":
    probe_gym_space()

