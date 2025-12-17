import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# We need to add 'ede' to path so 'level_replay' and 'analysis' can be imported
sys.path.append("/root/test/ede")

from level_replay.model import Policy

def create_ckpt():
    obs_shape = (3, 64, 64)
    num_actions = 15 # Maze usually has 15 actions
    
    actor_critic = Policy(
        obs_shape,
        num_actions,
        arch="large",
        base_kwargs={"recurrent": True, "hidden_size": 256},
    )
    
    state_dict = actor_critic.state_dict()
    
    torch.save(state_dict, "dummy_ckpt.pt")
    print("Saved dummy_ckpt.pt")

if __name__ == "__main__":
    create_ckpt()

