## Meta-RL Kinematic Procgen Experiments

This repository contains experiment code for a **meta-RL version of Kinematic** on the Procgen benchmark.
It is adapted from the original EDE implementation at [facebookresearch/ede](https://github.com/facebookresearch/ede)
and uses the Procgen environments from [openai/procgen](https://github.com/openai/procgen).

Author: Ventus.

### Environment and setup

- Use a Conda environment named `ede` for all runs.
- Install dependencies for level-replay, baselines, and Procgen following the instructions in the upstream repositories:
  - [facebookresearch/ede](https://github.com/facebookresearch/ede)
  - [openai/procgen](https://github.com/openai/procgen)

Example setup:

```bash
conda create -n ede python=3.10
conda activate ede

# Install level-replay + baselines + Procgen as in the original EDE repo.
# Then, inside this repository:
pip install -r requirements.txt
```

All commands below assume:

```bash
conda activate ede
```

Only Procgen environments are needed for these experiments; Crafter and other environments are not required.

### Train PPO on Procgen (maze)

```bash
CUDA_VISIBLE_DEVICES=1 python train_ppo.py \
  --env_name=maze \
  --num_processes=32 \
  --num_train_seeds=200 \
  --num_env_steps=10000000 \
  --log_interval=50 \
  --record_eval_video=True \
  --eval_video_envs=maze \
  --eval_video_episodes=3 \
  --eval_video_freq=20000 \
  --num_test_seeds=10 \
  --final_num_test_seeds=1000 \
  --wandb_project=off-policy-procgen \
  --model_path=models/ppo-maze-run1-final.tar \
  --save_interval=60 \
  --verbose
```

### Train recurrent PPO (GRU) on Procgen (maze)

This uses the same convolutional encoder as the feedforward PPO agent, but adds
an extra GRU layer on top of the latent features so the policy has a persistent
hidden state across time steps (i.e. it can remember information from previous
observations on the same level, maybe helpful for kinematics analysis?).

```bash
CUDA_VISIBLE_DEVICES=1 python -u train_ppo_recurrent.py \
  --env_name=maze \
  --arch=small \
  --num_processes=64 \
  --num_steps=256 \
  --num_env_steps=25000000 \
  --num_train_seeds=200 \
  --num_test_seeds=10 \
  --final_num_test_seeds=1000 \
  --hidden_size=256 \
  --lr=5e-4 \
  --eps=1e-5 \
  --clip_param=0.2 \
  --ppo_epoch=3 \
  --num_mini_batch=8 \
  --distribution_mode=easy \
  --log_interval=50 \
  --record_eval_video=True \
  --eval_video_envs=maze \
  --eval_video_episodes=3 \
  --eval_video_freq=20000 \
  --wandb_project=off-policy-procgen \
  --wandb_tags=\"ppo,rnn,maze\" \
  --model_path=models/ppo-recurrent-maze-run1-final.tar \
  --save_interval=60 \
  --save_model=True \
  --log_rnn_stats True \
  --log_rnn_traces False \
  --verbose
```

### Train Rainbow DQN (EDE) on Procgen (maze)

```bash
CUDA_VISIBLE_DEVICES=0 python train_rainbow.py \
  --env_name=maze \
  --num_processes=64 \
  --num_train_seeds=200 \
  --T_max=25000000 \
  --eval_freq=10000 \
  --wandb True \
  --wandb_project=off-policy-procgen \
  --model_path=models/dqn-maze-run1-final.tar \
  --save_interval=60 \
  --save_model=True
```

### Important arguments

- **`env_name`**: Procgen environment name (e.g. `maze`, `heist`).
- **`num_processes`**: number of parallel CPU environments; higher values increase throughput but use more CPU.
- **`num_train_seeds`**: how many Procgen levels are used for training; test levels are sampled outside this range.
- **`num_env_steps`** / **`T_max`**:
  - `num_env_steps`: total environment steps for PPO.
  - `T_max`: total environment steps for Rainbow DQN.
- **`log_interval`** / **`eval_freq`**:
  - `log_interval`: how often PPO logs training and test statistics (in updates).
  - `eval_freq`: how often DQN runs evaluation (in environment steps).
- **`record_eval_video`**, **`eval_video_envs`**, **`eval_video_episodes`**, **`eval_video_freq`**:
  - Enable periodic video evaluation and control which Procgen envs are recorded to WandB, how many episodes, and how often.
- **`num_test_seeds`**, **`final_num_test_seeds`**:
  - Number of held-out seeds used for periodic and final test evaluation.
- **`wandb`**, **`wandb_project`**, **`wandb_group`**, **`wandb_tags`**:
  - Configure Weights & Biases logging. Set `wandb` to `True` for online logging or use environment variables to control WandB mode.
- **`model_path`**, **`save_interval`**, **`save_model`**:
  - Where checkpoints are written and whether to save the final model. `save_interval` is in minutes of wall-clock time.


