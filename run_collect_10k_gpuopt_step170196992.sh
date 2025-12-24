#!/bin/bash
# =============================================================================
# Meta-RL Route Collection (Optimized with Filtering) - 20k tasks
#
# Based on: /root/backup/kinematics/run_collect.sh
# Updated for:
# - New checkpoint: meta-rl-maze-easy-n10k-trial10-dense-gpu-opt/model_step_170196992.tar
# - Collect 20,000 successful trajectories
# - Max steps / episode length: 128
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

EXP_NAME="run_random_seeds_10k_gpuopt_step170196992—attempt-2"
MODEL_CKPT="/root/logs/ppo/meta-rl-maze-easy-n10k-trial10-dense-gpu-opt/model_step_170196992.tar"
BASE_OUT_DIR="/root/backup/kinematics/experiments"

# =============================================================================
# COLLECTION PARAMETERS
# =============================================================================

NUM_TASKS=10000            # Target number of SUCCESSFUL trajectories
NUM_PROCESSES=128          # Parallel environments per batch
DEVICE="cuda:1"            # GPU device (cuda:0, cuda:1, etc.)
ADAPT_EPISODES=6           # Episodes for adaptation (sampling)
RECORD_EPISODES=2          # Episodes after adaptation (typically deterministic)
SEED_OFFSET=0              # Starting seed

MAX_STEPS=512              # Max steps per episode
# MAX_STEPS=128              # original
MAX_EP_LEN=256             # Max episode length to keep (also used by selector)
# MAX_EP_LEN=128             # original


DISTRIBUTION_MODE="easy"   # "easy" or "hard"
REQUIRE_SUCCESS=1          # 1 = only keep successful trajectories

# SPEED OPTIMIZATION PARAMS
BATCH_COMPLETION=0.9       # Don't wait for slowest 10% of envs
TASK_TIMEOUT=0             # Total steps per task (0 = auto)
EARLY_ABORT=1              # Abort tasks with no success after adapt phase

# =============================================================================
# DERIVED PATHS
# =============================================================================

EXP_DIR="${BASE_OUT_DIR}/${EXP_NAME}"
DATA_DIR="${EXP_DIR}/data"
LOGS_DIR="${EXP_DIR}/logs"
ROUTES_NPZ="${DATA_DIR}/routes.npz"
LOG_FILE="${LOGS_DIR}/collection.log"

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

. /root/miniconda3/etc/profile.d/conda.sh
conda activate ede
cd /root/backup/kinematics

mkdir -p "${DATA_DIR}" "${LOGS_DIR}"

# Suppress gym/numpy deprecation warnings
export PYTHONWARNINGS="ignore::DeprecationWarning:gym"

# Log to both console and file
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         META-RL ROUTE COLLECTION (Optimized)                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Experiment:    ${EXP_NAME}"
echo "Model:         ${MODEL_CKPT}"
echo "Output:        ${ROUTES_NPZ}"
echo "Started:       $(date)"
echo ""
echo "Parameters:"
echo "  ├── Device:               ${DEVICE}"
echo "  ├── Target trajectories:  ${NUM_TASKS}"
echo "  ├── Batch size:           ${NUM_PROCESSES}"
echo "  ├── Adapt episodes:       ${ADAPT_EPISODES}"
echo "  ├── Record episodes:      ${RECORD_EPISODES}"
echo "  ├── Max steps/episode:    ${MAX_STEPS}"
echo "  ├── Max episode length:   ${MAX_EP_LEN}"
echo "  ├── Require success:      ${REQUIRE_SUCCESS}"
echo "  └── Distribution mode:    ${DISTRIBUTION_MODE}"
echo ""
echo "Speed optimizations:"
echo "  ├── Batch completion:     ${BATCH_COMPLETION} (don't wait for slowest)"
echo "  ├── Task timeout:         ${TASK_TIMEOUT} (0=auto)"
echo "  └── Early abort:          ${EARLY_ABORT} (abort failed tasks)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# =============================================================================
# RUN COLLECTION
# =============================================================================

COLLECT_START=$(date +%s)

python -W ignore eval/collect_meta_routes.py \
    --env_name maze \
    --model_ckpt="${MODEL_CKPT}" \
    --out_npz="${ROUTES_NPZ}" \
    --device="${DEVICE}" \
    --num_tasks=${NUM_TASKS} \
    --num_processes=${NUM_PROCESSES} \
    --adapt_episodes=${ADAPT_EPISODES} \
    --record_episodes=${RECORD_EPISODES} \
    --seed_offset=${SEED_OFFSET} \
    --max_steps=${MAX_STEPS} \
    --max_ep_len=${MAX_EP_LEN} \
    --distribution_mode=${DISTRIBUTION_MODE} \
    --require_success=${REQUIRE_SUCCESS} \
    --batch_completion_threshold=${BATCH_COMPLETION} \
    --task_timeout=${TASK_TIMEOUT} \
    --early_abort=${EARLY_ABORT} \
    --xy_fail_policy="warn_only"

COLLECT_END=$(date +%s)
COLLECT_TIME=$((COLLECT_END - COLLECT_START))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    COLLECTION COMPLETE                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Time:          ${COLLECT_TIME}s"
echo "Output:        ${ROUTES_NPZ}"
echo "Finished:      $(date)"
echo ""

# =============================================================================
# QUICK ANALYSIS
# =============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Quick Analysis:"
echo ""

python -c "
import numpy as np
data = np.load('${ROUTES_NPZ}', allow_pickle=True)

seeds = data['routes_seed']
ep_lens = data['routes_ep_len']
success = data['routes_success']
meta = data['meta'].item() if 'meta' in data.files else {}

n = len(seeds)
unique = len(np.unique(seeds))

print(f'  Trajectories:     {n}')
print(f'  Unique seeds:     {unique} ({100*unique/max(1,n):.1f}%)')
print(f'  Success rate:     {100*success.sum()/max(1,n):.1f}%')
print(f'  Ep length:        min={ep_lens.min()}, max={ep_lens.max()}, mean={ep_lens.mean():.1f}')
if 'seeds_attempted' in meta:
    print(f'  Seeds attempted:  {meta[\"seeds_attempted\"]}')
    print(f'  Collection rate:  {100*n/max(1,meta[\"seeds_attempted\"]):.1f}%')
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. python analysis/trajectory_stats.py --routes_npz=${ROUTES_NPZ}"
echo "  2. python analysis/pkd_cycle_sampler.py --routes_npz=${ROUTES_NPZ} ..."
echo "  3. python analysis/cca_alignment.py ..."
echo ""


