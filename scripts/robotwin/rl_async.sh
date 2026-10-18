#!/bin/bash
# Asynchronous EXPO-FT on RoboTwin beat_block_hammer from an SFT checkpoint (train_pi_robo_async.py):
# NUM_ENVS env processes share one learner, which updates continuously while they roll out and
# hands each env the newest params at its next chunk. Reference defaults: N=K=8, edit scale 0.2, UTD 20.
# The first given GPU samples; the rest run the updates (batch 64 must divide by their count).
# Env k listens on PORT + k and renders on ENV_GPUS[k % len]. With ROBOTWIN_FIXED_SEED=1 every env
# replays the same scene; otherwise env k walks its own seed range (index 100 + k).
# Usage: [NUM_ENVS=n ENV_GPUS=i,j MEM_FRACTION=f EDIT_SCALE=b N_EDIT=k PORT=p ROBOTWIN_FIXED_SEED=1 MAX_STEPS=s]
#        rl_async.sh <gpus comma list> <sft_step_dir> <run_name>
set -uo pipefail
GPUS=$1; STEP_DIR=$2; RUN=$3
NUM_ENVS=${NUM_ENVS:-1}; ENV_GPUS=${ENV_GPUS:-${GPUS%%,*}}; MEM_FRACTION=${MEM_FRACTION:-0.85}
EDIT_SCALE=${EDIT_SCALE:-0.2}; N_EDIT=${N_EDIT:-8}; MAX_STEPS=${MAX_STEPS:-500000}
FIXED_SEED=${ROBOTWIN_FIXED_SEED:-0}
W=/root/data/atticux/agent-workspace/expo-ft
SRC=$W/src/expo-ft
O=$SRC/expo_ft/agents/vla/openpi
PORT=${PORT:-8400}
OUT=$W/rl/$RUN
mkdir -p $OUT
cd $SRC

(
  source /public/bin/network_accelerate >/dev/null
  export WANDB_API_KEY=$(cat /root/data/atticux/.wandb_key) WANDB_ENTITY=atticux
  # MEM_FRACTION applies to every learner GPU and must leave room for the envs rendering on it.
  export CUDA_VISIBLE_DEVICES=$GPUS XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION
  exec nice -n 5 $SRC/.venv/bin/python train_pi_robo_async.py \
    --config_task=configs/task/robotwin_beat_block_hammer.py \
    --dataset_path=$W/data/lerobot/robotwin/beat_block_hammer_aloha_clean_50 --num_data=50 \
    --offline_ratio=0 --max_steps=$MAX_STEPS --num_envs=$NUM_ENVS \
    --config=configs/model/expo_ft_pi_config.py \
    --config.N=8 --config.n_edit_samples=$N_EDIT --config.edit_scale=$EDIT_SCALE \
    --config.pi05_config_name=pi05_robotwin_beat_block_hammer_lora \
    --config.pi05_weight_loader_path=$STEP_DIR/params \
    --config.pi05_assets_dir=$STEP_DIR/assets \
    --config.pi05_asset_id=robotwin/beat_block_hammer_aloha_clean_50 \
    --project_name=expo-ft --run_name=$RUN --output_dir=$W/rl \
    --client_host=127.0.0.1 --client_port=$PORT --fsdp_devices=1 \
    --resume --checkpoint_model --checkpoint_buffer --checkpoint_interval=2000 --max_to_keep=2
) > $OUT/learner.log 2>&1 &
LEARNER_PID=$!
until grep -q "EnvClient listening" $OUT/learner.log 2>/dev/null; do
  kill -0 $LEARNER_PID 2>/dev/null || { echo "learner exited before listening"; exit 1; }
  sleep 5
done

# Training seeds come from their own range, away from the evaluation shards (indices 0-7).
# Each client redials until its port listens, so all can start now.
IFS=, read -ra ENV_GPU_LIST <<< "$ENV_GPUS"
ENV_PIDS=()
for ((k = 0; k < NUM_ENVS; k++)); do
  SEED_INDEX=$((FIXED_SEED == 1 ? 100 : 100 + k))
  CUDA_VISIBLE_DEVICES=${ENV_GPU_LIST[k % ${#ENV_GPU_LIST[@]}]} \
  VK_ICD_FILENAMES=/root/data/atticux/vulkan/nvidia_icd_egl.json ROBOTWIN_SEED_INDEX=$SEED_INDEX \
  ROBOTWIN_FIXED_SEED=$FIXED_SEED ROBOTWIN_ROOT=/root/data/atticux/robotwin PYTHONPATH=$SRC:$O/packages/xense-client/src \
    /root/data/atticux/robotwin/.venv/bin/python -m client.run_client --host 127.0.0.1 --port $((PORT + k)) \
    --config-task-path configs/task/robotwin_beat_block_hammer.py > $OUT/env_$k.log 2>&1 &
  ENV_PIDS+=($!)
done
trap 'kill ${ENV_PIDS[*]} $LEARNER_PID 2>/dev/null' EXIT
wait $LEARNER_PID
