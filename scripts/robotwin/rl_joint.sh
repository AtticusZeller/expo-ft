#!/bin/bash
# EXPO-FT joint update on RoboTwin beat_block_hammer from an SFT checkpoint (reference defaults:
# N=K=8, edit scale 0.2, UTD 20, 3 update calls per episode, updates after 10 episodes).
# The learner and the RoboTwin env endpoint share the given GPUs; the env renders on the first one.
# Usage: [ENV_GPU=i MEM_FRACTION=f EDIT_SCALE=b] rl_joint.sh <gpus comma list> <sft_step_dir> <run_name>
set -uo pipefail
GPUS=$1; STEP_DIR=$2; RUN=$3
# ENV_GPU renders on its own card so the learner can take MEM_FRACTION of every learner GPU.
ENV_GPU=${ENV_GPU:-${GPUS%%,*}}; MEM_FRACTION=${MEM_FRACTION:-0.85}; EDIT_SCALE=${EDIT_SCALE:-0.2}
W=/root/data/atticux/agent-workspace/expo-ft
SRC=$W/src/expo-ft
O=$SRC/expo_ft/agents/vla/openpi
PORT=8400
OUT=$W/rl/$RUN
mkdir -p $OUT
cd $SRC

(
  source /public/bin/network_accelerate >/dev/null
  export WANDB_API_KEY=$(cat /root/data/atticux/.wandb_key) WANDB_ENTITY=atticux
  # With the env on a learner GPU, MEM_FRACTION must leave it ~3.5 GB (rendering + CuRobo).
  export CUDA_VISIBLE_DEVICES=$GPUS XLA_PYTHON_CLIENT_MEM_FRACTION=$MEM_FRACTION
  exec nice -n 5 $SRC/.venv/bin/python train_pi_robo.py \
    --config_task=configs/task/robotwin_beat_block_hammer.py \
    --dataset_path=$W/data/lerobot/robotwin/beat_block_hammer_aloha_clean_50 --num_data=50 \
    --update_type=episode --num_updates=3 --offline_ratio=0 \
    --config=configs/model/expo_ft_pi_config.py \
    --config.N=8 --config.n_edit_samples=8 --config.edit_scale=$EDIT_SCALE \
    --config.pi05_config_name=pi05_robotwin_beat_block_hammer_lora \
    --config.pi05_weight_loader_path=$STEP_DIR/params \
    --config.pi05_assets_dir=$STEP_DIR/assets \
    --config.pi05_asset_id=robotwin/beat_block_hammer_aloha_clean_50 \
    --project_name=expo-ft --run_name=$RUN --output_dir=$W/rl \
    --client_host=127.0.0.1 --client_port=$PORT --fsdp_devices=1 \
    --resume --checkpoint_model --checkpoint_buffer --checkpoint_interval=2000
) > $OUT/learner.log 2>&1 &
LEARNER_PID=$!
until grep -q "EnvClient listening" $OUT/learner.log 2>/dev/null; do
  kill -0 $LEARNER_PID 2>/dev/null || { echo "learner exited before listening"; exit 1; }
  sleep 5
done

# Training seeds come from their own range, away from the evaluation shards (indices 0-7).
CUDA_VISIBLE_DEVICES=$ENV_GPU VK_ICD_FILENAMES=/root/data/atticux/vulkan/nvidia_icd_egl.json ROBOTWIN_SEED_INDEX=100 \
ROBOTWIN_ROOT=/root/data/atticux/robotwin PYTHONPATH=$SRC:$O/packages/xense-client/src \
  /root/data/atticux/robotwin/.venv/bin/python -m client.run_client --host 127.0.0.1 --port $PORT \
  --config-task-path configs/task/robotwin_beat_block_hammer.py > $OUT/env.log 2>&1 &
ENV_PID=$!
trap 'kill $ENV_PID $LEARNER_PID 2>/dev/null' EXIT
wait $LEARNER_PID
