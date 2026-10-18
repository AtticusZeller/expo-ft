#!/bin/bash
# Evaluate one SFT checkpoint with base actions only on RoboTwin (Easy): the learner-side eval and the
# RoboTwin env endpoint run as a pair on one GPU and talk over a local port.
# Usage: eval_sft_ckpt.sh <gpu> <ckpt_step_dir> <port> <num_episodes> <out_dir> [seed_index]
#   seed_index picks the seed range 100000 * (1 + seed_index); parallel shards use distinct indices.
#   <gpu> may be empty to run the learner on CPU (debugging).
#   TASK picks another RoboTwin task (configs/task/robotwin_task.py, official protocol); PI05_CONFIG and
#   ASSET_ID name the checkpoint's openpi config and norm-stats asset. The defaults are beat_block_hammer's.
set -uo pipefail
GPU=$1; STEP_DIR=$2; PORT=$3; EPISODES=$4; OUT=$5; SEED_INDEX=${6:-0}
TASK=${TASK:-beat_block_hammer}
PI05_CONFIG=${PI05_CONFIG:-pi05_robotwin_beat_block_hammer_lora}
ASSET_ID=${ASSET_ID:-robotwin/beat_block_hammer_aloha_clean_50}
W=/root/data/atticux/agent-workspace/expo-ft
SRC=$W/src/expo-ft
O=$SRC/expo_ft/agents/vla/openpi
if [ "$TASK" = beat_block_hammer ]; then CONFIG_TASK=configs/task/robotwin_beat_block_hammer.py
else CONFIG_TASK=configs/task/robotwin_task.py; fi
# Both sides load the task config, which reads these.
export ROBOTWIN_TASK=$TASK ROBOTWIN_ROOT=/root/data/atticux/robotwin
mkdir -p "$OUT"
cd $SRC

# Learner side first: run_client dials once and exits if nobody is listening yet.
# Leave GPU memory for the env process (rendering + CuRobo, ~3.5 GB).
if [ -z "$GPU" ]; then export JAX_PLATFORMS=cpu; fi
CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  $SRC/.venv/bin/python eval_droid_policy.py \
  --config configs/model/expo_ft_pi_config.py \
  --config.pi05_config_name=$PI05_CONFIG \
  --config.pi05_weight_loader_path=$STEP_DIR/params \
  --config.pi05_assets_dir=$STEP_DIR/assets \
  --config.pi05_asset_id=$ASSET_ID \
  --config_task $CONFIG_TASK \
  --dataset_path $W/data/lerobot/robotwin/${TASK}_aloha_clean_50 --num_data 1 \
  --only_base_actions --num_episodes $EPISODES --client_host 127.0.0.1 --client_port $PORT \
  > "$OUT/eval.log" 2>&1 &
EVAL_PID=$!
until grep -q "EnvClient listening" "$OUT/eval.log" 2>/dev/null; do
  kill -0 $EVAL_PID 2>/dev/null || { echo "eval exited before listening"; exit 1; }
  sleep 5
done

# Env endpoint: RoboTwin's venv, headless Vulkan through the EGL ICD.
CUDA_VISIBLE_DEVICES=$GPU VK_ICD_FILENAMES=/root/data/atticux/vulkan/nvidia_icd_egl.json ROBOTWIN_SEED_INDEX=$SEED_INDEX \
PYTHONPATH=$SRC:$O/packages/xense-client/src \
  /root/data/atticux/robotwin/.venv/bin/python -m client.run_client --host 127.0.0.1 --port $PORT \
  --config-task-path $CONFIG_TASK > "$OUT/env.log" 2>&1 &
ENV_PID=$!
trap 'kill $ENV_PID $EVAL_PID 2>/dev/null' EXIT
wait $EVAL_PID
RC=$?
grep -h "success_rate=" "$OUT/eval.log" | tail -1
exit $RC
