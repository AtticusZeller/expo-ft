#!/bin/bash
# Night run for the EXPO-FT simulation: wait for idle GPUs in the night window, train the RoboTwin pi0.5
# LoRA SFT on as many of them as divide the batch, evaluate the final checkpoint with the base policy
# (episodes split over the GPUs, one seed range per shard), then start the EXPO-FT joint update from it.
# Idle = memory.used < IDLE_MIB and utilization 0 on every sample for IDLE_MINUTES straight.
set -uo pipefail
W=/root/data/atticux/agent-workspace/expo-ft
SRC=$W/src/expo-ft
O=$SRC/expo_ft/agents/vla/openpi
CONFIG=pi05_robotwin_beat_block_hammer_lora
EXP=${EXP:-sft_bbh_clean50_h16}
RUN_DIR=$W/checkpoints/$CONFIG/$EXP
EVAL_EPISODES=${EVAL_EPISODES:-50}
RL_RUN=${RL_RUN:-expo_bbh_joint}
START=${START:-02:00}; DEADLINE=${DEADLINE:-03:30}
IDLE_MIB=1000; IDLE_MINUTES=10; POLL=30
# Kernel-parameter norm of the initialized model as train.py logs it (1803.249 in every CPU init);
# the guard allows +-5 for GPU numerics.
INIT_PARAM_NORM=1803.25
log() { echo "[$(date '+%F %T')] $*"; }
now_s() { date +%s; }

# ---- wait for the window and for idle GPUs --------------------------------------------------------
start_s=$(date -d "$START" +%s); [ $start_s -lt $(now_s) ] && start_s=$(date -d "tomorrow $START" +%s)
deadline_s=$(date -d "$(date -d @$start_s +%F) $DEADLINE" +%s)
log "waiting until $(date -d @$start_s '+%F %T'); deadline $(date -d @$deadline_s '+%F %T')"
while [ $(now_s) -lt $start_s ]; do sleep 60; done

declare -A streak
idle=()
while [ ${#idle[@]} -eq 0 ] && [ $(now_s) -lt $deadline_s ]; do
  while IFS=, read -r i mem util; do
    i=${i// /}; mem=${mem// /}; util=${util// /}
    if [ "$mem" -lt $IDLE_MIB ] && [ "$util" -eq 0 ]; then streak[$i]=$(( ${streak[$i]:-0} + POLL )); else streak[$i]=0; fi
  done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
  for i in "${!streak[@]}"; do [ ${streak[$i]} -ge $(( IDLE_MINUTES * 60 )) ] && idle+=($i); done
  [ ${#idle[@]} -eq 0 ] && sleep $POLL
done
if [ ${#idle[@]} -eq 0 ]; then
  log "no GPU idle for ${IDLE_MINUTES} min before the deadline; not training tonight"
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  exit 0
fi
IFS=$'\n' idle=($(sort -n <<<"${idle[*]}")); unset IFS
# Batch 64 must split evenly across devices.
n=1; for k in 8 4 2; do [ ${#idle[@]} -ge $k ] && { n=$k; break; }; done
gpus=$(IFS=,; echo "${idle[*]:0:$n}")
log "idle GPUs: ${idle[*]}; training on $gpus at expo-ft $(git -C $SRC rev-parse --short HEAD), openpi $(git -C $O rev-parse --short HEAD)"

# ---- train ------------------------------------------------------------------------------------------
TRAIN_LOG=$W/logs/train_${EXP}.log
(
  cd $O
  # W&B as the user's account; the platform accelerator is the only route to api.wandb.ai.
  source /public/bin/network_accelerate >/dev/null
  export WANDB_API_KEY=$(cat /root/data/atticux/.wandb_key) WANDB_ENTITY=atticux
  export CUDA_VISIBLE_DEVICES=$gpus XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 HF_LEROBOT_HOME=$W/data/lerobot
  exec nice -n 5 $SRC/.venv/bin/python scripts/train.py $CONFIG --exp-name=$EXP --project-name=expo-ft \
    --assets-base-dir=$W/assets --checkpoint-base-dir=$W/checkpoints --overwrite
) > $TRAIN_LOG 2>&1 &
TRAIN_PID=$!

# Guard: stop at once if step 0 does not start from the loaded weights or the loss goes non-finite.
checked=0
while kill -0 $TRAIN_PID 2>/dev/null; do
  if [ $checked -eq 0 ] && line=$(grep -m1 -E "^Step 0:" $TRAIN_LOG); then
    norm=$(sed -E 's/.*param_norm=([^ ,]+).*/\1/' <<<"$line"); loss=$(sed -E 's/.*loss=([^ ,]+).*/\1/' <<<"$line")
    if ! awk -v n="$norm" -v l="$loss" -v r=$INIT_PARAM_NORM 'BEGIN{exit !(n==n+0 && l==l+0 && (n-r)^2 < 25)}'; then
      log "ABORT: step 0 '$line' (expected param_norm ~ $INIT_PARAM_NORM and a finite loss)"; kill $TRAIN_PID; wait $TRAIN_PID; exit 1
    fi
    log "step 0 ok: $line"; checked=1
  fi
  if grep -qE "^Step [0-9]+:.*loss=(nan|inf)" $TRAIN_LOG; then
    log "ABORT: non-finite loss: $(grep -m1 -E '^Step [0-9]+:.*loss=(nan|inf)' $TRAIN_LOG)"; kill $TRAIN_PID; wait $TRAIN_PID; exit 1
  fi
  sleep 30
done
wait $TRAIN_PID; rc=$?
log "training exited rc=$rc; last: $(grep -E '^Step' $TRAIN_LOG | tail -1)"
[ $rc -eq 0 ] || exit $rc

# ---- evaluate the final checkpoint: shards over the GPUs we hold, disjoint seed ranges --------------
final=$(ls $RUN_DIR | grep -E '^[0-9]+$' | sort -n | tail -1)
IFS=, read -ra gl <<<"$gpus"
k=${#gl[@]}
log "evaluating step $final: $EVAL_EPISODES episodes over $k GPU shards"
pids=()
for ((j = 0; j < k; j++)); do
  eps=$(( EVAL_EPISODES / k + (j < EVAL_EPISODES % k ? 1 : 0) ))
  [ $eps -eq 0 ] && continue
  /root/data/atticux/eval_sft_ckpt.sh ${gl[$j]} $RUN_DIR/$final $((8300 + j)) $eps $RUN_DIR/evals_night/$final/shard$j $j \
    > $W/logs/eval_${EXP}_${final}_shard$j.log 2>&1 &
  pids+=($!)
done
wait "${pids[@]}"
succ=0; total=0
for f in $RUN_DIR/evals_night/$final/shard*/eval.log; do
  r=$(grep -oE "Evaluation complete: success_rate=[0-9.]+ \(([0-9]+)/([0-9]+)\)" $f | tail -1)
  log "$(basename $(dirname $f)): ${r:-no result}"
  [[ $r =~ \(([0-9]+)/([0-9]+)\) ]] && { succ=$((succ + BASH_REMATCH[1])); total=$((total + BASH_REMATCH[2])); }
done
log "SFT step $final base-policy success: $succ/$total"

# ---- EXPO-FT joint update from the final checkpoint ------------------------------------------------
# The reference run uses 4 GPUs data-parallel; take at most that many of ours.
rl_gpus=$(IFS=,; echo "${gl[*]:0:$(( k < 4 ? k : 4 ))}")
log "starting RL $RL_RUN on GPUs $rl_gpus from step $final"
/root/data/atticux/rl_joint.sh $rl_gpus $RUN_DIR/$final $RL_RUN > $W/logs/rl_${RL_RUN}.log 2>&1
log "RL exited rc=$?"
