#!/bin/bash
# Night run for the EXPO-FT simulation: wait for idle GPUs in the night window, train the RoboTwin pi0.5
# LoRA SFT on as many of them as divide the batch, evaluate the last checkpoint with the base policy
# (episodes split over the GPUs, one seed range per shard), then start the EXPO-FT joint update from it.
# SFT length matches the sample count of RoboTwin's official pi0.5 LoRA baseline (30k steps x batch 32);
# only the latest checkpoint is kept on disk (saved every SAVE_EVERY steps so a stop or crash still leaves
# one), training stops at TRAIN_STOP regardless, and only that checkpoint is evaluated. Below LOW_PCT the
# script stops before RL and the overnight checks take over (longer SFT first). RL stops at
# RL_STOP so the GPUs are free before the daytime jobs.
# Idle = memory.used < IDLE_MIB and utilization 0 on every sample for IDLE_MINUTES straight.
set -uo pipefail
W=/root/data/atticux/agent-workspace/expo-ft
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$W/src/expo-ft
O=$SRC/expo_ft/agents/vla/openpi
CONFIG=pi05_robotwin_beat_block_hammer_lora
EXP=${EXP:-sft_bbh_clean50_15k}
RUN_DIR=$W/checkpoints/$CONFIG/$EXP
EVAL_EPISODES=${EVAL_EPISODES:-50}
RL_RUN=${RL_RUN:-expo_bbh_joint}
START=${START:-01:30}; DEADLINE=${DEADLINE:-03:30}; TRAIN_STOP=${TRAIN_STOP:-06:30}; RL_STOP=${RL_STOP:-09:00}
STEPS=${STEPS:-15000}; SAVE_EVERY=1000; LOW_PCT=30
# Larger than any run, so no step is kept besides the latest.
KEEP_EVERY=1000000
IDLE_MIB=1000; IDLE_MINUTES=${IDLE_MINUTES:-10}; POLL=30
# Kernel-parameter norm of the initialized model as train.py logs it (1803.249 in every CPU init);
# the guard allows +-5 for GPU numerics.
INIT_PARAM_NORM=1803.25
log() { echo "[$(date '+%F %T')] $*"; }
now_s() { date +%s; }
# PIDs under $1, children before parents.
descendants() { local c; for c in $(pgrep -P $1); do descendants $c; echo $c; done; }

# SKIP_TRAIN=<gpu list> re-runs only the evaluation and RL of an already trained EXP on those GPUs.
if [ -z "${SKIP_TRAIN:-}" ]; then
# ---- wait for the window and for idle GPUs --------------------------------------------------------
start_s=$(date -d "$START" +%s); [ $start_s -lt $(now_s) ] && start_s=$(date -d "tomorrow $START" +%s)
day=$(date -d @$start_s +%F)
deadline_s=$(date -d "$day $DEADLINE" +%s); train_stop_s=$(date -d "$day $TRAIN_STOP" +%s); rl_stop_s=$(date -d "$day $RL_STOP" +%s)
log "waiting until $(date -d @$start_s '+%F %T'); deadline $(date -d @$deadline_s '+%F %T'); SFT $STEPS steps, stop by $TRAIN_STOP; RL stops at $RL_STOP"
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
    --assets-base-dir=$W/assets --checkpoint-base-dir=$W/checkpoints ${TRAIN_MODE:---overwrite} --num-workers=${WORKERS:-32} \
    --num-train-steps=$STEPS --save-interval=$SAVE_EVERY --keep-period=$KEEP_EVERY ${EXTRA_TRAIN_ARGS:-}
) > $TRAIN_LOG 2>&1 &
TRAIN_PID=$!

# Log lines carry a timestamp prefix, so the patterns are unanchored. TRAIN_MODE=--resume continues a
# stopped run (no step-0 line then, so only the NaN check applies).
# Guard: stop at once if step 0 does not start from the loaded weights or the loss goes non-finite.
checked=0; capped=0
while kill -0 $TRAIN_PID 2>/dev/null; do
  if [ $(now_s) -ge $train_stop_s ]; then
    log "TRAIN_STOP $TRAIN_STOP reached at '$(grep -E 'Step [0-9]+:' $TRAIN_LOG | tail -1)'; stopping training"
    kill $(descendants $TRAIN_PID) $TRAIN_PID 2>/dev/null; wait $TRAIN_PID; capped=1; break
  fi
  if [ $checked -eq 0 ] && line=$(grep -m1 -E "Step 0:" $TRAIN_LOG); then
    norm=$(sed -E 's/.*param_norm=([^ ,]+).*/\1/' <<<"$line"); loss=$(sed -E 's/.*loss=([^ ,]+).*/\1/' <<<"$line")
    if ! awk -v n="$norm" -v l="$loss" -v r=$INIT_PARAM_NORM 'BEGIN{exit !(n==n+0 && l==l+0 && (n-r)^2 < 25)}'; then
      log "ABORT: step 0 '$line' (expected param_norm ~ $INIT_PARAM_NORM and a finite loss)"; kill $TRAIN_PID; wait $TRAIN_PID; exit 1
    fi
    log "step 0 ok: $line"; checked=1
  fi
  if grep -qE "Step [0-9]+:.*loss=(nan|inf)" $TRAIN_LOG; then
    log "ABORT: non-finite loss: $(grep -m1 -E 'Step [0-9]+:.*loss=(nan|inf)' $TRAIN_LOG)"; kill $TRAIN_PID; wait $TRAIN_PID; exit 1
  fi
  sleep 30
done
if [ $capped -eq 0 ]; then
  wait $TRAIN_PID; rc=$?
  log "training exited rc=$rc; last: $(grep -E 'Step [0-9]+:' $TRAIN_LOG | tail -1)"
  [ $rc -eq 0 ] || exit $rc
fi
else
  gpus=$SKIP_TRAIN; day=$(date +%F); rl_stop_s=$(date -d "$day $RL_STOP" +%s)
  log "SKIP_TRAIN: evaluating $EXP on GPUs $gpus; RL stops at $RL_STOP"
fi

# ---- evaluate: shards over the GPUs we hold, disjoint seed ranges -----------------------------------
IFS=, read -ra gl <<<"$gpus"
k=${#gl[@]}
# Evaluates the base policy of checkpoint step $1 and sets succ/total.
eval_step() {
  local step=$1 j eps f r pids=()
  log "evaluating step $step: $EVAL_EPISODES episodes over $k GPU shards"
  for ((j = 0; j < k; j++)); do
    eps=$(( EVAL_EPISODES / k + (j < EVAL_EPISODES % k ? 1 : 0) ))
    [ $eps -eq 0 ] && continue
    $HERE/eval_sft_ckpt.sh ${gl[$j]} $RUN_DIR/$step $((8300 + j)) $eps $RUN_DIR/evals_night/$step/shard$j $j \
      > $W/logs/eval_${EXP}_${step}_shard$j.log 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  succ=0; total=0
  for f in $RUN_DIR/evals_night/$step/shard*/eval.log; do
    r=$(grep -oE "Evaluation complete: success_rate=[0-9.]+ \(([0-9]+)/([0-9]+)\)" $f | tail -1)
    log "$(basename $(dirname $f)): ${r:-no result}"
    [[ $r =~ \(([0-9]+)/([0-9]+)\) ]] && { succ=$((succ + BASH_REMATCH[1])); total=$((total + BASH_REMATCH[2])); }
  done
  log "SFT step $step base-policy success: $succ/$total"
}
pct() { if [ $total -gt 0 ]; then echo $(( 100 * succ / total )); else echo -1; fi; }

# Orbax renames a checkpoint to its bare step number only once it is complete.
final=$(ls $RUN_DIR | grep -E '^[0-9]+$' | sort -n | tail -1)
[ -n "$final" ] || { log "no complete checkpoint saved"; exit 1; }
eval_step $final; p=$(pct); chosen=$final
if [ $p -lt $LOW_PCT ]; then
  log "step $final below ${LOW_PCT}% or no result; stopping before RL for the overnight checks to take over"
  exit 2
fi

[ -n "${NO_RL:-}" ] && { log "NO_RL set; stopping after the evaluation"; exit 0; }

# ---- EXPO-FT joint update from the chosen checkpoint -----------------------------------------------
# The reference run uses 4 GPUs data-parallel; take at most that many of ours.
rl_gpus=$(IFS=,; echo "${gl[*]:0:$(( k < 4 ? k : 4 ))}")
log "starting RL $RL_RUN on GPUs $rl_gpus from step $chosen"
$HERE/rl_joint.sh $rl_gpus $RUN_DIR/$chosen $RL_RUN > $W/logs/rl_${RL_RUN}.log 2>&1 &
RL_PID=$!
while kill -0 $RL_PID 2>/dev/null && [ $(now_s) -lt $rl_stop_s ]; do sleep 60; done
if kill -0 $RL_PID 2>/dev/null; then
  # The learner, the env endpoint and its CuRobo planners all sit under rl_joint.sh.
  log "RL_STOP $RL_STOP reached; stopping RL"
  kill $(descendants $RL_PID) 2>/dev/null; sleep 20; kill $RL_PID 2>/dev/null
fi
wait $RL_PID
log "RL exited rc=$?"
