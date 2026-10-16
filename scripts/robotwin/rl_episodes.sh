#!/bin/bash
# Print the finished episodes (W&B step, success, length) of an RL run; train_pi_robo.py logs them to W&B
# only, not to learner.log. The run id is in <output_dir>/<run_name>/checkpoints/wandb_id.txt.
# Usage: rl_episodes.sh <wandb run id> [entity/project]
set -uo pipefail
RUN_ID=$1; PROJECT=${2:-atticux/expo-ft}
source /public/bin/network_accelerate >/dev/null
export WANDB_API_KEY=$(cat /root/data/atticux/.wandb_key)
timeout 120 /root/data/atticux/agent-workspace/expo-ft/src/expo-ft/.venv/bin/python - "$PROJECT/$RUN_ID" <<'PY' 2>&1 | tail -1
import sys
import wandb

keys = ["_step", "training/success", "training/length"]
rows = [(r["_step"], r["training/success"], r["training/length"]) for r in wandb.Api().run(sys.argv[1]).scan_history(keys=keys)]
print(len(rows), "episodes, successes", sum(s for _, s, _ in rows), "last 6:", rows[-6:])
PY
