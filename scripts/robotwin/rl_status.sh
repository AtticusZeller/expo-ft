#!/bin/bash
# Summarize RL runs from W&B, where train_pi_robo.py logs episodes and updates (learner.log does not):
# success per 10 episodes (the first 10 run before updates), recent success lengths, and the latest
# critic/actor metrics with the Q and critic-loss trend. Run ids are in <output_dir>/<run>/checkpoints/wandb_id.txt.
# Usage: rl_status.sh <wandb run id>... (project atticux/expo-ft)
set -uo pipefail
source /public/bin/network_accelerate >/dev/null
export WANDB_API_KEY=$(cat /root/data/atticux/.wandb_key)
timeout 180 /root/data/atticux/agent-workspace/expo-ft/src/expo-ft/.venv/bin/python - "$@" <<'PY' 2>&1 | grep -v "Loaded credentials"
import sys

import numpy as np
import wandb

KEYS = ["critic_loss", "q", "q_min", "q_max", "target_q_mean", "actor_loss", "edit_q", "entropy",
        "temperature", "select_ratio_with_edit", "actor_param_norm"]

api = wandb.Api()
for run_id in sys.argv[1:]:
    r = api.run(f"atticux/expo-ft/{run_id}")
    eps = [(x["training/success"], x["training/length"])
           for x in r.scan_history(keys=["_step", "training/success", "training/length"])]
    s = [e[0] for e in eps]
    print(f"== {r.name} ({run_id}) state={r.state} runtime={r.summary.get('_runtime', 0) / 3600:.2f}h "
          f"episodes={len(s)} success={int(sum(s))}")
    print("  per10:", " ".join(f"{int(sum(s[i:i + 10]))}/{len(s[i:i + 10])}" for i in range(0, len(s), 10)))
    print("  last10 success lengths:", [l for ok, l in eps[-10:] if ok])
    hist = list(r.scan_history(keys=["_step", "training/q", "training/critic_loss"]))
    if hist:
        print("  updates logged:", len(hist), "latest:",
              {k: round(v, 4) for k in KEYS if isinstance(v := r.summary.get(f"training/{k}"), (int, float))})
        idx = np.linspace(0, len(hist) - 1, min(8, len(hist))).astype(int)
        print("  q trend:", [round(hist[i]["training/q"], 3) for i in idx])
        print("  critic_loss trend:", [round(hist[i]["training/critic_loss"], 4) for i in idx])
PY
