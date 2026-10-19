"""Any RoboTwin 2.0 task, named by ROBOTWIN_TASK, Easy setting (demo_clean).

The learner and the env process both load this file, so both need ROBOTWIN_TASK and ROBOTWIN_ROOT.
robotwin_beat_block_hammer.py stays the config for that task's single-task runs.
"""

import os

import yaml

from configs.task import robotwin_base

try:
    from client.envs.robotwin_env import RoboTwinEnv
except Exception:
    print("Not importing robotwin env [module]")


def get_config():
    config = robotwin_base.get_config()

    try:
        config.env = RoboTwinEnv
    except Exception:
        print("Not importing robotwin env [env]")

    task = os.environ["ROBOTWIN_TASK"]
    config.env_name = f"robotwin_{task}"
    config.robotwin_task = task
    # ROBOTWIN_TASK_CONFIG=demo_randomized evaluates the Hard setting (clutter, random background and
    # light, table height +-3 cm), which RoboTwin's leaderboard pairs with unseen instructions.
    config.task_config_name = os.environ.get("ROBOTWIN_TASK_CONFIG", "demo_clean")
    config.robotwin_root = os.environ.get("ROBOTWIN_ROOT", "../robotwin")
    # Seeds start at 100000 * (1 + seed_index); parallel eval shards pick disjoint ranges.
    config.seed_index = int(os.environ.get("ROBOTWIN_SEED_INDEX", 0))
    config.fixed_seed = os.environ.get("ROBOTWIN_FIXED_SEED", "0") == "1"
    # Official protocol by default: only some tasks have a single-attempt judge
    # (client/envs/robotwin_attempts.py), and the co-trained base is picked under the official
    # protocol for every task (user 2026-10-18).
    config.single_attempt = os.environ.get("ROBOTWIN_SINGLE_ATTEMPT", "0") == "1"
    # ROBOTWIN_SEED_TABLE=<json> takes seeds and instructions from a precomputed table
    # (scripts/robotwin/make_seed_table.py) instead of running the expert at every reset.
    config.seed_table = os.environ.get("ROBOTWIN_SEED_TABLE", "")
    config.instruction_type = os.environ.get("ROBOTWIN_INSTRUCTION_TYPE", "seen")
    # Fallback only; each episode's instruction comes from RoboTwin.
    config.language_instruction = task.replace("_", " ")
    config.image_size = (224, 224)

    # RoboTwin's step limit for this task, as its evaluation uses.
    with open(os.path.join(config.robotwin_root, "env_cfg", "task_config", "_eval_step_limit.yml"),
              encoding="utf-8") as f:
        config.auto_reset_steps = int(yaml.safe_load(f)[task])
    # No wall-clock pacing in simulation: the learner never sleeps between steps.
    config.control_hz = float("inf")

    return config
