"""RoboTwin 2.0 beat_block_hammer, Easy setting (demo_clean)."""

import os

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

    config.env_name = "robotwin_beat_block_hammer"
    config.robotwin_task = "beat_block_hammer"
    config.task_config_name = "demo_clean"
    # The RoboTwin checkout the env process runs against; a sibling of expo-ft in this workspace.
    config.robotwin_root = os.environ.get("ROBOTWIN_ROOT", "../robotwin")
    # Seeds start at 100000 * (1 + seed_index). The env process reads this file itself, so parallel eval shards
    # and online training pick disjoint seed ranges through ROBOTWIN_SEED_INDEX.
    config.seed_index = int(os.environ.get("ROBOTWIN_SEED_INDEX", 0))
    # ROBOTWIN_FIXED_SEED=1 replays the first valid seed of that range every episode (single-scene debugging).
    config.fixed_seed = os.environ.get("ROBOTWIN_FIXED_SEED", "0") == "1"
    # One strike per episode for training and evaluation alike: a miss ends the episode as a
    # failure (user 2026-10-18). ROBOTWIN_SINGLE_ATTEMPT=0 restores RoboTwin's protocol, which
    # lets the policy retry until step_lim.
    config.single_attempt = os.environ.get("ROBOTWIN_SINGLE_ATTEMPT", "1") == "1"
    # ROBOTWIN_STEP_LIMIT=n ends an episode as a failure after n steps instead of step_lim (400), with
    # retries allowed inside it (user 2026-10-18: 150, near the demonstrations' ~120). 0 keeps step_lim.
    config.step_limit = int(os.environ.get("ROBOTWIN_STEP_LIMIT", 0))
    config.instruction_type = "seen"
    # Fallback instruction and the learner's default prompt; per-episode instructions come from RoboTwin.
    config.language_instruction = "Pick up the hammer and beat the block."
    config.image_size = (224, 224)

    # RoboTwin's step limit for this task (env_cfg/task_config/_eval_step_limit.yml).
    config.auto_reset_steps = config.step_limit or 400
    # No wall-clock pacing in simulation: the learner never sleeps between steps.
    config.control_hz = float("inf")

    return config
