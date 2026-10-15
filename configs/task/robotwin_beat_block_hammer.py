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
    # Evaluation seeds start at 100000 * (1 + seed_index); training and evaluation should use different indices.
    config.seed_index = 0
    config.instruction_type = "seen"
    # Fallback instruction and the learner's default prompt; per-episode instructions come from RoboTwin.
    config.language_instruction = "Pick up the hammer and beat the block."
    config.image_size = (224, 224)

    # RoboTwin's step limit for this task (env_cfg/task_config/_eval_step_limit.yml).
    config.auto_reset_steps = 400
    # No wall-clock pacing in simulation: the learner never sleeps between steps.
    config.control_hz = float("inf")

    return config
