"""Shared settings for RoboTwin 2.0 tasks (Aloha-AgileX, simulated).

Per-task files add `env`, `env_name`, `language_instruction`, `auto_reset_steps` and
`control_hz` once the RoboTwin env endpoint exists.
"""

import ml_collections
import numpy as np


def get_config():
    config = ml_collections.ConfigDict()

    # Selects the LeRobot v3.0 prefill loader in train_pi_robo / eval_droid_policy.
    config.env_type = "robotwin"

    # Two arms, each 6 joint targets + 1 gripper, in the dataset's action order.
    config.action_dim = 14
    # Action-shaped zero used for replay-buffer shapes.
    config.example_action = np.zeros((1, config.action_dim))

    # Head camera plus both wrists; RoboTwin tasks may use either arm.
    config.critic_camera_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

    # The xyz+gripper edit assumes DROID's [xyz, rot, gripper] layout; joint actions have no such split.
    config.edit_action_xyzg = False

    return config
