"""RoboTwin 2.0 task behind the expo-ft environment RPC (served by client/run_client.py).

Runs in RoboTwin's own Python environment. Observations use the feature names of RoboTwin's
LeRobot v3.0 conversion so that replay prefill (expo_ft/env/lerobot_utils.py) and live rollouts
go through the same openpi transforms.
"""

import importlib
import os
import random
import sys

import numpy as np
import yaml

from client.envs.utils import process_image_for_obs

# RoboTwin camera name -> LeRobot feature name used by XPolicyLab's v3.0 conversion.
CAMERA_KEYS = {
    "head_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}


def load_task_args(robotwin_root, task_name, task_config_name):
    """Build the kwargs RoboTwin's `setup_demo` expects, as scripts/eval_policy_xpolicylab.py does."""
    config_dir = os.path.join(robotwin_root, "env_cfg", "task_config")
    with open(os.path.join(config_dir, f"{task_config_name}.yml"), encoding="utf-8") as f:
        args = yaml.safe_load(f)
    with open(os.path.join(config_dir, "_embodiment_config.yml"), encoding="utf-8") as f:
        embodiments = yaml.safe_load(f)
    with open(os.path.join(config_dir, "_camera_config.yml"), encoding="utf-8") as f:
        cameras = yaml.safe_load(f)

    args["task_name"] = task_name
    args["task_config"] = task_config_name
    args["data_type"].update(rgb=True, qpos=True, endpose=True)
    head = cameras[args["camera"]["head_camera_type"]]
    args["head_camera_h"], args["head_camera_w"] = head["h"], head["w"]

    embodiment = args["embodiment"]
    if len(embodiment) != 1:
        raise ValueError(f"Only single-embodiment configs are supported; got {embodiment}")
    robot_file = embodiments[embodiment[0]]["file_path"]
    with open(os.path.join(robot_file, "config.yml"), encoding="utf-8") as f:
        robot_config = yaml.safe_load(f)
    args.update(
        left_robot_file=robot_file,
        right_robot_file=robot_file,
        dual_arm_embodied=True,
        left_embodiment_config=robot_config,
        right_embodiment_config=robot_config,
        # eval_mode loads step_lim from _eval_step_limit.yml, as RoboTwin's evaluation does.
        eval_mode=True,
        # Rollouts are driven over RPC: no viewer, no data saving, no RoboTwin-side video.
        render_freq=0,
        save_data=False,
        eval_video_log=False,
    )
    return args


class RoboTwinEnv:
    """One RoboTwin task, reset and judged by the official evaluation protocol.

    reset() walks seeds from 100000 * (1 + seed_index), skipping seeds whose expert
    demonstration fails, exactly like RoboTwin's evaluation; each reset uses the next seed.
    """

    def __init__(self, robotwin_root, robotwin_task, task_config_name="demo_clean", seed_index=0,
                 instruction_type="seen", image_size=(224, 224), language_instruction=None, **_):
        self.robotwin_root = os.path.abspath(robotwin_root)
        # RoboTwin loads assets through paths relative to its root.
        os.chdir(self.robotwin_root)
        for path in (self.robotwin_root, os.path.join(self.robotwin_root, "description", "utils")):
            if path not in sys.path:
                sys.path.insert(0, path)

        self.task_name = robotwin_task
        self.args = load_task_args(self.robotwin_root, robotwin_task, task_config_name)
        self.instruction_type = instruction_type
        self.default_instruction = language_instruction or robotwin_task
        self.image_size = tuple(image_size)
        self.next_seed = 100000 * (1 + int(seed_index))
        self.episode_id = 0
        self.seed = None
        self.instruction = None
        self.task = None

    def _new_task(self):
        module = importlib.import_module(f"envs.{self.task_name}")
        return getattr(module, self.task_name)()

    def _setup(self, task, seed):
        task.setup_demo(now_ep_num=self.episode_id, seed=seed, is_test=True, **self.args)

    def _expert_succeeds(self, seed):
        """Run the scripted expert on `seed`; RoboTwin evaluation only keeps seeds it can solve."""
        from envs.utils.create_actor import UnStableError

        task = self._new_task()
        try:
            self._setup(task, seed)
            episode_info = task.play_once()
            ok = bool(task.plan_success and task.check_success())
        except UnStableError:
            return False, None
        except Exception as e:  # The official evaluation also skips seeds the expert crashes on.
            print(f"[RoboTwinEnv] expert error on seed {seed}: {type(e).__name__}: {e}")
            return False, None
        finally:
            task.close_env()
        return ok, episode_info

    def _instruction(self, episode_info, seed):
        """Same instruction choice as RoboTwin's build_instruction for the configured split."""
        from generate_episode_instructions import generate_episode_descriptions

        try:
            results = generate_episode_descriptions(self.task_name, [episode_info.get("info", {})], 1, seed=seed)
            candidates = results[0].get(self.instruction_type)
            if candidates:
                return random.Random(f"{seed}:select").choice(candidates)
        except Exception as e:
            print(f"[RoboTwinEnv] instruction generation failed: {e}")
        return self.default_instruction

    def reset(self):
        if self.task is not None:
            self.task.close_env()
            self.task = None
        while True:
            seed = self.next_seed
            self.next_seed += 1
            ok, episode_info = self._expert_succeeds(seed)
            if ok:
                break
        task = self._new_task()
        self._setup(task, seed)
        self.task, self.seed = task, seed
        self.instruction = self._instruction(episode_info, seed)
        self.episode_id += 1
        return self.get_observation()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.task.take_action(action, action_type="qpos")
        return {"executed_action": action}

    def get_observation(self):
        raw = self.task.get_obs()
        obs = {
            key: process_image_for_obs(raw["observation"][camera]["rgb"], image_size=self.image_size)
            for camera, key in CAMERA_KEYS.items()
        }
        obs["observation.state"] = np.asarray(raw["joint_action"]["vector"], dtype=np.float32)
        obs["prompt"] = self.instruction
        return obs

    def get_info_for_step(self):
        success = bool(self.task.eval_success)
        done = success or self.task.take_action_cnt >= self.task.step_lim
        return done, success, 1.0 if success else 0.0, 0.0 if done else 1.0

    def render(self):
        return self.task.get_obs()["observation"]["head_camera"]["rgb"]
