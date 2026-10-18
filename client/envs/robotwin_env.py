"""RoboTwin 2.0 task behind the expo-ft environment RPC (served by client/run_client.py).

Runs in RoboTwin's own Python environment. Observations use the feature names and the CHW image
layout of RoboTwin's LeRobot v3.0 conversion so that replay prefill (expo_ft/env/lerobot_utils.py)
and live rollouts go through the same openpi transforms.
"""

import importlib
import json
import os
import random
import sys

import numpy as np
import yaml

from client.envs.robotwin_attempts import ATTEMPT_JUDGES
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
    With fixed_seed every reset replays the first valid seed instead. With single_attempt the
    episode also ends as a failure once the task's first attempt is over without success
    (client/envs/robotwin_attempts.py). A positive step_limit lowers the task's step_lim, so
    retries count only while they fit in it.

    With seed_table (a JSON file from scripts/robotwin/make_seed_table.py) reset takes the next
    precomputed seed and instruction instead of running the expert, as RLinf does: the table holds
    the seeds the expert check would have accepted, in the same order, so only reset time changes.

    One task object serves every expert check and episode, as RoboTwin's evaluation does:
    setup_demo then reuses the robot and its curobo planners (~18 s to build) across scenes.
    """

    def __init__(self, robotwin_root, robotwin_task, task_config_name="demo_clean", seed_index=0,
                 instruction_type="seen", image_size=(224, 224), language_instruction=None,
                 fixed_seed=False, single_attempt=False, step_limit=0, seed_table="", **_):
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
        self.fixed_seed = bool(fixed_seed)
        if single_attempt and robotwin_task not in ATTEMPT_JUDGES:
            raise ValueError(f"No single-attempt judge for RoboTwin task {robotwin_task}")
        self.judge_cls = ATTEMPT_JUDGES[robotwin_task] if single_attempt else None
        self.judge = None
        self.step_limit = int(step_limit)
        self.table = self._load_table(seed_table) if seed_table else None
        self.episode_id = 0
        self.seed = None
        self.instruction = None
        module = importlib.import_module(f"envs.{self.task_name}")
        self.task = getattr(module, self.task_name)()

    def _setup(self, seed):
        self.task.setup_demo(now_ep_num=self.episode_id, seed=seed, is_test=True, **self.args)

    def _expert_succeeds(self, seed):
        """Run the scripted expert on `seed`; RoboTwin evaluation only keeps seeds it can solve."""
        from envs.utils.create_actor import UnStableError

        task = self.task
        try:
            self._setup(seed)
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

    def _load_table(self, path):
        with open(path, encoding="utf-8") as f:
            table = json.load(f)
        if (table["task"], table["task_config"], table["instruction_type"]) != (
                self.task_name, self.args["task_config"], self.instruction_type):
            raise ValueError(f"Seed table {path} is for {table['task']}/{table['task_config']}/"
                             f"{table['instruction_type']}")
        # Entries are (seed, instruction), in seed order; start at this env's seed range.
        return [tuple(e) for e in table["seeds"] if e[0] >= self.next_seed]

    def _reset_from_table(self):
        from envs.utils.create_actor import UnStableError

        while True:
            if not self.table:
                raise RuntimeError("Seed table exhausted")
            self.seed, self.instruction = self.table[0] if self.fixed_seed else self.table.pop(0)
            try:
                self._setup(self.seed)
                return
            except UnStableError:  # As RLinf: an unstable scene moves on to the next seed.
                self.task.close_env()
                if self.fixed_seed:
                    self.table.pop(0)

    def reset(self):
        self.task.close_env()
        if self.table is not None:
            self._reset_from_table()
            self._after_setup()
            return self.get_observation()
        # With fixed_seed the expert check and instruction of the first valid seed are reused,
        # which also skips the expert rollout (~1 s on the reused task object) on every later reset.
        if not (self.fixed_seed and self.seed is not None):
            while True:
                seed = self.next_seed
                self.next_seed += 1
                ok, episode_info = self._expert_succeeds(seed)
                if ok:
                    break
            self.seed = seed
            self.instruction = self._instruction(episode_info, seed)
        self._setup(self.seed)
        self._after_setup()
        return self.get_observation()

    def _after_setup(self):
        if self.step_limit > 0:
            self.task.step_lim = min(self.task.step_lim, self.step_limit)
        self.judge = self.judge_cls() if self.judge_cls else None
        self.episode_id += 1

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        self.task.take_action(action, action_type="qpos")
        return {"executed_action": action}

    def get_observation(self):
        raw = self.task.get_obs()
        obs = {
            key: process_image_for_obs(raw["observation"][camera]["rgb"], image_size=self.image_size).transpose(2, 0, 1)
            for camera, key in CAMERA_KEYS.items()
        }
        obs["observation.state"] = np.asarray(raw["joint_action"]["vector"], dtype=np.float32)
        obs["prompt"] = self.instruction
        return obs

    def get_info_for_step(self):
        success = bool(self.task.eval_success)
        failed_attempt = (
            not success and self.judge is not None and self.judge.attempt_failed(self.task)
        )
        done = success or failed_attempt or self.task.take_action_cnt >= self.task.step_lim
        return done, success, 1.0 if success else 0.0, 0.0 if done else 1.0

    def render(self):
        return self.task.get_obs()["observation"]["head_camera"]["rgb"]
