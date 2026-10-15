"""Convert RoboTwin 2.0 demonstrations (per-episode HDF5 + instruction JSON) into a LeRobot v3.0 dataset.

Follows XPolicyLab's transform_lerobot_v30_format.py: state[t] is the joint vector of frame t and
action[t] the joint vector of frame t+1, so the last frame only serves as a target; cameras are
decoded with RoboTwin's decode_image_bit; each episode's task is one of its instructions, picked
with a fixed seed. Frames are stored as images rather than video so readers need no FFmpeg.
The action column is named `actions`, the key expo-ft's replay buffer inserts, so one openpi data
config repacks both the SFT dataset and replay transitions.

Example:
    HF_LEROBOT_HOME=<lerobot home> python scripts/convert_robotwin_to_lerobot.py \
        --raw_dir <.../beat_block_hammer/aloha-agilex_clean_50> --robotwin_root <robotwin checkout> \
        --repo_id robotwin/beat_block_hammer_aloha_clean_50
"""

import argparse
import importlib.util
import json
import os
import random
import re

import h5py
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

# RoboTwin camera group -> feature name used by XPolicyLab's conversion and the expo-ft env endpoint.
CAMERAS = {
    "head_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}


def load_decode_image_bit(robotwin_root):
    """RoboTwin stores frames in two byte formats; only its own decoder tells them apart."""
    path = os.path.join(robotwin_root, "data", "decode_image_bit.py")
    spec = importlib.util.spec_from_file_location("robotwin_decode_image_bit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.decode_image_bit


def episode_ids(raw_dir):
    names = os.listdir(os.path.join(raw_dir, "data"))
    return sorted(int(m.group(1)) for n in names if (m := re.fullmatch(r"episode(\d+)\.hdf5", n)))


def read_episode(raw_dir, episode, decode_image_bit):
    with h5py.File(os.path.join(raw_dir, "data", f"episode{episode}.hdf5"), "r") as f:
        joints = np.asarray(f["joint_action/vector"], dtype=np.float32)
        images = {key: decode_image_bit(f[f"observation/{cam}/rgb"][()]) for cam, key in CAMERAS.items()}
    with open(os.path.join(raw_dir, "instructions", f"episode{episode}.json"), encoding="utf-8") as f:
        instructions = json.load(f)["seen"]
    return joints, images, instructions


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--robotwin_root", required=True)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--num_episodes", type=int, default=0, help="0 = all episodes")
    # Aloha-AgileX saves a frame every save_freq sim steps; XPolicyLab reads this rate from collect_freq.
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    decode_image_bit = load_decode_image_bit(args.robotwin_root)
    episodes = episode_ids(args.raw_dir)
    if args.num_episodes > 0:
        episodes = episodes[: args.num_episodes]

    joints, images, _ = read_episode(args.raw_dir, episodes[0], decode_image_bit)
    features = {
        "observation.state": {"dtype": "float32", "shape": (joints.shape[1],), "names": None},
        "actions": {"dtype": "float32", "shape": (joints.shape[1],), "names": None},
    }
    for key, frames in images.items():
        features[key] = {"dtype": "image", "shape": frames.shape[1:], "names": ["height", "width", "channels"]}
    dataset = LeRobotDataset.create(args.repo_id, fps=args.fps, robot_type="aloha-agilex", features=features,
                                    use_videos=False)

    rng = random.Random(args.seed)
    for episode in tqdm(episodes):
        joints, images, instructions = read_episode(args.raw_dir, episode, decode_image_bit)
        task = rng.choice(instructions)
        for t in range(len(joints) - 1):
            frame = {"observation.state": joints[t], "actions": joints[t + 1], "task": task}
            frame.update({key: frames[t] for key, frames in images.items()})
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()
    print(f"Wrote {len(episodes)} episodes, {dataset.num_frames} frames to {dataset.root}")


if __name__ == "__main__":
    main()
