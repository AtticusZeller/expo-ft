"""Load LeRobot v3.0 demonstrations as replay transitions (used for the RoboTwin prefill)."""

import os

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from tqdm import tqdm

# Bookkeeping columns every LeRobot frame carries; they are not observations.
_INDEX_KEYS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
# Action column written by scripts/convert_robotwin_to_lerobot.py: the same key the replay buffer
# inserts, so a single openpi repack serves the SFT dataset and replay transitions.
ACTION_KEY = "actions"


def _frame_to_observation(frame, camera_keys):
    """Turn a decoded LeRobot frame into the observation dict the env endpoint sends.

    Keys stay the dataset feature names, so the openpi data config that repacks the
    dataset for SFT also repacks replay observations. Images stay CHW, as LeRobot yields
    them and as openpi's Aloha transforms expect, and go back to uint8, which is what the
    replay buffer stores.
    """
    obs = {}
    for key, value in frame.items():
        if key == ACTION_KEY or key in _INDEX_KEYS or key == "task":
            continue
        value = np.asarray(value)
        if key in camera_keys:
            value = np.round(value * 255).astype(np.uint8)
        obs[key] = value
    obs["prompt"] = frame["task"]
    return obs


def _episode_transitions(steps):
    """Label one demonstration, given as (observation, action) pairs, as a successful episode.

    r=1 and done land on the last step only, matching `process_droid_dataset`.
    """
    T = len(steps)
    transitions = []
    for t, (obs, action) in enumerate(steps):
        last = float(t == T - 1)
        transitions.append({
            "observations": obs,
            "actions": action,
            "rewards": np.float32(last),
            "masks": np.float32(1.0 - last),
            "dones": np.float32(last),
        })
    return transitions


def process_lerobot_dataset(
    datapath,
    task_config,
    episode_indices=None,
    num_data=None):
    """Same contract as `process_droid_dataset`, for a local LeRobot v3.0 dataset root."""
    del task_config  # Kept for call-site parity with process_droid_dataset.
    repo_id = os.path.basename(os.path.normpath(datapath))
    meta = LeRobotDatasetMetadata(repo_id, root=datapath)
    episodes = list(range(meta.total_episodes))
    if episode_indices is not None:
        episodes = [i for i in episode_indices if 0 <= i < meta.total_episodes]
    elif num_data is not None and num_data > 0:
        episodes = episodes[:num_data]

    print(f"Find {meta.total_episodes} episodes; using {len(episodes)}")

    # PyAV bundles its own FFmpeg; torchcodec's default needs system FFmpeg libraries the training hosts lack.
    dataset = LeRobotDataset(repo_id, root=datapath, episodes=episodes, video_backend="pyav")
    camera_keys = set(meta.camera_keys)
    by_episode = {}
    for i in tqdm(range(len(dataset))):
        frame = dataset[i]
        # Convert right away: decoded float images are ~4x the size of the uint8 ones we keep.
        step = (
            int(frame["frame_index"]),
            _frame_to_observation(frame, camera_keys),
            np.asarray(frame[ACTION_KEY], dtype=np.float32),
        )
        by_episode.setdefault(int(frame["episode_index"]), []).append(step)

    data = []
    for ep in episodes:
        steps = sorted(by_episode[ep], key=lambda s: s[0])
        data.extend(_episode_transitions([(obs, action) for _, obs, action in steps]))
    return data
