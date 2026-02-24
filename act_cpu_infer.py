#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
在 CPU 上用 ACTPolicy 控制 SO101Follower，
并通过最新版 record_loop API 录制数据集（或只控制，不保存）。

你需要修改下面这几个地方：
- POLICY_PATH: 你的 pretrained_model 路径
- REPO_ID: 你想保存到本地的 HF 样式数据集 id
- 机械臂串口 port
- 相机 index_or_path
"""

import logging
from pprint import pformat

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
)
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
    log_say,
)
# from lerobot.utils.visualization_utils import init_rerun


NUM_EPISODES = 5
FPS = 30
EPISODE_TIME_SEC = 60
TASK_DESCRIPTION = "My CPU ACTPolicy eval task"

# 1) Your ACTPolicy pre-trained model path (the pretrained_model directory you used to export to ONNX)
POLICY_PATH = "./outputs/train/act_so101_test1/checkpoints/100000/pretrained_model"

# 2) The repo_id for the recorded dataset (this name is also needed locally)
REPO_ID = "baiwen/eval_lerobot_cpu"

# 3) Robotic Arm Serial Port & Camera Configuration
ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "follower_arm"

CAMERA_CONFIG = {
    # up camera
    "up": OpenCVCameraConfig(index_or_path=11, width=640, height=480, fps=FPS),
    # front camera
    "front": OpenCVCameraConfig(index_or_path=13, width=640, height=480, fps=FPS),
}

# Whether to save data to disk; if you just want to test the control flow first, you can set it to False
SAVE_DATASET = True

# =====================================================


def main():
    init_logging()
    logging.info("=== ACTPolicy CPU control SO101Follower start ===")
    logging.info(f"POLICY_PATH = {POLICY_PATH}")
    logging.info(f"REPO_ID     = {REPO_ID}")

    policy_cfg = PreTrainedConfig.from_pretrained(POLICY_PATH)
    policy_cfg.device = "cpu"

    robot_cfg = SO101FollowerConfig(
        port=ROBOT_PORT,
        id=ROBOT_ID,
        cameras=CAMERA_CONFIG,
    )
    robot = SO101Follower(robot_cfg)

    # First, construct the default processors (this needs to be done before dataset_features).
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    from lerobot.utils.constants import ACTION, OBS_STR

    # Initial features: Starting from hardware action_features / observation_features
    from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
    from lerobot.datasets.utils import combine_feature_dicts

    action_initial_features = create_initial_features(action=robot.action_features)
    obs_initial_features = create_initial_features(observation=robot.observation_features)

    # Let the pipeline pass the 'initial features' through the processor to get the final features that will actually be written to the dataset.
    action_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=teleop_action_processor,
        initial_features=action_initial_features,
        use_videos=True,
    )

    obs_dataset_features = aggregate_pipeline_dataset_features(
        pipeline=robot_observation_processor,
        initial_features=obs_initial_features,
        use_videos=True,
    )

    dataset_features = combine_feature_dicts(action_dataset_features, obs_dataset_features)

    logging.info("Dataset features (keys):\n" + pformat(list(dataset_features.keys())))

    # 3) Create/Open dataset (only to provide meta for policy & record data)
    if SAVE_DATASET:
        sanity_check_dataset_name(REPO_ID, policy_cfg)

        dataset = LeRobotDataset.create(
            repo_id=REPO_ID,
            fps=FPS,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_processes=0,
            image_writer_threads=4 * len(robot.cameras),
        )
    else:
        dataset = None

    # 4) build policy + pre/post processor
    policy = make_policy(policy_cfg, ds_meta=(dataset.meta if dataset is not None else None))

    if dataset is not None and getattr(dataset.meta, "stats", None) is not None:
        dataset_stats = rename_stats(dataset.meta.stats, rename_map={})
    else:
        dataset_stats = None

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=POLICY_PATH,
        dataset_stats=dataset_stats,
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": {}},
        },
    )

    logging.info("Robot action_features:\n" + pformat(robot.action_features))
    logging.info("Robot observation_features:\n" + pformat(robot.observation_features))

    listener, events = init_keyboard_listener()
    print("no use rerun,Visualization is off")

    # 6) Connect the robotic arm
    robot.connect()
    print("SO101Follower connected (CPU ACTPolicy).")

    try:
        for episode_idx in range(NUM_EPISODES):
            log_say(
                f"Running CPU ACTPolicy, eval episode {episode_idx + 1}/{NUM_EPISODES}",
                play_sounds=False,
            )

            from lerobot.teleoperators import Teleoperator

            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=None,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=False,
            )

            if SAVE_DATASET and dataset is not None:
                dataset.save_episode()

            if events["stop_recording"]:
                print("stop_recording flag set, break.")
                break

        log_say("CPU ACTPolicy eval done.", play_sounds=False)

    finally:
        robot.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()
        print("Robot disconnected, exit.")


if __name__ == "__main__":
    main()
