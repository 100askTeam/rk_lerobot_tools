#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np
import argparse
import os
import glob
from collections import deque

import cv2
import torch
from torch import Tensor

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

try:
    from rknnlite.api import RKNNLite
    print("using: RKNNLite (NPU)")
except ImportError:
    print("rknnlite not found, please check NPU runtime installation!")
    exit()

class RDK_ACTConfig:
    def __init__(self, device="cpu", n_action_steps=50):
        self.device = device
        self.n_action_steps = n_action_steps
        self.use_amp = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--npu-act-path',
        type=str,
        default='rknn_tools/npu_onnx_export',
        help='Path to NPU ACTPolicy folder (contains .rknn + npu_output/*.npy).'
    )
    parser.add_argument('--fps', type=int, default=15, help='FPS for inference loop')
    parser.add_argument('--n-action-steps', type=int, default=50, help='Number of action steps')
    parser.add_argument('--num-steps', type=int, default=300, help='Total timesteps to run')
    opt = parser.parse_args()

    camera_names = detect_cameras_from_model(opt.npu_act_path)
    print(f"Detected cameras from model: {camera_names}")

    if set(camera_names) == {"up", "front"}:
        camera_names = ["up", "front"]
    print(f"Using camera order for NPU: {camera_names}")

    camera_config = {
        "up": OpenCVCameraConfig(index_or_path=11, width=640, height=480, fps=opt.fps),
        "front": OpenCVCameraConfig(index_or_path=13, width=640, height=480, fps=opt.fps),
    }

    missing_cameras = set(camera_names) - set(camera_config.keys())
    if missing_cameras:
        print(f"Error: Missing camera configurations for: {missing_cameras}")
        print("Please add configurations for these cameras in the camera_config section:")
        for camera in missing_cameras:
            print(f'    "{camera}": OpenCVCameraConfig(index_or_path=X, width=640, height=480, fps={opt.fps}),')
        return

    caps = {}
    for name in camera_names:
        cfg = camera_config[name]
        cap = cv2.VideoCapture(cfg.index_or_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        if not cap.isOpened():
            print(f"Failed to open camera {name} at index {cfg.index_or_path}")
            return
        caps[name] = cap
        print(f"Camera '{name}' -> index/path: {cfg.index_or_path}")

    policy = RDK_NPU_ACTPolicy_Dynamic(opt.npu_act_path, opt.n_action_steps, camera_names)

    period = 1.0 / opt.fps
    print("Start NPU ACTPolicy inference loop (press Ctrl+C to stop)...")

    try:
        for step in range(opt.num_steps):
            t0 = time.time()

            batch: dict[str, Tensor] = {}
            for name in camera_names:
                ret, frame = caps[name].read()
                if not ret:
                    print(f"Failed to read frame from camera {name}")
                    return
                # BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # HWC -> CHW, [0,1]
                img = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                img = img.unsqueeze(0)  # [1,C,H,W]
                batch[f"observation.images.{name}"] = img

            state_dim = policy.state_mean.numel()
            state = torch.zeros(1, state_dim, dtype=torch.float32)
            batch["observation.state"] = state
            with torch.no_grad():
                action = policy.select_action(batch)  # Tensor, [1, action_dim]

            print(f"[step {step}] action shape: {tuple(action.shape)}")
            print(f"[step {step}] action[0, :5]: {action[0, :5].cpu().numpy()}")

            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        for cap in caps.values():
            cap.release()
        print("Cameras released, exit.")


def detect_cameras_from_model(npu_act_path: str):
    '''
    Automatically infer which cameras are in npu_output/*.npy.
    Rules:
    - Match *_mean.npy
    - Exclude files starting with action_ or state_
    '''
    camera_names = []
    npu_output_dir = os.path.join(npu_act_path, "npu_output")
    mean_files = glob.glob(os.path.join(npu_output_dir, "*_mean.npy"))

    for mean_file in mean_files:
        filename = os.path.basename(mean_file)
        if filename.startswith("action_") or filename.startswith("state_"):
            continue

        # 去掉 "_mean.npy"
        camera_name = filename.replace("_mean.npy", "")
        std_file = os.path.join(npu_output_dir, f"{camera_name}_std.npy")
        if os.path.exists(std_file):
            camera_names.append(camera_name)

    if not camera_names:
        print("Warning: No camera configuration detected, using default up+front")
        camera_names = ["up", "front"]

    return camera_names


class RDK_NPU_ACTPolicy_Dynamic:
    def __init__(self, npu_act_model_path: str, n_action_steps: int, camera_names):
        self.config = RDK_ACTConfig(device="cpu", n_action_steps=n_action_steps)
        self.n_action_steps = n_action_steps
        self._action_queue = deque([], maxlen=self.n_action_steps)
        self.camera_names = camera_names

        print(f"Initializing NPU policy (RK3576) with cameras: {camera_names}")

        npu_output_dir = os.path.join(npu_act_model_path, "npu_output")

        self.camera_params = {}
        for camera_name in camera_names:
            std_path = os.path.join(npu_output_dir, f"{camera_name}_std.npy")
            mean_path = os.path.join(npu_output_dir, f"{camera_name}_mean.npy")
            if os.path.exists(std_path) and os.path.exists(mean_path):
                self.camera_params[camera_name] = {
                    "std": torch.tensor(np.load(std_path), dtype=torch.float32) + 1e-8,
                    "mean": torch.tensor(np.load(mean_path), dtype=torch.float32),
                }
                print(f"Loaded normalization params for camera '{camera_name}'")
            else:
                raise FileNotFoundError(f"Missing normalization files for camera: {camera_name}")

        state_std_path = os.path.join(npu_output_dir, "state_std.npy")
        state_mean_path = os.path.join(npu_output_dir, "state_mean.npy")
        for fp in [state_std_path, state_mean_path]:
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Required state stats file not found: {fp}")

        self.state_std = torch.tensor(np.load(state_std_path), dtype=torch.float32) + 1e-8
        self.state_mean = torch.tensor(np.load(state_mean_path), dtype=torch.float32)

        action_std_unnorm_path = os.path.join(npu_output_dir, "action_std_unnormalize.npy")
        action_mean_unnorm_path = os.path.join(npu_output_dir, "action_mean_unnormalize.npy")
        for fp in [action_std_unnorm_path, action_mean_unnorm_path]:
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Required action unnormalize file not found: {fp}")

        self.action_std_unnormalize = torch.tensor(
            np.load(action_std_unnorm_path), dtype=torch.float32
        )
        self.action_mean_unnormalize = torch.tensor(
            np.load(action_mean_unnorm_path), dtype=torch.float32
        )

        for camera_name in camera_names:
            params = self.camera_params[camera_name]
            assert not torch.isinf(params["std"]).any(), f"Invalid std for {camera_name}"
            assert not torch.isinf(params["mean"]).any(), f"Invalid mean for {camera_name}"
        assert not torch.isinf(self.state_std).any(), "Invalid state_std"
        assert not torch.isinf(self.state_mean).any(), "Invalid state_mean"
        assert not torch.isinf(self.action_std_unnormalize).any(), "Invalid action_std_unnormalize"
        assert not torch.isinf(self.action_mean_unnormalize).any(), "Invalid action_mean_unnormalize"

        ve_rknn_path = os.path.join(
            npu_act_model_path, "NPU_ACTPolicy_VisionEncoder", "NPU_ACTPolicy_VisionEncoder.rknn"
        )
        tf_rknn_path = os.path.join(
            npu_act_model_path, "NPU_ACTPolicy_TransformerLayers", "NPU_ACTPolicy_TransformerLayers.rknn"
        )
        if not os.path.exists(ve_rknn_path):
            raise FileNotFoundError(f"Vision encoder RKNN not found: {ve_rknn_path}")
        if not os.path.exists(tf_rknn_path):
            raise FileNotFoundError(f"Transformer RKNN not found: {tf_rknn_path}")

        # VisionEncoder RKNNLite
        self.ve_rknn = RKNNLite()
        assert self.ve_rknn.load_rknn(ve_rknn_path) == 0, "load vision rknn failed!"
        assert self.ve_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) == 0, "init vision runtime failed!"

        # TransformerLayers RKNNLite
        self.tf_rknn = RKNNLite()
        assert self.tf_rknn.load_rknn(tf_rknn_path) == 0, "load transformer rknn failed!"
        assert self.tf_rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) == 0, "init transformer runtime failed!"

        self.cnt = 0
        print("NPU RKNN models loaded successfully.")

    def reset(self):
        self._action_queue.clear()

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        batch = self.normalize_inputs(batch)

        if len(self._action_queue) == 0:
            begin_time = time.time()

            state = batch["observation.state"].detach().cpu().numpy().astype("float32")

            vision_features = []
            for camera_name in self.camera_names:
                key = f"observation.images.{camera_name}"
                cam_tensor = batch[key].detach().cpu()   # [1, C, H, W]
                
                if cam_tensor.shape[1] == 3:
                    cam_tensor = cam_tensor.repeat(1, 3, 1, 1)  # -> [1, 9, H, W]

                cam_input = cam_tensor.numpy().astype("float32")
                outputs = self.ve_rknn.inference(inputs=[cam_input])
                if outputs is None:
                    raise RuntimeError("VisionEncoder RKNN inference failed, got None outputs")
                vision_features.append(outputs[0])

            tf_inputs = [state]
            for _, feat in zip(self.camera_names, vision_features):
                tf_inputs.append(feat.astype("float32"))

            tf_outputs = self.tf_rknn.inference(inputs=tf_inputs)
            action_output = tf_outputs[0]  # [1, chunk_size, action_dim]
            actions = torch.from_numpy(action_output)[:, :self.n_action_steps]

            cost_ms = 1000 * (time.time() - begin_time)
            print(
                f"{self.cnt} NPU ACT Policy Time (dynamic {len(self.camera_names)} cameras): "
                + "\033[1;31m"
                + f"{cost_ms:.2f} ms"
                + "\033[0m"
            )
            self.cnt += 1

            actions = self.unnormalize_outputs({"action": actions})["action"]
            # [1, n_action_steps, action_dim] -> [n_action_steps, 1, action_dim]
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    def normalize_inputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # state: (state - mean) / std
        state = batch["observation.state"]
        batch["observation.state"] = (state - self.state_mean.to(state.device)) / self.state_std.to(state.device)

        for camera_name in self.camera_names:
            key = f"observation.images.{camera_name}"
            if key not in batch:
                continue
            img = batch[key]
            params = self.camera_params[camera_name]
            mean = params["mean"].to(img.device)
            std = params["std"].to(img.device)
            batch[key] = (img - mean) / std

        return batch

    def unnormalize_outputs(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        act = batch["action"]
        mean = self.action_mean_unnormalize.to(act.device)
        std = self.action_std_unnormalize.to(act.device)
        batch["action"] = act * std + mean
        return batch


def _no_stats_error_str(name: str) -> str:
    return (
        f"`{name}` is infinity. You should either initialize with `stats` as an argument, "
        "or use a pretrained model."
    )


if __name__ == '__main__':
    main()

