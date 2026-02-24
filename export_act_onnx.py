#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025, WuChao D-Robotics.
#
# LeRobot ACTPolicy -> ONNX export

import logging
import os
import shutil
import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
import onnx

from copy import deepcopy
from termcolor import colored
from onnxsim import simplify
from pprint import pformat

from lerobot.policies.act.modeling_act import *
from lerobot.datasets.factory import make_dataset
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig

NPU_VisionEncoder = "NPU_ACTPolicy_VisionEncoder"
NPU_TransformerLayers = "NPU_ACTPolicy_TransformerLayers"


@parser.wrap()
def main(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    argp = argparse.ArgumentParser()
    argp.add_argument(
        '--act-path',
        type=str,
        default='./outputs/train/act_so101_test1/checkpoints/100000/pretrained_model',
        help='Path to LeRobot ACT Policy checkpoint folder.'
    )
    """
    # example: --act-path pretrained_model
    ./pretrained_model/
    ├── config.json
    ├── model.safetensors
    └── train_config.json
    """
    argp.add_argument(
        '--export-path',
        type=str,
        default='npu_onnx_export',
        help='Root folder to save exported ONNX models and npy params.'
    )
    argp.add_argument(
        '--cal-num',
        type=int,
        default=400,
        help='Number of samples used for ONNX simplification calibration (if onnx-simplifier is enabled).'
    )
    argp.add_argument(
        '--onnx-sim',
        type=bool,
        default=True,
        help='Whether to run onnx-simplifier on exported models.'
    )

    opt = argp.parse_args([])
    logging.info(f"opt: {opt}")

    if os.path.exists(opt.export_path):
        shutil.rmtree(opt.export_path)

    vision_ws = os.path.join(opt.export_path, NPU_VisionEncoder)
    trans_ws = os.path.join(opt.export_path, NPU_TransformerLayers)

    onnx_name_ve = NPU_VisionEncoder + ".onnx"
    onnx_path_ve = os.path.join(vision_ws, onnx_name_ve)
    onnx_name_tf = NPU_TransformerLayers + ".onnx"
    onnx_path_tf = os.path.join(trans_ws, onnx_name_tf)

    npu_output_name = "npu_output"
    npu_output_path = os.path.join(opt.export_path, npu_output_name)

    up_std_path = os.path.join(npu_output_path, "up_std.npy")
    up_mean_path = os.path.join(npu_output_path, "up_mean.npy")
    front_std_path = os.path.join(npu_output_path, "front_std.npy")
    front_mean_path = os.path.join(npu_output_path, "front_mean.npy")
    state_std_path = os.path.join(npu_output_path, "state_std.npy")
    state_mean_path = os.path.join(npu_output_path, "state_mean.npy")
    action_std_unnorm_path = os.path.join(
        npu_output_path, "action_std_unnormalize.npy"
    )
    action_mean_unnorm_path = os.path.join(
        npu_output_path, "action_mean_unnormalize.npy"
    )

    os.makedirs(vision_ws, exist_ok=True)
    logging.info(colored(f"mkdir: {vision_ws} Success.", 'green'))
    os.makedirs(trans_ws, exist_ok=True)
    logging.info(colored(f"mkdir: {trans_ws} Success.", 'green'))
    os.makedirs(npu_output_path, exist_ok=True)
    logging.info(colored(f"mkdir: {npu_output_path} Success.", 'green'))

    policy = ACTPolicy.from_pretrained(opt.act_path).cpu().eval()
    logging.info(colored(f"Load ACT Policy Model: {opt.act_path} Success.", 'green'))

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = make_dataset(cfg)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_size=1,
        shuffle=True,                                                                                    
        sampler=None,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    logging.info(colored(f"Load ACT Policy Dataset:\n{dataset} Success.", 'green'))

    kvs = ['observation.images.up', 'observation.images.front', 'observation.state']
    batch = next(iter(dataloader))
    batch = dict(filter(lambda item: item[0] in kvs, batch.items()))

    _ = policy.select_action(deepcopy(batch))

    '''
    up_std = policy.normalize_inputs.buffer_observation_images_up.std.data.detach().cpu().numpy()
    up_mean = policy.normalize_inputs.buffer_observation_images_up.mean.data.detach().cpu().numpy()
    front_std = policy.normalize_inputs.buffer_observation_images_front.std.data.detach().cpu().numpy()
    front_mean = policy.normalize_inputs.buffer_observation_images_front.mean.data.detach().cpu().numpy()
    state_std = policy.normalize_inputs.buffer_observation_state.std.data.detach().cpu().numpy()
    state_mean = policy.normalize_inputs.buffer_observation_state.mean.data.detach().cpu().numpy()
    action_std_unnorm = policy.unnormalize_outputs.buffer_action.std.data.detach().cpu().numpy()
    action_mean_unnorm = policy.unnormalize_outputs.buffer_action.mean.data.detach().cpu().numpy()
    
    np.save(up_std_path, up_std)
    np.save(up_mean_path, up_mean)
    np.save(front_std_path, front_std)
    np.save(front_mean_path, front_mean)
    np.save(state_std_path, state_std)
    np.save(state_mean_path, state_mean)
    np.save(action_std_unnorm_path, action_std_unnorm)
    np.save(action_mean_unnorm_path, action_mean_unnorm)
    logging.info(colored("Save pre/post-process params Success.", 'green'))
    '''
    
    stats = None
    if hasattr(dataset, "stats") and dataset.stats is not None:
        stats = dataset.stats
    elif hasattr(dataset, "meta") and hasattr(dataset.meta, "stats") and dataset.meta.stats is not None:
        stats = dataset.meta.stats
    else:
        raise RuntimeError(
            "Stats cannot be found in the dataset object (neither dataset.stats nor dataset.meta.stats), so the mean/std of preprocessing and postprocessing cannot be exported. "
            "Please first confirm where the statistical information is stored in the current version of LeRobot."
        )

    def _get_mean_std(feature_key: str):
        if feature_key not in stats:
            raise KeyError(f"stats no key='{feature_key}', use keys: {list(stats.keys())}")
        s = stats[feature_key]
        mean = s["mean"]
        std = s["std"]

        if isinstance(mean, torch.Tensor):
            mean = mean.detach().cpu().numpy()
        else:
            mean = np.asarray(mean)

        if isinstance(std, torch.Tensor):
            std = std.detach().cpu().numpy()
        else:
            std = np.asarray(std)

        return mean, std

    up_mean,   up_std   = _get_mean_std("observation.images.up")
    front_mean,front_std= _get_mean_std("observation.images.front")
    state_mean,state_std= _get_mean_std("observation.state")
    
    action_mean_unnorm, action_std_unnorm = _get_mean_std("action")

    np.save(up_std_path,                 up_std)
    np.save(up_mean_path,                up_mean)
    np.save(front_std_path,              front_std)
    np.save(front_mean_path,             front_mean)
    np.save(state_std_path,              state_std)
    np.save(state_mean_path,             state_mean)
    np.save(action_std_unnorm_path,      action_std_unnorm)
    np.save(action_mean_unnorm_path,     action_mean_unnorm)
    logging.info(colored("Save pre/post-process params (from dataset.stats) Success.", 'green'))
    
    if hasattr(policy, "normalize_inputs"):
        batch = policy.normalize_inputs(batch)

    m_VisionEncoder = NPU_ACTPolicy_VisionEncoder(policy)
    m_VisionEncoder.eval()

    up_imgs = batch['observation.images.up']
    front_imgs = batch['observation.images.front']
    vision_feat1 = m_VisionEncoder(up_imgs)
    vision_feat2 = m_VisionEncoder(front_imgs)

    m_TransformerLayers = NPU_ACTPolicy_TransformerLayers(policy)
    m_TransformerLayers.eval()

    state = batch["observation.state"]
    actions = m_TransformerLayers(state, vision_feat1, vision_feat2)
    np.save(os.path.join(opt.export_path, "new_actions.npy"),
            actions.detach().cpu().numpy())

    logging.info(colored("Export VisionEncoder to ONNX ...", 'yellow'))
    torch.onnx.export(
        m_VisionEncoder,
        up_imgs,                 # 输入 tensor (NCHW)
        onnx_path_ve,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['Vision_Features'],
        dynamic_axes=None,
    )
    onnx_sim(onnx_path_ve, opt.onnx_sim)
    logging.info(colored(f"Export {onnx_path_ve} Success.", 'green'))


    logging.info(colored("Export TransformerLayers to ONNX ...", 'yellow'))
    torch.onnx.export(
        m_TransformerLayers,
        (state, vision_feat1, vision_feat2),
        onnx_path_tf,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['states', 'up_features', 'front_features'],
        output_names=['Actions'],
        dynamic_axes=None,
    )
    onnx_sim(onnx_path_tf, opt.onnx_sim)
    logging.info(colored(f"Export {onnx_path_tf} Success.", 'green'))



def onnx_sim(onnx_path, do_simplify: bool = True):
    if not do_simplify:
        return
    logging.info(colored(f"Simplify ONNX: {onnx_path}", 'yellow'))
    model_onnx = onnx.load(onnx_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    model_onnx, check = simplify(
        model_onnx,
        dynamic_input_shape=False,
        input_shapes=None)
    assert check, 'onnx-simplifier check failed'
    onnx.save(model_onnx, onnx_path)
    logging.info(colored(f"Simplify ONNX Done: {onnx_path}", 'green'))


class NPU_ACTPolicy_VisionEncoder(nn.Module):
    '''
    In the dataset, only keep the backbone encoder_img_feat_input_proj,
    Input: NCHW image (already normalized)
    Output: Vision Feature Map (B, C, H, W) → for Transformer use
    '''
    def __init__(self, act_policy):
        super().__init__()
        self.backbone = deepcopy(act_policy.model.backbone)
        self.encoder_img_feat_input_proj = deepcopy(
            act_policy.model.encoder_img_feat_input_proj
        )

    def forward(self, images):
        cam_features = self.backbone(images)["feature_map"]
        cam_features = self.encoder_img_feat_input_proj(cam_features)
        return cam_features


class NPU_ACTPolicy_TransformerLayers(nn.Module):
    '''
    Combine the encoder, decoder, and action_head in ACT into a single subnet.
    Input:
    - states: [T/B, state_dim]
    - vision_feature1: NPU_ACTPolicy_VisionEncoder(up_images)
    - vision_feature2: NPU_ACTPolicy_VisionEncoder(front_images)
    Output:
    - actions: [chunk_size, B, action_dim]
    '''
    def __init__(self, act_policy):
        super().__init__()
        self.model = deepcopy(act_policy.model)

    def forward(self, states, vision_feature1, vision_feature2):
        # latent token
        latent_sample = torch.zeros(
            [1, self.model.config.latent_dim], dtype=torch.float32,
            device=states.device
        )

        encoder_in_tokens = [self.model.encoder_latent_input_proj(latent_sample)]
        encoder_in_pos_embed = self.model.encoder_1d_feature_pos_embed.weight.unsqueeze(1).unbind(dim=0)
        encoder_in_tokens.append(self.model.encoder_robot_state_input_proj(states))

        all_cam_features = []
        all_cam_pos_embeds = []

        vision_features = [vision_feature1, vision_feature2]
        for vf in vision_features:
            cam_pos_embed = self.model.encoder_cam_feat_pos_embed(vf)
            all_cam_features.append(vf)
            all_cam_pos_embeds.append(cam_pos_embed)

        tokens = []
        for token in encoder_in_tokens:
            tokens.append(token.view(1, 1, self.model.config.dim_model))
        all_cam_features = torch.cat(all_cam_features, axis=-1).permute(2, 3, 0, 1).view(-1, 1, self.model.config.dim_model)
        tokens.append(all_cam_features)
        encoder_in_tokens = torch.cat(tokens, axis=0)

        pos_embeds = []
        for pos_embed in encoder_in_pos_embed:
            pos_embeds.append(pos_embed.view(1, 1, self.model.config.dim_model))
        all_cam_pos_embeds = torch.cat(all_cam_pos_embeds, axis=-1).permute(2, 3, 0, 1).view(-1, 1, self.model.config.dim_model)
        pos_embeds.append(all_cam_pos_embeds)
        encoder_in_pos_embed = torch.cat(pos_embeds, axis=0)

        encoder_out = self.model.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_in = torch.zeros(
            (self.model.config.chunk_size, 1, self.model.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.model.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.model.decoder_pos_embed.weight.unsqueeze(1),
        )

        decoder_out = decoder_out.transpose(0, 1)
        actions = self.model.action_head(decoder_out)
        return actions


def lerobotTensor2cvmat(tensor):
    '''
    Keep your original debug tools (in case visualization is needed later)
    tensor: [1, C, H, W], range [0,1]
    '''
    img = (tensor * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0, :, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


if __name__ == "__main__":
    init_logging()
    main()
