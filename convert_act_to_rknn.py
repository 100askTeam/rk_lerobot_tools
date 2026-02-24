# convert_act_to_rknn.py

import os
from rknn.api import RKNN


VISION_ONNX = 'npu_onnx_export/NPU_ACTPolicy_VisionEncoder/NPU_ACTPolicy_VisionEncoder.onnx'
TRANS_ONNX  = 'npu_onnx_export/NPU_ACTPolicy_TransformerLayers/NPU_ACTPolicy_TransformerLayers.onnx'

VISION_RKNN = 'npu_onnx_export/NPU_ACTPolicy_VisionEncoder/NPU_ACTPolicy_VisionEncoder.rknn'
TRANS_RKNN  = 'npu_onnx_export/NPU_ACTPolicy_TransformerLayers/NPU_ACTPolicy_TransformerLayers.rknn'

TARGET_PLATFORM = 'rk3576'


# VisionEncoder: images -> Vision_Features

def build_vision_encoder_rknn():
    print('==== Build VisionEncoder RKNN ====')
    rknn = RKNN(verbose=True)

    rknn.config(
        target_platform=TARGET_PLATFORM,
        mean_values=[[0, 0, 0]],
        std_values=[[1, 1, 1]],
    )

    ret = rknn.load_onnx(model=VISION_ONNX)
    if ret != 0:
        print('Load VisionEncoder ONNX failed!')
        return

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build VisionEncoder RKNN failed!')
        return

    ret = rknn.export_rknn(VISION_RKNN)
    if ret != 0:
        print('Export VisionEncoder RKNN failed!')
        return

    rknn.release()
    print('==== VisionEncoder RKNN Done ====')


# TransformerLayers: (states, up_features, front_features) -> Actions -----

def build_transformer_rknn():
    print('==== Build TransformerLayers RKNN ====')
    rknn = RKNN(verbose=True)

    rknn.config(
        target_platform=TARGET_PLATFORM,
    )

    ret = rknn.load_onnx(model=TRANS_ONNX)
    if ret != 0:
        print('Load TransformerLayers ONNX failed!')
        return

    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build TransformerLayers RKNN failed!')
        return

    ret = rknn.export_rknn(TRANS_RKNN)
    if ret != 0:
        print('Export TransformerLayers RKNN failed!')
        return

    rknn.release()
    print('==== TransformerLayers RKNN Done ====')


if __name__ == '__main__':
    os.makedirs(os.path.dirname(VISION_RKNN), exist_ok=True)
    os.makedirs(os.path.dirname(TRANS_RKNN), exist_ok=True)

    build_vision_encoder_rknn()
    build_transformer_rknn()
