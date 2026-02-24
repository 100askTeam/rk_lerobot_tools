# inspect_onnx_nodes.py
import onnx

def inspect_model(path):
    print(f'=== {path} ===')
    model = onnx.load(path)
    graph = model.graph

    print('Inputs:')
    for i in graph.input:
        shape = [
            d.dim_value if d.HasField('dim_value') else 'None'
            for d in i.type.tensor_type.shape.dim
        ]
        print(f'  {i.name}: {shape}')

    print('Outputs:')
    for o in graph.output:
        shape = [
            d.dim_value if d.HasField('dim_value') else 'None'
            for d in o.type.tensor_type.shape.dim
        ]
        print(f'  {o.name}: {shape}')

    for v in graph.value_info:
        shape = [
            d.dim_value if d.HasField('dim_value') else 'None'
            for d in v.type.tensor_type.shape.dim
        ]
        print(f'  {v.name}: {shape}')

if __name__ == '__main__':
    inspect_model('npu_onnx_export/NPU_ACTPolicy_VisionEncoder/NPU_ACTPolicy_VisionEncoder.onnx')
    inspect_model('npu_onnx_export/NPU_ACTPolicy_TransformerLayers/NPU_ACTPolicy_TransformerLayers.onnx')
