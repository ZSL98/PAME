import onnx
from onnx.tools import update_model_dims
import numpy as np
import onnx.helper as helper
from onnx import shape_inference, TensorProto
import sys

ONNX_DTYPE = {
    0: TensorProto.FLOAT,
    1: TensorProto.FLOAT,
    2: TensorProto.UINT8,
    3: TensorProto.INT8,
    4: TensorProto.UINT16,
    5: TensorProto.INT16,
    6: TensorProto.INT32,
    7: TensorProto.INT64,
    8: TensorProto.STRING,
    9: TensorProto.BOOL
}

# load model
onnx_model = onnx.load("./models/vgg16.onnx")
graph = onnx_model.graph

# rewrite the input tensor of graph
input_tensor = graph.input[0]
input_shape = input_tensor.type.tensor_type.shape.dim
input_tensor_new = onnx.helper.make_tensor_value_info(name = input_tensor.name, elem_type = 1, 
                                                      shape = [1, input_shape[1].dim_value, input_shape[2].dim_value, input_shape[3].dim_value])
graph.input.remove(input_tensor)
graph.input.insert(0, input_tensor_new)

# append all tensor infos to graph input
weight_infos = []
tensors = graph.initializer
for i, tensor in enumerate(tensors):
    value_info = helper.make_tensor_value_info(tensor.name, ONNX_DTYPE[tensor.data_type], tensor.dims)
    weight_infos.append(value_info)
    graph.input.insert(i+1, value_info) # because 0 is for placeholder, so start index is 1

# run node shape inference
node = graph.node
value_info = graph.value_info
print("Before shape inference: \n")
print(value_info)
print("------------------------------------------------------------")
print("After shape inference: \n")
inferred_onnx_model = shape_inference.infer_shapes(onnx_model)
onnx.checker.check_model(onnx_model)
inferred_graph = inferred_onnx_model.graph
inferred_value_info = inferred_graph.value_info
print(inferred_value_info)
onnx.save(inferred_onnx_model,"./models_out/new_vgg.onnx")