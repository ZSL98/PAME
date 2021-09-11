import torch
import torch.nn as nn
from networks import resnet_s1, resnet_s2
import torchvision.models as models

dummy_input1 = torch.randn(1, 3, 224, 224)
dummy_input2 = torch.randn(1, 1024, 14, 14)
model_s1 = resnet_s1(start_point=10, end_point=13)
model_s2 = resnet_s2(start_point=10, end_point=13)

# model = models.resnet101(pretrained=True)

input_names = ["input"]
s1_output_names = ["output1", "exit_output"]
torch.onnx.export(model_s1, dummy_input1, "resnet_s1.onnx",input_names=input_names, output_names=s1_output_names,
                  verbose=True,dynamic_axes={
                                      'input': {0: 'batch_size'},
                                      'output1': {0: 'batch_size'},
                                      'exit_output': {0: 'batch_size'},
                                  },opset_version=11)

s2_output_names = ["final_output"]
torch.onnx.export(model_s2, dummy_input2, "resnet_s2.onnx",input_names=input_names, output_names=s2_output_names,
                  verbose=True,dynamic_axes={
                                      'input': {0: 'batch_size'},
                                      'final_output': {0: 'batch_size'},
                                  },opset_version=11)
exit()