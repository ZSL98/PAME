import torch
import torch.nn as nn
from networks import resnet_s1, resnet_s2
import torchvision.models as models

dummy_input = torch.randn(1, 3, 224, 224)
model_s1 = resnet_s1(start_point=10, end_point=13)

# model = models.resnet101(pretrained=True)

input_names = ["input"]
output_names = ["output1", "output2"]
torch.onnx.export(model_s1, dummy_input, "resnet_s1.onnx",input_names=input_names, output_names=output_names,
                  verbose=True,dynamic_axes={
                                      'input': {0: 'batch_size'},
                                      'output1': {0: 'batch_size'},
                                      'output2': {0: 'batch_size'},
                                  },opset_version=11)
exit()