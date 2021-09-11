import torch
import torch.nn as nn
from networks import resnet_s1, resnet_s2
import torchvision.models as models

dummy_input = torch.randn(1, 3, 224, 224)

model = models.resnet101(pretrained=True)

class construct_net(object):
    def __init__(self, start_point: int = 0, end_point: int = 0, 
                        backbone: str = 'resnet') -> None:
        super().__init__()
        self.start_point = start_point
        self.end_point = end_point

    def construct_net_s1(self):
        if self.backbone == 'resnet':
            return resnet_s1(self.start_point, self.end_point)

    def construct_net_s2(self):
        if self.backbone == 'resnet':
            return resnet_s2(self.start_point, self.end_point)


if __name__ == '__main__':
    inst = construct_net()
    inst.start_point = 


input_names = ["input"]
output_names = ["output"]
torch.onnx.export(model, dummy_input, "resnet101.onnx",input_names=input_names, output_names=output_names,
                  verbose=True,dynamic_axes={
                                      'input': {0: 'batch_size'},
                                      'output': {0: 'batch_size'},
                                  },opset_version=11)