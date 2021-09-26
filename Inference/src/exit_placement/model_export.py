import torch
import torch.nn as nn
from networks import resnet_s1, resnet_s2, unipose_s1, unipose_s2
import torchvision.models as models

model = models.resnet101(pretrained=True)

class construct_net(object):
    def __init__(self, start_point: int = 0, end_point: int = 0, exit_type: bool = False,
                        backbone: str = 'unipose') -> None:
        super().__init__()
        self.backbone = backbone
        self.start_point = start_point
        self.end_point = end_point
        self.exit_type = exit_type

    def construct_net_s1(self):
        if self.backbone == 'resnet':
            return resnet_s1(start_point=self.start_point, end_point=self.end_point, simple_exit=self.exit_type)
        elif self.backbone == 'unipose':
            return unipose_s1(start_point=self.start_point, end_point=self.end_point)

    def construct_net_s2(self):
        if self.backbone == 'resnet':
            return resnet_s2(start_point=self.start_point, end_point=self.end_point)
        elif self.backbone == 'unipose':
            return unipose_s2(start_point=self.start_point, end_point=self.end_point)

def model_export_func(start_point, end_point, exit_type=False):

    inst = construct_net(start_point=start_point, end_point=end_point, exit_type=exit_type)
    dummy_input1 = torch.randn(1, 3, 513, 513)
    s1_model = inst.construct_net_s1()
    s1_model.eval()
    if False:
        dummy_input2 = s1_model(dummy_input1)
    else:
        dummy_input2, low_level_feat, x_exit = s1_model(dummy_input1)
    s2_model = inst.construct_net_s2()
    s2_model.eval()

    print("Split point: " + str(start_point))
    # print(dummy_input1.shape)
    print(dummy_input2.shape)
    print(low_level_feat.shape)


    if False:
        s1_input_names = ["input"]
        s1_output_names = ["output1", "exit_output"]
        torch.onnx.export(s1_model, dummy_input1, 
                            "/home/slzhang/projects/ETBA/Inference/src/exit_placement/models/resnet_s1.onnx",
                            input_names=s1_input_names, output_names=s1_output_names,
                            verbose=False,dynamic_axes={
                                        'input': {0: 'batch_size'},
                                        'output1': {0: 'batch_size'},
                                        'exit_output': {0: 'batch_size'},
                                    },opset_version=11)

        s2_input_names = ["input"]
        s2_output_names = ["final_output"]
        torch.onnx.export(s2_model, dummy_input2[0], 
                            "/home/slzhang/projects/ETBA/Inference/src/exit_placement/models/resnet_s2.onnx",
                            input_names=s2_input_names, output_names=s2_output_names,
                            verbose=False,dynamic_axes={
                                        'input': {0: 'batch_size'},
                                        'final_output': {0: 'batch_size'},
                                    },opset_version=11)
    else:
        s1_input_names = ["input"]
        s1_output_names = ["output1", "exit_output"]
        torch.onnx.export(s1_model, dummy_input1, 
                            "/home/slzhang/projects/ETBA/Inference/src/exit_placement/models/unipose_s1.onnx",
                            input_names=s1_input_names, output_names=s1_output_names,
                            verbose=False,dynamic_axes={
                                        'input': {0: 'batch_size'},
                                        'output1': {0: 'batch_size'},
                                        'exit_output': {0: 'batch_size'},
                                    },opset_version=11)

        s2_input_names = ["input", "low_level_feat"]
        s2_output_names = ["final_output"]
        torch.onnx.export(s2_model, (dummy_input2, low_level_feat),
                        "/home/slzhang/projects/ETBA/Inference/src/exit_placement/models/unipose_s2.onnx",
                        input_names=s2_input_names, output_names=s2_output_names,
                        verbose=False,dynamic_axes={
                                      'input': {0: 'batch_size'},
                                      'low_level_feat': {0: 'batch_size'},
                                      'final_output': {0: 'batch_size'},
                                  },opset_version=11)

if __name__ == '__main__':
    model_export_func(13, 17, False)