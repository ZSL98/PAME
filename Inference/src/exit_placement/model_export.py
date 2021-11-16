import torch
import torch.nn as nn
from networks import resnet_s1, resnet_s2, posenet_s1, posenet_s2
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from wav2vec2_model import Wav2Vec2_with_exit_s1, Wav2Vec2_with_exit_s2
from modeling_bert import BertWithExit_s1, BertWithExit_s2
from ocrnet_with_exit import SpatialOCRNet_s1, SpatialOCRNet_s2, SpatialOCRNet
import torchvision.models as models
import copy

model = models.resnet101(pretrained=True)

class construct_net(object):
    def __init__(self, start_point: int = 0, end_point: int = 0, exit_type: bool = False,
                        backbone: str = 'Wav2Vec2') -> None:
        super().__init__()
        self.backbone = backbone
        self.start_point = start_point
        self.end_point = end_point
        self.exit_type = exit_type

    def construct_net_s1(self):
        if self.backbone == 'resnet':
            return resnet_s1(start_point=self.start_point, end_point=self.end_point, simple_exit=self.exit_type)
        elif self.backbone == 'posenet':
            return posenet_s1(start_point=self.start_point, end_point=self.end_point)
        elif self.backbone == 'bert':
            model = BertWithExit_s1.from_pretrained('bert-base-uncased')
            model.add_exit(start_point=self.start_point, end_point=self.end_point)
            return model
        elif self.backbone == 'Wav2Vec2':
            tokenizer = Wav2Vec2CTCTokenizer("/home/slzhang/projects/ETBA/Inference/src/exit_placement/vocab.json",
                                                 unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
            feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            model_CTC = Wav2Vec2_with_exit_s1.from_pretrained(
                "facebook/wav2vec2-base", 
                gradient_checkpointing=True, 
                ctc_loss_reduction="mean", 
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            model_CTC.add_exit(start_point=self.start_point, end_point=self.end_point)
            return model_CTC
        elif self.backbone == 'openseg':
            return SpatialOCRNet_s1(self.start_point)

    def construct_net_s2(self):
        if self.backbone == 'resnet':
            return resnet_s2(start_point=self.start_point, end_point=self.end_point)
        elif self.backbone == 'posenet':
            return posenet_s2(start_point=self.start_point, end_point=self.end_point)
        elif self.backbone == 'bert':
            model = BertWithExit_s2.from_pretrained('bert-base-uncased')
            model.add_exit(start_point=self.start_point, end_point=self.end_point)
            return model
        elif self.backbone == 'Wav2Vec2':
            tokenizer = Wav2Vec2CTCTokenizer("/home/slzhang/projects/ETBA/Inference/src/exit_placement/vocab.json",
                                             unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
            feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            model_CTC = Wav2Vec2_with_exit_s2.from_pretrained(
                "facebook/wav2vec2-base", 
                gradient_checkpointing=True, 
                ctc_loss_reduction="mean", 
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            model_CTC.add_exit(start_point=self.start_point, end_point=self.end_point)
            return model_CTC
        elif self.backbone == 'openseg':
            return SpatialOCRNet_s2(self.start_point)

def model_export_func(model_name, start_point, end_point, exit_type=False):

    inst = construct_net(start_point=start_point, end_point=end_point, exit_type=exit_type, backbone = model_name)
    # dummy_input1 = torch.randn(1, 3, 513, 513)
    if inst.backbone == "resnet" or inst.backbone == "posenet":
        dummy_input1 = torch.randn(1, 3, 384, 384)
    elif inst.backbone == "openseg":
        dummy_input1 = torch.randn(1, 3, 1024, 2048)
        # dummy_input1 = torch.randn(1, 3, 384, 384)
    elif inst.backbone == "Wav2Vec2":
        dummy_input1 = torch.randn(1, 10000)

    s1_model = inst.construct_net_s1()
    s1_model.eval()

    if inst.backbone == "resnet":
        dummy_input2 = s1_model(dummy_input1)
        dummy_input2 = dummy_input2[0]
    elif inst.backbone == "posenet":
        dummy_input2, x_exit = s1_model(dummy_input1)
        # dummy_input2 = dummy_input2[0]
    elif inst.backbone == "Wav2Vec2":
        dummy_input2 = torch.randn(1, 624, 768)
    elif inst.backbone == "openseg":
        # x_dsn, x, x_moveon = s1_model(dummy_input1)
        dummy_input2 = torch.randn(1, 1024, 129, 257)
        # dummy_input2 = torch.randn(1, 1024, 48, 48)
        # print(x_moveon.shape)

    s2_model = inst.construct_net_s2()
    s2_model.eval()

    # tmp_model = SpatialOCRNet()
    # s2_input_names = ["input"]
    # # s2_output_names = ["output_dsn", "output"]
    # s2_output_names = ["output"]
    # torch.onnx.export(tmp_model, dummy_input1,
    #                     "/home/slzhang/projects/ETBA/Inference/src/exit_placement/models/" + inst.backbone + "_s0.onnx",
    #                 input_names=s2_input_names, output_names=s2_output_names,
    #                 verbose=False,dynamic_axes={
    #                                 'input': {0: 'batch_size'},
    #                                 'output': {0: 'batch_size'},
    #                                 # 'output': {0: 'batch_size'},
    #                             },opset_version=11)
    # exit()

    print("Split point: " + str(start_point))
    # print(dummy_input1.shape)
    # print(dummy_input2.shape)

    if inst.backbone == "resnet" or inst.backbone == "posenet" or inst.backbone == "Wav2Vec2":
        s1_input_names = ["input"]
        s1_output_names = ["output1", "exit_output"]
        torch.onnx.export(s1_model, dummy_input1, 
                            "/home/slzhang/projects/ETBA/Inference/src/exit_placement/models/" + inst.backbone + "_s1.onnx",
                            input_names=s1_input_names, output_names=s1_output_names,
                            verbose=False,dynamic_axes={
                                        'input': {0: 'batch_size'},
                                        'output1': {0: 'batch_size'},
                                        'exit_output': {0: 'batch_size'},
                                    },opset_version=11)

        s2_input_names = ["input"]
        s2_output_names = ["final_output"]
        torch.onnx.export(s2_model, dummy_input2,
                            "/home/slzhang/projects/ETBA/Inference/src/exit_placement/models/" + inst.backbone + "_s2.onnx",
                        input_names=s2_input_names, output_names=s2_output_names,
                        verbose=False,dynamic_axes={
                                        'input': {0: 'batch_size'},
                                        'final_output': {0: 'batch_size'},
                                    },opset_version=11)
    elif inst.backbone == "openseg":
        s1_input_names = ["input"]
        # s1_output_names = ["output_dsn", "output", "x_moveon"]
        s1_output_names = ["output_dsn", "output"]
        torch.onnx.export(s1_model, dummy_input1, 
                            "/home/slzhang/projects/ETBA/Inference/src/exit_placement/models/" + inst.backbone + "_s1.onnx",
                            input_names=s1_input_names, output_names=s1_output_names,
                            verbose=False,dynamic_axes={
                                        'input': {0: 'batch_size'},
                                        'output_dsn': {0: 'batch_size'},
                                        'output': {0: 'batch_size'},
                                        # 'x_moveon': {0: 'batch_size'},
                                    },opset_version=11)

        s2_input_names = ["input"]
        s2_output_names = ["output_dsn", "output"]
        torch.onnx.export(s2_model, dummy_input2,
                            "/home/slzhang/projects/ETBA/Inference/src/exit_placement/models/" + inst.backbone + "_s2.onnx",
                        input_names=s2_input_names, output_names=s2_output_names,
                        verbose=False,dynamic_axes={
                                        'input': {0: 'batch_size'},
                                        'output_dsn': {0: 'batch_size'},
                                        'output': {0: 'batch_size'},
                                    },opset_version=11)
    elif inst.backbone == "bert":
        from transformers.convert_graph_to_onnx import load_graph_from_args, infer_shapes, ensure_valid_input

        pipeline_name = "feature-extraction"
        framework = "pt"
        model_name = "bert-base-uncased"
        tokenizer = "bert-base-uncased"
        nlp = load_graph_from_args(pipeline_name, framework, model_name, tokenizer)

        input_names, output_names, dynamic_axes, tokens = infer_shapes(nlp, "pt")
        ordered_input_names, s1_model_args = ensure_valid_input(nlp.model, tokens, input_names)
        dynamic_axes.pop('output_0')
        dynamic_axes.pop('output_1')

        s1_output_names = ["output1", "exit_output"]
        s1_dynamic_axes = copy.deepcopy(dynamic_axes)
        s1_dynamic_axes['output1'] = {0: 'batch', 1: 'sequence'}
        s1_dynamic_axes['exit_output'] = {0: 'batch'}

        torch.onnx.export(
            s1_model,
            s1_model_args,
            f="/home/slzhang/projects/ETBA/Inference/src/exit_placement/models/" + inst.backbone + "_s1.onnx",
            input_names=ordered_input_names,
            output_names=s1_output_names,
            dynamic_axes=s1_dynamic_axes,
            do_constant_folding=True,
            enable_onnx_checker=True,
            opset_version=11,
        )

        s2_output_names = ["final_output"]
        s2_dynamic_axes = copy.deepcopy(dynamic_axes)
        s2_dynamic_axes.pop('token_type_ids')
        s2_dynamic_axes['final_output'] = {0: 'batch'}
        s2_model_args = (torch.Tensor(1, 7, 768), torch.Tensor(1, 7))

        torch.onnx.export(
            s2_model,
            s2_model_args,
            f="/home/slzhang/projects/ETBA/Inference/src/exit_placement/models/" + inst.backbone + "_s2.onnx",
            input_names=ordered_input_names[:2],
            output_names=s2_output_names,
            dynamic_axes=s2_dynamic_axes,
            do_constant_folding=True,
            enable_onnx_checker=True,
            opset_version=11,
        )

    # elif inst.backbone == "Wav2Vec2":
    #     pass


if __name__ == '__main__':
    model_export_func('resnet', 1, 1, False)