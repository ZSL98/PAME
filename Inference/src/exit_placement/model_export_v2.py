import torch
import torch.nn as nn
from networks_v2 import resnet_s1, resnet_s2, posenet_s1, posenet_s2, backbone_s2, backbone_s3, backbone_init
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
from wav2vec2_model import Wav2Vec2_with_exit_s1, Wav2Vec2_with_exit_s2, Wav2Vec2_with_dual_exit
from modeling_bert import BertWithExit_s1, BertWithExit_s2, BertWithDualExit
from ocrnet_with_exit import SpatialOCRNet_s1, SpatialOCRNet_s2, SpatialOCRNet
import torchvision.models as models
import copy

model = models.resnet101(pretrained=True)

class construct_net(object):
    def __init__(self, begin_point: int = 0, split_point: int = 0, backbone: str = 'resnet') -> None:
        super().__init__()
        self.backbone = backbone
        self.begin_point = begin_point
        self.split_point = split_point

    def construct_net_init(self):
        if self.backbone == 'resnet' or self.backbone == 'posenet':
            return backbone_init(layers=[3, 4, 23, 3],
                                split_point_s1=self.begin_point, 
                                split_point_s2=self.begin_point, 
                                split_point_s3=self.begin_point)

    def construct_net_s1(self):
        if self.backbone == 'resnet':
            if self.begin_point == 0:
                return resnet_s1(layers=[3, 4, 23, 3],
                                begin_point=self.begin_point,
                                split_point_s1=self.split_point, 
                                split_point_s2=self.split_point, 
                                split_point_s3=self.split_point, 
                                is_init=True)
            else:
                return resnet_s1(layers=[3, 4, 23, 3],
                                begin_point=self.begin_point,
                                split_point_s1=self.split_point, 
                                split_point_s2=self.split_point, 
                                split_point_s3=self.split_point, 
                                is_init=False)

        elif self.backbone == 'posenet':
            if self.begin_point == 0:
                return posenet_s1(layers=[3, 4, 23, 3],
                                begin_point=self.begin_point,
                                split_point_s1=self.split_point, 
                                split_point_s2=self.split_point, 
                                split_point_s3=self.split_point, 
                                is_init=True)
            else:
                return posenet_s2(layers=[3, 4, 23, 3],
                                begin_point=self.begin_point,
                                split_point_s1=self.split_point, 
                                split_point_s2=self.split_point, 
                                split_point_s3=self.split_point, 
                                is_init=False)

        elif self.backbone == 'bert':
            if self.begin_point == 0:
                model = BertWithExit_s1.from_pretrained('bert-base-uncased')
                model.add_exit(start_point=self.split_point, end_point=self.split_point)
                return model
            else:
                model = BertWithDualExit.from_pretrained('bert-base-uncased')
                model.add_exit(num_hidden_layers=self.split_point)
                return model   

        elif self.backbone == 'Wav2Vec2':
            tokenizer = Wav2Vec2CTCTokenizer("/home/slzhang/projects/ETBA/Inference/src/exit_placement/vocab.json",
                                                 unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
            feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

            if self.begin_point == 0:
                model_CTC = Wav2Vec2_with_exit_s1.from_pretrained(
                    "facebook/wav2vec2-base", 
                    gradient_checkpointing=True, 
                    ctc_loss_reduction="mean", 
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
                model_CTC.add_exit(start_point=self.split_point, end_point=self.split_point)
            else:
                model_CTC = Wav2Vec2_with_dual_exit.from_pretrained(
                    "facebook/wav2vec2-base", 
                    gradient_checkpointing=True, 
                    ctc_loss_reduction="mean", 
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
                model_CTC.add_exit(num_hidden_layers=self.split_point)                
            return model_CTC

        elif self.backbone == 'openseg':
            return SpatialOCRNet_s1(self.start_point)

    def construct_net_s2(self):
        if self.backbone == 'resnet':
            return resnet_s2(layers=[3, 4, 23, 3],
                             begin_point=self.begin_point,
                             split_point_s1=self.split_point, 
                             split_point_s2=self.split_point, 
                             split_point_s3=self.split_point, 
                               )
        elif self.backbone == 'posenet':
            return posenet_s2(layers=[3, 4, 23, 3],
                             begin_point=self.begin_point,
                             split_point_s1=self.split_point, 
                             split_point_s2=self.split_point, 
                             split_point_s3=self.split_point, 
                               )
        elif self.backbone == 'bert':
            model = BertWithExit_s2.from_pretrained('bert-base-uncased')
            model.add_exit(end_point=self.begin_point+self.split_point)
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
            model_CTC.add_exit(end_point=self.begin_point+self.split_point)
            return model_CTC
        elif self.backbone == 'openseg':
            return SpatialOCRNet_s2(self.start_point)

    def construct_net_s3(self):
        if self.backbone == 'resnet' or self.backbone == 'posenet':
            return backbone_s3(layers=[3, 4, 23, 3],
                               split_point_s1=self.split_point_s1,
                               split_point_s2=self.split_point_s2, 
                               split_point_s3=self.split_point_s3
                               )


def model_export_func(model_name, begin_point, split_point, exit_type=False):

    inst = construct_net(begin_point=begin_point, split_point=split_point, backbone = model_name)

    # dummy_input1 = torch.randn(1, 3, 513, 513)
    if inst.backbone == "resnet":
        dummy_input1 = torch.randn(1, 3, 224, 224)
        if begin_point != 0:
            model_init = inst.construct_net_init()
            model_init.eval()
            dummy_input1 = model_init(dummy_input1)
            dummy_input1 = dummy_input1[0]

    elif inst.backbone == "posenet":
        dummy_input1 = torch.randn(1, 3, 384, 384)
        if begin_point != 0:
            model_init = inst.construct_net_init()
            model_init.eval()
            dummy_input1 = model_init(dummy_input1)
            dummy_input1 = dummy_input1[0]

    elif inst.backbone == "openseg":
        dummy_input1 = torch.randn(1, 3, 1024, 2048)
        # dummy_input1 = torch.randn(1, 3, 384, 384)
    elif inst.backbone == "Wav2Vec2":
        dummy_input1 = torch.randn(1, 10000)
        if begin_point != 0:
            dummy_input1 = torch.randn(1, 31, 768)

    s1_model = inst.construct_net_s1()
    s1_model.eval()

    if inst.backbone == "resnet":
        dummy_input2 = s1_model(dummy_input1)
        dummy_input2 = dummy_input2[0]
    elif inst.backbone == "posenet":
        dummy_input2, x_exit = s1_model(dummy_input1)
        # dummy_input2 = dummy_input2[0]
    elif inst.backbone == "Wav2Vec2":
        dummy_input2 = torch.randn(1, 31, 768)
    elif inst.backbone == "openseg":
        # x_dsn, x, x_moveon = s1_model(dummy_input1)
        dummy_input2 = torch.randn(1, 1024, 129, 257)
        # dummy_input2 = torch.randn(1, 1024, 48, 48)
        # print(x_moveon.shape)

    s2_model = inst.construct_net_s2()
    s2_model.eval()

    # tmp_model = SpatialOCRNet()
    # tmp_model = model
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

    print("Begin point: " + str(begin_point))
    print("Split point: " + str(split_point))
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

        if begin_point == 0:
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
        else:
            s1_output_names = ["output1", "exit_output"]
            s1_dynamic_axes = copy.deepcopy(dynamic_axes)
            s1_dynamic_axes.pop('token_type_ids')
            s1_dynamic_axes['output1'] = {0: 'batch', 1: 'sequence'}
            s1_dynamic_axes['exit_output'] = {0: 'batch'}
            s1_model_args = (torch.Tensor(1, 7, 768), torch.Tensor(1, 7))

            torch.onnx.export(
                s1_model,
                s1_model_args,
                f="/home/slzhang/projects/ETBA/Inference/src/exit_placement/models/" + inst.backbone + "_s1.onnx",
                input_names=ordered_input_names[:2],
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

def model_export_func_backup(model_name, split_point_s1, split_point_s2, split_point_s3):

    inst = construct_net(split_point_s1=split_point_s1, split_point_s2=split_point_s2, split_point_s3=split_point_s3, backbone = model_name)
    # dummy_input1 = torch.randn(1, 3, 513, 513)
    if inst.backbone == "resnet" or inst.backbone == "posenet":
        dummy_input1 = torch.randn(1, 3, 224, 224)
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

    if inst.backbone == "resnet" or inst.backbone == "posenet":
        dummy_input3 = s2_model(dummy_input2)
    elif inst.backbone == "Wav2Vec2":
        dummy_input3 = torch.randn(1, 624, 768)
    elif inst.backbone == "openseg":
        # x_dsn, x, x_moveon = s1_model(dummy_input1)
        dummy_input3 = torch.randn(1, 1024, 129, 257)
        # dummy_input2 = torch.randn(1, 1024, 48, 48)
        # print(x_moveon.shape)

    s3_model = inst.construct_net_s3()
    s3_model.eval()

    print("split_point_s1: " + str(split_point_s1))
    print("split_point_s2: " + str(split_point_s2))
    print("split_point_s3: " + str(split_point_s3))
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
        s2_output_names = ["trans_output"]
        torch.onnx.export(s2_model, dummy_input2,
                            "/home/slzhang/projects/ETBA/Inference/src/exit_placement/models/" + inst.backbone + "_s2.onnx",
                        input_names=s2_input_names, output_names=s2_output_names,
                        verbose=False,dynamic_axes={
                                        'input': {0: 'batch_size'},
                                        'final_output': {0: 'batch_size'},
                                    },opset_version=11)

        s3_input_names = ["trans_input"]
        s3_output_names = ["final_output"]
        torch.onnx.export(s3_model, dummy_input3,
                            "/home/slzhang/projects/ETBA/Inference/src/exit_placement/models/" + inst.backbone + "_s3.onnx",
                        input_names=s3_input_names, output_names=s3_output_names,
                        verbose=False,dynamic_axes={
                                        'trans_input': {0: 'batch_size'},
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
    model_export_func('posenet', 0, 2)
