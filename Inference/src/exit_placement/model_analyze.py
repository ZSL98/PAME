import torch
import torch.nn as nn
from networks import resnet_s1, resnet_s2, posenet_s1, posenet_s2, PoseResNet
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC
from wav2vec2_model import Wav2Vec2_with_exit_s1, Wav2Vec2_with_exit_s2
from modeling_bert import BertWithExit_s1, BertWithExit_s2
import torchvision.models as models
import copy
from ptflops import get_model_complexity_info
from transformers import BertForSequenceClassification, BertTokenizer
from ocrnet_with_exit import SpatialOCRNet_with_exit, SpatialOCRNet


from functools import partial

start_point = 0
end_point = 0

def _input_constructor(input_shape, tokenizer):
    max_length = input_shape[1]
    
    model_input_ids = []
    model_attention_mask = []
    model_token_type_ids = []
    for _ in range(input_shape[0]):
        inp_seq = ""
        inputs = tokenizer.encode_plus(
            inp_seq,
            add_special_tokens=True,
            truncation_strategy='longest_first',
        )
        print(inputs)

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1] * len(input_ids)
        padding_length = max_length - len(input_ids)
        pad_token = tokenizer.pad_token_id
        pad_token_segment_id = 0
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length
        model_input_ids.append(input_ids)
        model_attention_mask.append(attention_mask)
        model_token_type_ids.append(token_type_ids)

    labels = torch.tensor([1] * input_shape[0])
    # Batch size input_shape[0], sequence length input_shape[128]
    inputs = {
        "input_ids": torch.tensor(model_input_ids),
        "token_type_ids": torch.tensor(model_token_type_ids),
        "attention_mask": torch.tensor(model_attention_mask),
    }
    inputs.update({"labels": labels})
    print([(k, v.size()) for k,v in inputs.items()])
    return inputs

with torch.cuda.device(0):
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    # tok = BertTokenizer.from_pretrained('bert-base-uncased')
    # macs, params = get_model_complexity_info(model, (1, 2), as_strings=True, input_constructor=partial(_input_constructor, tokenizer=tok),
    #                                         print_per_layer_stat=True, verbose=True)

    model = SpatialOCRNet()
    # model = PoseResNet()
    # model = models.resnet101(pretrained=True)
    # model = resnet_s1(start_point=8, end_point=8, simple_exit=False)
    macs, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)

    # model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')
    # macs, params = get_model_complexity_info(model, (20000,), as_strings=True,
    #                                         print_per_layer_stat=True, verbose=True)

    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))