from typing import OrderedDict
import torch
import argparse

# resnet_s1 includes post_layers while partial_resnet does not
# resnet_s1 has dual-heads
from Mytrain.networks import resnet_s1, partial_resnet

# SpatialOCRNet_with_exit has dual-heads, while SpatialOCRNet_with_only_exit has only one exit
from openseg.lib.models.nets.ocrnet_with_exit import SpatialOCRNet_with_only_exit, SpatialOCRNet_with_exit
from openseg.lib.utils.tools.configer import Configer
from openseg.lib.datasets.data_loader import DataLoader
from openseg.segmentor.tools.data_helper import DataHelper
from openseg.segmentor.tools.evaluator import get_evaluator
from openseg.segmentor.etrainer import ETrainer
from openseg import main

# PoseResNetwthExit has dual-heads
from pose_estimation.lib.models.pose_resnet import PoseResNetwthExit, PoseResNetwthOnlyExit


def load_openseg():
    # load the backbone of the network with dual-heads

    net_wth_finalhead = torch.load("/home/slzhang/projects/ETBA/Train/openseg/checkpoints/cityscapes/ocrnet_resnet101_s33_latest.pth")
    # for k,v in net_wth_finalhead['state_dict'].items():
    #     print(k)

    config = Configer(configs="/home/slzhang/projects/ETBA/Train/openseg/configs/cityscapes/R_101_D_8_with_exit.json")
    split_point = config.get("network", "split_point")
    net_wth_eehead = torch.load("/home/slzhang/projects/ETBA/Train/openseg/checkpoints/cityscapes/ocrnet_resnet101_s"+str(split_point)+"_latest.pth")
    # for k,v in net_wth_eehead['state_dict'].items():
    #     print(k)


    net_wth_dualheads = SpatialOCRNet_with_exit(config)
    # for k,v in net_wth_dualheads.state_dict().items():
    #     print(k)

    dict_finalhead = net_wth_finalhead['state_dict'].copy()
    dict_eehead = net_wth_eehead['state_dict'].copy()
    dict_dualhead = OrderedDict()

    for k,v in net_wth_dualheads.state_dict().items():
        if 'num_batches_tracked' not in k:
            if 'backbone_s1.resinit' in k:
                dict_dualhead[k] = dict_finalhead['module.backbone'+k[11:]]
            elif 'backbone_s1.pre' in k:
                dict_dualhead[k] = dict_finalhead['module.backbone.'+k[16:]]
            elif 'backbone_s1.exit' in k:
                dict_dualhead[k] = dict_eehead['module.'+k]
            elif 'backbone_s2.layer3' in k:
                new_idx = int(k.split('.')[2]) + config.get("network", "split_point")-7
                dict_dualhead[k] = dict_finalhead['module.backbone.layer3.'+str(new_idx)+'.'+'.'.join(k.split('.')[3:])]
            elif 'backbone_s2.layer4' in k:
                dict_dualhead[k] = dict_finalhead['module.backbone.'+k[12:]]
            elif 'conv_3x3_s1' in k:
                dict_dualhead[k] = dict_eehead['module.conv_3x3' + k[11:]]
            elif 'conv_3x3_s2' in k:
                dict_dualhead[k] = dict_finalhead['module.conv_3x3' + k[11:]]
            elif 'head_s1' in k:
                dict_dualhead[k] = dict_eehead['module.'+k.split('.')[0][:-3]+'.'+'.'.join(k.split('.')[1:])]
            elif 'head_s2' in k:
                dict_dualhead[k] = dict_finalhead['module.'+k.split('.')[0][:-3]+'.'+'.'.join(k.split('.')[1:])]     

    net_wth_dualheads.load_state_dict(dict_dualhead)
    return net_wth_dualheads

def eval_openseg(net):
    net = net.cuda()
    net.eval()
    config = Configer(configs="/home/slzhang/projects/ETBA/Train/openseg/configs/cityscapes/R_101_D_8_with_exit.json")
    abs_data_dir = "/home/slzhang/projects/ETBA/Train/openseg/data/cityscapes"
    config.update(["data", "data_dir"], abs_data_dir)
    config.add(["gpu"], [0, 1, 2, 3])
    config.add(["network", "gathered"], "n")
    config.add(["network", "resume"], None)
    config.add(["optim", "group_method"], None)
    config.add(["data", "include_val"], False)
    config.add(["data", "include_coarse"], False)
    config.add(["data", "include_atr"], False)
    config.add(["data", "only_coarse"], False)
    config.add(["data", "only_mapillary"], False)
    config.add(["data", "drop_last"], True)
    config.add(["network", "loss_balance"], False)
    data_loader = DataLoader(config)
    etrainer = ETrainer(config)
    data_helper = DataHelper(config, etrainer)
    val_loader = data_loader.get_valloader()
    evaluator = get_evaluator(config, etrainer)

    for j, data_dict in enumerate(val_loader):
        (inputs, targets), batch_size = data_helper.prepare_data(data_dict)
        with torch.no_grad():
            outputs = net(*inputs)
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
            print(len(outputs[0]))
            # final_outputs = supervise(outputs)
            # evaluator.update_score(outputs, data_dict["meta"])


if __name__ == '__main__':
    net = load_openseg()
    eval_openseg(net)



# load_backbone()

# load state dict from the stage one model

# load_head()
