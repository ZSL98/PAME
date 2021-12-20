##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Shulai Zhang
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2018
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pdb
from typing import OrderedDict
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

import sys
sys.path.append("/home/slzhang/projects/ETBA/Train/openseg")
from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.tools.module_helper import ModuleHelper
from lib.models.backbones.resnet.resnet_backbone_with_exit import backbone_s1, backbone_s2, Bottleneck

class SpatialOCRNet_with_only_exit(nn.Module):
    """
    Object-Contextual Representations for Semantic Segmentation,
    Yuan, Yuhui and Chen, Xilin and Wang, Jingdong
    """
    def __init__(self, configer):
        self.inplanes = 128
        self.for_finetune = None
        super(SpatialOCRNet_with_only_exit, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.split_point = self.configer.get('network', 'split_point')
        # self.backbone_s1 = BackboneSelector(configer).get_backbone()
        split_point = self.configer.get("network", "split_point")
        self.backbone_s1 = backbone_s1(start_point=split_point, end_point=split_point, bn_type='implace_abn')

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [1024, 2048]
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )

        from lib.models.modules.spatial_ocr_block import SpatialGather_Module, SpatialOCR_Module
        self.spatial_context_head = SpatialGather_Module(self.num_classes)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=512, 
                                                  key_channels=256, 
                                                  out_channels=512,
                                                  scale=1,
                                                  dropout=0.05, 
                                                  bn_type=self.configer.get('network', 'bn_type'))

        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )
        self.init_weights()

    def init_weights(self):
        # model_pretrained = models.resnet101(pretrained=True)
        # new_list = list (self.backbone_s1.state_dict().keys())
        # trained_list = list (model_pretrained.state_dict().keys())

        # dict_trained = model_pretrained.state_dict().copy()
        # dict_new = self.backbone_s1.state_dict().copy()
        
        # checkpoint = torch.load("/home/slzhang/projects/ETBA/Train/openseg/pretrained_model/resnet101-5d3b4d8f.pth")
        # dict_s1 = self.backbone_s1.state_dict().copy()

        # head_checkpoint = torch.load("/home/slzhang/projects/ETBA/Train/openseg/checkpoints/cityscapes/ocrnet_resnet101_s" + \
        #                             str(self.split_point+1) + "_latest.pth")
        head_checkpoint = torch.load("/home/slzhang/projects/ETBA/Train/openseg/checkpoints/spatial_ocrnet_deepbase_resnet101_dilated8_1_latest.pth")
        # head_checkpoint_2 = torch.load("/home/slzhang/projects/ETBA/Train/openseg/checkpoints/cityscapes/ocrnet_resnet101_s8_latest_trained_2.pth")

        dict_trained = head_checkpoint['state_dict'].copy()
        # dict_trained_2 = head_checkpoint_2['state_dict'].copy()
        # dict_new = self.state_dict().copy()
        dict_new = OrderedDict()        
        
        for k,v in self.state_dict().items():
            if 'num_batches_tracked' not in k:
                if 'backbone_s1.resinit' in k:
                    # copy the backbone parameters
                    dict_new[k] = dict_trained['module.backbone'+k[11:]]
                elif 'backbone_s1.pre' in k:
                    # copy the backbone parameters
                    dict_new[k] = dict_trained['module.backbone.'+k[16:]]
                elif 'head' in k or 'conv_3x3' in k:
                    # copy the head parameters
                    dict_new[k] = dict_trained['module.'+k]
                elif 'exit' in k and 'num' not in k:
                    # copy the exit parameters
                    dict_new[k] = dict_trained['module.backbone.layer4'+k[16:]]

        # dict_new['backbone_s1.resinit.conv1.weight'] = dict_trained_2['module.backbone_s1.resinit.conv1.weight']
        # dict_new['backbone_s1.resinit.bn1.weight'] = dict_trained_2['module.backbone_s1.resinit.bn1.weight']
        # dict_new['backbone_s1.resinit.bn1.bias'] = dict_trained_2['module.backbone_s1.resinit.bn1.bias']
        # dict_new['backbone_s1.resinit.bn1.running_mean'] = dict_trained_2['module.backbone_s1.resinit.bn1.running_mean']
        # dict_new['backbone_s1.resinit.bn1.running_var'] = dict_trained_2['module.backbone_s1.resinit.bn1.running_var']
        # for k,v in self.state_dict().items():
        #     if 'num_batches_tracked' not in k:
        #         if 'backbone_s1.resinit' in k:
        #             dict_new[k] = dict_trained_2['module.backbone_s1'+k[11:]]
        #         elif 'backbone_s1.pre' in k:
        #             dict_new[k] = dict_trained_2['module.backbone_s1.pre_'+k[16:]]
        #         elif 'head' in k or 'conv_3x3' in k:
        #             dict_new[k] = dict_trained['module.'+k]

        # for k,v in self.state_dict().items():
        #     for k1, v1 in head_checkpoint_2['state_dict'].items():
        #         if k==k1[7:]:
        #             dict_new[k] = dict_trained_2[k1]      

        # for k,v in self.state_dict().items():
        #     if 'backbone' not in k and 'num' not in k:
        #         dict_new[k] = dict_trained['module.'+k]

        # freeze the backbone parameters
        for k,v in self.named_parameters():
            if 'backbone' in k and 'exit' not in k:
                v.requires_grad=False
            else:
                v.requires_grad=True
        
        self.load_state_dict(dict_new)
        # self.backbone_s1.load_state_dict(dict_new)
        #     for k_s1,v_s1 in self.backbone_s1.state_dict().items():
        #         if k == k_s1[4:]:
        #             dict_s1[k_s1] = checkpoint[k]

        # self.backbone_s1.load_state_dict(dict_s1)

    def forward(self, x_):
        x = self.backbone_s1(x_)
        x1_dsn = self.dsn_head(x[-2])
        x1 = self.conv_3x3(x[-1])
        context1 = self.spatial_context_head(x1, x1_dsn)
        x1 = self.spatial_ocr_head(x1, context1)
        x1 = self.head(x1)
        x1_dsn = F.interpolate(x1_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x1 = F.interpolate(x1, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)

        return  x1_dsn, x1

# TODO: Class below NOT finished yet
class SpatialOCRNet_with_multi_exit(nn.Module):
    def __init__(self, configer):
        super(SpatialOCRNet_with_multi_exit, self).__init__()
        self.configer = configer
        self.ori_backbone = nn.ModuleList()
        self.backbone = nn.ModuleList()
        self.exit_list = self.configer.get('network', 'exit_list')
        for i in range(len(self.exit_list)):
            self.ori_backbone.append(SpatialOCRNet_with_only_exit(self.configer))
            state_dict = torch.load('/home/slzhang/projects/ETBA/Train/openseg/checkpoints/cityscapes_metric_controlled/split_point_{}/model_best.pth'.format(self.exit_list[i]))
            new_dict = OrderedDict()

            for k,v in self.ori_backbone[i].state_dict().items():
                new_dict[k] = state_dict['module.'+k]
            self.ori_backbone[i].load_state_dict(new_dict, strict=True)

        ori_backbone_copy = copy.deepcopy(self.ori_backbone)

        for i in range(len(self.exit_list)):
            if i == 0:
                flatt_model = nn.Sequential(*list(ori_backbone_copy[i].backbone_s1.children())[:-1])
                self.backbone.append(flatt_model)
            else:
                print('-------------------')
                backbone = nn.Sequential()
                last_bottleneck_num = 0
                for layer in ori_backbone_copy[i-1].backbone_s1.named_modules():
                    if isinstance(layer[1], Bottleneck) and 'exit' not in layer[0]:
                        last_bottleneck_num = last_bottleneck_num + 1

                cnt = 0
                for layer in ori_backbone_copy[i].backbone_s1.named_modules():
                    if isinstance(layer[1], Bottleneck) and 'exit' not in layer[0]:
                        cnt = cnt + 1
                        if cnt > last_bottleneck_num:
                            backbone.add_module(layer[0].replace('.',' '), layer[1])

                self.backbone.append(backbone)
            for k,v in self.backbone[i].named_parameters():
                v.requires_grad=True
            for k,v in self.ori_backbone[i].named_parameters():
                v.requires_grad=True

    def forward(self, x):
        x_ = x
        output = []
        for i in range(len(self.exit_list)):
            x = self.backbone_s1(x)
            x1_dsn = self.dsn_head(x[-2])
            x1 = self.conv_3x3(x[-1])
            context1 = self.spatial_context_head(x1, x1_dsn)
            x1 = self.spatial_ocr_head(x1, context1)
            x1 = self.head(x1)
            x1_dsn = F.interpolate(x1_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
            x1 = F.interpolate(x1, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)

        return output

class SpatialOCRNet_with_exit(nn.Module):
    """
    Object-Contextual Representations for Semantic Segmentation,
    Yuan, Yuhui and Chen, Xilin and Wang, Jingdong
    """
    def __init__(self, configer):
        self.inplanes = 128
        self.for_finetune = True
        super(SpatialOCRNet_with_exit, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone_s1, self.backbone_s2 = BackboneSelector(configer).get_backbone()
        # self.init_weights()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [1024, 2048]
        self.conv_3x3_s1 = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )

        self.conv_3x3_s2 = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
        )

        from lib.models.modules.spatial_ocr_block import SpatialGather_Module, SpatialOCR_Module
        self.spatial_context_head_s1 = SpatialGather_Module(self.num_classes)
        self.spatial_ocr_head_s1 = SpatialOCR_Module(in_channels=512, 
                                                  key_channels=256, 
                                                  out_channels=512,
                                                  scale=1,
                                                  dropout=0.05, 
                                                  bn_type=self.configer.get('network', 'bn_type'))

        self.head_s1 = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head_s1 = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

        self.spatial_context_head_s2 = SpatialGather_Module(self.num_classes)
        self.spatial_ocr_head_s2 = SpatialOCR_Module(in_channels=512, 
                                                  key_channels=256, 
                                                  out_channels=512,
                                                  scale=1,
                                                  dropout=0.05, 
                                                  bn_type=self.configer.get('network', 'bn_type'))

        self.head_s2 = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head_s2 = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def init_weights(self):
        checkpoint = torch.load("/home/slzhang/projects/ETBA/Train/openseg/pretrained_model/resnet101-5d3b4d8f.pth")
        dict_s1 = self.backbone_s1.state_dict().copy()
        dict_s2 = self.backbone_s2.state_dict().copy()

        for k,v in checkpoint.items():
            for k_s1,v_s1 in self.backbone_s1.state_dict().items():
                if k == k_s1[4:]:
                    dict_s1[k_s1] = checkpoint[k]

            for k_s2,v_v2 in self.backbone_s2.state_dict().items():
                if k.split(".")[0] == k_s2.split(".")[0] and k.split(".")[2:] == k_s2.split(".")[2:] and k.split(".")[0] == "layer3" and int(k.split(".")[1]) == int(k_s2.split(".")[1])+self.backbone_s2.end_point-7:
                    dict_s2[k_s2] = checkpoint[k]
                elif k.split(".") == k_s2.split(".")[1:]:
                    dict_s2[k_s2] = checkpoint[k]

        self.backbone_s1.load_state_dict(dict_s1)
        self.backbone_s2.load_state_dict(dict_s2)

    def forward(self, x_):
        x = self.backbone_s1(x_)
        # print(x[-1].shape)
        # print(x[-2].shape)
        x1_dsn = self.dsn_head_s1(x[-2])
        x1 = self.conv_3x3_s1(x[-1])
        # print(x1_dsn.shape)
        # print(x1.shape)
        context1 = self.spatial_context_head_s1(x1, x1_dsn)
        x1 = self.spatial_ocr_head_s1(x1, context1)
        x1 = self.head_s1(x1)
        x1_dsn = F.interpolate(x1_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x1 = F.interpolate(x1, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)

        x2 = self.backbone_s2(x[-2])
        # print(x2[-1].shape)
        # print(x2[-2].shape)
        x2_dsn = self.dsn_head_s2(x2[-2])
        x2 = self.conv_3x3_s2(x2[-1])
        # print(x2_dsn.shape)
        # print(x2.shape)
        context2 = self.spatial_context_head_s2(x2, x2_dsn)
        x2 = self.spatial_ocr_head_s2(x2, context2)
        x2 = self.head_s2(x2)
        x2_dsn = F.interpolate(x2_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x2 = F.interpolate(x2, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return  x1_dsn, x1, x2_dsn, x2

    def load_parameters(self):
        net_wth_finalhead = torch.load("/home/slzhang/projects/ETBA/Train/openseg/checkpoints/spatial_ocrnet_deepbase_resnet101_dilated8_1_latest.pth")
        split_point = self.configer.get('network', 'split_point')
        print(split_point)
        net_wth_eehead = torch.load("/home/slzhang/projects/ETBA/Train/openseg/checkpoints/cityscapes_metric_controlled/split_point_{}/ocrnet_resnet101_s{}_max_performance.pth".format(split_point,split_point))

        dict_finalhead = net_wth_finalhead['state_dict'].copy()
        dict_eehead = net_wth_eehead['state_dict'].copy()
        dict_dualhead = OrderedDict()

        for k,v in net_wth_eehead['state_dict'].items():
            print(k)

        for k,v in self.state_dict().items():
            if 'num_batches_tracked' not in k:
                if 'backbone_s1.resinit' in k:
                    dict_dualhead[k] = dict_finalhead['module.backbone'+k[11:]]
                elif 'backbone_s1.pre' in k:
                    dict_dualhead[k] = dict_finalhead['module.backbone.'+k[16:]]
                elif 'backbone_s1.exit' in k:
                    dict_dualhead[k] = dict_eehead['module.'+k]
                elif 'backbone_s2.layer3' in k:
                    new_idx = int(k.split('.')[2]) + split_point-7
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


        self.load_state_dict(dict_dualhead)
        # for k,v in self.named_parameters():
        #     if v.requires_grad == True:
        #         print(k)


class ASPOCRNet(nn.Module):
    """
    Object-Contextual Representations for Semantic Segmentation,
    Yuan, Yuhui and Chen, Xilin and Wang, Jingdong
    """
    def __init__(self, configer):
        self.inplanes = 128
        super(ASPOCRNet, self).__init__()
        self.configer = configer
        self.num_classes = self.configer.get('data', 'num_classes')
        self.backbone = BackboneSelector(configer).get_backbone()

        # extra added layers
        if "wide_resnet38" in self.configer.get('network', 'backbone'):
            in_channels = [2048, 4096] 
        else:
            in_channels = [1024, 2048]

        # we should increase the dilation rates as the output stride is larger
        from lib.models.modules.spatial_ocr_block import SpatialOCR_ASP_Module
        self.asp_ocr_head = SpatialOCR_ASP_Module(features=2048, 
                                                  hidden_features=256, 
                                                  out_features=256,
                                                  num_classes=self.num_classes,
                                                  bn_type=self.configer.get('network', 'bn_type'))

        self.head = nn.Conv2d(256, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            ModuleHelper.BNReLU(512, bn_type=self.configer.get('network', 'bn_type')),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

    def forward(self, x_):
        x = self.backbone(x_)
        x_dsn = self.dsn_head(x[-2])
        x = self.asp_ocr_head(x[-1], x_dsn)
        x = self.head(x)
        x_dsn = F.interpolate(x_dsn, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        x = F.interpolate(x, size=(x_.size(2), x_.size(3)), mode="bilinear", align_corners=True)
        return  x_dsn, x
