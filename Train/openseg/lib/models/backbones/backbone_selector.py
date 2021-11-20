##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Donny You, RainbowSecret
## Microsoft Research
## yuyua@microsoft.com
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.models.backbones.resnet.resnet_backbone import ResNetBackbone
from lib.models.backbones.hrnet.hrnet_backbone import HRNetBackbone
from lib.models.backbones.resnet.resnet_backbone_with_exit import backbone_s1, backbone_s2
from lib.utils.tools.logger import Logger as Log


class BackboneSelector(object):
    def __init__(self, configer):
        self.configer = configer

    def get_backbone(self, **params):
        backbone = self.configer.get("network", "backbone")
        split_point = self.configer.get("network", "split_point")

        # when using tensorrt, remember to switch bn_type to 'torchbn' and modify module.py
        if 'only_exit' in backbone:
            return backbone_s1(start_point=split_point, end_point=split_point, bn_type='implace_abn')

        if 'exit' in backbone:
            model = list()
            model.append(backbone_s1(start_point=split_point, end_point=split_point))
            model.append(backbone_s2(start_point=split_point, end_point=split_point))
            return model

        model = None
        if (
            "resnet" in backbone or "resnext" in backbone or "resnest" in backbone
        ) and "senet" not in backbone:
            model = ResNetBackbone(self.configer)(**params)

        elif "hrne" in backbone:
            model = HRNetBackbone(self.configer)(**params)

        else:
            Log.error("Backbone {} is invalid.".format(backbone))
            exit(1)

        return model
