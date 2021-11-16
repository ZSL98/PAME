# resnet_s1 includes post_layers while partial_resnet does not

from Mytrain.networks import resnet_s1, partial_resnet
from openseg.lib.models.nets.ocrnet_with_exit import SpatialOCRNet_with_only_exit, SpatialOCRNet_with_exit
from pose_estimation.lib.models.pose_resnet import PoseResNetwthExit, PoseResNetwthOnlyExit

# load backbone with the final head.

load_backbone()

# load state dict from the stage one model

load_head()
