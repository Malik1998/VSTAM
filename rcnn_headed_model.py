import torch
import torch.nn as nn
from base_model import get_model_by_name
from identify import IdentifyModel
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.structures.image_list import ImageList

@BACKBONE_REGISTRY.register()
class RCNNHeadedModel(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.res4 = get_model_by_name("BaseBackbone")()
        self.out_channels = 2048

    def forward(self, image, imgs_suplementary=None):
        concated = []
        for main_img, stacked_img in zip(image, imgs_suplementary):
            concated.append(torch.unsqueeze(torch.cat([
                torch.unsqueeze(main_img, 0), stacked_img], 0), 0))
        concated = torch.cat(concated, 0)
        return {"res4": self.res4(concated)}

    def output_shape(self):
        return {"res4": ShapeSpec(channels=2048 * 1, stride=16)}

