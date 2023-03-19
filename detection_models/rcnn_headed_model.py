import torch
from base_backbone.base_backbone import BaseBackbone
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


@BACKBONE_REGISTRY.register()
class RCNNHeadedModel(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.res4 = BaseBackbone()
        self.out_channels = self.res4.feature_embedder_out

    def forward(self, image, imgs_suplementary=None):
        # Because we can send only one image to detectron2 to default detection
        # so we sent first image and concatenate previous frames here
        concated = []
        for main_img, stacked_img in zip(image, imgs_suplementary):
            concated.append(torch.unsqueeze(torch.cat([
                torch.unsqueeze(main_img, 0), stacked_img], 0), 0))
        concated = torch.cat(concated, 0)
        return {"res4": self.res4(concated)}

    def output_shape(self):
        return {"res4": ShapeSpec(channels=2048, stride=16)}

