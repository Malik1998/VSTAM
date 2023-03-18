import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

class FasterRCNNHeadSimple(nn.Module):
    def __init__(self,
                 backbone,
                 num_classes=30,
                 ):
        super(FasterRCNNHeadSimple, self).__init__()
        self.anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                                aspect_ratios=((0.5, 1.0, 2.0),))
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                             output_size=7,
                                                             sampling_ratio=2)
        self.faster_rcnn = FasterRCNN(backbone,
                                num_classes=num_classes,
                                rpn_anchor_generator=self.anchor_generator,
                                box_roi_pool=self.roi_pooler)

    def forward(self, input, **kwargs):

        return self.faster_rcnn(input, **kwargs)



