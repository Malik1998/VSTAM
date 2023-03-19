import detectron2.modeling.backbone
from detectron2.modeling import ResNet

import torch
import torch.nn as nn

class Resnet50Custom(ResNet):
    def __init__(self, pretrained=True, **kwargs):
        super(Resnet50Custom, self).__init__(**kwargs)
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', "resnet50", pretrained=pretrained)
        self.resnet.layer4[0].conv2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(2, 2), dilation=2, bias=False)
        self.resnet.layer4[0].downsample = nn.Conv2d(1024, 2048, kernel_size=1, bias=False)
        self.resnet.avgpool = IdentifyModel()
        self.resnet.fc = IdentifyModel()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x

from detectron2.config import get_cfg
from detectron2.modeling import build_model, build_backbone
from detectron2 import model_zoo

import yaml

with open("resnet.yml", "r") as stream:
    try:
        cfg_add = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

cfg = get_cfg()
# cfg.merge_from_file("resnet.yml")
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.RESNETS.RES5_DILATION = 2
# cfg.MODEL.RESNETS.STRIDE_IN_1X1 = False

print(cfg)
model = build_backbone(cfg)
print(model)
print(model.backbone)
print(model.backbone.bottom_up.res5[0].shortcut )
# model.backbone.bottom_up.res5[0].shortcut.stride = 1
# model.backbone.bottom_up.res5[0].shortcut.padding = (2, 2)
# model.backbone.bottom_up.res5[0].shortcut.dilation = 2

# model.backbone.bottom_up.res5[0].conv1.stride = 1
# model.backbone.bottom_up.res5[0].conv1.padding = (2, 2)
# model.backbone.bottom_up.res5[0].conv1.dilation = 2

from detectron2.structures.image_list import ImageList
BATCH_SIZE = 1
SEQUENCE_LENGTH = 5
SIDE_SIZE = 512 + 96
IMG_SIZE = (SIDE_SIZE, SIDE_SIZE)
CH_IN = 3
imgs_suplementary = torch.ones((BATCH_SIZE, CH_IN, *IMG_SIZE)).to("cuda")
out = model.backbone(imgs_suplementary)
print(out.keys())
print([v.shape for v in out.values()])
