import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.modeling import build_model, build_backbone
from detectron2.structures.image_list import ImageList
from rcnn_headed_model import *

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

BATCH_SIZE = 8
SEQUENCE_LENGTH = 5
IMG_SIZE = (128, 128)
CH_IN = 3
imgs_suplementary = torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, CH_IN, *IMG_SIZE)).to("cuda")
imgs = torch.ones((BATCH_SIZE, CH_IN, *IMG_SIZE)).to("cuda")
imgs = ImageList(imgs, [IMG_SIZE for i in range(BATCH_SIZE)])
boxes = torch.FloatTensor([[0, 0, 32, 32], [1, 1, 20, 20]]).to("cuda")
labels = torch.IntTensor([0, 1]).to("cuda")
targets = 8 * [{
    "boxes": boxes,
    "labels": labels
}]
cfg.MODEL.BACKBONE.NAME = 'RCNNHeadedModel'   # or set it in the config file
cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 512

# backbone = build_backbone(cfg)
model = build_model(cfg)
model.eval()
features = model.backbone(imgs, imgs_suplementary=imgs_suplementary)
proposals, _ = model.proposal_generator(imgs, features)
instances, _ = model.roi_heads(imgs, features, proposals)
print(len(instances))