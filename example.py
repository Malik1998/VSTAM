import torch
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.structures.image_list import ImageList
from detection_models.rcnn_headed_model import *

BATCH_SIZE = 1
SEQUENCE_LENGTH = 5
ONE_SIDE_SIZE = 324
IMG_SIZE = (ONE_SIDE_SIZE, ONE_SIDE_SIZE)
CH_IN = 3

imgs_suplementary = torch.ones((BATCH_SIZE, SEQUENCE_LENGTH, CH_IN, *IMG_SIZE)).to("cuda")
imgs = torch.ones((BATCH_SIZE, CH_IN, *IMG_SIZE)).to("cuda")
imgs = ImageList(imgs, [IMG_SIZE for i in range(BATCH_SIZE)])

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.BACKBONE.NAME = 'RCNNHeadedModel'
cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 512
model = build_model(cfg)
model.eval()
features = model.backbone(imgs, imgs_suplementary=imgs_suplementary)
proposals, _ = model.proposal_generator(imgs, features)
instances, _ = model.roi_heads(imgs, features, proposals)
print(instances)