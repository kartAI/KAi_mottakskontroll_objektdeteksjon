from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
import os
import torch
from datetime import datetime

setup_logger()

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Define backbone: choose "R_50" or "R_101"
backbone = "R_50"  # 🔁 Change to "R_101" for second run

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/mask_rcnn_{backbone}_FPN_3x.yaml"))
cfg.MODEL.DEVICE = device.type

# cfg.INPUT.AUG = [
#     {"name": "RandomFlip", "prob": 0.5, "horizontal": True},
#     {"name": "RandomBrightness", "intensity_range": [0.8, 1.2]},
#     {"name": "RandomContrast", "intensity_range": [0.8, 1.2]},
#     {"name": "RandomRotation", "angle": [-10, 10]},
#     {"name": "RandomExtent", "scale_range": [0.9, 1.1]},
#     {"name": "RandomApply", "transform_list": [{"name": "GaussianBlur", "kernel_size": [3, 3], "sigma_range": [0.1, 2.0]}], "prob": 0.3}, # Add this block
# ]

# Timestamped output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cfg.OUTPUT_DIR = os.path.join("output", f"phase3_step5_box2_mask025_{backbone}_{timestamp}")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Dataset setup
cfg.DATASETS.TRAIN = ("building_dataset_train",)
cfg.DATASETS.TEST = ("building_dataset_val",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

# Training hyperparameters
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025            # ✅ oppdatert for Fase 1
cfg.SOLVER.MAX_ITER = 30000              # ✅ baseline
cfg.SOLVER.STEPS = (20000, 26000)
cfg.SOLVER.WARMUP_ITERS = 1500
cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.TEST.EVAL_PERIOD = 1000

# Model behavior
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-InstanceSegmentation/mask_rcnn_{backbone}_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 2
cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT = 0.25

# # Optional: AMP (mixed precision)
# cfg.SOLVER.AMP.ENABLED = True
# cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
# cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
# cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
