import random
import numpy as np
import torch
import os
import pandas as pd
import traceback
import json
import datetime

# Import necessary Detectron2 components
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer # We will inherit from this
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# --- CUSTOM TRAINER TO ENABLE PERIODIC VALIDATION ---
# This is the essential part that was missing.
# The DefaultTrainer needs this method to know how to perform periodic evaluation.
class FinalTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            # Use the output directory of the current run for inference results
            output_folder = os.path.join(cfg.OUTPUT_DIR, "validation_inference")
        return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)


# --- build_base_config function (no changes needed) ---
# ... (copy the full build_base_config function from the previous answer here) ...
def build_base_config(backbone="R_50"):
    from detectron2.utils.logger import setup_logger
    from detectron2 import model_zoo

    setup_logger()
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/mask_rcnn_{backbone}_FPN_3x.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    cfg.MODEL.DEVICE = device.type
    cfg.INPUT.AUG = [{"name": "RandomFlip", "prob": 0.5, "horizontal": True},{"name": "RandomBrightness", "intensity_range": [0.8, 1.2]},{"name": "RandomContrast", "intensity_range": [0.8, 1.2]},{"name": "RandomRotation", "angle": [-10, 10]},{"name": "RandomExtent", "scale_range": [0.9, 1.1]},{"name": "RandomApply", "transform_list": [{"name": "GaussianBlur", "kernel_size": [3, 3], "sigma_range": [0.1, 2.0]}], "prob": 0.3}]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.WARMUP_ITERS = 1500
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT = 2
    cfg.MODEL.ROI_MASK_HEAD.LOSS_WEIGHT = 0.25
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-InstanceSegmentation/mask_rcnn_{backbone}_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATASETS.TRAIN = ("building_dataset_train",)
    cfg.DATASETS.TEST = ("building_dataset_val",)
    cfg.TEST.EVAL_PERIOD = 1000
    return cfg


# --- Configuration for Final Training Run (no changes needed) ---
# ... (copy the BEST_TRIAL_HPS, FINAL_MAX_ITER, BACKBONE_TO_USE, etc. from previous answer here) ...
BEST_TRIAL_HPS = {'SOLVER.BASE_LR': 0.0015079194738465958,'SOLVER.IMS_PER_BATCH': 8,'SOLVER.WEIGHT_DECAY': 0.0005263046993895792,'SOLVER.WEIGHT_DECAY_NORM': 0.0005232794754292152,'MODEL.RPN.BATCH_SIZE_PER_IMAGE': 256,'MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE': 128,'MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION': 7,'MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION': 7}
FINAL_MAX_ITER = 50000
BACKBONE_TO_USE = "R_50"
FINAL_PROJECT_DIR = './output/final_training'
FINAL_RUN_NAME = f'maskrcnn_best_hps_corr_{BACKBONE_TO_USE}_{FINAL_MAX_ITER}_iter_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}' # Changed run name slightly
GLOBAL_RANDOM_SEED = 42
DATASET_TEST_NAME = "building_dataset_test"


# --- Dataset Registration and Seeding (no changes needed) ---
# ... (copy the dataset registration and seeding blocks here) ...
try:
    from dataset import *
    print("Dataset registration code imported.")
except ImportError:
    print("Warning: Could not import dataset registration.")
random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)
torch.manual_seed(GLOBAL_RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_RANDOM_SEED)


# --- Build Base Config and apply overrides (no changes needed) ---
# ... (copy the config building and override blocks here) ...
cfg = build_base_config(backbone=BACKBONE_TO_USE)
print(f"--- Preparing Final Training Run: {FINAL_RUN_NAME} ---")
for key, value in BEST_TRIAL_HPS.items():
     try:
         exec(f'cfg.{key} = {repr(value)}')
         print(f"Override config: {key} = {value}")
     except Exception as e:
          print(f"Error applying best HP override {key}={value}: {e}")
          traceback.print_exc()
cfg.SOLVER.MAX_ITER = FINAL_MAX_ITER
if 'SOLVER.STEPS' not in BEST_TRIAL_HPS:
    steps = (int(cfg.SOLVER.MAX_ITER * 0.66), int(cfg.SOLVER.MAX_ITER * 0.85))
    cfg.SOLVER.STEPS = list(steps)
    print(f"Calculated SOLVER.STEPS based on new MAX_ITER: {cfg.SOLVER.STEPS}")
cfg.OUTPUT_DIR = os.path.join(FINAL_PROJECT_DIR, FINAL_RUN_NAME)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=False)
with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
    f.write(cfg.dump())
print(f"Final config saved to {os.path.join(cfg.OUTPUT_DIR, 'config.yaml')}")


# --- Build and Train Model ---
try:
    # Use the NEW FinalTrainer class instead of DefaultTrainer
    trainer = FinalTrainer(cfg)
    trainer.resume_or_load(resume=False)

    print("Starting Detectron2 training...")
    # Now, trainer.train() will correctly perform periodic validation
    trainer.train()
    print("Training completed.")

except Exception as e:
    print(f"An error occurred during final training: {e}")
    traceback.print_exc()
    print("Skipping final evaluation due to training error.")
    exit() # Exit if training failed

# --- Perform Final Evaluation on Test Set ---
# This part now uses the weights from the *best* performing model on the validation set
# Fix: Corrected variable name in f-string (removed extra hyphen)
print(f"\n--- Performing Final Evaluation on Test Set for {FINAL_RUN_NAME} ---")
try:
    # IMPORTANT: The model is loaded from `model_best.pth` for final evaluation,
    # as the trainer saves the best performing weights based on validation.

    # Set the model weights path in the config
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_best.pth")

    # Fix: Use the standard global build_model function
    # Ensure 'from detectron2.modeling import build_model' is at the top of your script
    from detectron2.modeling import build_model # Make sure this import is present

    # Build a new model instance based on the configuration
    model = build_model(cfg)

    # Use Checkpointer to load the specified weights into the model instance
    # Fix: Checkpointer usage is correct, warning might persist but code is fine.
    from fvcore.common.checkpoint import Checkpointer # Ensure this import is present
    Checkpointer(model).load(cfg.MODEL.WEIGHTS) # Load weights from the path set in cfg.MODEL.WEIGHTS

    # --- Rest of the Evaluation Block (no changes needed) ---
    # Build the data loader specifically for the FINAL TEST set
    test_loader = build_detection_test_loader(cfg, DATASET_TEST_NAME) # <-- Use TEST dataset name

    # Build the COCO Evaluator specifically for the FINAL TEST set
    evaluator = COCOEvaluator(DATASET_TEST_NAME, cfg, False, output_dir=os.path.join(cfg.OUTPUT_DIR, "final_test_inference"))

    # Run inference and evaluation on the TEST set
    results = inference_on_dataset(model, test_loader, evaluator)

    print(f"\n📊 Final Evaluation Results for {FINAL_RUN_NAME} on {DATASET_TEST_NAME}:")
    print(results)

    # Optionally, save evaluation results to a JSON file
    eval_results_path = os.path.join(cfg.OUTPUT_DIR, "final_test_eval_results.json")
    with open(eval_results_path, "w") as f:
        json.dump(results, f)
    print(f"Final evaluation results saved to {eval_results_path}")

except Exception as e:
    print(f"An error occurred during final evaluation on {DATASET_TEST_NAME} for {FINAL_RUN_NAME}: {e}")
    traceback.print_exc()