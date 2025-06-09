import os
import random
import numpy as np
import torch
import pandas as pd
import traceback
from datetime import datetime

from config import cfg  # Your custom config setup
from dataset import *    # Your dataset registration
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Custom trainer WITHOUT W&B logging
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = "./output"
        return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)

# --- Random Search Setup ---
NUM_TRIALS = 30
GLOBAL_RANDOM_SEED = 42
OUTPUT_BASE = './output/random_search'
VAL_DATASET_NAME = 'building_dataset_val'
OPTIMIZATION_METRIC = 'bbox/AP50'

search_space = {
    'BASE_LR': ('log_uniform', 1e-5, 1e-2),
    'IMS_PER_BATCH': ('categorical', [2, 4, 8]),
    'MAX_ITER': ('categorical', [10000]),
    'WEIGHT_DECAY': ('uniform', 0.0001, 0.001),
    'WEIGHT_DECAY_NORM': ('uniform', 0.0, 0.001),
    'RPN.BATCH_SIZE_PER_IMAGE': ('categorical', [64, 128, 256]),
    'ROI_HEADS.BATCH_SIZE_PER_IMAGE': ('categorical', [64, 128, 256]),
    'ROI_BOX_HEAD.POOLER_RESOLUTION': ('categorical', [7, 14]),
    'ROI_MASK_HEAD.POOLER_RESOLUTION': ('categorical', [7, 14]),
}

results_list = []

# Set seeds for reproducibility
random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)
torch.manual_seed(GLOBAL_RANDOM_SEED)

os.makedirs(OUTPUT_BASE, exist_ok=True)

for i in range(NUM_TRIALS):
    trial_seed = GLOBAL_RANDOM_SEED + i
    trial_random = random.Random(trial_seed)
    trial_np = np.random.default_rng(trial_seed)

    # --- Sample hyperparameters ---
    sampled_hps = {}
    for hp, (hp_type, *hp_values) in search_space.items():
        if hp_type == 'uniform':
            low, high = hp_values
            sampled_hps[hp] = trial_random.uniform(low, high)
        elif hp_type == 'log_uniform':
            low, high = hp_values
            sampled_hps[hp] = float(np.exp(trial_np.uniform(np.log(low), np.log(high))))
        elif hp_type == 'categorical':
            options = hp_values[0]
            sampled_hps[hp] = trial_random.choice(options)
        else:
            raise ValueError(f"Unknown HP type: {hp_type}")

    max_iter = sampled_hps['MAX_ITER']
    steps = (int(max_iter * 0.66), int(max_iter * 0.85))

    run_name = f"trial_{i+1}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_output_dir = os.path.join(OUTPUT_BASE, f"{run_name}_{timestamp}")

    if os.path.exists(trial_output_dir):
        print(f"\n--- Trial {i+1} already exists, skipping. ---")
        continue

    print(f"\n--- Starting Trial {i+1}/{NUM_TRIALS} ---")
    print("Sampled HPs:", sampled_hps)

    # --- Build config ---
    trial_cfg = cfg.clone()  # Important: work on a clone!
    trial_cfg.OUTPUT_DIR = trial_output_dir
    trial_cfg.SOLVER.BASE_LR = sampled_hps['BASE_LR']
    trial_cfg.SOLVER.MAX_ITER = max_iter
    trial_cfg.SOLVER.STEPS = steps
    trial_cfg.SOLVER.IMS_PER_BATCH = sampled_hps['IMS_PER_BATCH']
    trial_cfg.SOLVER.WEIGHT_DECAY = sampled_hps['WEIGHT_DECAY']
    trial_cfg.SOLVER.WEIGHT_DECAY_NORM = sampled_hps['WEIGHT_DECAY_NORM']
    trial_cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = sampled_hps['RPN.BATCH_SIZE_PER_IMAGE']
    trial_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = sampled_hps['ROI_HEADS.BATCH_SIZE_PER_IMAGE']
    trial_cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = sampled_hps['ROI_BOX_HEAD.POOLER_RESOLUTION']
    trial_cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = sampled_hps['ROI_MASK_HEAD.POOLER_RESOLUTION']

    os.makedirs(trial_cfg.OUTPUT_DIR, exist_ok=True)

    try:
        trainer = MyTrainer(trial_cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

        evaluator = COCOEvaluator(VAL_DATASET_NAME, trial_cfg, False, output_dir=trial_cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(trial_cfg, VAL_DATASET_NAME)
        results = inference_on_dataset(trainer.model, val_loader, evaluator)

        metric_value = results['bbox'].get('AP50', None)
        print(f"Validation bbox/AP50: {metric_value}")

        results_list.append({
            'trial': i + 1,
            **sampled_hps,
            'metric': metric_value,
            'output_dir': trial_output_dir,
        })

    except Exception as e:
        print(f"Error in trial {i + 1}: {e}")
        traceback.print_exc()
        results_list.append({
            'trial': i + 1,
            **sampled_hps,
            'metric': None,
            'output_dir': trial_output_dir,
        })

# --- Summarize Results ---
if results_list:
    df = pd.DataFrame(results_list)
    print("\n--- Random Search Results ---")
    print(df)
    df.to_csv(os.path.join(OUTPUT_BASE, 'random_search_results.csv'), index=False)
else:
    print("No successful trials to report.")
