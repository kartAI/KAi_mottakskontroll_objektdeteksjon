import torch
from ultralytics import YOLO
import os
import random
import numpy as np
import traceback

# --- Configuration ---
DATA_YAML_PATH = 'data_new/yolo_dataset_filtered/data.yaml' # Path to your YOLO dataset config
BASE_MODEL = 'yolov8s-seg.pt'            # Start from pre-trained weights

# Define the exact search space that was used in your random search script
# We need this structure to know *which* parameters were tuned.
search_space = {
    'lr0': ('log_uniform', 1e-5, 1e-2),
    'lrf': ('uniform', 0.01, 0.2),
    'optimizer': ('categorical', ['SGD', 'AdamW']),
    'weight_decay': ('uniform', 0.0, 0.001),
    'batch': ('categorical', [4, 8, 16]),
    'patience': ('categorical', [20, 30]),
    'degrees': ('uniform', 0.0, 10.0),
    'scale': ('uniform', 0.5, 1.5),
    'mosaic': ('uniform', 0.0, 1.0),
    'mixup': ('uniform', 0.0, 0.2),
}

# --- BEST HYPERPARAMETERS FOUND FROM RANDOM SEARCH ---
# IMPORTANT: Manually input the actual best values found from your analysis
# of the random search results (e.g., from the best_run variable or the CSV).
# These are the optimal values for the HPs *in your search_space*.
BEST_HPS_FROM_SEARCH = {
    'lr0': 0.0005797509593958689,      # Example value, replace with your actual best lr0
    'lrf': 0.052595240220992025,       # Example value, replace with your actual best lrf
    'optimizer': 'AdamW',              # Example value, replace with your actual best optimizer
    'weight_decay': 2.2758574308230805e-05, # Example value, replace with your actual best weight_decay
    'batch': 8,                        # Example value, replace with your actual best batch
    'patience': 20,                    # Example value, replace with your actual best patience
    'degrees': 6.798960761851487,      # Example value, replace with your actual best degrees
    'scale': 0.5105745488894674,         # Example value, replace with your actual best scale
    'mosaic': 0.8283118716249926,      # Example value, replace with your actual best mosaic
    'mixup': 0.07565241379284726,       # Example value, replace with your actual best mixup
    # Add other HPs from your search_space here with their best values
    # ...
}
# Ensure batch is an integer
BEST_HPS_FROM_SEARCH['batch'] = int(BEST_HPS_FROM_SEARCH['batch'])


FINAL_EPOCHS = 300                                       # <-- Set your desired higher number of epochs here
IMG_SIZE = 1024                                          # Use the same image size as tuning
FINAL_PROJECT_NAME = 'yolov8_final_training_tuned_hps' # New project name
FINAL_RUN_NAME = 'best_hp_config'                        # A descriptive name
FINAL_RANDOM_SEED = 42                                   # Use the same seed

# --- Set Seeds for Reproducibility of the Final Training Run ---
# Seeding Python, NumPy, and Torch for the final run
print(f"Setting global random seed for final training to {FINAL_RANDOM_SEED}")
random.seed(FINAL_RANDOM_SEED)
np.random.seed(FINAL_RANDOM_SEED)
torch.manual_seed(FINAL_RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(FINAL_RANDOM_SEED)
    # Optional: for stricter CUDNN reproducibility (can slow down)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


print("--- Preparing Final Training Run with Best Tuned Hyperparameters ---")

# --- Prepare Training Arguments ---
# Start with essential arguments, then add the tuned HPs.
# Any arguments *not* explicitly listed here will use Ultralytics/YOLOv8 defaults.
final_train_args = {
    'task': 'segment', # Essential argument
    'mode': 'train',   # Essential argument
    'data': DATA_YAML_PATH,
    'epochs': FINAL_EPOCHS,       # <-- Your desired higher number of epochs
    'imgsz': IMG_SIZE,
    'project': FINAL_PROJECT_NAME,
    'name': FINAL_RUN_NAME,
    'exist_ok': False, # Set to True if you might re-run and want to resume/overwrite
    'verbose': True,   # You might want verbose output for the final run
    'seed': FINAL_RANDOM_SEED, # Pass the seed to Ultralytics training
    'val': True, # Keep validation enabled
    'save': True, # Save weights
    'plots': True, # Save training plots
    'save_period': -1, # Save only best. Set to >0 to save every N epochs
    # ... include any other essential, non-tuned parameters you need fixed ...

    # --- Override with Best HPs from the search_space ---
    # This unpacks the dictionary containing ONLY the HPs you tuned
    **BEST_HPS_FROM_SEARCH,

}

print("Final Training Arguments:")
# Print args nicely
for key, value in final_train_args.items():
    print(f"  {key}: {value}")

# --- Load Model and Start Training ---
try:
    model = YOLO(BASE_MODEL) # Load the base model (e.g., yolov8s-seg.pt)

    print("\nStarting final training...")
    results = model.train(**final_train_args)
    print("\nFinal training completed.")

    # You can optionally print/log final metrics from the results object here
    if hasattr(results, 'results_dict') and results.results_dict:
        print("\nFinal Training Metrics:")
        # Print relevant validation metrics (e.g., box mAP50, mask mAP50)
        # Use the same metric names you checked for optimization and others of interest
        metric_names_to_print = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                                'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']
        for metric_name in metric_names_to_print:
             if metric_name in results.results_dict:
                 print(f"  {metric_name}: {results.results_dict[metric_name]:.4f}")


except Exception as e:
    print(f"An error occurred during final training: {e}")
    traceback.print_exc()