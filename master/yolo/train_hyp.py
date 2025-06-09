import random
import numpy as np
import torch
from ultralytics import YOLO
import yaml
import os
import pandas as pd
import traceback # Import traceback for detailed error logging

# --- Configuration ---
DATA_YAML_PATH = 'data_new/yolo_dataset_filtered/data.yaml' # Path to your YOLO dataset config
BASE_MODEL = 'yolov8s-seg.pt'            # Start from pre-trained weights
NUM_TRIALS = 50                          # Number of random configurations to test
EPOCHS_PER_TRIAL = 30                    # Epochs for each tuning trial
IMG_SIZE = 1024
PROJECT_NAME = 'yolov8_random_search_seeded_1' # Base directory for results (USE THE SAME NAME AS BEFORE)
GLOBAL_RANDOM_SEED = 42                  # Your chosen seed value (USE THE SAME SEED AS BEFORE)

# Define the search space (make sure this is the same as your interrupted run)
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

# Metric to optimize (e.g., validation mAP50 for box)
OPTIMIZATION_METRIC = 'metrics/mAP50(B)' # Check Ultralytics results.csv for exact name

# List to store results (will only collect results from trials run in THIS session)
results_list = []

# --- Set Global Seeds for Reproducibility ---
# This ensures the *sequence* of trials (and the specific HP combination sampled in each trial)
# is the same if you re-run the script from scratch with the same GLOBAL_RANDOM_SEED.
# This is crucial for generating the *same* sequence of HPs for the remaining trials.
print(f"Setting global random seed for sampling to {GLOBAL_RANDOM_SEED}")
random.seed(GLOBAL_RANDOM_SEED)
np.random.seed(GLOBAL_RANDOM_SEED)

# Setting torch seed globally for reproducibility of the training process itself
# (though this seed is also passed explicitly to model.train)
torch.manual_seed(GLOBAL_RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(GLOBAL_RANDOM_SEED)
    # Optional: for stricter CUDNN reproducibility (can slow down)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


print(f"Starting Random Search (resume mode) for {NUM_TRIALS} trials...")
print(f"Checking for existing trial directories in 'runs/{PROJECT_NAME}'...")

# Ensure the base project directory exists
base_project_dir = os.path.join('runs', PROJECT_NAME)
os.makedirs(base_project_dir, exist_ok=True) # Use exist_ok=True here as we are resuming

for i in range(NUM_TRIALS):
    # --- Set Trial-Specific Seed for Hyperparameter Sampling ---
    # This ensures that the sampling process for THIS trial is unique based on its index,
    # and importantly, will produce the SAME sequence of HPs as the original run
    # due to the fixed GLOBAL_RANDOM_SEED and trial index 'i'.
    trial_sampling_seed = GLOBAL_RANDOM_SEED + i
    # Create trial-specific RNGs using this seed
    trial_random = random.Random(trial_sampling_seed)
    trial_np_random = np.random.Generator(np.random.PCG64(trial_sampling_seed))

    # --- 1. Sample Hyperparameters ---
    # Sample HPs using the trial-specific RNGs. We MUST sample HPs for *every* trial (0 to NUM_TRIALS-1)
    # in the loop, even if we skip training, to ensure the random state advances correctly
    # for subsequent trials.
    sampled_hps = {}
    for hp, (hp_type, *hp_values) in search_space.items():
        if hp_type == 'uniform':
            low, high = hp_values
            sampled_hps[hp] = trial_random.uniform(low, high) # Use trial_random
        elif hp_type == 'log_uniform':
            low, high = hp_values
            sampled_hps[hp] = float(np.exp(trial_np_random.uniform(np.log(low), np.log(high)))) # Use trial_np_random
        elif hp_type == 'categorical':
            options = hp_values[0]
            sampled_hps[hp] = trial_random.choice(options) # Use trial_random
        else:
            raise ValueError(f"Unknown HP type: {hp_type}")

    # --- 2. Determine Run Name and Output Directory ---
    run_name_parts = [f'trial_{i+1}'] # Use i+1 for directory name
    for k, v in sampled_hps.items():
        if isinstance(v, float):
            run_name_parts.append(f"{k}_{v:.4f}".replace('.', 'p'))
        else:
            run_name_parts.append(f"{k}_{v}")
    run_name = '_'.join(run_name_parts).replace('-', 'neg')

    trial_output_dir = os.path.join(base_project_dir, run_name)

    # --- CHECK IF TRIAL ALREADY EXISTS ---
    if os.path.exists(trial_output_dir):
        print(f"\n--- Trial {i+1}/{NUM_TRIALS} already exists ({run_name}), skipping training. ---")
        # Optionally, you could try to load the metric from the existing run's results.csv
        # and append it to results_list if needed, but the recovery script is better for full analysis later.
        # For simplicity here, we just skip training and move on.
        continue # Skip the rest of the loop for this trial

    print(f"\n--- Starting Trial {i+1}/{NUM_TRIALS} ({run_name}) ---")
    print("Sampled HPs:", sampled_hps)
    print(f"Using trial sampling seed: {trial_sampling_seed}") # Print here when actually starting trial

    # --- 3. Prepare Training Arguments ---
    train_args = {
        'data': DATA_YAML_PATH,
        'epochs': EPOCHS_PER_TRIAL,
        'imgsz': IMG_SIZE,
        'project': PROJECT_NAME,
        'name': run_name,
        'exist_ok': False, # Ensure a new directory is created (shouldn't exist if we got here)
        'verbose': False,  # Reduce console output during training
        'patience': sampled_hps['patience'],
        'seed': GLOBAL_RANDOM_SEED, # Pass the *global* seed for training reproducibility
        **{k: v for k, v in sampled_hps.items() if k != 'patience'},
    }

    if isinstance(train_args['batch'], float):
         train_args['batch'] = int(train_args['batch'])

    print("Training args:", train_args)

    # --- 4. Run Training ---
    try:
        # os.makedirs(trial_output_dir, exist_ok=False) # Ultralytics train handles directory creation with exist_ok

        model = YOLO(BASE_MODEL)
        results = model.train(**train_args)
        print("Training completed.")

        # --- 5. Extract Performance Metric ---
        validation_metric_value = None
        if hasattr(results, 'results_dict') and results.results_dict:
             best_metrics = results.results_dict
             validation_metric_value = best_metrics.get(OPTIMIZATION_METRIC, None)

        if validation_metric_value is None: # Fallback to reading csv
            results_csv_path = os.path.join(trial_output_dir, 'results.csv')
            if os.path.exists(results_csv_path):
                try:
                    df_results = pd.read_csv(results_csv_path)
                    df_results.columns = df_results.columns.str.strip()
                    if not df_results.empty and OPTIMIZATION_METRIC in df_results.columns:
                         validation_metric_value = df_results[OPTIMIZATION_METRIC].iloc[-1]
                         print(f"Extracted metric from results.csv: {validation_metric_value:.4f}")
                    else:
                         print(f"Warning: Metric column '{OPTIMIZATION_METRIC}' not found in results.csv for trial {i+1}.")
                except Exception as csv_e:
                    print(f"Error reading results.csv for trial {i+1}: {csv_e}")


        if validation_metric_value is not None:
             print(f"Final Validation Metric ({OPTIMIZATION_METRIC}): {validation_metric_value:.4f}")
             results_list.append({
                 'trial': i + 1,
                 **sampled_hps,
                 'metric': validation_metric_value,
                 'run_dir': os.path.join(PROJECT_NAME, run_name)
             })
        else:
            print(f"Warning: Could not determine validation metric for trial {i+1}.")
            results_list.append({
                 'trial': i + 1,
                 **sampled_hps,
                 'metric': None, # Indicate failure or missing metric
                 'run_dir': os.path.join(PROJECT_NAME, run_name)
             })


    except Exception as e:
        print(f"Error during trial {i+1}: {e}")
        traceback.print_exc() # Print full traceback for debugging
        results_list.append({
            'trial': i + 1,
            **sampled_hps,
            'metric': None, # Indicate failure
            'run_dir': os.path.join(PROJECT_NAME, run_name)
        })


# --- 6. Analyze Results (only for trials run in this session) ---
print("\n--- Random Search Session Complete ---")
# The results_list here only contains results from trials that ran in *this* script execution.
# To get the overall best across ALL trials (including previous runs),
# you should use the recovery script method described previously AFTER this script finishes.
if results_list:
    session_results_df = pd.DataFrame(results_list)
    print("\nResults from this session:")
    print(session_results_df)
    # Note: Finding the 'best' here only considers trials from THIS session.
    # Use the recovery script to find the best across ALL trials.
else:
    print("\nNo new trials were completed in this session.")


# The best_run logic and saving to PROJECT_NAME_random_search_results.csv
# from the *original* script should ideally operate on *all* trials.
# Running the recovery script *after* this resuming script finishes is the recommended
# way to get the complete set of results and find the overall best.

# Example of how you would run the recovery script after this finishes:
# python your_recovery_script_name.py