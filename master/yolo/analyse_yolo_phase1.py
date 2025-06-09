import os
import pandas as pd

# Path to your YOLOv8 training results
BASE_DIR = "yolo_phase1"

# Metrics to extract
METRICS = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)"]

results = []

# Loop through each subfolder (each model run)
for run_name in sorted(os.listdir(BASE_DIR)):
    run_path = os.path.join(BASE_DIR, run_name, "results.csv")
    if not os.path.isfile(run_path):
        continue

    try:
        df = pd.read_csv(run_path)
        last = df.iloc[-1]  # last epoch

        results.append({
            "model": run_name,
            "precision(B)": last.get("metrics/precision(B)", None),
            "recall(B)": last.get("metrics/recall(B)", None),
            "mAP50(B)": last.get("metrics/mAP50(B)", None),
        })

    except Exception as e:
        print(f"⚠️ Failed to read {run_name}: {e}")

# Convert to DataFrame and sort by recall(B) (you can change this)
df_results = pd.DataFrame(results)
df_results = df_results.sort_values("recall(B)", ascending=False)

# Print results
print(f"{'Model':<25} {'Precision(B)':>12} {'Recall(B)':>10} {'mAP50(B)':>10}")
print("=" * 60)
for _, row in df_results.iterrows():
    print(f"{row['model']:<25} {row['precision(B)']:.3f}     {row['recall(B)']:.3f}     {row['mAP50(B)']:.3f}")
