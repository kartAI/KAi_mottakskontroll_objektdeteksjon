from ultralytics import YOLO
import os
import csv

# Paths
model_path = "yolo_phase1/yolov8s_baseline/weights/best.pt"
output_dir = "yolo_phase1/predictions_s"
source_dir = "data_new/yolo_dataset_filtered/test/images"
count_log_path = os.path.join(output_dir, "prediction_counts.csv")

# Create output dir
os.makedirs(output_dir, exist_ok=True)

# Load model
model = YOLO(model_path)

# Run prediction
results = model.predict(
    source=source_dir,
    save=True,
    save_txt=True,
    save_conf=False,
    project=output_dir,
    name="predict3",
    imgsz=1024,
    conf=0.5,
    iou=0.5,
    stream=False,
    show=False,
    retina_masks=True,
    verbose=True
)

# Save counts
with open(count_log_path, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "num_predictions"])
    for result in results:
        image_name = os.path.basename(result.path)
        num_predictions = len(result.masks) if result.masks is not None else 0
        writer.writerow([image_name, num_predictions])

print(f"✅ Predictions saved in: {output_dir}")
print(f"📄 Count log saved at: {count_log_path}")
