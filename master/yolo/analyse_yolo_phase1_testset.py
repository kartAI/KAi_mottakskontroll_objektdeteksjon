from ultralytics import YOLO
import os

# --- Configuration ---
model_path = "yolov8_final_training_tuned_hps/best_hp_config/weights/best.pt"
data_yaml = "data_new/yolo_dataset_filtered/data.yaml"  # Must include a 'test:' entry
img_size = 1024
conf_threshold = 0.001 # Use low threshold for mAP evaluation
iou_threshold = 0.5

# --- Load the trained model ---
model = YOLO(model_path)

# --- Run evaluation on the test set ---
metrics = model.val(
    data=data_yaml,
    split="test",             # Evaluates on the test split, not validation
    imgsz=img_size,
    conf=conf_threshold,
    iou=iou_threshold,
    save_json=True,           # Saves COCO results JSON
    verbose=True,
    plots= True,
)

# --- Print out core metrics ---
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP75: {metrics.box.map75:.4f}")
print(f"mAP@[.5:.95]: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")

print(f"Mask mAP50: {metrics.seg.map50:.4f}")
print(f"Mask mAP75: {metrics.seg.map75:.4f}")
print(f"Mask mAP@[.5:.95]: {metrics.seg.map:.4f}")
print(f"Mask Precision: {metrics.seg.mp:.4f}")
print(f"Mask Recall: {metrics.seg.mr:.4f}")
