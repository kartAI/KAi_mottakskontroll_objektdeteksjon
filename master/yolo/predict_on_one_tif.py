import os
from ultralytics import YOLO

# --- CONFIGURATION ---
model_path = "yolov8_final_training_tuned_hps/best_hp_config/weights/best.pt"
image_path = "data/eksport_data/32-2-479-091-31.tif"
output_dir = "single_tiff_prediction_output"
confidence_threshold = 0.5
iou_threshold = 0.5
img_size = 1024

# --- Load YOLOv8 model ---
model = YOLO(model_path)

# --- Run prediction ---
results = model.predict(
    source=image_path,
    imgsz=img_size,
    conf=confidence_threshold,
    iou=iou_threshold,
    save=True,
    save_txt=True,
    save_conf=True,
    project=output_dir,
    name="predict_on_tiff",
    verbose=True,
)

# --- Report summary ---
print(f"✅ Predictions saved to: {os.path.join(output_dir, 'predict_on_tiff')}")
for result in results:
    print(f"Image: {result.path}")
    print(f"Number of detections: {len(result.boxes)}")
