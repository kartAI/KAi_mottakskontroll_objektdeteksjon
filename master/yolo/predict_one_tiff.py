from ultralytics import YOLO

model = YOLO('yolov8_final_training_tuned_hps/best_hp_config/weights/best.pt')

metrics = model.val(
    data='data_custom_tile_subset/data.yaml',
    split='test',       # or 'val'
    imgsz=1024,
    conf=0.5,
    iou=0.5,
    save_json=True,
    verbose=True,
    plots=True,
)
