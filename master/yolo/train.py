from ultralytics import YOLO

model = YOLO('yolov8s-seg.pt')

model.train(
    data='data_new/yolo_dataset/data.yaml',
    epochs=100,
    imgsz=1024,
    batch=8,
    lr0=0.001,
    optimizer='SGD',
    patience=20,
    project='yolo_phase1',
    name='yolov8s_baseline',
    val=True
)
