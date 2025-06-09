import os
import numpy as np
import pandas as pd
from PIL import Image

# === CONFIG ===
pred_dir = "yolo_phase1/predictions_final/predict_final/labels"  # YOLO predictions (.txt)
gt_dir = "data_new/yolo_dataset_filtered/test/labels"   # YOLO ground truth (.txt)
image_dir = "data_new/yolo_dataset_filtered/test/images"
iou_thresh = 0.5
class_id = 0  # building

results = []

def load_yolo_boxes(txt_path, width, height):
    """Load YOLO bbox annotations into [x1, y1, x2, y2] pixel format."""
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5 or int(parts[0]) != class_id:
                continue
            x_center, y_center, w, h = map(float, parts[1:5])
            x1 = (x_center - w / 2) * width
            y1 = (y_center - h / 2) * height
            x2 = (x_center + w / 2) * width
            y2 = (y_center + h / 2) * height
            boxes.append([x1, y1, x2, y2])
    return boxes

def compute_box_tp_fp_fn(pred_boxes, gt_boxes, iou_thresh=0.5):
    matched_gt = set()
    tp = 0
    for pred in pred_boxes:
        best_iou = 0
        best_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            xx1 = max(pred[0], gt[0])
            yy1 = max(pred[1], gt[1])
            xx2 = min(pred[2], gt[2])
            yy2 = min(pred[3], gt[3])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            area_pred = (pred[2] - pred[0]) * (pred[3] - pred[1])
            area_gt = (gt[2] - gt[0]) * (gt[3] - gt[1])
            union = area_pred + area_gt - inter
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_idx)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn

# === MAIN LOOP ===
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]

for fname in image_files:
    name_stem = os.path.splitext(fname)[0]
    image_path = os.path.join(image_dir, fname)

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"⚠️ Skipping {fname}: {e}")
        continue

    gt_path = os.path.join(gt_dir, f"{name_stem}.txt")
    pred_path = os.path.join(pred_dir, f"{name_stem}.txt")

    gt_boxes = load_yolo_boxes(gt_path, width, height)
    pred_boxes = load_yolo_boxes(pred_path, width, height)

    tp, fp, fn = compute_box_tp_fp_fn(pred_boxes, gt_boxes, iou_thresh)

    results.append({
        "file_name": fname,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    })

# === RESULTS ===
df = pd.DataFrame(results)
df["precision"] = df["TP"] / (df["TP"] + df["FP"] + 1e-6)
df["recall"] = df["TP"] / (df["TP"] + df["FN"] + 1e-6)
df_sorted = df.sort_values(by="FP", ascending=False)

print(df_sorted.head(50))

# Optionally save:
# df_sorted.to_csv("yolo_bbox_fp_fn_report.csv", index=False)
