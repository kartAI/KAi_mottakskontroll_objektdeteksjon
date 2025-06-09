import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# === CONFIG ===
pred_dir = "yolo_phase1/predictions_s/predict3/labels"  # YOLO predictions (.txt)
gt_dir = "data_new/yolo_dataset_filtered/test/labels"   # YOLO ground truth (.txt)
image_dir = "data_new/yolo_dataset_filtered/test/images"
iou_thresh = 0.5
class_id = 0  # building

results = []

def yolo_polygon_to_mask(polygon, width, height):
    """Convert YOLO polygon (normalized) to binary mask."""
    polygon_px = [(x * width, y * height) for x, y in polygon]
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(polygon_px, outline=1, fill=1)
    return np.array(img, dtype=np.uint8)

def load_yolo_masks(txt_path, width, height):
    """Load YOLO polygon annotations into binary masks."""
    masks = []
    if not os.path.exists(txt_path):
        return masks
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7 or int(parts[0]) != class_id:
                continue
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                coords = coords[:-1]
            points = np.array(coords, dtype=np.float32).reshape(-1, 2)
            masks.append(yolo_polygon_to_mask(points, width, height))
    return masks

def compute_mask_tp_fp_fn(pred_masks, gt_masks, iou_thresh=0.5):
    matched_gt = set()
    tp = 0
    for pred in pred_masks:
        best_iou = 0
        best_idx = -1
        for i, gt in enumerate(gt_masks):
            if i in matched_gt:
                continue
            intersection = np.logical_and(pred, gt).sum()
            union = np.logical_or(pred, gt).sum()
            iou = intersection / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_idx)
    fp = len(pred_masks) - tp
    fn = len(gt_masks) - len(matched_gt)
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

    gt_masks = load_yolo_masks(gt_path, width, height)
    pred_masks = load_yolo_masks(pred_path, width, height)

    tp, fp, fn = compute_mask_tp_fp_fn(pred_masks, gt_masks, iou_thresh)

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
# df["f1"] = 2 * df["precision"] * df["recall"] / (df["precision"] + df["recall"] + 1e-6)
df_sorted = df.sort_values(by="FP", ascending=False)

print(df_sorted.head(50))

# # Optionally save
# df_sorted.to_csv("yolo_fp_fn_report.csv", index=False)
