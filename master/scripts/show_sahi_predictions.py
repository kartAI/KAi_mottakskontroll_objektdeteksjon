import os
import json
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Config ---
tif_path = "data/eksport_data/32-2-479-091-31.tif"
gt_json_path = "coco_annotations.json"
pred_json_path = "evaluation_outputs_SAHI_best/pred_coco_annotations.json"
output_image_path = "debug_bbox_eval_final.png"
iou_threshold = 0.5

# --- Helper: IoU computation ---
def compute_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    union_area = w1 * h1 + w2 * h2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

# --- Load image ---
with rasterio.open(tif_path) as src:
    image = src.read([1, 2, 3]).transpose(1, 2, 0)

# Get image dimensions
height, width, _ = image.shape
dpi = 100
figsize = (width / dpi, height / dpi)

# --- Load annotations ---
with open(gt_json_path) as f:
    gt_data = json.load(f)
with open(pred_json_path) as f:
    preds = json.load(f)

# Handle both flat list or COCO-style annotation dict
if isinstance(preds, dict) and "annotations" in preds:
    preds = preds["annotations"]

pred_boxes = [
    p for p in preds
    if p.get("image_id") == 1 and "bbox" in p and isinstance(p["bbox"], list) and len(p["bbox"]) == 4
]
gt_boxes = [
    ann["bbox"] for ann in gt_data["annotations"]
    if ann["image_id"] == 1 and isinstance(ann["bbox"], list) and len(ann["bbox"]) == 4
]

# --- Matching ---
matched_gt = set()
matched_pred = set()
matches = []

for pred_idx, pred in enumerate(pred_boxes):
    pred_bbox = pred["bbox"]
    best_iou = 0
    best_gt_idx = -1

    for gt_idx, gt_bbox in enumerate(gt_boxes):
        iou = compute_iou(pred_bbox, gt_bbox)
        if iou > best_iou:
            best_iou = iou
            best_gt_idx = gt_idx

    if best_iou >= iou_threshold:
        matches.append((pred_idx, best_gt_idx, best_iou))
        matched_pred.add(pred_idx)
        matched_gt.add(best_gt_idx)

# --- Count TP, FP, FN ---
tp = len(matches)
fp = len(pred_boxes) - tp
fn = len(gt_boxes) - tp

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

# --- Visualization ---
fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
ax.imshow(image)

# True Positives (Yellow)
for pred_idx, gt_idx, iou in matches:
    x, y, w, h = pred_boxes[pred_idx]["bbox"]
    rect = patches.Rectangle((x, y), w, h, linewidth=6, edgecolor='yellow', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y - 4, f"{iou:.2f}", color='yellow', fontsize=10)

# False Positives (Red)
for i, pred in enumerate(pred_boxes):
    if i not in matched_pred:
        x, y, w, h = pred["bbox"]
        rect = patches.Rectangle((x, y), w, h, linewidth=10, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

# False Negatives (Green)
for i, gt_bbox in enumerate(gt_boxes):
    if i not in matched_gt:
        x, y, w, h = gt_bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=10, edgecolor='lime', facecolor='none', linestyle='solid')
        ax.add_patch(rect)

# Legend
legend_patches = [
    patches.Patch(color='yellow', label='True Positives'),
    patches.Patch(color='red', label='False Positives'),
    patches.Patch(color='lime', label='False Negatives'),
]
ax.legend(handles=legend_patches, loc='lower left', fontsize=100)

ax.set_axis_off()
plt.tight_layout(pad=0)
plt.savefig(output_image_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
plt.close()

# --- Output Metrics ---
print("✅ Debug image saved to:", output_image_path)
print("\n--- Detection Stats ---")
print(f"True Positives (TP):     {tp}")
print(f"False Positives (FP):    {fp}")
print(f"False Negatives (FN):    {fn}")
print(f"Precision:               {precision:.4f}")
print(f"Recall:                  {recall:.4f}")
