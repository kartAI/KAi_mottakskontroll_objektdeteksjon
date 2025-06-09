import os
import json
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
from collections import defaultdict

# --- Config ---
tif_path = "data/eksport_data/32-2-479-091-31.tif"
gt_json_path = "coco_annotations.json"
pred_json_path = "evaluation_outputs/pred_coco_annotations.json"
output_image_path = "debug_combined_eval.png"
box_iou_threshold = 0.5
mask_iou_threshold = 0.5

# --- Helpers ---
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

def compute_polygon_iou(poly1, poly2):
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    if not poly1.intersects(poly2):
        return 0.0
    inter_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    return inter_area / union_area if union_area > 0 else 0.0

def match_polygons(pred_polys, gt_polys, iou_threshold=0.3):
    pred_shapes = [Polygon(p) for p in pred_polys]
    gt_shapes = [Polygon(p) for p in gt_polys]
    pred_to_gt = defaultdict(list)
    gt_to_pred = defaultdict(list)

    for pred_idx, pred_shape in enumerate(pred_shapes):
        for gt_idx, gt_shape in enumerate(gt_shapes):
            iou = compute_polygon_iou(pred_shape, gt_shape)
            if iou >= iou_threshold:
                pred_to_gt[pred_idx].append(gt_idx)
                gt_to_pred[gt_idx].append(pred_idx)

    tp = []
    merged = []
    split = []
    matched_pred = set()
    matched_gt = set()

    for pred_idx, gts in pred_to_gt.items():
        if len(gts) == 1 and len(gt_to_pred[gts[0]]) == 1:
            tp.append((pred_idx, gts[0]))
            matched_pred.add(pred_idx)
            matched_gt.add(gts[0])
        elif len(gts) > 1:
            merged.append((pred_idx, gts))
            matched_pred.add(pred_idx)
            matched_gt.update(gts)

    for gt_idx, preds in gt_to_pred.items():
        if len(preds) > 1 and gt_idx not in matched_gt:
            split.append((gt_idx, preds))
            matched_gt.add(gt_idx)
            matched_pred.update(preds)

    false_positives = [i for i in range(len(pred_polys)) if i not in matched_pred]
    false_negatives = [i for i in range(len(gt_polys)) if i not in matched_gt]

    return {
        "true_positives": tp,
        "merged": merged,
        "split": split,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

# --- Load image ---
with rasterio.open(tif_path) as src:
    image = src.read([1, 2, 3]).transpose(1, 2, 0)

height, width, _ = image.shape
dpi = 100
figsize = (width / dpi, height / dpi)

# --- Load annotations ---
with open(gt_json_path) as f:
    gt_data = json.load(f)
with open(pred_json_path) as f:
    pred_data = json.load(f)

if isinstance(pred_data, dict) and "annotations" in pred_data:
    pred_annotations = pred_data["annotations"]
else:
    pred_annotations = pred_data

gt_annotations = [
    ann for ann in gt_data["annotations"]
    if ann["image_id"] == 1
    and "bbox" in ann and isinstance(ann["bbox"], list) and len(ann["bbox"]) == 4
    and "segmentation" in ann and ann["segmentation"] and ann["segmentation"][0]
]

pred_annotations = [
    ann for ann in pred_annotations
    if ann.get("image_id") == 1
    and "bbox" in ann and isinstance(ann["bbox"], list) and len(ann["bbox"]) == 4
    and "segmentation" in ann and ann["segmentation"] and ann["segmentation"][0]
]

gt_boxes = [ann["bbox"] for ann in gt_annotations]
gt_polys = [np.array(ann["segmentation"][0]).reshape(-1, 2) for ann in gt_annotations]

pred_boxes = [ann["bbox"] for ann in pred_annotations]
pred_polys = [
    np.array(ann["segmentation"][0]).reshape(-1, 2)
    for ann in pred_annotations
    if "segmentation" in ann and ann["segmentation"] and ann["segmentation"][0]
]

# --- Stage 1: Box-based matching ---
matched_gt = set()
matched_pred = set()
tp_box = []

for pred_idx, pred_box in enumerate(pred_boxes):
    best_iou = 0
    best_gt_idx = -1
    for gt_idx, gt_box in enumerate(gt_boxes):
        iou = compute_iou(pred_box, gt_box)
        if iou > best_iou:
            best_iou = iou
            best_gt_idx = gt_idx
    if best_iou >= box_iou_threshold:
        tp_box.append((pred_idx, best_gt_idx))
        matched_pred.add(pred_idx)
        matched_gt.add(best_gt_idx)

# --- Prepare unmatched for polygon check ---
unmatched_pred_idxs = [i for i in range(len(pred_polys)) if i not in matched_pred]
unmatched_gt_idxs = [i for i in range(len(gt_polys)) if i not in matched_gt]

unmatched_pred_polys = [pred_polys[i] for i in unmatched_pred_idxs]
unmatched_gt_polys = [gt_polys[i] for i in unmatched_gt_idxs]

# --- Stage 2: Polygon-based refinement ---
poly_results = match_polygons(unmatched_pred_polys, unmatched_gt_polys, iou_threshold=mask_iou_threshold)

# Re-index polygons back to global idx space
def remap_indices(local_idx_list, idx_map):
    return [idx_map[i] for i in local_idx_list]

tp_mask = [(unmatched_pred_idxs[p], unmatched_gt_idxs[g]) for (p, g) in poly_results["true_positives"]]
merged = [(unmatched_pred_idxs[p], [unmatched_gt_idxs[g] for g in gs]) for (p, gs) in poly_results["merged"]]
split = [(unmatched_gt_idxs[g], [unmatched_pred_idxs[p] for p in ps]) for (g, ps) in poly_results["split"]]
fp_idxs = remap_indices(poly_results["false_positives"], unmatched_pred_idxs)
fn_idxs = remap_indices(poly_results["false_negatives"], unmatched_gt_idxs)

# --- Final counts ---
tp = len(tp_box) + len(tp_mask) + len(merged) + len(split)
fp = len(pred_polys) - (len(tp_box) + len(tp_mask) + len(merged) + len(split))
fn = len(gt_polys) - (len(tp_box) + len(tp_mask) + len(merged) + len(split))
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

# --- Visualization ---
fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
ax.imshow(image)

def draw_poly(poly, color, style='solid', linewidth=2):
    if not np.array_equal(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    ax.plot(*poly.T, color=color, linewidth=linewidth, linestyle=style)

# True Positives (yellow)
for pred_idx, _ in tp_box + tp_mask:
    draw_poly(pred_polys[pred_idx], 'yellow', 'solid', 2)

# Merged (blue)
for pred_idx, _ in merged:
    draw_poly(pred_polys[pred_idx], 'blue', 'dotted', 2)

# Split (cyan)
for gt_idx, _ in split:
    draw_poly(gt_polys[gt_idx], 'cyan', 'dotted', 2)

# False Positives (red dashed)
for idx in fp_idxs:
    draw_poly(pred_polys[idx], 'red', 'dashed', 2)

# False Negatives (lime solid)
for idx in fn_idxs:
    draw_poly(gt_polys[idx], 'lime', 'solid', 2)

# Legend
legend_patches = [
    patches.Patch(color='yellow', label='True Positives'),
    patches.Patch(color='blue', label='Merged Prediction'),
    patches.Patch(color='cyan', label='Split Ground Truth'),
    patches.Patch(color='red', label='False Positives'),
    patches.Patch(color='lime', label='False Negatives'),
]
ax.legend(handles=legend_patches, loc='lower left', fontsize=10)

ax.set_axis_off()
plt.tight_layout(pad=0)
plt.savefig(output_image_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
plt.close()

# --- Output ---
print("✅ Debug image saved to:", output_image_path)
print("\n--- Detection Stats ---")
print(f"True Positives (TP):     {tp}")
print(f"False Positives (FP):    {fp}")
print(f"False Negatives (FN):    {fn}")
print(f"Precision:               {precision:.4f}")
print(f"Recall:                  {recall:.4f}")
