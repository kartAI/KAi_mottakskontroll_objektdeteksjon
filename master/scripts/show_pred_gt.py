import os
import json
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon

# --- Config ---
tif_path = "data/eksport_data/32-2-479-091-31.tif"
gt_json_path = "coco_annotations.json"
pred_json_path = "evaluation_outputs_SAHI_best/pred_coco_annotations.json"

gt_output_image = "ground_truth_polygons.png"
pred_output_image = "prediction_polygons.png"

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
    preds = json.load(f)

if isinstance(preds, dict) and "annotations" in preds:
    preds = preds["annotations"]

gt_polygons = [
    ann["segmentation"][0] for ann in gt_data["annotations"]
    if ann["image_id"] == 1 and "segmentation" in ann and len(ann["segmentation"]) > 0
]
pred_polygons = [
    ann["segmentation"][0] for ann in preds
    if ann.get("image_id") == 1 and "segmentation" in ann and len(ann["segmentation"]) > 0
]

# --- Draw Ground Truth Polygons ---
fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
ax.imshow(image)

for seg in gt_polygons:
    coords = np.array(seg).reshape(-1, 2)
    polygon = MplPolygon(coords, edgecolor='yellow', facecolor='none', linewidth=8)
    ax.add_patch(polygon)

ax.set_title("Ground Truth Polygons")
ax.set_axis_off()
plt.savefig(gt_output_image, dpi=dpi, bbox_inches='tight', pad_inches=0)
plt.close()
print("✅ Saved GT polygons image to:", gt_output_image)

# --- Draw Prediction Polygons ---
fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
ax.imshow(image)

for seg in pred_polygons:
    coords = np.array(seg).reshape(-1, 2)
    polygon = MplPolygon(coords, edgecolor='red', facecolor='none', linewidth=8)
    ax.add_patch(polygon)

ax.set_title("Predicted Polygons")
ax.set_axis_off()
plt.savefig(pred_output_image, dpi=dpi, bbox_inches='tight', pad_inches=0)
plt.close()
print("✅ Saved prediction polygons image to:", pred_output_image)
