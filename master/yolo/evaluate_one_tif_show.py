import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

# --- Config ---
tile_dir = "data_new/yolo_dataset_filtered/test/images/"  # folder with tiled images
gt_dir = "data_new/yolo_dataset_filtered/test/labels"    # folder with GT .txt files
pred_dir = "yolo_phase1/predictions_s/predict3/labels"  # YOLO predicted labels
output_image = "debug_combined_eval.png"
tile_size = 1024
iou_threshold = 0.5

# --- Helper functions ---

def parse_yolo_polygon_line(line, tile_w, tile_h):
    coords = list(map(float, line.strip().split()[1:]))
    if len(coords) % 2 != 0 or len(coords) < 6:
        return None
    xys = np.array(coords).reshape(-1, 2)
    xys[:, 0] *= tile_w
    xys[:, 1] *= tile_h
    return xys

def polygon_to_bbox(polygon):
    x_coords, y_coords = polygon[:, 0], polygon[:, 1]
    x, y = np.min(x_coords), np.min(y_coords)
    w, h = np.max(x_coords) - x, np.max(y_coords) - y
    return [x, y, w, h]

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

# --- Gather tiles ---
tiles = [f for f in os.listdir(tile_dir) if f.lower().endswith((".jpg", ".png")) and "32-2-479-091-31" in f]
tiles_info = []

for tile_name in tiles:
    base = os.path.splitext(tile_name)[0]
    parts = base.split("_")
    try:
        if len(parts) >= 4:
            x_off, y_off = int(parts[-2]), int(parts[-1])
            tiles_info.append((tile_name, x_off, y_off))
        else:
            print(f"[SKIP] {tile_name}: not enough parts")
    except ValueError:
        print(f"[SKIP] {tile_name}: invalid coordinate parts '{parts[-2:]}'")


max_x = max([x for _, x, _ in tiles_info])
max_y = max([y for _, _, y in tiles_info])
canvas_w = max_x + tile_size
canvas_h = max_y + tile_size

# --- Create blank canvas ---
fig, ax = plt.subplots(figsize=(canvas_w // 100, canvas_h // 100), dpi=100)
ax.set_xlim(0, canvas_w)
ax.set_ylim(canvas_h, 0)
ax.set_axis_off()

# --- Process each tile ---
for tile_name, x_offset, y_offset in tqdm(tiles_info):
    img_path = os.path.join(tile_dir, tile_name)
    base = os.path.splitext(tile_name)[0]
    gt_path = os.path.join(gt_dir, base + ".txt")
    pred_path = os.path.join(pred_dir, base + ".txt")

    img = Image.open(img_path)
    ax.imshow(img, extent=(x_offset, x_offset + tile_size, y_offset + tile_size, y_offset))

    # Parse GT
    gt_polys = []
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            for line in f:
                poly = parse_yolo_polygon_line(line, tile_size, tile_size)
                if poly is not None:
                    gt_polys.append((poly, polygon_to_bbox(poly)))

    # Parse Predictions
    pred_polys = []
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            for line in f:
                poly = parse_yolo_polygon_line(line, tile_size, tile_size)
                if poly is not None:
                    pred_polys.append((poly, polygon_to_bbox(poly)))

    # Matching
    matched_gt = set()
    matched_pred = set()
    for pred_idx, (pred_poly, pred_box) in enumerate(pred_polys):
        best_iou, best_gt_idx = 0.0, -1
        for gt_idx, (gt_poly, gt_box) in enumerate(gt_polys):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_iou >= iou_threshold:
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)
            # Draw TP (yellow)
            shifted = pred_poly + np.array([x_offset, y_offset])
            ax.plot(*shifted.T, color='yellow', linewidth=1.5)

    # Draw FP (red)
    for i, (poly, _) in enumerate(pred_polys):
        if i not in matched_pred:
            shifted = poly + np.array([x_offset, y_offset])
            ax.plot(*shifted.T, color='red', linewidth=1.0, linestyle='--')

    # Draw FN (lime)
    for i, (poly, _) in enumerate(gt_polys):
        if i not in matched_gt:
            shifted = poly + np.array([x_offset, y_offset])
            ax.plot(*shifted.T, color='lime', linewidth=1.0)

# --- Save final image ---
plt.tight_layout()
plt.savefig(output_image, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved combined visualization to: {output_image}")
