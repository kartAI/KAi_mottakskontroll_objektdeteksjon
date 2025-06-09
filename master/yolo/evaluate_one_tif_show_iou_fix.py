import os
import json # json is imported but not used, can be removed
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches # patches is imported but not used, can be removed
from PIL import Image
from tqdm import tqdm

# --- Config ---
tile_dir = "data_new/yolo_dataset_filtered/test/images/"  # folder with tiled images
gt_dir = "data_new/yolo_dataset_filtered/test/labels"    # folder with GT .txt files
pred_dir = "yolo_phase1/predictions_s/predict3/labels"  # YOLO predicted labels
output_image = "debug_combined_eval.png"
tile_size = 1024
iou_threshold = 0.5 # Threshold for considering a match for visualization

# --- Helper functions ---
def parse_yolo_polygon_line(line, tile_w, tile_h):
    # Assuming the line format is 'class_id x1 y1 x2 y2 ...'
    parts = line.strip().split()
    if len(parts) < 7: # Need at least class_id and 3 points (6 coords)
        # print(f"Skipping line with insufficient coordinates: {line.strip()}") # Optional debug
        return None
    try:
        # class_id = int(parts[0]) # We don't use class_id for visualization here
        coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0:
            # print(f"Skipping line with odd number of coordinates: {line.strip()}") # Optional debug
            return None
        xys = np.array(coords).reshape(-1, 2)
        # Convert normalized coordinates [0, 1] to pixel coordinates [0, tile_size]
        xys[:, 0] *= tile_w
        xys[:, 1] *= tile_h
        return xys
    except ValueError:
        # print(f"Skipping line with non-float coordinates: {line.strip()}") # Optional debug
        return None

def polygon_to_bbox(polygon):
    # Check if polygon is valid
    if polygon is None or polygon.shape[0] < 3:
        return None
    x_coords, y_coords = polygon[:, 0], polygon[:, 1]
    x, y = np.min(x_coords), np.min(y_coords)
    w, h = np.max(x_coords) - x, np.max(y_coords) - y
    # Ensure width and height are non-negative
    w = max(0, w)
    h = max(0, h)
    return [x, y, w, h]

def compute_iou(box1, box2):
    # Ensure boxes are valid (list/tuple of 4 numbers)
    if not (isinstance(box1, (list, tuple)) and len(box1) == 4 and
            isinstance(box2, (list, tuple)) and len(box2) == 4):
        return 0.0

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Handle potential zero width/height boxes gracefully
    if w1 <= 0 or h1 <= 0 or w2 <= 0 or h2 <= 0:
        return 0.0

    # Convert width/height to xmax/ymax
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2

    # Calculate intersection coordinates
    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1_max, x2_max)
    yb = min(y1_max, y2_max)

    # Calculate intersection area
    inter_area = max(0, xb - xa) * max(0, yb - ya)

    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # Avoid division by zero
    return inter_area / union_area if union_area > 0 else 0.0

# --- Gather tiles ---
# Find tiles matching the pattern, assuming pattern is in the filename
# For broader selection, remove or change the filter `and "32-2-479-091-31" in f`
tiles = [f for f in os.listdir(tile_dir) if f.lower().endswith((".jpg", ".png"))]
# Example: Filter for a specific area or tile index pattern
# Adjust the pattern matching logic based on your filename structure if needed
filtered_tiles = []
for f in tiles:
    # Assuming filename format ends with _{x_offset}_{y_offset}.ext
    # Or similar format where offsets can be parsed.
    # Example: image_name_1024_2048.png -> x_off=1024, y_off=2048
    base = os.path.splitext(f)[0]
    parts = base.split("_")
    try:
        # Attempt to parse the last two parts as integers (offsets)
        if len(parts) >= 2:
            x_off_str, y_off_str = parts[-2], parts[-1]
            x_off = int(x_off_str)
            y_off = int(y_off_str)
            # Optional: Add more specific filtering here if needed, e.g., check range
            # if f"32-2-479-091-31_{x_off_str}_{y_off_str}" in f: # Example specific filter
            filtered_tiles.append((f, x_off, y_off))
        else:
             print(f"[SKIP] {f}: filename does not contain sufficient parts for offset parsing (e.g., 'name_x_y.ext')")
    except ValueError:
        print(f"[SKIP] {f}: could not parse offset integers from '{parts[-2:]}'")

if not filtered_tiles:
    print("No tiles found matching the expected filename pattern and filter.")
    exit() # Exit if no tiles are found to prevent errors

tiles_info = filtered_tiles

# Determine canvas size
# Find the maximum extent reached by any tile
max_x_coord = 0
max_y_coord = 0
if tiles_info:
    max_x_coord = max([x for _, x, _ in tiles_info])
    max_y_coord = max([y for _, _, y in tiles_info])

canvas_w = max_x_coord + tile_size
canvas_h = max_y_coord + tile_size

print(f"Canvas size: {canvas_w}x{canvas_h}")

# --- Create blank canvas ---
# Adjust figsize based on canvas size, maybe scale it down for display
# DPI controls the resolution of the output image
fig, ax = plt.subplots(figsize=(canvas_w / 200, canvas_h / 200), dpi=200) # Reduced scale for screen, high dpi for file
ax.set_xlim(0, canvas_w)
ax.set_ylim(canvas_h, 0) # Inverted y-axis for image coordinates (origin top-left)
ax.set_aspect('equal', adjustable='box') # Keep aspect ratio square
ax.set_axis_off()
plt.subplots_adjust(left=0, right=1, top=1, bottom=0) # Remove padding

# --- Process each tile ---
# Sort tiles by offset to draw them in order (optional, but can be nice)
# tiles_info.sort(key=lambda item: (item[2], item[1])) # Sort by y then x offset

print(f"Processing {len(tiles_info)} tiles...")

for tile_name, x_offset, y_offset in tqdm(tiles_info):
    img_path = os.path.join(tile_dir, tile_name)
    base = os.path.splitext(tile_name)[0]
    gt_path = os.path.join(gt_dir, base + ".txt")
    pred_path = os.path.join(pred_dir, base + ".txt")

    try:
        img = Image.open(img_path).convert("RGB") # Ensure RGB for consistent display
    except FileNotFoundError:
        print(f"Warning: Image not found for tile {tile_name}. Skipping.")
        continue
    except Exception as e:
        print(f"Warning: Could not open image {tile_name}: {e}. Skipping.")
        continue

    # Draw the image onto the canvas
    # extent is (left, right, bottom, top) for matplotlib's origin (bottom-left)
    # But we want top-left origin, so it's (left, right, top, bottom) in our coordinate system
    ax.imshow(img, extent=(x_offset, x_offset + tile_size, y_offset + tile_size, y_offset), origin='upper') # Use origin='upper'

    # Parse GT
    gt_data = [] # Store (polygon, bbox) tuples
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            for line in f:
                poly = parse_yolo_polygon_line(line, tile_size, tile_size)
                if poly is not None:
                    bbox = polygon_to_bbox(poly)
                    if bbox is not None: # Only add if bbox is valid
                        gt_data.append((poly, bbox))
    # Initialize detection status for GTs
    is_gt_detected = [False] * len(gt_data)

    # Parse Predictions
    pred_data = [] # Store (polygon, bbox) tuples
    if os.path.exists(pred_path):
        with open(pred_path) as f:
            for line in f:
                # Add confidence score parsing if needed, but not required for this matching
                # parts = line.strip().split()
                # confidence = float(parts[5]) if len(parts) > 6 else 1.0 # Assuming confidence is after coords
                poly = parse_yolo_polygon_line(line, tile_size, tile_size)
                if poly is not None:
                    bbox = polygon_to_bbox(poly)
                    if bbox is not None: # Only add if bbox is valid
                         pred_data.append((poly, bbox)) # Store (poly, bbox)

    # Initialize correctness status for Predictions
    is_pred_correct = [False] * len(pred_data)

    # --- Perform Relaxed Matching for Visualization ---
    # Check pairwise IoU to determine detection/correctness status
    for gt_idx, (gt_poly, gt_box) in enumerate(gt_data):
        for pred_idx, (pred_poly, pred_box) in enumerate(pred_data):
            iou = compute_iou(gt_box, pred_box)
            if iou >= iou_threshold:
                # This prediction matches this GT sufficiently
                is_gt_detected[gt_idx] = True
                # This GT matches this prediction sufficiently
                is_pred_correct[pred_idx] = True

    # --- Draw Polygons ---

    # Draw False Negatives (Undetected GTs)
    for gt_idx, (gt_poly, gt_box) in enumerate(gt_data):
        if not is_gt_detected[gt_idx]:
            shifted = gt_poly + np.array([x_offset, y_offset])
            # Close the polygon for drawing if it has enough points
            if shifted.shape[0] >= 2:
                 ax.plot(*shifted.T, color='lime', linewidth=1.5, linestyle='-', label='FN (GT)' if 'FN (GT)' not in [l.get_label() for l in ax.get_lines()] else "")
                 # Add a closing segment if necessary
                 if not np.array_equal(shifted[0], shifted[-1]):
                      ax.plot([shifted[-1, 0], shifted[0, 0]], [shifted[-1, 1], shifted[0, 1]], color='lime', linewidth=1.5, linestyle='-')


    # Draw True Positives (Correct Predictions) and False Positives (Incorrect Predictions)
    for pred_idx, (pred_poly, pred_box) in enumerate(pred_data):
        shifted = pred_poly + np.array([x_offset, y_offset])
        # Choose color based on whether prediction was marked as "correct"
        if is_pred_correct[pred_idx]:
            color = 'yellow' # TP
            label = 'TP (Pred)' if 'TP (Pred)' not in [l.get_label() for l in ax.get_lines()] else ""
        else:
            color = 'red' # FP
            label = 'FP (Pred)' if 'FP (Pred)' not in [l.get_label() for l in ax.get_lines()] else ""

        # Close the polygon for drawing if it has enough points
        if shifted.shape[0] >= 2:
            ax.plot(*shifted.T, color=color, linewidth=1.0, linestyle='--', label=label)
            # Add a closing segment if necessary
            if not np.array_equal(shifted[0], shifted[-1]):
                 ax.plot([shifted[-1, 0], shifted[0, 0]], [shifted[-1, 1], shifted[0, 1]], color=color, linewidth=1.0, linestyle='--')


# Add a legend to explain colors
handles, labels = ax.get_legend_handles_labels()
if handles:
    # Filter unique labels and their corresponding handles if labels were repeated
    unique_labels = []
    unique_handles = []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
    ax.legend(unique_handles, unique_labels, loc='lower right')


# --- Save final image ---
# Use bbox_inches='tight' to remove extra whitespace around the plot
# Use pad_inches=0 to remove padding added by tight layout
plt.savefig(output_image, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close(fig) # Close the figure to free memory

print(f"✅ Saved combined visualization to: {output_image}")