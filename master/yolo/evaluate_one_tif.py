import os
import glob
import cv2
from shapely.geometry import Polygon
from shapely.errors import TopologicalError

def load_yolo_seg(filepath, img_width, img_height):
    polygons = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                print(f"[SKIP] {os.path.basename(filepath)} line {i+1}: invalid polygon")
                continue
            try:
                coords = list(map(float, parts[1:]))  # skip class_id
                abs_coords = [(coords[i] * img_width, coords[i+1] * img_height) for i in range(0, len(coords), 2)]
                poly = Polygon(abs_coords)
                if not poly.is_valid or poly.area < 1.0:
                    print(f"[SKIP] {os.path.basename(filepath)} line {i+1}: still invalid or too small")
                    continue
                polygons.append(poly)
            except Exception as e:
                print(f"[SKIP] {os.path.basename(filepath)} line {i+1}: parse error {e}")
                continue
    return polygons

def compute_iou(poly1, poly2):
    try:
        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return inter / union if union > 0 else 0
    except TopologicalError:
        return 0

def evaluate_for_prefix(prefix, gt_dir, pred_dir, img_shape, iou_thresh=0.5):
    stats = {"TP": 0, "FP": 0, "FN": 0}
    gt_files = sorted(glob.glob(os.path.join(gt_dir, f"{prefix}*.txt")))
    print(f"Found {len(gt_files)} ground truth files for prefix '{prefix}'")

    for gt_file in gt_files:
        filename = os.path.basename(gt_file)
        pred_file = os.path.join(pred_dir, filename)

        gt_polys = load_yolo_seg(gt_file, *img_shape)
        pred_polys = load_yolo_seg(pred_file, *img_shape) if os.path.exists(pred_file) else []

        print(f"{filename}: {len(gt_polys)} GT polygons")
        print(f"{filename}: {len(pred_polys)} predicted polygons")

        matched_gt = set()
        matched_pred = set()

        for i, pred_poly in enumerate(pred_polys):
            for j, gt_poly in enumerate(gt_polys):
                if j in matched_gt:
                    continue
                if compute_iou(pred_poly, gt_poly) >= iou_thresh:
                    matched_gt.add(j)
                    matched_pred.add(i)
                    break

        TP = len(matched_pred)
        FP = len(pred_polys) - TP
        FN = len(gt_polys) - TP

        stats["TP"] += TP
        stats["FP"] += FP
        stats["FN"] += FN

    precision = stats["TP"] / (stats["TP"] + stats["FP"]) if (stats["TP"] + stats["FP"]) > 0 else 0
    recall = stats["TP"] / (stats["TP"] + stats["FN"]) if (stats["TP"] + stats["FN"]) > 0 else 0

    stats["Precision"] = round(precision, 4)
    stats["Recall"] = round(recall, 4)
    return stats

if __name__ == "__main__":
    prefix = "32-2-479-091-31"
    img_shape = (1024, 1024)  # width, height of your tiles

    gt_dir = "data_new/yolo_dataset_filtered/test/labels"
    pred_dir = "yolo_phase1/predictions_s/predict3/labels"

    results = evaluate_for_prefix(prefix, gt_dir, pred_dir, img_shape)
    print(results)
