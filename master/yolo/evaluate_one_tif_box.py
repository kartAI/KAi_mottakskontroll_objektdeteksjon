import os
import glob

def load_yolo_boxes(filepath, img_width, img_height):
    boxes = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 5:
                print(f"[SKIP] {os.path.basename(filepath)} line {i+1}: invalid bbox line")
                continue
            try:
                class_id, cx, cy, w, h = map(float, parts[:5])
                x_min = (cx - w / 2) * img_width
                y_min = (cy - h / 2) * img_height
                x_max = (cx + w / 2) * img_width
                y_max = (cy + h / 2) * img_height
                boxes.append([x_min, y_min, x_max, y_max])
            except Exception as e:
                print(f"[SKIP] {os.path.basename(filepath)} line {i+1}: parse error {e}")
                continue
    return boxes

def compute_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    xi1 = max(x1_min, x2_min)
    yi1 = max(y1_min, y2_min)
    xi2 = min(x1_max, x2_max)
    yi2 = min(y1_max, y2_max)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = max(0, x1_max - x1_min) * max(0, y1_max - y1_min)
    box2_area = max(0, x2_max - x2_min) * max(0, y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0

def evaluate_for_prefix(prefix, gt_dir, pred_dir, img_shape, iou_thresh=0.5):
    stats = {"TP": 0, "FP": 0, "FN": 0}
    gt_files = sorted(glob.glob(os.path.join(gt_dir, f"{prefix}*.txt")))
    print(f"Found {len(gt_files)} ground truth files for prefix '{prefix}'")

    for gt_file in gt_files:
        filename = os.path.basename(gt_file)
        pred_file = os.path.join(pred_dir, filename)

        gt_boxes = load_yolo_boxes(gt_file, *img_shape)
        pred_boxes = load_yolo_boxes(pred_file, *img_shape) if os.path.exists(pred_file) else []

        print(f"{filename}: {len(gt_boxes)} GT boxes")
        print(f"{filename}: {len(pred_boxes)} predicted boxes")

        matched_gt = set()
        matched_pred = set()

        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                if compute_iou(pred_box, gt_box) >= iou_thresh:
                    matched_gt.add(j)
                    matched_pred.add(i)
                    break

        TP = len(matched_pred)
        FP = len(pred_boxes) - TP
        FN = len(gt_boxes) - TP

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
    pred_dir = "yolo_phase1/predictions_final/predict_final/labels"

    results = evaluate_for_prefix(prefix, gt_dir, pred_dir, img_shape)
    print(results)
