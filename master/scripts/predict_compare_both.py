import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json

from shapely.geometry import Polygon
from PIL import ImageDraw, Image

# --- Utility Functions ---

def yolo_polygon_to_mask(polygon, width, height):
    """Convert YOLO-style polygon (normalized) to binary mask."""
    polygon_px = [(x * width, y * height) for x, y in polygon]
    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(polygon_px, outline=1, fill=1)
    return np.array(img, dtype=np.uint8)

def load_yolo_polygons(txt_path, width, height):
    """Load YOLO .txt polygon annotations into binary masks."""
    masks = []
    if not os.path.exists(txt_path):
        return masks
    with open(txt_path, "r") as f:
        lines = f.readlines()
    for line_num, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 7:
            print(f"⚠️ Skipping short line in {txt_path}, line {line_num}")
            continue

        class_id, *coords = parts
        coords = list(map(float, coords))

        if len(coords) % 2 != 0:
            print(f"⚠️ Odd number of coords in {txt_path}, line {line_num}. Trimming last value.")
            coords = coords[:-1]

        try:
            points = np.array(coords, dtype=np.float32).reshape(-1, 2)
            masks.append(yolo_polygon_to_mask(points, width, height))
        except Exception as e:
            print(f"❌ Failed to process line {line_num} in {txt_path}: {e}")
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

def compute_tp_fp_fn(gt_anns, dt_anns, iou_thresh=0.5):
    tp = 0
    fp = 0
    matched_gt = set()
    for i, dt in enumerate(dt_anns):
        best_iou = 0
        best_gt_idx = -1
        dt_box = dt["bbox"]
        dt_box = [dt_box[0], dt_box[1], dt_box[0]+dt_box[2], dt_box[1]+dt_box[3]]
        for j, gt in enumerate(gt_anns):
            if j in matched_gt:
                continue
            gt_box = gt["bbox"]
            gt_box = [gt_box[0], gt_box[1], gt_box[0]+gt_box[2], gt_box[1]+gt_box[3]]
            xx1 = max(gt_box[0], dt_box[0])
            yy1 = max(gt_box[1], dt_box[1])
            xx2 = min(gt_box[2], dt_box[2])
            yy2 = min(gt_box[3], dt_box[3])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            union = (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1]) + (dt_box[2]-dt_box[0])*(dt_box[3]-dt_box[1]) - inter
            iou = inter / union if union > 0 else 0
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        if best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    fn = len(gt_anns) - len(matched_gt)
    return tp, fp, fn

def draw_yolo_label(image, label_path):
    h, w = image.shape[:2]
    overlay = image.copy()
    if not os.path.exists(label_path):
        return overlay
    with open(label_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        coords = list(map(float, line.strip().split()[1:]))
        if len(coords) >= 6:
            points = np.array(coords, dtype=np.float32).reshape(-1, 2)
            points *= [w, h]
            cv2.polylines(overlay, [points.astype(np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)
    return overlay

# --- Config ---
coco_img_dir = "data_new/coco_dataset_filtered/test/images"
coco_ann_path = "data_new/coco_dataset_filtered/test/coco_annotations.json"
yolo_label_dir = "data_new/yolo_dataset_filtered/test/labels"
yolo_pred_dir = "yolo_phase1/predictions_s/predict3"
yolo_counts_csv = "yolo_phase1/predictions_s/prediction_counts.csv"
output_dir = "output/comparisons_4panel"
os.makedirs(output_dir, exist_ok=True)

# --- Load YOLO prediction counts ---
df_counts = pd.read_csv(yolo_counts_csv)
yolo_counts = dict(zip(df_counts["image_name"], df_counts["num_predictions"]))

# --- Register COCO test set ---
dataset_name = "building_dataset_test"
def register_building_instances(name, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(thing_classes=["building"])
register_building_instances(dataset_name, coco_ann_path, coco_img_dir)

# --- Setup Detectron2 predictor ---
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "output/phase3_step5_box2_mask025_R_50_20250501_222218/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda"
predictor = DefaultPredictor(cfg)

# --- Load COCO dicts ---
dataset_dicts = DatasetCatalog.get(dataset_name)

# --- Run visualization ---
print(f"🔹 Processing {len(dataset_dicts)} images...")
for d in dataset_dicts:
    file_name = os.path.basename(d["file_name"])
    name_stem = os.path.splitext(file_name)[0]

    image = cv2.imread(d["file_name"])
    if image is None:
        print(f"❌ Failed to load image {d['file_name']}")
        continue
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # --- YOLO paths
    yolo_label_path = os.path.join(yolo_label_dir, f"{name_stem}.txt")
    yolo_pred_txt = os.path.join(yolo_pred_dir, "labels", f"{name_stem}.txt")


    # --- Load YOLO masks
    yolo_label_masks = load_yolo_polygons(yolo_label_path, w, h)
    print(f"🔍 Loading YOLO predictions from: {yolo_pred_txt}")
    yolo_pred_masks = load_yolo_polygons(yolo_pred_txt, w, h)

    # --- Debug
    print(f"{file_name} — Pred: {len(yolo_pred_masks)}, GT: {len(yolo_label_masks)}")
    if len(yolo_pred_masks) == 0:
        print("⚠️ No YOLO prediction polygons loaded.")
    if len(yolo_label_masks) == 0:
        print("⚠️ No YOLO label polygons loaded.")

    # --- YOLO visualization
    yolo_pred_path = os.path.join(yolo_pred_dir, file_name)
    yolo_pred_image = cv2.imread(yolo_pred_path)
    if yolo_pred_image is None:
        yolo_pred_image = np.ones_like(image) * 128
        cv2.putText(yolo_pred_image, "Missing YOLO Prediction", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    yolo_label_image = draw_yolo_label(image.copy(), yolo_label_path)
    yolo_label_image = cv2.cvtColor(yolo_label_image, cv2.COLOR_BGR2RGB)

    # --- Mask R-CNN prediction
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    v_pred = Visualizer(image_rgb, MetadataCatalog.get(dataset_name), scale=1.0)
    v_pred = v_pred.draw_instance_predictions(instances)
    maskrcnn_pred_image = v_pred.get_image()

    # --- COCO label visualization
    v_gt = Visualizer(image_rgb, MetadataCatalog.get(dataset_name), scale=1.0)
    v_gt = v_gt.draw_dataset_dict(d)
    coco_gt_image = v_gt.get_image()

    # --- Metrics
    gt_anns = d["annotations"]
    pred_boxes = instances.pred_boxes.tensor.numpy()
    dt_anns = [{"bbox": [x1, y1, x2 - x1, y2 - y1]} for x1, y1, x2, y2 in pred_boxes]

    tp_rcnn, fp_rcnn, _ = compute_tp_fp_fn(gt_anns, dt_anns)
    tp_yolo, fp_yolo, _ = compute_mask_tp_fp_fn(yolo_pred_masks, yolo_label_masks)

    # --- Plot 2x2 panel
    fig, axs = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
    axs[0, 0].imshow(yolo_pred_image[:, :, ::-1])
    axs[0, 0].set_title(f"YOLO Prediction (TP:{tp_yolo}, FP:{fp_yolo})", fontsize=12)
    axs[0, 1].imshow(yolo_label_image)
    axs[0, 1].set_title(f"YOLO Label ({len(yolo_label_masks)} buildings)", fontsize=12)
    axs[1, 0].imshow(maskrcnn_pred_image)
    axs[1, 0].set_title(f"Mask R-CNN Prediction (TP:{tp_rcnn}, FP:{fp_rcnn})", fontsize=12)
    axs[1, 1].imshow(coco_gt_image)
    axs[1, 1].set_title(f"COCO Label ({len(gt_anns)} buildings)", fontsize=12)
    for ax in axs.flat:
        ax.axis("off")

    output_path = os.path.join(output_dir, f"{name_stem}_4panel.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {output_path}")

print("✅ Done with all 4-panel visualizations.")
