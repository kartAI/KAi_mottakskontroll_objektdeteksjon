import json
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import pandas as pd

# === CONFIG ===
ground_truth_path = "data_new/coco_dataset_filtered/test/coco_annotations.json"
predictions_path = "output/evaluation_results/coco_instances_results.json"
iou_thresh = 0.5
cat_id = 1  # building

# === LOAD DATA ===
coco_gt = COCO(ground_truth_path)
coco_dt = coco_gt.loadRes(predictions_path)

img_ids = coco_gt.getImgIds()
results = []

for img_id in img_ids:
    gt_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[cat_id])
    dt_ids = coco_dt.getAnnIds(imgIds=[img_id], catIds=[cat_id])
    gt_anns = coco_gt.loadAnns(gt_ids)
    dt_anns = coco_dt.loadAnns(dt_ids)

    gt_masks = [coco_gt.annToRLE(ann) for ann in gt_anns]
    dt_masks = [coco_dt.annToRLE(ann) for ann in dt_anns]

    matched_gt = set()
    matched_dt = set()

    for i, dt_mask in enumerate(dt_masks):
        iscrowd_flags = [gt.get("iscrowd", 0) for gt in gt_anns]
        ious = maskUtils.iou([dt_mask], gt_masks, iscrowd_flags)[0]
        best_idx = -1
        best_iou = 0

        for j, iou in enumerate(ious):
            if j in matched_gt:
                continue
            if iou >= iou_thresh and iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_idx != -1:
            matched_dt.add(i)
            matched_gt.add(best_idx)

    tp = len(matched_gt)
    fp = len(dt_masks) - len(matched_dt)
    fn = len(gt_masks) - len(matched_gt)

    img_info = coco_gt.loadImgs([img_id])[0]
    file_name = img_info.get("file_name", f"id_{img_id}")

    results.append({
        "image_id": img_id,
        "file_name": file_name,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    })

# === OUTPUT ===
df = pd.DataFrame(results)
df_sorted = df.sort_values(by="FP", ascending=False)

print(df_sorted.head(50))
# df_sorted.to_csv("mask_rcnn_tp_fp_fn_segmentation.csv", index=False)
