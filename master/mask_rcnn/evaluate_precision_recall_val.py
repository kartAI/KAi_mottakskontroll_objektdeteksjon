import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# CHANGE THESE:
ground_truth_path = "data_new/coco_dataset_filtered/val/coco_annotations.json"
predictions_path = "output/phase2_box15_mask05__R_50_20250428_125647/coco_instances_results.json"  # default from COCOEvaluator

# Load COCO ground truth and detections
coco_gt = COCO(ground_truth_path)
coco_dt = coco_gt.loadRes(predictions_path)

# Init COCOeval
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.evaluate()
coco_eval.accumulate()

# Get image and annotation IDs
img_ids = coco_gt.getImgIds()
iou_thresh = 0.5
cat_id = 1  # 'building'

tp = 0
fp = 0
fn = 0

for img_id in img_ids:
    gt_ids = coco_gt.getAnnIds(imgIds=[img_id], catIds=[cat_id])
    dt_ids = coco_dt.getAnnIds(imgIds=[img_id], catIds=[cat_id])
    gt_anns = coco_gt.loadAnns(gt_ids)
    dt_anns = coco_dt.loadAnns(dt_ids)

    matched_gt = set()
    matched_dt = set()

    for i, dt in enumerate(dt_anns):
        best_iou = 0
        best_gt_idx = -1
        for j, gt in enumerate(gt_anns):
            gt_box = gt["bbox"]
            dt_box = dt["bbox"]

            # Convert [x,y,w,h] to [x1,y1,x2,y2]
            gt_box = [gt_box[0], gt_box[1], gt_box[0]+gt_box[2], gt_box[1]+gt_box[3]]
            dt_box = [dt_box[0], dt_box[1], dt_box[0]+dt_box[2], dt_box[1]+dt_box[3]]

            # Compute IoU
            xx1 = max(gt_box[0], dt_box[0])
            yy1 = max(gt_box[1], dt_box[1])
            xx2 = min(gt_box[2], dt_box[2])
            yy2 = min(gt_box[3], dt_box[3])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            union = (gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1]) + (dt_box[2]-dt_box[0])*(dt_box[3]-dt_box[1]) - inter
            iou = inter / union if union > 0 else 0

            if iou >= iou_thresh and j not in matched_gt and iou > best_iou:
                best_iou = iou
                best_gt_idx = j

        if best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
            matched_dt.add(i)
        else:
            fp += 1

    fn += len(gt_anns) - len(matched_gt)

# Final metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"✅ Precision: {precision:.3f}")
print(f"✅ Recall:    {recall:.3f}")
print(f"TP: {tp}, FP: {fp}, FN: {fn}")
