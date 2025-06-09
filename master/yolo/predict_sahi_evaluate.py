import os
import json
import rasterio
import geopandas as gpd
from shapely.geometry import box
from sahi.predict import get_sliced_prediction
from sahi.auto_model import AutoDetectionModel
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ---------------- CONFIG ---------------- #
model_path = "yolov8_final_training_tuned_hps/best_hp_config/weights/best.pt"
image_path = "data/eksport_data/32-2-479-091-31.tif"
gpkg_path = "data/Bygning.gpkg"

output_dir = "evaluation_outputs_SAHI_best"
os.makedirs(output_dir, exist_ok=True)

gt_coco_file = os.path.join(output_dir, "gt_coco_annotations.json")
pred_coco_file = os.path.join(output_dir, "pred_coco_annotations.json")

category_id = 1
category_name = "building"
image_id = 1
device = "cuda:0"  # or "cpu"

# ---------------- PREDICTION ---------------- #
print("Loading model...")
model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path=model_path,
    confidence_threshold=0.5,
    device=device,
)

print("Running sliced prediction with SAHI...")
result = get_sliced_prediction(
    image=image_path,
    detection_model=model,
    slice_height=1024,
    slice_width=1024,
    overlap_height_ratio=0.1,
    overlap_width_ratio=0.1,
    perform_standard_pred=False,
)
print("Prediction complete.")

# Convert to flat COCO-style prediction list (bbox-only)
# sahi_preds = result.to_coco_predictions(image_id=image_id)
# pred_list = []
# for idx, pred in enumerate(sahi_preds):
#     if "bbox" not in pred or len(pred["bbox"]) != 4:
#         continue
#     pred_list.append({
#         "id": idx + 1,
#         "image_id": image_id,
#         "category_id": 1,
#         "bbox": pred["bbox"],
#         "score": pred.get("score", 0.5)
#     })

# print(f"Saved {len(pred_list)} bbox predictions.")

# with open(pred_coco_file, "w") as f:
#     json.dump(pred_list, f, indent=4)

sahi_anns = result.to_coco_annotations()

# Add category_id if missing (SAHI sometimes omits it)
for ann in sahi_anns:
    ann["image_id"] = image_id
    ann["category_id"] = category_id
    ann["iscrowd"] = 0  # optional, but required by COCOeval

with open(pred_coco_file, "w") as f:
    json.dump(sahi_anns, f, indent=4)

print(f"Saved {len(sahi_anns)} predictions with masks.")

# ---------------- GT GENERATION ---------------- #
def generate_gt_annotations(image_path, gpkg_path, image_id, category_id):
    annotations = []
    gdf = gpd.read_file(gpkg_path)
    with rasterio.open(image_path) as src:
        img_width, img_height = src.width, src.height
        transform = src.transform
        if gdf.crs != src.crs:
            print(f"Reprojecting GPKG from {gdf.crs} to {src.crs}")
            gdf = gdf.to_crs(src.crs)

        bounds = box(*src.bounds)
        gdf = gdf[gdf.intersects(bounds)]
        gdf["geometry"] = gdf["geometry"].apply(lambda g: g.buffer(0) if not g.is_valid else g)

        ann_id = 1
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom.is_empty:
                continue
            polygons = [geom] if geom.geom_type == "Polygon" else geom.geoms
            for poly in polygons:
                if not poly.is_valid or poly.is_empty or not poly.exterior:
                    continue

                segmentation = []
                for x, y in list(poly.exterior.coords):
                    px, py = ~transform * (x, y)
                    px = max(0, min(px, img_width - 1))
                    py = max(0, min(py, img_height - 1))
                    segmentation.extend([float(px), float(py)])

                minx, miny, maxx, maxy = poly.bounds
                x0, y0 = ~transform * (minx, maxy)
                x1, y1 = ~transform * (maxx, miny)
                x_min, y_min = min(x0, x1), min(y0, y1)
                w, h = abs(x1 - x0), abs(y1 - y0)

                if w <= 1 or h <= 1:
                    continue

                annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],
                    "area": w * h,
                    "bbox": [x_min, y_min, w, h],
                    "iscrowd": 0
                })
                ann_id += 1

        image_info = {
            "id": image_id,
            "file_name": os.path.basename(image_path),
            "width": img_width,
            "height": img_height
        }

    return annotations, image_info

print("Generating GT annotations...")
gt_annotations, image_info = generate_gt_annotations(image_path, gpkg_path, image_id, category_id)

gt_data = {
    "info": {"description": "GT", "version": "1.0"},
    "licenses": [],
    "categories": [{"id": category_id, "name": category_name}],
    "images": [image_info],
    "annotations": gt_annotations
}

with open(gt_coco_file, "w") as f:
    json.dump(gt_data, f, indent=4)

print(f"Saved {len(gt_annotations)} GT annotations.")

# # ---------------- EVALUATION ---------------- #
# print("Starting COCO evaluation...")
# try:
#     coco_gt = COCO(gt_coco_file)
#     coco_dt = coco_gt.loadRes(pred_coco_file)

#     coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
#     coco_eval.params.imgIds = [image_id]
#     coco_eval.params.catIds = [category_id]

#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()

#     mean_ap = coco_eval.stats[0]
#     ap_05 = coco_eval.stats[1]
#     average_recall = coco_eval.stats[8]
#     ar_05 = coco_eval.stats[9]

#     print("\nCalculated Metrics:")
#     print(f"mAP (IoU=0.50:0.95): {mean_ap:.4f}")
#     print(f"mAP@0.5 (IoU=0.50):  {ap_05:.4f}")
#     print(f"Average Recall (IoU=0.50:0.95): {average_recall:.4f}")
#     print(f"Average Recall@0.5 (IoU=0.50): {ar_05:.4f}")
#     # print(pred_list[:3])
#     print(gt_annotations[:3])

# except Exception as e:
#     print(f"\nAn error occurred during evaluation: {e}")
