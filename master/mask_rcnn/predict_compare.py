import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import load_coco_json

def register_building_instances(name, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        thing_classes=["building"],
        thing_dataset_id_to_contiguous_id={1: 0},
    )
    return name

if __name__ == '__main__':
    cfg = get_cfg()

    # Define dataset names and paths
    dataset_name_test = "building_dataset_test"
    test_json_path = "data_new/coco_dataset_filtered/test/coco_annotations.json"
    test_image_root = "data_new/coco_dataset_filtered/test/images"

    # Register dataset
    register_building_instances(dataset_name_test, test_json_path, test_image_root)
    print(MetadataCatalog.get(dataset_name_test))

    # Load configuration
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "output/phase3_step5_box2_mask025_R_50_20250501_222218/model_final.pth"  # <<< CHANGE to correct path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda"

    predictor = DefaultPredictor(cfg)

    # Create output directories
    output_dir = "./output/comparisons_phase3_step5"
    os.makedirs(output_dir, exist_ok=True)

    # Get dataset dictionary
    dataset_dicts = DatasetCatalog.get(dataset_name_test)
    
    print(f"🔹 Processing {len(dataset_dicts)} images...")

    for i, d in enumerate(dataset_dicts):
        image_path = d["file_name"]
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Warning: Could not load image {image_path}")
            continue
        
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Count ground truth buildings
        num_gt_buildings = len(d["annotations"]) if "annotations" in d else 0

        # Visualize ground truth
        v_gt = Visualizer(image_rgb, MetadataCatalog.get(dataset_name_test), scale=1.2)
        v_gt = v_gt.draw_dataset_dict(d)
        gt_image = v_gt.get_image()

        # Run inference and visualize predictions
        outputs = predictor(image)
        num_pred_buildings = len(outputs["instances"])

        v_pred = Visualizer(image_rgb, MetadataCatalog.get(dataset_name_test), scale=1.2)
        v_pred = v_pred.draw_instance_predictions(outputs["instances"].to("cpu"))
        prediction_image = v_pred.get_image()

        # Plot side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

        axes[0].imshow(gt_image)
        axes[0].set_title(f"Ground Truth (Buildings: {num_gt_buildings})", fontsize=14)
        axes[0].axis("off")

        axes[1].imshow(prediction_image)
        axes[1].set_title(f"Predictions (Buildings: {num_pred_buildings})", fontsize=14)
        axes[1].axis("off")

        # Save figure
        original_filename = os.path.basename(image_path).split(".")[0]
        comparison_path = os.path.join(output_dir, f"comparison_{original_filename}.png")
        plt.savefig(comparison_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"✅ Saved: {comparison_path} (GT: {num_gt_buildings}, Pred: {num_pred_buildings})")

    print("✅ Finished processing all comparisons!")
