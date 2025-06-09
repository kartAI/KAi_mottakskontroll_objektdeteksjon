import torch
import cv2
import os
import numpy as np
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
    # cfg.DATALOADER.NUM_WORKERS = 0  # Avoid multiprocessing issues on macOS

    # Define dataset names and paths
    dataset_name_test = "building_dataset_test"
    test_json_path = "data_new/coco_dataset_filtered/test/coco_annotations.json"
    test_image_root = "data_new/coco_dataset_filtered/test/images"


    # Register dataset
    register_building_instances(dataset_name_test, test_json_path, test_image_root)
    print(MetadataCatalog.get(dataset_name_test))

    # Load configuration
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = "output/phase2_box15_mask05__R_50_20250428_125647/model_final.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda"


    predictor = DefaultPredictor(cfg)

    # Register dataset again if needed
    dataset_name = "building_dataset_test"
    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, dataset_name)

    # Run evaluation
    metrics = inference_on_dataset(predictor.model, val_loader, evaluator)
    print(metrics)

    # 🔹 Create output directory for visualized images
    output_dir = "./output/predictions_box15mask05"
    os.makedirs(output_dir, exist_ok=True)

    # 🔹 Get dataset dictionary to fetch image paths
    dataset_dicts = DatasetCatalog.get(dataset_name_test)
    
    print(f"🔹 Processing {len(dataset_dicts)} images...")

    for i, d in enumerate(dataset_dicts):  # Process all images
        image_path = d["file_name"]
        image = cv2.imread(image_path)  # Load the image
        if image is None:
            print(f"❌ Warning: Could not load image {image_path}")
            continue
        
        # Run inference
        outputs = predictor(image)
        
        # Visualize predictions
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(dataset_name_test), scale=1.2)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # Convert image back to OpenCV format
        result_image = v.get_image()[:, :, ::-1]

        # Save the image
        original_filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, original_filename)
        cv2.imwrite(output_path, result_image)
        print(f"✅ Saved: {output_path}")

    print("✅ Finished processing all predictions!")
