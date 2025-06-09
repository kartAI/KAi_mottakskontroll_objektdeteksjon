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
from detectron2.structures import Instances
from detectron2.utils.logger import setup_logger # To control logging level

# setup_logger() # Uncomment to see more detectron2 logs
# setup_logger(name="detectron2.data.datasets.coco", level=20) # Set level=20 (INFO) for specific module

def register_building_instances(name, json_file, image_root, original_building_id):
    """
    Registers the dataset and sets metadata, ensuring the original_id to contiguous_id
    mapping is available for load_coco_json.
    """
    # Set metadata FIRST. This makes the mapping available immediately.
    MetadataCatalog.get(name).set(
        thing_classes=["building"], # Your class name(s)
        # Mapping from original COCO ID from your JSON (original_building_id)
        # to the 0-indexed contiguous ID used by the model (0).
        thing_dataset_id_to_contiguous_id={original_building_id: 0},
        evaluator_type="coco" # Indicate that it's a COCO-formatted dataset
    )
    print(f"Metadata for '{name}' set with mapping {{original_id {original_building_id}: contiguous_id 0}}.")

    # Then register the data loading function.
    # Crucially, pass dataset_name=name to load_coco_json so it can retrieve the metadata
    # we just set, allowing it to apply the id mapping during loading.
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, dataset_name=name))

    print(f"Dataset '{name}' registered.")
    # Return the name for convenience
    return name

if __name__ == '__main__':
    # --- Configuration ---
    # Define dataset names and paths - Using paths from your last script
    dataset_name_test = "building_dataset_test"
    test_json_path = "data_new/coco_dataset_filtered/test/coco_annotations.json"
    test_image_root = "data_new/coco_dataset_filtered/test/images"

    # --- IMPORTANT: Confirm this ID matches your JSON categories section ---
    # Your JSON says {"id": 1, "name": "building"}, so this should be 1.
    ORIGINAL_BUILDING_ID_IN_JSON = 1 # <--- Correct based on your feedback

    # Output directory for evaluation results and visualizations - Using paths from your last script
    OUTPUT_BASE_DIR = "./output"
    EVALUATION_OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, "evaluation_results") # Keeping evaluation separate
    PREDICTION_VIS_DIR = os.path.join(OUTPUT_BASE_DIR, "predictions_phase3_step5") # Your specified visualization directory

    # Ensure output directories exist
    os.makedirs(EVALUATION_OUTPUT_DIR, exist_ok=True)
    os.makedirs(PREDICTION_VIS_DIR, exist_ok=True)


    # --- Register Dataset ---
    # Pass the actual original ID from the JSON during registration
    if dataset_name_test not in DatasetCatalog.list():
         register_building_instances(dataset_name_test, test_json_path, test_image_root, ORIGINAL_BUILDING_ID_IN_JSON)
    else:
         print(f"Dataset '{dataset_name_test}' already registered. Ensuring metadata is correct.")
         # Even if registered, re-set metadata to be absolutely sure it's correct and linked
         MetadataCatalog.get(dataset_name_test).set(
             thing_classes=["building"],
             thing_dataset_id_to_contiguous_id={ORIGINAL_BUILDING_ID_IN_JSON: 0},
             evaluator_type="coco"
         )
         print(f"Confirmed metadata for '{dataset_name_test}' is set with {{original_id {ORIGINAL_BUILDING_ID_IN_JSON}: contiguous_id 0}}.")

    # Print final metadata to confirm
    print("Final Metadata:", MetadataCatalog.get(dataset_name_test))


    # --- Debugging Dataset Contents (Check Category IDs) ---
    # Add this section to inspect the category_id in the loaded data
    print("\n--- Debugging Dataset Contents ---")
    # Getting the dataset dicts here forces the lambda function (and thus load_coco_json) to run
    # This is where we check if the mapping is applied correctly during load
    try:
        dataset_dicts_debug = DatasetCatalog.get(dataset_name_test)
        print(f"Checking category_ids for annotations in '{dataset_name_test}':")
        processed_images_count = 0
        annotation_check_limit = 10 # Limit total annotations checked across images
        annotations_checked = 0

        for i, d in enumerate(dataset_dicts_debug): # Iterate through images
            if annotations_checked >= annotation_check_limit: break # Stop after checking enough annotations

            # Ensure the dictionary has the expected keys and format
            if "annotations" not in d:
                # print(f"Warning: Image {d.get('image_id', i+1)} ({os.path.basename(d['file_name'])}) has no 'annotations' key.")
                continue # Skip images without annotations key

            if len(d["annotations"]) == 0:
                # print(f"Image {d.get('image_id', i+1)} ({os.path.basename(d['file_name'])}) has 0 annotations.")
                continue # Skip images with no annotations

            print(f"Image {d.get('image_id', i+1)}, file: {os.path.basename(d['file_name'])}")
            # Check the category_id exactly as stored in the dictionary
            for j, anno in enumerate(d["annotations"]):
                 if annotations_checked >= annotation_check_limit: break # Stop after checking enough annotations globally
                 anno_category_id = anno.get('category_id')
                 print(f"  - Annotation {j+1}: category_id: {anno_category_id}, bbox_mode: {anno.get('bbox_mode')}")

                 # Sanity check: Should be 0 if mapping worked
                 if anno_category_id != 0:
                     print(f"  !!! Unexpected category_id: {anno_category_id}. Expected 0 based on mapping {{original_id {ORIGINAL_BUILDING_ID_IN_JSON}: contiguous_id 0}}.")

                 annotations_checked += 1

            processed_images_count += 1

        print(f"--- Debugging Complete ({annotations_checked} annotations checked in {processed_images_count} images) ---\n")
    except Exception as e:
        print(f"\n!!! Error during dataset debugging: {e}")
        print("This might indicate an issue with loading or parsing the dataset.")
        # Continue script execution to see if the evaluator still throws the error


    # --- Load Configuration and Model ---
    cfg = get_cfg()

    # It's best to merge from the *exact* config used for training this model if you saved it.
    # cfg.merge_from_file("output/phase3_step5_box2_mask025_R_50_20250501_222218/config.yaml")
    # If not using a saved training config, load from model_zoo and set overridden parameters:
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    # Set number of classes (important!) - must match training
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # Set model weights - Using weights path from your last script
    cfg.MODEL.WEIGHTS = "output/phase3_step5_box2_mask025_R_50_20250501_222218/model_final.pth"

    # Set device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {cfg.MODEL.DEVICE}")


    # --- Run Evaluation to get Metrics (P, R, mAP50) ---

    # IMPORTANT: Set the SCORE_THRESH_TEST LOW for proper evaluation.
    # The COCOEvaluator needs to see detections across a range of scores.
    # Setting it too high here filters out detections before evaluation.
    # The evaluator internally uses different thresholds, but the predictor
    # needs to produce low-scoring detections for them to be considered.
    # A value like 0.0 or 0.05 is typical for full evaluation.
    # Ensure this threshold is applied before creating the predictor used for evaluation.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Use a low threshold for evaluation


    # Create predictor with the *low* threshold for evaluation
    # Re-creating the predictor ensures it uses the updated cfg
    predictor_eval = DefaultPredictor(cfg)

    # Setup evaluator for COCO metrics
    # Pass the dataset name. The evaluator will get metadata (like the mapping) from the dataset name.
    # Simplified COCOEvaluator call is less error-prone with metadata.
    evaluator = COCOEvaluator(dataset_name_test, output_dir=EVALUATION_OUTPUT_DIR)

    # Build test data loader
    test_loader = build_detection_test_loader(cfg, dataset_name_test)

    # Run evaluation on the dataset
    print("\n--- Running Evaluation ---")
    # inference_on_dataset takes the model (from predictor_eval), data loader, and evaluator
    metrics = inference_on_dataset(predictor_eval.model, test_loader, evaluator)

    # --- Print Evaluation Results ---
    print("\n--- Evaluation Results (COCO Metrics) ---")
    print(metrics)

    # Extract and print specific metrics you are interested in
    print("\nSpecific Metrics for Boxes:")
    # These keys are standard outputs from COCOEvaluator
    if 'bbox_AP50' in metrics:
        print(f"mAP@0.5 (bbox_AP50): {metrics['bbox_AP50']:.4f}")
    else:
        print("bbox_AP50 not found in results (check evaluator setup or if evaluation completed).")

    if 'bbox_AP' in metrics:
         print(f"mAP@[0.5:0.95] (bbox_AP): {metrics['bbox_AP']:.4f}")

    # AR@100 (Average Recall at max 100 detections) is the closest standard
    # metric to "Recall" reported by COCOEvaluator.
    # A single Precision/Recall number depends on a specific confidence threshold,
    # which COCO mAP/AR integrates over.
    if 'bbox_AR@100' in metrics:
         print(f"Average Recall (AR@100 bbox): {metrics['bbox_AR@100']:.4f}")
         print("(Note: A single 'Precision' and 'Recall' value is threshold-dependent.)")


    # --- Visualization (Optional - uses a potentially higher confidence threshold for clarity) ---
    print(f"\n--- Running Visualization ---")

    # Set a higher threshold *just for visualization* if desired.
    # Detections below this threshold won't be shown.
    VISUALIZATION_THRESHOLD = 0.5 # Set your desired threshold for visualization

    # Create a separate config and predictor for visualization
    cfg_vis = get_cfg()
    # It's best to merge from the *exact* config used for training if you saved it.
    # cfg_vis.merge_from_file("output/phase3_step5_box2_mask025_R_50_20250501_222218/config.yaml")
    # Otherwise, use the model_zoo base and set overridden parameters:
    cfg_vis.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg_vis.MODEL.WEIGHTS = cfg.MODEL.WEIGHTS # Use the same trained weights
    cfg_vis.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg_vis.MODEL.DEVICE = cfg.MODEL.DEVICE
    cfg_vis.MODEL.ROI_HEADS.SCORE_THRESH_TEST = VISUALIZATION_THRESHOLD # Apply visualization threshold

    predictor_vis = DefaultPredictor(cfg_vis)

    # Get dataset dictionary again to iterate through images for visualization
    # This should be fast as data is likely cached after registration
    dataset_dicts = DatasetCatalog.get(dataset_name_test)
    metadata = MetadataCatalog.get(dataset_name_test) # Get metadata for visualizer

    print(f"Processing {len(dataset_dicts)} images for visualization (threshold > {VISUALIZATION_THRESHOLD})...")

    for i, d in enumerate(dataset_dicts):
        image_path = d["file_name"]
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Warning: Could not load image {image_path}")
            continue

        # Run inference using the predictor configured for visualization threshold
        outputs = predictor_vis(image)

        # The predictor_vis already applied the threshold, so no need to filter here unless you want a different threshold
        instances_to_visualize = outputs["instances"].to("cpu")

        v = Visualizer(image[:, :, ::-1], metadata, scale=1.2) # Use the metadata
        v = v.draw_instance_predictions(instances_to_visualize)

        # Convert image back to OpenCV format (BGR)
        result_image = v.get_image()[:, :, ::-1]

        # Save the image
        original_filename = os.path.basename(image_path)
        output_path = os.path.join(PREDICTION_VIS_DIR, original_filename)
        cv2.imwrite(output_path, result_image)
        # print(f"✅ Saved: {output_path}") # Optional: reduce console spam

    print(f"\n✅ Finished visualization! Images saved to {PREDICTION_VIS_DIR}")
    print(f"Detailed evaluation results (JSON) saved to {EVALUATION_OUTPUT_DIR}/coco_instances_results.json")

