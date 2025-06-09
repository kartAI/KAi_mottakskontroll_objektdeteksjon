import detectron2
import albumentations as A
import cv2
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data import detection_utils as utils
import torch

# Define dataset names and paths
dataset_name = "building_dataset_train"
train_json_path = "data_new/coco_dataset/train/coco_annotations.json"
train_image_root = "data_new/coco_dataset/train/images"

dataset_name_val = "building_dataset_val"
val_json_path = "data_new/coco_dataset/val/coco_annotations.json"
val_image_root = "data_new/coco_dataset/val/images"

dataset_name_test = "building_dataset_test"
test_json_path = "data_new/coco_dataset/test/coco_annotations.json"
test_image_root = "data_new/coco_dataset/test/images"

# Register datasets
def register_building_instances(name, json_file, image_root):
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        thing_classes=["building"],
        thing_dataset_id_to_contiguous_id={1: 0},
    )
    return name

building_dataset_train_name = register_building_instances(dataset_name, train_json_path, train_image_root)
building_dataset_val_name = register_building_instances(dataset_name_val, val_json_path, val_image_root)
building_dataset_test_name = register_building_instances(dataset_name_test, test_json_path, test_image_root)

print(f"Registered training dataset: {building_dataset_train_name}")
print(f"Registered validation dataset: {building_dataset_val_name}")
print(f"Registered test dataset: {building_dataset_test_name}")

if __name__ == "__main__":
    # Test dataset registration
    dataset_dicts = DatasetCatalog.get("building_dataset_train")
    print(f"Number of training samples: {len(dataset_dicts)}")
    dataset_dicts_val = DatasetCatalog.get("building_dataset_val")
    print(f"Number of validation samples: {len(dataset_dicts_val)}")
    dataset_dicts_test = DatasetCatalog.get("building_dataset_test")
    print(f"Number of test samples: {len(dataset_dicts_test)}")
