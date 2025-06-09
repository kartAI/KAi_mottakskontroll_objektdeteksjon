import os
import json
from preprocessing_yolo_new import DatasetProcessor

# Config
split_json_path = "data/tile_splits.json"
tiff_folder = "data/eksport_data"
gpkg_path = "data/Bygning.gpkg"
output_folder = "data_new/yolo_dataset_filtered_new"

# Load split list
with open(split_json_path) as f:
    tile_splits = json.load(f)

for split_name in ["train", "val", "test"]:
    print(f"\n🔄 Starting {split_name.upper()} split")
    tiff_list = tile_splits[split_name]

    processor = DatasetProcessor(
        tiff_folder=tiff_folder,
        gpkg_path=gpkg_path,
        output_folder=output_folder,
        tile_size=1024,
        overlap_ratio=0.1,
        split_name=split_name  # Automatically sets include_empty_tiles
    )

    for tiff_name in tiff_list:
        print(f"📦 Processing {tiff_name}")
        try:
            processor.tile_images(tiff_name)
        except Exception as e:
            print(f"❌ Error processing {tiff_name}: {e}")

# Generate metadata
processor.generate_data_yaml()
