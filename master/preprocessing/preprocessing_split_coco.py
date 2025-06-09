import os
import json
from preprocessing.preprocessing_coco import DatasetProcessor

# Config
tiff_folder = "data/eksport_data"
gpkg_path = "data/Bygning.gpkg"
split_json_path = "data/tile_splits.json"
output_root = "data_new/coco_dataset_filtered"

# Load split list
with open(split_json_path) as f:
    tile_splits = json.load(f)

for split_name in ["train", "val", "test"]:
    print(f"\n🔄 Starting {split_name.upper()} split")
    tiff_list = tile_splits[split_name]

    output_folder = os.path.join(output_root, split_name)
    os.makedirs(os.path.join(output_folder, "images"), exist_ok=True)

    # include_empty = (split_name == "test")  # Match logic with YOLO
    include_empty = False  # Match logic with YOLO

    processor = DatasetProcessor(
        tiff_folder=tiff_folder,
        gpkg_path=gpkg_path,
        output_folder=output_folder,
        tile_size=1024,
        overlap_ratio=0.1,
        include_empty_tiles=include_empty,
    )

    for tiff_name in tiff_list:
        print(f"📦 Processing {tiff_name}")
        try:
            processor.tile_images(tiff_name)
        except Exception as e:
            print(f"❌ Error processing {tiff_name}: {e}")

    processor.generate_coco_json()
    print(f"✅ Finished {split_name.upper()} split")
