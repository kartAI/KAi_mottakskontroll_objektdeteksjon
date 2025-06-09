import os
import glob
import json
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.merge import merge
from rasterio.enums import Resampling
from matplotlib.patches import Patch
from tqdm import tqdm

# Config
tiff_folder = "data/eksport_data"
output_path = "merged_visualization_overlay.png"
split_json_path = "data/tile_splits.json"
rescale_factor = 0.1
max_files = None  # Set to e.g. 10 for testing

# Load split info
with open(split_json_path, "r") as f:
    split_info = json.load(f)

# Helper to downsample and return data + transform
def resample_raster(src, scale):
    if src.count < 3:
        print(f"Skipping {src.name}: not enough bands")
        return None, None

    new_height = int(src.height * scale)
    new_width = int(src.width * scale)

    data = src.read(
        out_shape=(src.count, new_height, new_width),
        resampling=Resampling.bilinear
    )

    new_transform = src.transform * src.transform.scale(
        (src.width / data.shape[-1]),
        (src.height / data.shape[-2])
    )

    return data, new_transform

# Collect files
tiff_files = sorted(glob.glob(os.path.join(tiff_folder, "*.tif")))
if max_files:
    tiff_files = tiff_files[:max_files]

resampled_datasets = []
source_infos = []  # Store data for bounding boxes

for path in tqdm(tiff_files, desc="Processing TIFFs"):
    with rasterio.open(path) as src:
        data, transform = resample_raster(src, rescale_factor)
        if data is not None:
            profile = src.profile
            profile.update({
                'height': data.shape[1],
                'width': data.shape[2],
                'transform': transform
            })

            memfile = rasterio.io.MemoryFile()
            with memfile.open(**profile) as tmp_ds:
                tmp_ds.write(data)
            dataset = memfile.open()
            resampled_datasets.append(dataset)

            # Split label
            filename = os.path.basename(path)
            if filename in split_info["train"]:
                label = "train"
                color = "red"
            elif filename in split_info["val"]:
                label = "val"
                color = "green"
            elif filename in split_info["test"]:
                label = "test"
                color = "blue"
            else:
                label = "unknown"
                color = "gray"

            source_infos.append({
                "dataset": dataset,
                "label": label,
                "color": color
            })

# Merge
mosaic, mosaic_transform = merge(resampled_datasets)
rgb = mosaic[:3].transpose(1, 2, 0)

# Replace no-data with white
nodata_mask = np.all(rgb == 0, axis=-1)
rgb[nodata_mask] = [255, 255, 255]

# Plot
fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
ax.imshow(rgb.astype("uint8"))

# Draw overlays based on true mosaic transform
for info in source_infos:
    ds = info["dataset"]
    bounds = ds.bounds

    # Convert geo coords to pixel coords in the merged mosaic
    col_start, row_start = ~mosaic_transform * (bounds.left, bounds.top)
    col_end, row_end = ~mosaic_transform * (bounds.right, bounds.bottom)

    x = col_start
    y = row_start
    width = col_end - col_start
    height = row_end - row_start

    ax.add_patch(plt.Rectangle(
        (x, y), width, height,
        edgecolor=info["color"],
        facecolor=info["color"],
        linewidth=0.5,
        alpha=0.3  # 30% visible, you can adjust this
    ))

# Legend
legend_patches = [
    Patch(edgecolor='red', label='Train', fill=False, linewidth=1.5),
    Patch(edgecolor='green', label='Val', fill=False, linewidth=1.5),
    Patch(edgecolor='blue', label='Test', fill=False, linewidth=1.5)
]
ax.legend(handles=legend_patches, loc='lower left', fontsize='small')
ax.axis('off')
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

# Clean up
for ds in resampled_datasets:
    ds.close()
