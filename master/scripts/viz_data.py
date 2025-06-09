import os
import glob
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm



# Set the path to your TIFF folder and output PNG file
tiff_folder = "data/eksport_data"
output_path = "merged_visualization_w.png"
rescale_factor = 0.05  # 10% of original resolution
max_files = None  # 🔧 Change to None to process ALL TIFFs

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

# Gather TIFF files
tiff_files = sorted(glob.glob(os.path.join(tiff_folder, "*.tif")))
if max_files:
    tiff_files = tiff_files[:max_files]

resampled_datasets = []

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
            resampled_datasets.append(memfile.open())

# Merge and save
mosaic, _ = merge(resampled_datasets)
rgb = mosaic[:3].transpose(1, 2, 0)

nodata_mask = np.all(rgb == 0, axis=-1)
rgb[nodata_mask] = [255, 255, 255]

fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
ax.imshow(rgb.astype("uint8"))
# ax.set_title(f" Downsampled Merged Visualization ({len(tiff_files)} files)")
ax.axis('off')
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')

# Clean up
for ds in resampled_datasets:
    ds.close()
