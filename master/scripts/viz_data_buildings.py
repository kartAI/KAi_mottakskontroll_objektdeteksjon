import os
import glob
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import box
from shapely.affinity import affine_transform

# Config
tiff_folder = "data/eksport_data"
output_path = "merged_with_buildings_px.png"
rescale_factor = 0.05  # 5% resolution
max_files = None  # Use None to run full dataset
buildings_path = "data/Bygning.gpkg"
image_crs = "EPSG:25832"

# Downsampling function
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

# Collect TIFF files
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

# Merge
mosaic, mosaic_transform = merge(resampled_datasets)
rgb = mosaic[:3].transpose(1, 2, 0)

# Replace no-data with white
nodata_mask = np.all(rgb == 0, axis=-1)
rgb[nodata_mask] = [255, 255, 255]

# Plot mosaic
fig, ax = plt.subplots(figsize=(16, 16), facecolor='white')
ax.imshow(rgb.astype("uint8"))
ax.axis('off')

# Lock axis extent
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

# Load buildings
buildings = gpd.read_file(buildings_path).to_crs(image_crs)
print(f"Total buildings in full GPKG: {len(buildings)}")

# Clip buildings to mosaic extent
mosaic_bounds = rasterio.transform.array_bounds(
    rgb.shape[0], rgb.shape[1], mosaic_transform
)
minx, miny, maxx, maxy = mosaic_bounds
mosaic_box = box(minx, miny, maxx, maxy)
buildings_in_view = buildings[buildings.intersects(mosaic_box)]
print(f"Buildings inside current mosaic: {len(buildings_in_view)}")

# Transform building geometries to pixel space
inv_transform = ~mosaic_transform
affine_mat = [
    inv_transform.a, inv_transform.b,
    inv_transform.d, inv_transform.e,
    inv_transform.xoff, inv_transform.yoff
]
buildings_px = buildings_in_view.copy()
buildings_px["geometry"] = buildings_px["geometry"].apply(
    lambda geom: affine_transform(geom, affine_mat)
)

# Optional: plot centroids
centroids = buildings_px.centroid
for pt in centroids:
    ax.plot(pt.x, pt.y, marker='o', color='orange', markersize=1)

# Restore axis limits
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

# Cleanup
for ds in resampled_datasets:
    ds.close()
