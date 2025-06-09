import os
import pathlib
import random
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from PIL import Image
import cv2
import shutil

base_dir = "dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "labels"), exist_ok=True)

tiff_dir = "data/images"
gpkg_path = "data/Bygning.gpkg"
geopackage = gpd.read_file(gpkg_path)

tile_id = 0  
all_images = [] 

for tiff_path in pathlib.Path(tiff_dir).glob("*.tif"):
    print(f"Processing {tiff_path}...")

    with rasterio.open(tiff_path) as src:
        tiff_bounds = src.bounds
        tiff_transform = src.transform
        tiff_crs = src.crs
        tiff_shape = (src.height, src.width)
        tiff_data = src.read()

    geopackage = geopackage.to_crs(tiff_crs)

    bbox = gpd.GeoDataFrame({"geometry": [box(*tiff_bounds)]}, crs=tiff_crs)
    clipped_gpkg = gpd.overlay(geopackage, bbox, how="intersection")

    if clipped_gpkg.empty:
        print(f"Warning: No valid geometries found for {tiff_path}. Skipping...")
        continue

    shapes = ((geom, 1) for geom in clipped_gpkg.geometry)

    mask = rasterize(
        shapes,
        out_shape=tiff_shape,
        transform=tiff_transform,
        fill=0,
        dtype="uint8",
    )

    tile_size = 1024
    step_size = int(tile_size * 0.9)  

    for row_start in range(0, tiff_shape[0], step_size):
        for col_start in range(0, tiff_shape[1], step_size):
            row_end = min(row_start + tile_size, tiff_shape[0])
            col_end = min(col_start + tile_size, tiff_shape[1])

            mask_tile = mask[row_start:row_end, col_start:col_end]
            raster_tile = tiff_data[:, row_start:row_end, col_start:col_end]

            if np.sum(mask_tile) == 0:
                continue

            raster_tile_jpg = np.moveaxis(raster_tile, 0, -1).astype(np.uint8)
            image_name = f"raster_tile_{tile_id}.jpg"
            image_path = os.path.join(train_dir, "images", image_name) 
            raster_image = Image.fromarray(raster_tile_jpg)
            raster_image.save(image_path)
            all_images.append(image_path)

            label_name = f"raster_tile_{tile_id}.txt"
            label_path = os.path.join(train_dir, "labels", label_name)  

            contours, _ = cv2.findContours(mask_tile, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            with open(label_path, "w") as f:
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    img_width, img_height = raster_tile_jpg.shape[1], raster_tile_jpg.shape[0]
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    width = w / img_width
                    height = h / img_height
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            tile_id += 1

random.shuffle(all_images)
train_split = int(0.7 * len(all_images))
val_split = int(0.15 * len(all_images)) + train_split

train_images = all_images[:train_split]
val_images = all_images[train_split:val_split]
test_images = all_images[val_split:]

for image_path in train_images:
    label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
    shutil.move(image_path, os.path.join(train_dir, "images", os.path.basename(image_path)))
    shutil.move(label_path, os.path.join(train_dir, "labels", os.path.basename(label_path)))

for image_path in val_images:
    label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
    shutil.move(image_path, os.path.join(val_dir, "images", os.path.basename(image_path)))
    shutil.move(label_path, os.path.join(val_dir, "labels", os.path.basename(label_path)))

for image_path in test_images:
    label_path = image_path.replace("images", "labels").replace(".jpg", ".txt")
    shutil.move(image_path, os.path.join(test_dir, "images", os.path.basename(image_path)))
    shutil.move(label_path, os.path.join(test_dir, "labels", os.path.basename(label_path)))


yaml_path = os.path.join(base_dir, "data.yaml")
classes = ["building"]  
with open(yaml_path, "w") as yaml_file:
    yaml_file.write(f"train: {os.path.abspath(os.path.join(train_dir, 'images'))}\n")
    yaml_file.write(f"val: {os.path.abspath(os.path.join(val_dir, 'images'))}\n")
    yaml_file.write(f"test: {os.path.abspath(os.path.join(test_dir, 'images'))}\n")
    yaml_file.write(f"nc: {len(classes)}\n")
    yaml_file.write(f"names: {classes}\n")

print("YOLO dataset structure created successfully.")
