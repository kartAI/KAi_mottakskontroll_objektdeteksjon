import os
import random
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
import geopandas as gpd
from PIL import Image

class DatasetProcessor:
    def __init__(self, tiff_folder, gpkg_path, output_folder, tile_size=1024, overlap_ratio=0.1):
        self.tiff_folder = tiff_folder
        self.gpkg_path = gpkg_path
        self.output_folder = output_folder
        self.tile_size = tile_size
        self.overlap = int(tile_size * overlap_ratio)

        self.split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        self.dataset_folders = {
            split: {
                "images": os.path.join(output_folder, split, "images"),
                "labels": os.path.join(output_folder, split, "labels"),
            }
            for split in self.split_ratios
        }
        
        for split, paths in self.dataset_folders.items():
            os.makedirs(paths["images"], exist_ok=True)
            os.makedirs(paths["labels"], exist_ok=True)

        self.gdf = gpd.read_file(gpkg_path)

    def assign_split(self):
        """ Randomly assign tiles to train, val, or test based on split ratios """
        rand = random.random()
        cumulative = 0
        for split, ratio in self.split_ratios.items():
            cumulative += ratio
            if rand <= cumulative:
                return split
        return "train"  

    def tile_images(self, tiff_name):
        tiff_path = os.path.join(self.tiff_folder, tiff_name)
        
        with rasterio.open(tiff_path) as src:
            img_width, img_height = src.width, src.height

            for y in range(0, img_height, self.tile_size - self.overlap):
                for x in range(0, img_width, self.tile_size - self.overlap):
                    if x + self.tile_size > img_width:
                        x = img_width - self.tile_size
                    if y + self.tile_size > img_height:
                        y = img_height - self.tile_size

                    window = Window(x, y, self.tile_size, self.tile_size)
                    tile_data = src.read(window=window)

    
                    transform = src.window_transform(window)
                    minx, miny = transform * (0, 0)  
                    maxx, maxy = transform * (self.tile_size, self.tile_size)  
                    tile_bounds = box(minx, miny, maxx, maxy)


                    segmentation_annotations = self.generate_segmentation_labels(tile_bounds, transform)


                    if not segmentation_annotations:
                        print(f"Skipped empty tile at {x}, {y}")
                        continue

                    split = self.assign_split()
                    
                    tile_filename = f"{os.path.splitext(tiff_name)[0]}_tile_{y}_{x}.jpg"
                    tile_img_path = os.path.join(self.dataset_folders[split]["images"], tile_filename)
                    tile_img = Image.fromarray(tile_data.transpose(1, 2, 0))
                    tile_img.save(tile_img_path, "JPEG")
                    print(f"Saved tile at {tile_img_path}")

                    label_filename = f"{os.path.splitext(tile_filename)[0]}.txt"
                    label_path = os.path.join(self.dataset_folders[split]["labels"], label_filename)
                    with open(label_path, "w") as label_file:
                        label_file.write("\n".join(segmentation_annotations))
                    print(f"Saved YOLO label at {label_path}")

    def generate_segmentation_labels(self, tile_bounds, transform):
        clipped_gdf = self.gdf[self.gdf.intersects(tile_bounds)].copy()
        clipped_gdf['geometry'] = clipped_gdf['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)

        if clipped_gdf.empty:
            return []

        segmentation_annotations = []
        for _, row in clipped_gdf.iterrows():
            geometry = row.geometry

            polygons = [geometry] if geometry.geom_type == 'Polygon' else geometry.geoms
            
            for polygon in polygons:
                vertices = []
                for x, y in polygon.exterior.coords:
                    col, row = ~transform * (x, y) 

                    col = col / self.tile_size
                    row = row / self.tile_size
                    
                    threshold = 0.001
                    col = max(0, min(col, 1)) if col >= threshold else 0
                    row = max(0, min(row, 1)) if row >= threshold else 0

                    col = round(col, 3)
                    row = round(row, 3)

                    vertices.append(f"{col} {row}")

                annotation = f"0 " + " ".join(vertices)
                segmentation_annotations.append(annotation)

        return segmentation_annotations

    def generate_data_yaml(self):
        """ Generate the data.yaml file for YOLO """
        yaml_path = os.path.join(self.output_folder, "data.yaml")
        with open(yaml_path, "w") as yaml_file:
            yaml_file.write(f"train: {os.path.join(self.output_folder, 'train/images')}\n")
            yaml_file.write(f"val: {os.path.join(self.output_folder, 'val/images')}\n")
            yaml_file.write(f"test: {os.path.join(self.output_folder, 'test/images')}\n")
            yaml_file.write("nc: 1\n")  # Number of classes
            yaml_file.write("names: ['building']\n")
        print(f"Generated data.yaml at {yaml_path}")


tiff_folder = "data/images"
gpkg_path = "data/Bygninger_Flate.gpkg"
output_folder = "data/dataset"

processor = DatasetProcessor(tiff_folder, gpkg_path, output_folder)

for tiff_name in os.listdir(tiff_folder):
    if tiff_name.endswith(".tif"):
        print(f"Processing {tiff_name}...")
        try:
            processor.tile_images(tiff_name)
        except Exception as e:
            print(f"Error processing {tiff_name}: {e}")

processor.generate_data_yaml()
