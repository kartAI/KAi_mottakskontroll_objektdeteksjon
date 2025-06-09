import os
import json
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
import geopandas as gpd
from PIL import Image, ImageDraw
from tile_utils import should_skip_tile


class DatasetProcessor:
    def __init__(self, tiff_folder, gpkg_path, output_folder, tile_size=1024, overlap_ratio=0.1, include_empty_tiles=False):
        self.tiff_folder = tiff_folder
        self.gpkg_path = gpkg_path
        self.output_folder = output_folder
        self.tile_size = tile_size
        self.overlap = int(tile_size * overlap_ratio)
        self.include_empty_tiles = include_empty_tiles

        self.image_output_folder = os.path.join(output_folder, "images")
        self.mask_output_folder = os.path.join(output_folder, "masks")
        self.annotation_output_path = os.path.join(output_folder, "coco_annotations.json")

        os.makedirs(self.image_output_folder, exist_ok=True)
        os.makedirs(self.mask_output_folder, exist_ok=True)

        self.gdf = gpd.read_file(gpkg_path)
        self.images_info = []
        self.annotations_info = []
        self.annotation_id_counter = 1
        self.image_id_counter = 1

    def generate_binary_mask(self, polygons):
        mask = Image.new("L", (self.tile_size, self.tile_size), 0)
        draw = ImageDraw.Draw(mask)
        for poly in polygons:
            coords = [(x, y) for x, y in poly]
            draw.polygon(coords, outline=255, fill=255)
        return mask

    def tile_images(self, tiff_name):
        tiff_path = os.path.join(self.tiff_folder, tiff_name)

        with rasterio.open(tiff_path) as src:
            img_width, img_height = src.width, src.height
            image_base_filename = os.path.splitext(tiff_name)[0]

            for y in range(0, img_height, self.tile_size - self.overlap):
                for x in range(0, img_width, self.tile_size - self.overlap):
                    if x + self.tile_size > img_width:
                        x = img_width - self.tile_size
                    if y + self.tile_size > img_height:
                        y = img_height - self.tile_size

                    window = Window(x, y, self.tile_size, self.tile_size)
                    tile_data = src.read(window=window)
                    transform = src.window_transform(window)
                    tile_bounds = box(*transform * (0, 0), *transform * (self.tile_size, self.tile_size))

                    segmentation_result = self.generate_segmentation_labels(tile_bounds, transform, self.image_id_counter)
                    annotations = segmentation_result["annotations"]

                    clipped_gdf = self.gdf[self.gdf.intersects(tile_bounds)].copy()

                    if should_skip_tile(clipped_gdf, annotations, self.include_empty_tiles):
                        continue

                    tile_filename = f"{image_base_filename}_tile_{y}_{x}.jpg"
                    tile_img_path = os.path.join(self.image_output_folder, tile_filename)
                    tile_img = Image.fromarray(tile_data.transpose(1, 2, 0))
                    tile_img.save(tile_img_path, "JPEG")

                    self.images_info.append({
                        "id": self.image_id_counter,
                        "file_name": tile_filename,
                        "width": self.tile_size,
                        "height": self.tile_size,
                    })

                    if annotations:
                        self.annotations_info.extend(annotations)

                        # Save binary mask
                        pixel_polygons = []
                        for ann in annotations:
                            seg = ann['segmentation'][0]  # flat list
                            coords = list(zip(seg[::2], seg[1::2]))
                            pixel_polygons.append(coords)

                        mask_img = self.generate_binary_mask(pixel_polygons)
                        mask_filename = f"{os.path.splitext(tile_filename)[0]}_mask.png"
                        mask_path = os.path.join(self.mask_output_folder, mask_filename)
                        mask_img.save(mask_path)

                    self.image_id_counter += 1

    def generate_segmentation_labels(self, tile_bounds, transform, image_id):
        clipped_gdf = self.gdf[self.gdf.intersects(tile_bounds)].copy()
        clipped_gdf['geometry'] = clipped_gdf['geometry'].apply(lambda geom: geom.buffer(0) if not geom.is_valid else geom)

        if clipped_gdf.empty:
            return {'annotations': []}

        coco_annotations = []
        for _, row in clipped_gdf.iterrows():
            geometry = row.geometry
            polygons = [geometry] if geometry.geom_type == 'Polygon' else geometry.geoms

            for polygon in polygons:
                if polygon.is_empty or not polygon.exterior:
                    continue

                segmentation = []
                for x, y in list(polygon.exterior.coords)[:-1]:
                    col, row_pix = ~transform * (x, y)
                    col = max(0, min(col, self.tile_size))
                    row_pix = max(0, min(row_pix, self.tile_size))
                    segmentation.extend([col, row_pix])

                if not segmentation:
                    continue

                minx, miny, maxx, maxy = polygon.bounds
                x0, y0 = ~transform * (minx, miny)
                x1, y1 = ~transform * (maxx, maxy)
                x_min, x_max = sorted([int(x0), int(x1)])
                y_min, y_max = sorted([int(y0), int(y1)])
                x_min = max(0, min(x_min, self.tile_size))
                x_max = max(0, min(x_max, self.tile_size))
                y_min = max(0, min(y_min, self.tile_size))
                y_max = max(0, min(y_max, self.tile_size))

                annotation = {
                    "id": self.annotation_id_counter,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": [segmentation],
                    "area": polygon.area,
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "iscrowd": 0,
                }
                coco_annotations.append(annotation)
                self.annotation_id_counter += 1

        return {'annotations': coco_annotations}

    def generate_coco_json(self):
        coco_output = {
            "info": {
                "description": "Building Segmentation Dataset",
                "version": "1.0",
                "year": 2024,
                "contributor": "Marianne Stålen",
                "date_created": "2025-01-01",
            },
            "licenses": [],
            "categories": [{"id": 1, "name": "building", "supercategory": "building"}],
            "images": self.images_info,
            "annotations": self.annotations_info,
        }

        with open(self.annotation_output_path, "w") as outfile:
            json.dump(coco_output, outfile)
        print(f"📝 Generated COCO annotation file at {self.annotation_output_path}")
