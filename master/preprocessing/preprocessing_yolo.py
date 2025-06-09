import os
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
import geopandas as gpd
from PIL import Image, ImageDraw
from tile_utils import should_skip_tile


class DatasetProcessor:
    def __init__(self, tiff_folder, gpkg_path, output_folder, tile_size=1024, overlap_ratio=0.1, split_name=None):
        self.tiff_folder = tiff_folder
        self.gpkg_path = gpkg_path
        self.output_folder = output_folder
        self.tile_size = tile_size
        self.overlap = int(tile_size * overlap_ratio)
        self.split_name = split_name
        self.include_empty_tiles = False

        self.image_folder = os.path.join(output_folder, split_name, "images")
        self.label_folder = os.path.join(output_folder, split_name, "labels")
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.label_folder, exist_ok=True)

        self.gdf = gpd.read_file(gpkg_path)

    def generate_binary_mask(self, polygons, tile_size):
        mask = Image.new("L", (tile_size, tile_size), 0)
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

                    clipped_gdf = self.gdf[self.gdf.intersects(tile_bounds)].copy()
                    clipped_gdf['geometry'] = clipped_gdf['geometry'].apply(lambda g: g.buffer(0) if not g.is_valid else g)

                    segmentation_annotations = self.generate_segmentation_labels(clipped_gdf, transform)

                    if should_skip_tile(clipped_gdf, segmentation_annotations, self.include_empty_tiles):
                        continue

                    pixel_polygons = []
                    for _, row in clipped_gdf.iterrows():
                        geometry = row.geometry
                        poly_list = [geometry] if geometry.geom_type == 'Polygon' else geometry.geoms
                        for polygon in poly_list:
                            exterior_coords = list(polygon.exterior.coords)[:-1]
                            pixel_coords = [~transform * (x, y) for x, y in exterior_coords]
                            pixel_polygons.append(pixel_coords)

                    tile_filename = f"{image_base_filename}_tile_{y}_{x}.jpg"
                    tile_img_path = os.path.join(self.image_folder, tile_filename)
                    tile_img = Image.fromarray(tile_data.transpose(1, 2, 0))
                    tile_img.save(tile_img_path, "JPEG")

                    mask_img = self.generate_binary_mask(pixel_polygons, self.tile_size)
                    mask_path = os.path.join(self.image_folder, f"{os.path.splitext(tile_filename)[0]}_mask.png")
                    mask_img.save(mask_path)

                    label_path = os.path.join(self.label_folder, f"{os.path.splitext(tile_filename)[0]}.txt")
                    with open(label_path, "w") as f:
                        f.write("\n".join(segmentation_annotations))

    def generate_segmentation_labels(self, clipped_gdf, transform):
        if clipped_gdf.empty:
            return []

        segmentation_annotations = []
        for _, row in clipped_gdf.iterrows():
            geometry = row.geometry
            polygons = [geometry] if geometry.geom_type == 'Polygon' else geometry.geoms

            for polygon in polygons:
                if polygon.is_empty or not polygon.exterior:
                    continue

                vertices = []
                for x, y in polygon.exterior.coords:
                    col, row_pix = ~transform * (x, y)
                    col = max(0, min(col / self.tile_size, 1))
                    row_pix = max(0, min(row_pix / self.tile_size, 1))
                    vertices.append(f"{round(col, 4)} {round(row_pix, 4)}")

                annotation = "0 " + " ".join(vertices)
                segmentation_annotations.append(annotation)

        return segmentation_annotations

    def generate_data_yaml(self):
        yaml_path = os.path.join(self.output_folder, "data.yaml")
        with open(yaml_path, "w") as yaml_file:
            yaml_file.write(f"train: {os.path.join(self.output_folder, 'train/images')}\n")
            yaml_file.write(f"val: {os.path.join(self.output_folder, 'val/images')}\n")
            yaml_file.write(f"test: {os.path.join(self.output_folder, 'test/images')}\n")
            yaml_file.write("nc: 1\n")
            yaml_file.write("names: ['building']\n")
