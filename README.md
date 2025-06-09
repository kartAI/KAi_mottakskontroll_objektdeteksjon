# Deep Learning for Building Detection for Completeness Check in Mottakskontroll – Master Thesis

This repository contains the code developed as part of a master's thesis at NTNU. The project investigates how deep learning can support the Norwegian Mapping Authority’s *mottakskontroll* (completeness check) for FKB-building data by detecting buildings in aerial imagery using instance segmentation models.

## Project Structure

All the code produces in the master's thesis is in the master folder. The master folder have the following folders: 

 - mask_r_cnn: All code related to Mask R-CNN using Detectron2
- yolo: All code related to YOLOv8-seg using Ultralytics
- preprocessing: Scripts to tile TIFFs and generate training data
- scripts: Miscellaneous utilities (visualization, evaluation, etc.)

## Models and Frameworks

The following models and tools are used in the project:

- **YOLOv8-seg** [Ultralytics](https://github.com/ultralytics/ultralytics) 
- **Mask R-CNN** [Detectron2](https://github.com/facebookresearch/detectron2)
- **SAHI** – [Sliced Aided Hyper Inference(SAHI)](https://github.com/obss/sahi) to improve predictions near tile edges 

## Installation

This project uses Python 3.12+. Dependencies are declared in the `pyproject.toml` file. Minimal installation example:

```toml
[project]
name = "master"
version = "0.1.0"
description = "Deep learning models for building detection in aerial images"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.6.0",
    "torchvision>=0.21.0",
]
```
To install dependencies, use a tool like pip or poetry:
```
pip install torch torchvision
````

⚠️ Additional installation steps are required for Ultralytics and Detectron2. Follow their official guides based on your OS and CUDA version.

## Data Requirements
To run the full pipeline, you need:

- Georeferenced TIFF images
- FKB building data in polygon format stored as a GeoPackage (.gpkg)

These are processed into training tiles and annotations using the preprocessing/ scripts. Both COCO and YOLO format datasets are supported.

## Usage
Preprocess the data
Use the scripts in preprocessing/ to tile the input imagery and convert annotations to COCO or YOLO formats.

Train models

Run training scripts in yolo/ for YOLOv8-seg

Run training scripts in mask_r_cnn/ for Mask R-CNN (Detectron2)


## Acknowledgements
This project was conducted as part of a master’s thesis in Engineering and ICT at NTNU, in collaboration with the Norwegian Mapping Authority (Kartverket) and the KartAI initiative.

## Note
If you use this code or build upon it, please cite the frameworks used:

Ultralytics YOLOv8
Detectron2
SAHI

This repository is intended for academic and research use. 

