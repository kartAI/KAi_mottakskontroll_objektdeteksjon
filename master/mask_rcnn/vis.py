import random
import cv2
import matplotlib.pyplot as plt

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from dataset import building_dataset_train_name # Import dataset name from dataset.py

# Load a sample from the training dataset
dataset_dicts = DatasetCatalog.get(building_dataset_train_name)

# Pick a random image
sample = random.choice(dataset_dicts)
img = cv2.imread(sample["file_name"])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Visualize
visualizer = Visualizer(img, metadata=MetadataCatalog.get(building_dataset_train_name), scale=1.2)
out = visualizer.draw_dataset_dict(sample)
plt.figure(figsize=(10, 10))
plt.imshow(out.get_image())
plt.axis("off")
plt.show()