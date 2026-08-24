import random
import os
import shutil
import cv2
import numpy as np
import json
from pathlib import Path
import glob
from ultralytics import YOLO
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from split import get_random_split, generate_split, get_split_set
from labeling_pipeline import get_image_index, get_image_slices, replicate_structure, process_dataset, image_slicing_pipeline
from train import train_yolo
from test import test_yolo

RAW_PATH = 'raw/' # path to directory of raw images
LABELS_PATH = 'labelme/' # path to directory of LabelMe JSONs
COST_MATRIX = [[0, 2.0], # cost matrix for binary model predictions
               [1.0, 0]]

# Get train, val, test split
get_split_set(source_dir=RAW_PATH, dest_dir='dataset-rgb/')

# Generate dataset of image slices from raw images, using LabelMe annotations to label slices
image_slicing_pipeline(source_dir='dataset-rgb/', target_dir='dataset-slices/', json_dir=LABELS_PATH, binary=True) # binary dataset
image_slicing_pipeline(source_dir='dataset-rgb/', target_dir='dataset-slices-multiclass/', json_dir=LABELS_PATH, binary=False) # multiclass dataset

# Train YOLO models on binary and multiclass datasets
binary_model_path = train_yolo(dataset_path='dataset-slices/', start_model='yolo26n-cls.pt', epochs=20, imgsz=256)
multiclass_model_path = train_yolo(dataset_path='dataset-slices-multiclass/', start_model='yolo26n-cls.pt', epochs=20, imgsz=256)

test_yolo(model_path=binary_model_path, dataset_path='dataset-slices/', num_classes=2, cost_matrix=COST_MATRIX)
test_yolo(model_path=multiclass_model_path, dataset_path='dataset-slices-multiclass/', num_classes=3)
