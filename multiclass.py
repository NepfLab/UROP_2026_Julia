"""
Performs multiclass classification (undamaged, damaged) on a marsh imagery dataset.
"""

from ultralytics import YOLO
import os
import torch
import cv2
from slices import create_slices, delete_slices

DATASET_PATH = 'dataset/' ## set to base path of dataset
## number of rows and cols to slice each image into
ROWS = 4
COLS = 4

model = YOLO('multiclass.pt')

all_imgs = [img for img in os.listdir(DATASET_PATH) if os.path.isfile(os.path.join(DATASET_PATH, img))]
all_imgs = sorted(all_imgs)

for img in all_imgs:
    img_path = os.path.join(DATASET_PATH, img)
    slices_dir = create_slices(rows=ROWS, cols=COLS, image_path=img_path)
    num_healthy, num_borderline, num_damaged = 0, 0, 0
    for slice in os.listdir(slices_dir):
        slice_path = os.path.join(slices_dir, slice)
        if not os.path.isfile(slice_path):
            continue

        results = model(slice_path, verbose=False)
        slice_pred = results[0].names[results[0].probs.top1]

        if slice_pred == 'healthy':
            num_healthy += 1
        elif slice_pred == 'borderline':
            num_borderline += 1
        else:
            num_damaged += 1

    print(f"Image: {img} | Predicted: {num_healthy} healthy slices, {num_borderline} borderline slices, {num_damaged} damaged slices")
    delete_slices()
