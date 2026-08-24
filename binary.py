"""
Performs binary classification (undamaged, damaged) on a marsh imagery dataset.
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

model = YOLO('binary.pt')
cost_matrix = [[0, 2.0],
               [1.0, 0]]

all_imgs = [img for img in os.listdir(DATASET_PATH) if os.path.isfile(os.path.join(DATASET_PATH, img))]
all_imgs = sorted(all_imgs)

for img in all_imgs:
    img_path = os.path.join(DATASET_PATH, img)
    slices_dir = create_slices(rows=ROWS, cols=COLS, image_path=img_path)
    num_undamaged, num_damaged = 0, 0
    for slice in os.listdir(slices_dir):
        slice_path = os.path.join(slices_dir, slice)
        if not os.path.isfile(slice_path):
            continue

        results = model(slice_path, verbose=False)

        probs = results[0].probs.data
        C = torch.tensor(cost_matrix, device=probs.device)

        # Compute expected risk vector for predicted choices
        expected_risks = torch.matmul(C.T, probs)

        # Select prediction that minimizes expected cost
        best_class_idx = torch.argmin(expected_risks).item()
        slice_pred = results[0].names[best_class_idx]

        if slice_pred == 'damaged':
            num_damaged += 1
        else:
            num_undamaged += 1

    print(f"Image: {img} | Predicted: {num_undamaged} undamaged slices, {num_damaged} damaged slices")
    delete_slices()
