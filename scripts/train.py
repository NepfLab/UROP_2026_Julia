from ultralytics import YOLO
import torch


def train_yolo(dataset_path, start_model='yolo26n-cls.pt', epochs=20, imgsz=256):
  """
  Trains YOLO model.
  Parameters:
  - bucket_name: name of GCS bucket
  - gcs_path: GCS path to save trained YOLO model at
  - dataset_path: local path to dataset (omit final '/')
  - start_model: YOLO model to start with
  - epochs: number of training epochs
  - imgsz: image size
  """

  model = YOLO(start_model)

  results = model.train(data=dataset_path, epochs=epochs, imgsz=imgsz)
  local_path = f"{results.save_dir}/weights/best.pt"
  return local_path
