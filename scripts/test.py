from ultralytics import YOLO
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import matplotlib.pyplot as plt


def test_yolo(model_path, dataset_path, classes=None,
              num_classes=2, confidence_threshold=0, cost_matrix=None,
              verbose=False):
  """
  Tests YOLO model.
  Parameters:
  - model_path: path to model weights
  - dataset_path: local path to dataset (omit final '/')
  - classes: list of classes
  - confidence_threshold: confidence threshold for predictions
  - cost_matrix: cost matrix as an array of arrays
  """

  if classes is None:
    if num_classes == 2:
      classes = ['damaged', 'undamaged']
    elif num_classes == 3:
      classes = ['healthy', 'borderline', 'damaged']
    else:
      raise ValueError('Invalid classes')

  model = YOLO(model_path)

  print("==========")

  total_correct = 0
  total_incorrect = 0
  total_unsure = 0
  total_avg_incorrect_conf = 0
  summary_stats = "SUMMARY STATISTICS\n"
  Y_true = []
  Y_pred = []

  for class_name in classes:
    test_dir = os.path.join(dataset_path, 'test', class_name)
    if not os.path.exists(test_dir):
      print('NO TEST SPLIT FOUND, TESTING ON ALL IMAGES')
      test_dir = os.path.join(dataset_path, class_name)
    correct_count = 0
    incorrect_count = 0
    unsure_count = 0
    avg_incorrect_conf = 0
    for img_file in os.listdir(test_dir):
      img_path = f"{test_dir}/{img_file}"
      img_name = os.path.basename(img_file)

      # get prediction and class probabilities
      try:
        results = model(img_path, verbose=False)
      except Exception:
        print(f'Could not predict for {img_path}')
        continue
      if not cost_matrix:
        predicted = results[0].names[results[0].probs.top1]
        confidence = results[0].probs.top1conf.item()

      else:
        probs = results[0].probs.data
        C = torch.tensor(cost_matrix, device=probs.device)

        # Compute expected risk vector for predicted choices
        expected_risks = torch.matmul(C.T, probs)

        # Select prediction that minimizes expected cost
        best_class_idx = torch.argmin(expected_risks).item()

        initial_predicted = results[0].names[results[0].probs.top1]
        predicted = results[0].names[best_class_idx]
        confidence = probs[best_class_idx].item()

        if initial_predicted != predicted and verbose:
          print(f"Img {img_name} result changed: {initial_predicted} --> {predicted}")

      Y_true.append(class_name)
      Y_pred.append(predicted)

      if confidence < confidence_threshold:
        pred_str = "NO PREDICTION"
        unsure_count += 1
      elif predicted == class_name:
        pred_str = "CORRECT"
        correct_count += 1
      else:
        pred_str = "INCORRECT"
        incorrect_count += 1
        avg_incorrect_conf += confidence
      if verbose or pred_str != "CORRECT":
        print(f"{pred_str} | Img: {img_name} | True: {class_name} | Predicted: {predicted} | Confidence: {confidence:.2f}")
    total_avg_incorrect_conf += avg_incorrect_conf
    if incorrect_count > 0:
      avg_incorrect_conf = f'{(avg_incorrect_conf / incorrect_count):.4f}'
    else:
      avg_incorrect_conf = 'NaN'
    print(f"CLASS: {class_name} | Unsure: {unsure_count} | Correct: {correct_count} | Incorrect: {incorrect_count} | Accuracy: {correct_count / (correct_count + incorrect_count)} | Avg Incorrect Prediction Confidence: {avg_incorrect_conf}")
    total_correct += correct_count
    total_incorrect += incorrect_count
    total_unsure += unsure_count
    summary_stats += f"{class_name}: accuracy {(correct_count / (correct_count + incorrect_count)):.4f} ({correct_count}/{correct_count + incorrect_count}), {unsure_count} unsure, avg incorrect prediction confidence {avg_incorrect_conf}\n"
    print("==========")

  if total_incorrect > 0:
    total_avg_incorrect_conf = f'{(total_avg_incorrect_conf / total_incorrect):.4f}'
  else:
    total_avg_incorrect_conf = 'NaN'
  print(f"TOTAL | Unsure: {total_unsure} | Correct: {total_correct} | Incorrect: {total_incorrect} | Accuracy: {total_correct / (total_correct + total_incorrect)}")
  print(f"Avg Incorrect Prediction Confidence: {total_avg_incorrect_conf}")
  print(summary_stats)
  cm = confusion_matrix(Y_true, Y_pred, labels=classes)
  display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
  display.plot()
  plt.show()
