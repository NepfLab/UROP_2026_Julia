import os
import cv2
import numpy as np
import json
from pathlib import Path
import glob


def get_image_index(image_path):
  """
  Extracts the image index from a raw file path.
  """
  endings = ['D.JPG', 'F.JPG', 'MS_G.TIF', 'MS_NIR.TIF', 'MS_R.TIF', 'MS_RE.TIF',
             'D.json', 'F.json', 'MS_G.json', 'MS_NIR.json', 'MS_R.json', 'MS_RE.json']
  for end in endings:
    if end in image_path:
      return int(image_path.split(end)[0][-5:-1])

  if 'slice' in os.path.basename(image_path):
    try:
      index = os.path.basename(image_path).split('_D_slice')[0]
      index = int(index.split('_')[-1])
      return index
    except Exception:
      try:
        index = int(os.path.basename(image_path).split('_')[0])
        return index
      except Exception:
        raise ValueError(f"Can't extract image index from path {image_path}")

  try:
    index = int(os.path.splitext(os.path.basename(image_path))[0])
    return index
  except Exception:
    raise ValueError(f"Can't extract image index from path {image_path}")


def get_image_slices(image_path, json_path, output_dir, rows=4, cols=4, threshold=0.05, verbose=False, binary=False):
  # Create output directory
  print(f"Threshold: {threshold}")
  os.makedirs(output_dir, exist_ok=True)

  # Load the original image
  img = cv2.imread(image_path)
  h, w, c = img.shape

  # Create a blank black mask matching the image dimensions
  mask = np.zeros((h, w, 3), dtype=np.uint8)

  # Define strict color mapping for classes (OpenCV uses BGR color order)
  class_colors = {
    "damaged": (0, 0, 255),    # Red in BGR
    "borderline": (0, 255, 255),    # Yellow in BGR
  }

  # Parse LabelMe JSON and draw polygons onto the blank mask
  if os.path.exists(json_path):
    with open(json_path, 'r') as f:
      data = json.load(f)

    for shape in data['shapes']:
      label = shape['label']
      points = np.array(shape['points'], dtype=np.int32)

      # Get the color for this damage class, default to white if not found
      color = class_colors.get(label, (255, 255, 255))

      # Draw the solid polygon onto our mask layer
      cv2.fillPoly(mask, [points], color)

  # Calculate slice dimensions
  slice_h = h // rows
  slice_w = w // cols

  # Slice both the image and mask simultaneously
  slice_count = 0
  for r in range(rows):
    for c in range(cols):
      y1, y2 = r * slice_h, (r + 1) * slice_h
      x1, x2 = c * slice_w, (c + 1) * slice_w

      # Crop the same coordinates from image and mask
      img_slice = img[y1:y2, x1:x2]
      mask_slice = mask[y1:y2, x1:x2]

      # Set threshold (threshold = percent of the total slice area that must match the color)
      PIXEL_THRESHOLD = int((slice_h * slice_w) * threshold)

      labels_found = []
      for class_name, color_bgr in class_colors.items():
        # Create a boolean map where pixels match this specific color
        match = np.all(mask_slice == color_bgr, axis=-1)

        # Count how many pixels are TRUE
        matching_pixel_count = np.sum(match)

        # Only assign the label if it meets minimum threshold
        if matching_pixel_count >= PIXEL_THRESHOLD:
          labels_found.append(class_name)

      # If no colored pixels passed the threshold, it's healthy background
      if not labels_found:
        labels_found.append("healthy")

      # Save the image slice
      if binary:
        label_string = 'damaged' if 'damaged' in labels_found else 'undamaged'
      else:
        label_string = "damaged" if "damaged" in labels_found else labels_found[0]
      image_name = Path(image_path).stem
      slice_filename = f"{image_name}_slice_{slice_count}_{label_string}.jpg"
      dest_path = os.path.join(output_dir, label_string, slice_filename)
      os.makedirs(os.path.dirname(dest_path), exist_ok=True)
      cv2.imwrite(dest_path, img_slice)

      if verbose:
        print(f"Saved {slice_filename}")
      slice_count += 1


def replicate_structure(original_file_path, base_source_dir, base_target_dir, new_filename_modifier=""):
  """
  Replicates a file's subfolder structure in a target directory
  and prepares a new file path with a modified name.
  """
  # Convert strings to Path objects
  orig_path = Path(original_file_path)
  source_base = Path(base_source_dir)
  target_base = Path(base_target_dir)

  # Get the relative path from the source base
  relative_path = orig_path.relative_to(source_base)

  # Get new file name
  if new_filename_modifier:
    new_filename = f"{new_filename_modifier}_{orig_path.name}"
  else:
    new_filename = f"{orig_path.name}"

  # Combine target base, the subfolder structure, and the new filename
  new_file_path = target_base / relative_path.parent / new_filename

  # Create the subfolders in the target directory if they don't exist
  new_file_path.parent.mkdir(parents=True, exist_ok=True)

  return new_file_path


def process_dataset(base_source_dir, base_target_dir, process_fn, new_filename_modifier=""):
  """
  Loops through all files in base_source_dir (including all subfolders),
  generates the replicated target path, and provides a hook to process the files.
  Calls process_fn to process the image and save the processed version.
  (process_fn should be a function taking the original file path and the
  destination file path.)
  """
  source_base = Path(base_source_dir)

  # Find every file recursively using .rglob('*')
  all_files = [f for f in source_base.rglob('*') if f.is_file()]

  print(f"Found {len(all_files)} files to process")

  for current_file in all_files:
    # Generate the new replicated path
    destination_path = replicate_structure(
      original_file_path=current_file,
      base_source_dir=base_source_dir,
      base_target_dir=base_target_dir,
      new_filename_modifier=new_filename_modifier
    )

    print(f"Processing {current_file.name} and saving to {destination_path}")
    process_fn(str(current_file), destination_path)


def image_slicing_pipeline(source_dir, target_dir, json_dir, rows=4, cols=4,
                           threshold=0.05, verbose=False, binary=False):
  def process_fn(orig_path, dest_path, rows=rows, cols=cols, threshold=threshold, binary=binary):
    dest_path = str(dest_path)
    json_path = os.path.join(json_dir, str(get_image_index(orig_path)) + '.json')
    get_image_slices(image_path=orig_path, json_path=json_path,
                     output_dir=os.path.dirname(dest_path), verbose=verbose, rows=rows,
                     cols=cols, threshold=threshold, binary=binary)

  process_dataset(
      base_source_dir=source_dir,
      base_target_dir=target_dir,
      process_fn=process_fn
  )
