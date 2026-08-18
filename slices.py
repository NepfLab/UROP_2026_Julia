import os
import cv2

OUTPUT_DIR = 'image-slices/'

def create_slices(rows, cols, image_path):
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=False)

    # Load the original image
    img = cv2.imread(image_path)
    h, w, c = img.shape

    # Calculate slice dimensions
    slice_h = h // rows
    slice_w = w // cols

    # Slice the image
    slice_count = 0
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r * slice_h, (r + 1) * slice_h
            x1, x2 = c * slice_w, (c + 1) * slice_w
            img_slice = img[y1:y2, x1:x2]

            # Save the image slice
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            slice_filename = f"{image_name}_slice_{slice_count}.jpg"
            dest_path = os.path.join(OUTPUT_DIR, slice_filename)
            cv2.imwrite(dest_path, img_slice)

            slice_count += 1

    return OUTPUT_DIR

def delete_slices():
    os.rmdir(OUTPUT_DIR)
