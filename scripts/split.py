import random
import os
import shutil

def get_random_split(indices_list, val_ratio=0.2, test_ratio=0.2,
                     MAX_TRIES=50, eps=0.05):
  """
  Get a random train, val, test split.
  Parameters:
  - indices_list: list of image indices
  - val_ratio: probability of being in validation set
  - test_ratio: probability of being in test set
  """

  num_indices = len(indices_list)
  train_ratio = 1 - val_ratio - test_ratio
  if train_ratio < 0:
    raise ValueError("val_ratio + test_ratio cannot exceed 1")
    return

  for i in range(MAX_TRIES):
    index_to_split = {}
    num_train, num_val, num_test = 0, 0, 0
    for i in indices_list:
      rand_num = random.random()
      if rand_num < train_ratio:
        index_to_split[i] = 'train'
        num_train += 1
      elif rand_num < train_ratio + val_ratio:
        index_to_split[i] = 'val'
        num_val += 1
      else:
        index_to_split[i] = 'test'
        num_test += 1

    good_split = ((max(val_ratio-eps, 0) * num_indices <= num_val and
                   num_val <= min(val_ratio+eps, 1) * num_indices) and
                  (max(test_ratio-eps, 0) * num_indices <= num_test) and
                  num_test <= min(test_ratio+eps, 1) * num_indices)

    if good_split:
      print(f"{num_train} train images, {num_val} val images, {num_test} test images")
      print(index_to_split)
      return index_to_split

  raise ValueError(f"Failed to generate a good split within {MAX_TRIES} tries")


def generate_split(dataset_path, val_ratio=0.2, test_ratio=0.2, MAX_TRIES=50, eps=0.05):
  names = []
  for img in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, img)
    if not os.path.isfile(img_path):
      continue
    names.append(img)
  return get_random_split(indices_list=names, val_ratio=val_ratio, test_ratio=test_ratio,
                          MAX_TRIES=MAX_TRIES, eps=eps)


def get_split_set(source_dir, dest_dir, val_ratio=0.2, test_ratio=0.2, MAX_TRIES=50, eps=0.05, verbose=True):
  name_to_split = generate_split(dataset_path=source_dir, val_ratio=val_ratio, test_ratio=test_ratio,
                                 MAX_TRIES=MAX_TRIES, eps=eps)
  for img in os.listdir(source_dir):
    img_path = os.path.join(source_dir, img)
    if not os.path.isfile(img_path):
      continue
    split = name_to_split[img]
    dest_path = os.path.join(dest_dir, split, img)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy(src=img_path, dst=dest_path)
    if verbose:
      print(f'Copied {img_path} to {dest_path}')
