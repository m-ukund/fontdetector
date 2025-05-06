import os
import shutil
import random

source_dir = '/mnt/block20/datasets/adobevfr'
target_dir = '/mnt/block20/datasets/adobevfr_split'
split_ratio = (0.7, 0.15, 0.15)  # train, val, eval

os.makedirs(target_dir, exist_ok=True)

# List class folders
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(files)

    n = len(files)
    n_train = int(split_ratio[0] * n)
    n_val = int(split_ratio[1] * n)

    subsets = {
        'training': files[:n_train],
        'validation': files[n_train:n_train + n_val],
        'evaluation': files[n_train + n_val:]
    }

    for subset, file_list in subsets.items():
        subset_class_dir = os.path.join(target_dir, subset, class_name)
        os.makedirs(subset_class_dir, exist_ok=True)
        for fname in file_list:
            src = os.path.join(class_path, fname)
            dst = os.path.join(subset_class_dir, fname)
            shutil.copy(src, dst)

print("Splitting completed.")
