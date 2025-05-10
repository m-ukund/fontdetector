import os
import shutil

DATASET_PATH = '/mnt/block100/Synthetic'  # where your dataset is
FONT_LIST_PATH = '/home/cc/fontdetector/fastapi_pt/fontsubset.txt'  # your 100-font list

# Load font subset (100 fonts)
with open(FONT_LIST_PATH, 'r') as f:
    font_subset = set(line.strip() for line in f if line.strip())

print(f"Loaded {len(font_subset)} fonts from subset.")

# Loop through each folder in Synthetic
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(folder_path):
        if folder not in font_subset:
            print(f"Deleting folder: {folder_path}")
            shutil.rmtree(folder_path)
        else:
            print(f"Keeping folder: {folder_path}")

print("Cleanup complete.")
