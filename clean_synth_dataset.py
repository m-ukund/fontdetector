import os
import shutil

DATASET_PATH = '/mnt/block100/Synthetic'
FONT_LIST_PATH = '/home/cc/fontdetector/fastapi_pt/fontsubset.txt'

# Load and normalize font subset (remove trailing Std if present)
with open(FONT_LIST_PATH, 'r') as f:
    font_subset = set(line.strip().removesuffix('Std') for line in f if line.strip())

print(f"Loaded {len(font_subset)} normalized fonts from subset.")
print("Subset sample:", list(font_subset)[:5])

# Loop through each folder
for folder in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, folder)
    if os.path.isdir(folder_path):
        matched_font = None
        for font in font_subset:
            if font in folder:
                matched_font = font
                break

        if matched_font:
            if '-Regular' in folder:
                print(f"Keeping: {folder} (matches {matched_font} and is Regular)")
            else:
                print(f"Deleting: {folder} (matches {matched_font} but not Regular)")
                # shutil.rmtree(folder_path)
        else:
            print(f"Deleting: {folder} (no match in font subset)")
            # shutil.rmtree(folder_path)

print("Dry run complete. Uncomment shutil.rmtree() to enable deletion.")
