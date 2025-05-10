import os
from pathlib import Path
import shutil

# Paths
extracted_dir = Path("/mnt/block100/adobevfr/evaluation_extracted")
filtered_dir = Path("/mnt/block100/adobevfr/evaluation_filtered")
subset_path = Path("/home/cc/fontdetector/fastapi_pt/fontsubset.txt")

# Load subset font list
with open(subset_path, "r") as f:
    allowed_fonts = set(line.strip() for line in f if line.strip())

# Create filtered output directory
filtered_dir.mkdir(parents=True, exist_ok=True)

# Loop through .txt files to filter
for label_file in extracted_dir.glob("*.txt"):
    with open(label_file, "r") as f:
        font_name = f.read().strip()

    if font_name in allowed_fonts:
        i = label_file.stem  # '0' from '0.txt'
        shutil.copy(extracted_dir / f"{i}.png", filtered_dir / f"{i}.png")
        shutil.copy(label_file, filtered_dir / f"{i}.txt")

print(f"Filtered images saved to: {filtered_dir}")
