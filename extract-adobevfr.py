import os
from pathlib import Path
from tqdm import tqdm
from PIL import Image

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# -------------------------------
# Dataset class
# -------------------------------

class AdobeVFRDataset(Dataset):
    def __init__(self, bcf_dir, split, all_fonts, subset_fonts):
        bcf_dir = Path(bcf_dir)
        self.bcf_file = bcf_dir / f"{split}.bcf"
        labels = self._read_u32(bcf_dir / f"{split}.label")

        all_fonts = self._read_lines(all_fonts)
        sub_fonts = self._read_lines(subset_fonts)

        orig2compact = {all_fonts.index(n): i for i, n in enumerate(sub_fonts)}
        keep = np.isin(labels, list(orig2compact))
        self.idx = np.nonzero(keep)[0]
        self.labels = np.array([orig2compact[x] for x in labels[self.idx]], dtype=np.int64)

        self._load_offsets()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        real_i = int(self.idx[i])
        raw = self._entry_bytes(real_i)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        img = Image.fromarray(img)
        return img, self.labels[i]

    def _load_offsets(self):
        with self.bcf_file.open("rb") as f:
            n = int(np.frombuffer(f.read(8), np.uint64)[0])
            sizes = np.frombuffer(f.read(8 * n), np.uint64)
        self._offs = np.append(np.uint64(0), np.add.accumulate(sizes).astype(np.uint64))

    def _entry_bytes(self, i):
        hdr = len(self._offs) * 8
        a = int(self._offs[i])
        b = int(self._offs[i + 1])
        with self.bcf_file.open("rb") as f:
            f.seek(hdr + a)
            return f.read(b - a)

    def _read_u32(self, path):
        return np.fromfile(path, np.uint32).astype(np.int32)

    def _read_lines(self, path):
        return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip()]

# -------------------------------
# Main Extraction Logic
# -------------------------------

bcf_dir = "/mnt/block100/adobevfr/evaluation"
output_dir = "/mnt/block100/adobevfr/evaluation_extracted"
all_fonts_path = "/mnt/block100/adobevfr/all_fonts.txt"
subset_fonts_path = "/home/cc/fontdetector/fastapi_pt/fontsubset.txt"

os.makedirs(output_dir, exist_ok=True)

dataset = AdobeVFRDataset(
    bcf_dir=bcf_dir,
    split="test",  # Refers to test.bcf and test.label
    all_fonts=all_fonts_path,
    subset_fonts=subset_fonts_path
)

subset_fonts = dataset._read_lines(subset_fonts_path)

for i in tqdm(range(len(dataset))):
    img, label = dataset[i]
    img.save(os.path.join(output_dir, f"{i}.png"))
    font_name = subset_fonts[label]
    with open(os.path.join(output_dir, f"{i}.txt"), "w") as f:
        f.write(font_name)

print(f"Extraction complete: {len(dataset)} images saved to {output_dir}")
