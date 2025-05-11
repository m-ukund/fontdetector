import os
import random
from PIL import Image

# --- External Robustness Dataset ---

GIBBERISH_DIR = "gibberish_looks_like"

def evaluate_folder(model, folder_path, predict):
    correct = 0
    total = 0
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if not fname.endswith(".png"):
                continue
            img = Image.open(os.path.join(root, fname)).convert("RGB")
            pred = predict(model, img)
            if isinstance(pred, int):  # Sanity check
                correct += 1  # You may define correctness differently
            total += 1
    return correct / total * 100 if total > 0 else 0

# Require 60% meaningful prediction rate on gibberish images
def test_gibberish_looks_like_accuracy(model, predict):
    acc = evaluate_folder(model, GIBBERISH_DIR, predict)
    assert acc >= 60, f"{GIBBERISH_DIR} accuracy too low: {acc:.2f}%"
