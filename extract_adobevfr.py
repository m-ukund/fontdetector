import os
import argparse
import numpy as np
from PIL import Image

def read_bcf(bcf_file):
    with open(bcf_file, 'rb') as f:
        n_images = int.from_bytes(f.read(4), 'little')
        images = []
        for _ in range(n_images):
            img_bytes = f.read(224 * 224)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8).reshape(224, 224)
            images.append(img_array)
    return images

def read_labels(label_file):
    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f]
    return labels

def save_images(images, labels, classes, out_dir, split_name):
    for idx, (img_array, label_idx) in enumerate(zip(images, labels)):
        class_name = classes[int(label_idx)]
        class_dir = os.path.join(out_dir, split_name, class_name)
        os.makedirs(class_dir, exist_ok=True)
        img = Image.fromarray(img_array)
        img.convert('L').save(os.path.join(class_dir, f'{idx}.png'))

def main():
    parser = argparse.ArgumentParser(description="Extract Adobe VFR BCF files into image folders")
    parser.add_argument('--source_dir', type=str, required=True, help='Path to directory containing BCF and label files')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to directory where extracted images will go')
    args = parser.parse_args()

    source_dir = args.source_dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # Read class names
    with open(os.path.join(source_dir, 'fontlist.txt'), 'r') as f:
        classes = [line.strip() for line in f]

    # Process splits
    splits = {
        'training': ('train.bcf', 'train.label'),
        'validation': ('val.bcf', 'val.label'),
        'evaluation': ('test.bcf', 'test.label')
    }

    for split_name, (bcf_file, label_file) in splits.items():
        print(f'Processing {split_name}...')
        images = read_bcf(os.path.join(source_dir, bcf_file))
        labels = read_labels(os.path.join(source_dir, label_file))
        save_images(images, labels, classes, out_dir, split_name)
        print(f'{split_name} done.')

if __name__ == '__main__':
    main()
