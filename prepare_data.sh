#!/bin/bash

# Create volumes if not exist
docker volume create adobe_vfr
docker volume create google_fonts
docker volume create synthetic_fonts

# Build the data-prep image
docker build -t font-data-prep .

# Start a container to download and extract datasets into volumes
docker run --rm -it \
  -v adobe_vfr:/data/adobe \
  -v google_fonts:/data/google \
  -v synthetic_fonts:/data/synthetic \
  font-data-prep bash -c "
    echo 'Downloading Adobe VFR...'
    kaggle datasets download -d luisgoncalo/adobe-visual-font-recognition -p /data/adobe &&
    unzip /data/adobe/adobe-visual-font-recognition.zip -d /data/adobe &&

    echo 'Downloading Google Fonts...'
    kaggle datasets download -d prasunroy/google-fonts-for-stefann -p /data/google &&
    unzip /data/google/google-fonts-for-stefann.zip -d /data/google &&

    echo 'Generating synthetic images...'
    python3 /workspace/generate_synthetic_images.py
"
