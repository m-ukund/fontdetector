#!/bin/bash

SOURCE_DIR="/mnt/block20/adobevfr"
TARGET_DIR="/mnt/block20/adobevfr_split"

# Create split directories
mkdir -p $TARGET_DIR/training
mkdir -p $TARGET_DIR/validation
mkdir -p $TARGET_DIR/evaluation

# Copy training files
cp $SOURCE_DIR/train.bcf $TARGET_DIR/training/
cp $SOURCE_DIR/train.label $TARGET_DIR/training/

# Copy validation files
cp $SOURCE_DIR/val.bcf $TARGET_DIR/validation/
cp $SOURCE_DIR/val.label $TARGET_DIR/validation/

# Copy evaluation (test) files
cp $SOURCE_DIR/test.bcf $TARGET_DIR/evaluation/
cp $SOURCE_DIR/test.label $TARGET_DIR/evaluation/

echo "Adobe VFR dataset split completed."
