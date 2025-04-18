#!/bin/bash

# This script orchestrates the font data preparation using Docker Compose.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Define the compose file name and volume names
COMPOSE_FILE="docker-compose-font-data.yaml"
VOL_ADOBE="adobe_vfr"
VOL_GOOGLE="google_fonts"
VOL_SYNTHETIC="synthetic_fonts"
PYTHON_SCRIPT="generate_synthetic_images.py"
KAGGLE_CRED_FILE="kaggle.json"

# --- Prerequisites Check ---
echo "--- Checking Prerequisites ---"

# 1. Check for Docker and Docker Compose
if ! command -v docker &> /dev/null; then
    echo "Error: Docker command could not be found. Please install Docker."
    exit 1
fi
if ! command -v docker compose &> /dev/null; then
    echo "Error: 'docker compose' command could not be found."
    echo "Please install Docker Compose V2 (usually included with Docker Desktop or install docker-compose-plugin)."
    # Optionally check for older 'docker-compose' if needed
    # if ! command -v docker-compose &> /dev/null; then
    #    echo "Error: Neither 'docker compose' nor 'docker-compose' found."
    #    exit 1
    # fi
    exit 1
fi
echo "Docker and Docker Compose found."

# 2. Ensure necessary files exist in the current directory
if [ ! -f "$COMPOSE_FILE" ]; then echo "Error: $COMPOSE_FILE not found!"; exit 1; fi
if [ ! -f "Dockerfile" ]; then echo "Error: Dockerfile not found!"; exit 1; fi
if [ ! -f "$PYTHON_SCRIPT" ]; then echo "Error: $PYTHON_SCRIPT not found!"; exit 1; fi
if [ ! -f "$KAGGLE_CRED_FILE" ]; then
    echo "Error: $KAGGLE_CRED_FILE not found!"
    echo "Please download your Kaggle API token from Your Account -> API -> Create New API Token"
    echo "and place the downloaded 'kaggle.json' file in this directory."
    exit 1
fi
# Check if kaggle.json copy lines are uncommented in Dockerfile (basic check)
if ! grep -qE '^\s*COPY\s+kaggle\.json\s+/root/\.kaggle/kaggle\.json' Dockerfile; then
    echo "Warning: The 'COPY kaggle.json ...' line seems to be commented out or missing in your Dockerfile."
    echo "Ensure Kaggle authentication is handled correctly within the Dockerfile for the build process."
    # Optionally exit here if you strictly require the COPY method
    # exit 1
fi
echo "Required files found."

# 3. Ensure Docker volumes exist (create if missing)
echo "Ensuring Docker volumes exist: $VOL_ADOBE, $VOL_GOOGLE, $VOL_SYNTHETIC..."
docker volume create "$VOL_ADOBE" > /dev/null
docker volume create "$VOL_GOOGLE" > /dev/null
docker volume create "$VOL_SYNTHETIC" > /dev/null
echo "Volumes checked/created."

echo "--- Starting Data Preparation ---"

# --- Execution ---
# Run docker compose:
# -f: Specify the compose file
# up: Create and start services
# --build: Build the image before starting if it's missing or outdated
# --remove-orphans: Clean up containers for services not defined in the compose file (good hygiene)
# --abort-on-container-exit: Stop the 'up' command immediately when the single container exits
#                            (suitable for one-off tasks like this)
docker compose -f "$COMPOSE_FILE" up --build --remove-orphans --abort-on-container-exit

echo ""
echo "--- Data Preparation Complete ---"
echo "Data should be available in Docker volumes:"
echo "- $VOL_ADOBE"
echo "- $VOL_GOOGLE"
echo "- $VOL_SYNTHETIC"
echo "You can inspect them using 'docker run --rm -v <volume_name>:/data alpine ls -l /data'"
echo "---------------------------------"