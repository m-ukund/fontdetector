# fontdetector/Dockerfile

# Start from a standard Python base image
FROM python:3.10-slim

# Set the main working directory inside the image
WORKDIR /workspace

# Install system packages needed: unzip
# Refresh package lists, install unzip without optional recommends, clean up lists
RUN apt-get update && \
    apt-get install -y --no-install-recommends unzip && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages: kaggle CLI and potential dependencies for your script
# --- Add any libraries needed by generate_synthetic_images.py here! ---
# Examples: Pillow for image manipulation, numpy for arrays
RUN pip install --no-cache-dir \
    kaggle \
    Pillow \
    numpy
    # Add other libraries like: fonttools, etc., if your script needs them

# --- Handle Kaggle API Credentials ---
# Place your downloaded 'kaggle.json' file in the same directory as this Dockerfile.
# Then, UNCOMMENT the following lines to copy it into the image:
# ------------------------------------
# Create the .kaggle directory with appropriate permissions
RUN mkdir -p /root/.kaggle && chmod 700 /root/.kaggle
# Copy the kaggle.json file into the image
COPY kaggle.json /root/.kaggle/kaggle.json
# Set required permissions for the credentials file
RUN chmod 600 /root/.kaggle/kaggle.json
# ------------------------------------
# >>>>> DOUBLE-CHECK YOU HAVE UNCOMMENTED THE 5 LINES ABOVE! <<<<<

# Copy your Python script into the image's working directory
COPY generate_synthetic_images.py .

# No CMD or ENTRYPOINT needed - command is specified in docker-compose.yaml
