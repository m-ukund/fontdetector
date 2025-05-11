import pytest
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

# -------- FONT-SPECIFIC SETTINGS --------
@pytest.fixture(scope="session")
def font_list():
    subset_font_path = "/home/jovyan/work/fontsubset.txt"
    with open(subset_font_path) as f:
        return [line.strip() for line in f if line.strip()]

@pytest.fixture(scope="session")
def font_to_index(font_list):
    return {name: idx for idx, name in enumerate(font_list)}

# -------- TRANSFORM --------
@pytest.fixture(scope="session")
def transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# -------- PREDICT SINGLE IMAGE --------
@pytest.fixture(scope="session")
def predict(transform):
    def predict_image(model, image):
        model.eval()
        with torch.no_grad():
            input_tensor = transform(image).unsqueeze(0).to(next(model.parameters()).device)
            output = model(input_tensor)
            return output.argmax(dim=1).item()
    return predict_image

# -------- LOAD FONT DETECTION MODEL --------
@pytest.fixture(scope="session")
def model():
    model_path = "models/mobilenetv2_canary.pt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

# -------- CUSTOM FONT DATASET --------
class FontDataset(Dataset):
    def __init__(self, img_dir, transform, font_to_index):
        self.img_dir = img_dir
        self.transform = transform
        self.font_to_index = font_to_index
        self.image_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        txt_filename = img_filename.replace(".png", ".txt")

        img_path = os.path.join(self.img_dir, img_filename)
        txt_path = os.path.join(self.img_dir, txt_filename)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        with open(txt_path) as f:
            font_name = f.read().strip()

        label = self.font_to_index[font_name]
        return image, label

# -------- LOAD TEST DATA --------
@pytest.fixture(scope="session")
def test_data(transform, font_to_index):
    font_data_dir = os.getenv("FONT_DATA_DIR", "/mnt/evaluation_filtered")
    dataset = FontDataset(font_data_dir, transform, font_to_index)
    return DataLoader(dataset, batch_size=32, shuffle=False)

# -------- MAKE ALL PREDICTIONS --------
@pytest.fixture(scope="session")
def predictions(model, test_data):
    dataset_size = len(test_data.dataset)
    all_predictions = np.empty(dataset_size, dtype=np.int64)
    all_labels = np.empty(dataset_size, dtype=np.int64)

    current_index = 0
    with torch.no_grad():
        for images, labels in test_data:
            images = images.to(next(model.parameters()).device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions[current_index:current_index + len(labels)] = predicted.cpu().numpy()
            all_labels[current_index:current_index + len(labels)] = labels.cpu().numpy()
            current_index += len(labels)

    return all_labels, all_predictions
