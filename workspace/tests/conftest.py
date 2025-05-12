import pytest
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import torch.nn as nn

# ───────────── deterministic preprocessing ─────────────────
class ResizeHeight:
    """Resize so the height equals `height`, keep aspect ratio."""
    def __init__(self, height: int = 105):
        self.height = height

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if h == 0:
            return Image.new(img.mode, (1, self.height), 255)
        new_w = max(1, round(self.height * w / h))
        return img.resize((new_w, self.height), Image.LANCZOS)


class Squeezing:
    """Deterministically squeeze horizontally by a fixed factor."""
    def __init__(self, ratio: float = 2.5):
        self.ratio = ratio

    def __call__(self, img: Image.Image) -> Image.Image:
        _, h = img.size
        return img.resize((max(1, round(h * self.ratio)), h),
                          Image.LANCZOS)

class CenterPatch:
    """Extract (or pad) a centred width-`step` patch."""
    def __init__(self, step: int = 105):
        self.step = step

    @staticmethod
    def _pad(img: Image.Image, width: int) -> Image.Image:
        w, h = img.size
        if w >= width:
            return img
        canvas = Image.new(img.mode, (width, h), 255)
        canvas.paste(img, (0, 0))
        return canvas

    def __call__(self, img: Image.Image) -> Image.Image:
        img = self._pad(img, self.step)
        w, h = img.size
        if w == self.step:
            return img
        sx = (w - self.step) // 2
        tile = img.crop((sx, 0, sx + self.step, h))
        return tile


class DeepFontAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 12, 2, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.encoder(x)

class DeepFont(nn.Module):
    def __init__(self, encoder: nn.Sequential, num_classes: int):
        super().__init__()
        self.encoder = encoder
        self.conv5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv7 = nn.Conv2d(256, 256, 3, 1, 1)
        self.flatten = nn.Flatten()
        hid = 4096
        self.fc1 = nn.Linear(256 * 12 * 12, hid)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hid, hid)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hid, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv7(x))
        x = self.flatten(x)
        x = self.drop1(torch.relu(self.fc1(x)))
        x = self.drop2(torch.relu(self.fc2(x)))
        return self.fc3(x)

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
    ResizeHeight(105),
    Squeezing(),
    CenterPatch(step=105),
    transforms.Grayscale(1),
    transforms.ToTensor(),
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
    model_path = "/mnt/font-detector/models/finetuned_n100.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = DeepFontAutoencoder()
    model = DeepFont(ae.encoder, num_classes=100)

    state_dict = torch.load(model_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.to(device)
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
