import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ---- Config ----
MODEL_PATH = "/mnt/bock20/best_resnet50.pth"
DATA_DIR = os.getenv("FONTS_DATA_DIR", "adobe_vfr")
BATCH_SIZE = 32

# ---- Device ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---- Transforms ----
val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---- Dataset & Loader ----
test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "evaluation"),
                                    transform=val_test_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---- Model Setup ----
num_classes = len(test_dataset.classes)
resnet50_model = models.resnet50(weights=None)  # We load weights manually
num_ftrs = resnet50_model.fc.in_features
resnet50_model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, num_classes)
)
resnet50_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
resnet50_model = resnet50_model.to(device)
resnet50_model.eval()

# ---- Evaluation ----
criterion = nn.CrossEntropyLoss()
total_loss, correct, total = 0.0, 0, 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = resnet50_model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_loss = total_loss / len(test_loader)
test_acc = correct / total

print(f"Final Test Loss: {test_loss:.4f}")
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
