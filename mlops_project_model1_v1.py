# -*- coding: utf-8 -*-
"""MLOps_project_model1_v1

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1v37hT6KpcYNCTMuP3adJzmfRici8AxKS
"""

import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import shutil
import random

config["data_dir"] = os.getenv("FONTS_DATA_DIR", "Fonts")

def prepare_dataset(adobe_path, synth_path, combined_path, split_ratio=(.7,.15,.15)):
  def merge_classes(from_dir, to_dir):
    for class_name in os.listdir(from_dir):
      src_cls = os.path.join(from_dir, class_name)
      dst_cls = os.path.join(to_dir, class_name)

      os.makedirs(dst_cls, exist_ok=True)

      for file_name in os.listdir(src_cls):
        shutil.copy(os.path.join(src_cls, file_name), os.path.join(dst_cls,file_name))
    tmp_dir = os.path.join(to_dir, "all")
    os.makedirs(tmp_dir, exist_ok=True)
    merge_classes(adobe_path, tmp_dir)
    merge_classes(synth_path, tmp_dir)

    classes = os.listdir(tmp_dir)
    for cls in classes:
      files = os.listdir(os.path.join(tmp_dir, cls))
      random.shuffle(files)
      n = len(files)
      n_tr, n_val = int(split_ratio[0]*n), int(split_ratio[1]*n)

      subsets = {
          "training": files[:n_tr],
          "validation": files[n_tr:n_tr+n_val],
          "evaluation": files[n_tr+n_val:]
      }

      for split, filelist in subsets.items():
        split_cls = os.path.join(combined_path, split, cls)
        os.makedirs(split_cls, exist_ok=True)
        for fname in filelist:
          shutil.copy(os.path.join(tmp_dir, cls, fname), os.path.join(split_cls, fname))
    shutil.rmtree(tmp_dir)

prepare_dataset("Fonts/training", "Fonts/evaluation", "Fonts")
### Configuration
config = {
    "initial_epochs": 5,
    "total_epochs": 20,
    "patience": 5,
    "batch_size": 32,
    "lr": 1e-4,
    "fine_tune_lr": 1e-5,
    "dropout_probability": 0.5,
    "random_horizontal_flip": 0.5,
    "random_rotation": 15,
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
    "color_jitter_saturation": 0.2,
    "color_jitter_hue": 0.1,
    "num_classes": 200,  # Update this with your number of fonts
    "data_dir": os.getenv("FONTS_DATA_DIR", "Fonts")
}

### Transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=config["random_horizontal_flip"]),
    transforms.RandomRotation(config["random_rotation"]),
    transforms.ColorJitter(
        brightness=config["color_jitter_brightness"],
        contrast=config["color_jitter_contrast"],
        saturation=config["color_jitter_saturation"],
        hue=config["color_jitter_hue"]
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

### Datasets & Loaders
train_dataset = datasets.ImageFolder(os.path.join(config["data_dir"], "training"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(config["data_dir"], "validation"), transform=val_test_transform)
test_dataset = datasets.ImageFolder(os.path.join(config["data_dir"], "evaluation"), transform=val_test_transform)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

### Model
resnet50_model = models.resnet50(weights='IMAGENET1K_V1')  # or weights=None if training from scratch
num_ftrs = resnet50_model.fc.in_features
resnet50_model.fc = nn.Sequential(
    nn.Dropout(config["dropout_probability"]),
    nn.Linear(num_ftrs, config["num_classes"])
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50_model = resnet50_model.to(device)

### Freeze Backbone Initially
for param in resnet50_model.parameters():
    param.requires_grad = False
for param in resnet50_model.fc.parameters():
    param.requires_grad = True

### Optimizer and Loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet50_model.fc.parameters(), lr=config["lr"])

### Training Function
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total

### Validation Function
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

### Phase 1: Train Head Only
best_val_loss = float('inf')
for epoch in range(config["initial_epochs"]):
    start_time = time.time()
    train_loss, train_acc = train(resnet50_model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(resnet50_model, val_loader, criterion, device)
    print(f"[Init Epoch {epoch+1}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Time: {time.time() - start_time:.2f}s")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(resnet50_model.state_dict(), "best_resnet50.pth")
        print("  ↳ Model saved.")

### Phase 2: Unfreeze and Fine-tune All
for param in resnet50_model.parameters():
    param.requires_grad = True

optimizer = optim.Adam(resnet50_model.parameters(), lr=config["fine_tune_lr"])
patience_counter = 0

for epoch in range(config["initial_epochs"], config["total_epochs"]):
    start_time = time.time()
    train_loss, train_acc = train(resnet50_model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(resnet50_model, val_loader, criterion, device)
    print(f"[Finetune Epoch {epoch+1}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Time: {time.time() - start_time:.2f}s")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(resnet50_model.state_dict(), "best_resnet50.pth")
        patience_counter = 0
        print("  ↳ Model saved.")
    else:
        patience_counter += 1
        print(f"  ↳ No improvement. Patience: {patience_counter}")
        if patience_counter >= config["patience"]:
            print("  ↳ Early stopping.")
            break

### Final Evaluation
resnet50_model.load_state_dict(torch.load("best_resnet50.pth"))
test_loss, test_acc = validate(resnet50_model, test_loader, criterion, device)
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")