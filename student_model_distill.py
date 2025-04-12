import os, time, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from collections import Counter

# --- CONFIGURATION ---
config = {
    "teacher_path": "teacher_model_resnet50.pth",
    "top_k_fonts": 50,
    "batch_size": 32,
    "epochs": 20,
    "alpha": 0.5,           # CrossEntropy weight
    "beta": 0.5,            # KLDiv weight
    "temperature": 2.0,     # Soften logits
    "lr": 1e-4,
    "data_dir": "google_fonts_synth",    # Should contain training/validation/evaluation
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- TRANSFORMS ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- LOAD FULL TRAIN DATASET TO FIND TOP FONTS ---
train_full = datasets.ImageFolder(os.path.join(config["data_dir"], "training"), transform=transform)
class_counts = Counter([label for _, label in train_full.samples])
top_k_classes = [c for c, _ in class_counts.most_common(config["top_k_fonts"])]

# Save label mapping
label_map = {i: train_full.classes[i] for i in top_k_classes}
with open("student_class_map.json", "w") as f:
    json.dump(label_map, f, indent=2)

# --- FILTER DATASET TO TOP 50 FONTS ONLY ---
def filter_top_k(dataset):
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label in top_k_classes]
    return Subset(dataset, indices)

train_dataset = filter_top_k(train_full)
val_dataset = filter_top_k(datasets.ImageFolder(os.path.join(config["data_dir"], "validation"), transform=transform))
test_dataset = filter_top_k(datasets.ImageFolder(os.path.join(config["data_dir"], "evaluation"), transform=transform))

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# --- LOAD TEACHER MODEL ---
teacher = torch.load(config["teacher_path"])
teacher.eval()
teacher.to(config["device"])

# --- DEFINE STUDENT MODEL (MobileNetV2) ---
student = models.mobilenet_v2(weights='IMAGENET1K_V1')
student.classifier[1] = nn.Linear(student.last_channel, config["top_k_fonts"])
student.to(config["device"])

# --- LOSSES ---
ce_loss = nn.CrossEntropyLoss()
kl_loss = nn.KLDivLoss(reduction='batchmean')  # for distillation

# --- OPTIMIZER ---
optimizer = optim.Adam(student.parameters(), lr=config["lr"])

# --- TRAINING LOOP WITH DISTILLATION ---
def distill_loss(student_logits, teacher_logits, true_labels, T, alpha, beta):
    # Standard cross entropy loss (hard labels)
    ce = ce_loss(student_logits, true_labels)

    # KL divergence on softened logits
    p = F.log_softmax(student_logits / T, dim=1)
    q = F.softmax(teacher_logits / T, dim=1)
    kl = kl_loss(p, q) * (T * T)

    return alpha * ce + beta * kl

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(config["device"]), y.to(config["device"])
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

print("Starting distillation training...")
for epoch in range(config["epochs"]):
    student.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(config["device"]), labels.to(config["device"])

        optimizer.zero_grad()

        with torch.no_grad():
            teacher_outputs = teacher(images)

        student_outputs = student(images)
        loss = distill_loss(student_outputs, teacher_outputs, labels,
                            config["temperature"], config["alpha"], config["beta"])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    val_acc = evaluate(student, val_loader)
    print(f"Epoch {epoch+1:02d}: Loss = {total_loss:.4f}, Val Acc = {val_acc:.4f}")

# --- SAVE STUDENT MODEL ---
torch.save(student.state_dict(), "student_mobilenetv2_distilled.pth")
print("Saved distilled student model to student_mobilenetv2_distilled.pth")

# --- FINAL TEST ---
test_acc = evaluate(student, test_loader)
print(f"Final Test Accuracy: {test_acc:.4f}")


# For live monitoring 
"""
# Use it only in student_model_distill.py if you want to split synthetic data
prepare_dataset(
    synth_path="google_fonts_synth/raw",
    combined_path="google_fonts_synth",
    split_ratio=(.8, .1, .1)
)
"""
