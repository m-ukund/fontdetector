import torch
from torchvision import models

# Load your trained ResNet-50 model
model = models.resnet50(weights=None)
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(model.fc.in_features, 200)  # replace 200 with actual class count if known
)

# Load weights
model.load_state_dict(torch.load("best_resnet50.pth"))
torch.save(model, "teacher_model_resnet50.pth")
print("Teacher model saved to teacher_model_resnet50.pth")
