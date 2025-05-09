from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import base64
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
from prometheus_fastapi_instrumentator import Instrumentator

# Initialize FastAPI
app = FastAPI(
    title="Font Detection API",
    description="API for classifying fonts from text images (100 fonts)",
    version="1.0.0"
)

# Define request and response models
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image

class PredictionResponse(BaseModel):
    prediction: str
    probability: float = Field(..., ge=0, le=1)

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the font detection model
MODEL_PATH = "font_model.pth"
try:
    model = torch.load(MODEL_PATH, map_location=device)
    model.to(device)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Define your 100 font class labels (example: Font1, Font2, ..., Font100)
classes = [f"Font{i+1}" for i in range(100)]
# 👉 OR if you have actual font names, replace with your list:
# classes = ["Arial", "Times New Roman", "Courier", ..., "Font100"]

# Image preprocessing
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img).unsqueeze(0)

@app.post("/predict", response_model=PredictionResponse)
def predict_image(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Preprocess
        image = preprocess_image(image).to(device)

        # Inference
        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, 1).item()
            confidence = probabilities[0, predicted_class].item()

        return PredictionResponse(
            prediction=classes[predicted_class],
            probability=confidence
        )

    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 input")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)
