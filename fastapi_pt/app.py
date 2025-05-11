from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import base64
import numpy as np
from PIL import Image
import io
import os
from prometheus_fastapi_instrumentator import Instrumentator
import tritonclient.http as httpclient

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

# Get environment variables
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "font_detector")

# Initialize Triton client
triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)

@app.post("/predict", response_model=PredictionResponse)
def predict_image(request: ImageRequest):
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        
        # Prepare inputs
        inputs = []
        inputs.append(httpclient.InferInput("INPUT_IMAGE", [1, 1], "BYTES"))
        input_data = np.array([[request.image]], dtype=object)
        inputs[0].set_data_from_numpy(input_data)

        # Prepare outputs
        outputs = []
        outputs.append(httpclient.InferRequestedOutput("FONT_LABEL", binary_data=False))
        outputs.append(httpclient.InferRequestedOutput("PROBABILITY", binary_data=False))

        # Run inference
        results = triton_client.infer(model_name=MODEL_NAME, inputs=inputs, outputs=outputs)

        # Get results
        predicted_class = results.as_numpy("FONT_LABEL")[0,0]
        probability = results.as_numpy("PROBABILITY")[0,0]

        return PredictionResponse(
            prediction=predicted_class,
            probability=float(probability)
        )

    except base64.binascii.Error:
        raise HTTPException(status_code=400, detail="Invalid base64 input")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)
