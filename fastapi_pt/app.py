from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import base64
import numpy as np
from PIL import Image
import io
import os
from prometheus_fastapi_instrumentator import Instrumentator
import tritonclient.http as httpclient
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Font Detection API",
    description="API for classifying fonts from text images (100 fonts)",
    version="1.0.0"
)

# Define request and response models
class ImageRequest(BaseModel):
    image: str  # Base64 encoded image
    model_version: Optional[str] = None  # Optional model version

class PredictionResponse(BaseModel):
    prediction: str
    probability: float = Field(..., ge=0, le=1)
    model_version: str

# Get environment variables
TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "font_detector")
DEFAULT_MODEL_VERSION = os.getenv("DEFAULT_MODEL_VERSION", "1")

# Initialize Triton client
try:
    triton_client = httpclient.InferenceServerClient(url=TRITON_SERVER_URL)
    logger.info(f"Connected to Triton server at {TRITON_SERVER_URL}")
except Exception as e:
    logger.error(f"Failed to connect to Triton server: {e}")
    raise

def validate_image(image_data: bytes) -> bool:
    """Validate that the image data is valid and can be opened."""
    try:
        Image.open(io.BytesIO(image_data))
        return True
    except Exception:
        return False

@app.get("/model/versions")
async def get_model_versions():
    """Get available model versions."""
    try:
        model_metadata = triton_client.get_model_metadata(MODEL_NAME)
        return {"versions": model_metadata.get("versions", [])}
    except Exception as e:
        logger.error(f"Error getting model versions: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving model versions")

@app.post("/predict", response_model=PredictionResponse)
async def predict_image(request: ImageRequest):
    try:
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image)
        except base64.binascii.Error:
            raise HTTPException(status_code=400, detail="Invalid base64 encoding")

        # Validate image
        if not validate_image(image_data):
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Use provided version or default
        model_version = request.model_version or DEFAULT_MODEL_VERSION

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
        try:
            results = triton_client.infer(
                model_name=MODEL_NAME,
                model_version=model_version,
                inputs=inputs,
                outputs=outputs
            )
        except Exception as e:
            logger.error(f"Triton inference error: {e}")
            raise HTTPException(status_code=503, detail="Model inference service unavailable")

        # Get results
        try:
            predicted_class = results.as_numpy("FONT_LABEL")[0,0]
            probability = results.as_numpy("PROBABILITY")[0,0]
        except Exception as e:
            logger.error(f"Error processing model output: {e}")
            raise HTTPException(status_code=500, detail="Error processing model output")

        return PredictionResponse(
            prediction=predicted_class,
            probability=float(probability),
            model_version=model_version
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Add health check endpoint
@app.get("/health")
async def health_check():
    try:
        # Check Triton server health
        if not triton_client.is_server_ready():
            raise HTTPException(status_code=503, detail="Triton server not ready")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

# Add Prometheus metrics
Instrumentator().instrument(app).expose(app)
