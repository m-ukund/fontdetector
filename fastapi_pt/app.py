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
from canary_eval import CanaryEvaluator
import uuid
import time

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

# Initialize canary evaluator
canary_evaluator = CanaryEvaluator(
    canary_traffic_percentage=0.1,  # Route 10% of traffic to canary
    performance_threshold=0.95,     # Canary must be at least 95% as good as production
    min_requests=100               # Minimum requests before comparing performance
)

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

@app.post("/predict")
async def predict(request: Request):
    try:
        # Get image data from request
        image_data = await request.body()
        
        # Generate prediction ID
        prediction_id = str(uuid.uuid4())
        
        # Determine if request should go to canary
        is_canary = canary_evaluator.should_route_to_canary()
        
        # Get prediction from appropriate model
        start_time = time.time()
        if is_canary:
            prediction, probability = get_canary_prediction(image_data)
        else:
            prediction, probability = get_production_prediction(image_data)
        latency = time.time() - start_time
        
        # Save prediction data
        canary_evaluator.save_prediction(
            prediction_id=prediction_id,
            image_data=image_data.decode(),
            prediction=prediction,
            probability=probability,
            is_canary=is_canary,
            latency=latency
        )
        
        # Update metrics (initially assuming prediction is correct)
        canary_evaluator.update_metrics(
            is_canary=is_canary,
            is_correct=True,  # Will be updated when feedback is received
            latency=latency
        )
        
        return {
            "prediction_id": prediction_id,
            "prediction": prediction,
            "probability": probability,
            "is_canary": is_canary
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback/{prediction_id}")
async def submit_feedback(prediction_id: str, feedback: Feedback):
    try:
        # Update metrics with feedback
        canary_evaluator.update_metrics(
            is_canary=feedback.is_canary,
            is_correct=feedback.is_correct,
            latency=feedback.latency,
            has_feedback=True
        )
        
        # Check if canary should be rolled back
        if feedback.is_canary and canary_evaluator.should_rollback():
            logger.warning("Canary performance below threshold - initiating rollback")
            # TODO: Implement rollback logic
            
        return {"status": "feedback recorded"}
        
    except Exception as e:
        logger.error(f"Error recording feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/canary/metrics")
async def get_canary_metrics():
    """Get current canary and production metrics"""
    try:
        metrics = canary_evaluator.get_metrics()
        comparison = canary_evaluator.compare_performance()
        return {
            "metrics": metrics,
            "comparison": comparison
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
