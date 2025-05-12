import random
import time
from datetime import datetime
import boto3
import os
from typing import Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CanaryEvaluator:
    def __init__(self, 
                 canary_traffic_percentage: float = 0.1,
                 performance_threshold: float = 0.95,
                 min_requests: int = 100):
        self.canary_traffic_percentage = canary_traffic_percentage
        self.performance_threshold = performance_threshold
        self.min_requests = min_requests
        
        # Initialize metrics storage
        self.prod_metrics = {
            "total_requests": 0,
            "correct_predictions": 0,
            "avg_latency": 0,
            "feedback_count": 0
        }
        
        self.canary_metrics = {
            "total_requests": 0,
            "correct_predictions": 0,
            "avg_latency": 0,
            "feedback_count": 0
        }
        
        # Initialize MinIO client for data storage
        self.s3 = boto3.client(
            's3',
            endpoint_url=os.environ.get('MINIO_URL', 'http://minio:9000'),
            aws_access_key_id=os.environ.get('MINIO_USER', 'your-access-key'),
            aws_secret_access_key=os.environ.get('MINIO_PASSWORD', 'your-secret-key'),
            region_name='us-east-1'
        )
        
    def should_route_to_canary(self) -> bool:
        """Decide whether to route request to canary model"""
        return random.random() < self.canary_traffic_percentage
    
    def save_prediction(self, 
                       prediction_id: str,
                       image_data: str,
                       prediction: str,
                       probability: float,
                       is_canary: bool,
                       latency: float):
        """Save prediction data to MinIO"""
        timestamp = datetime.now().isoformat()
        data = {
            "prediction_id": prediction_id,
            "timestamp": timestamp,
            "image": image_data,
            "prediction": prediction,
            "probability": probability,
            "is_canary": is_canary,
            "latency": latency
        }
        
        # Save to appropriate bucket
        bucket = "canary-data" if is_canary else "production-data"
        key = f"{bucket}/{prediction_id}.json"
        
        try:
            self.s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=str(data)
            )
        except Exception as e:
            logger.error(f"Failed to save prediction data: {e}")
    
    def update_metrics(self, 
                      is_canary: bool,
                      is_correct: bool,
                      latency: float,
                      has_feedback: bool = False):
        """Update performance metrics"""
        metrics = self.canary_metrics if is_canary else self.prod_metrics
        metrics["total_requests"] += 1
        if is_correct:
            metrics["correct_predictions"] += 1
        metrics["avg_latency"] = (
            (metrics["avg_latency"] * (metrics["total_requests"] - 1) + latency) 
            / metrics["total_requests"]
        )
        if has_feedback:
            metrics["feedback_count"] += 1
    
    def compare_performance(self) -> Tuple[bool, Dict[str, Any]]:
        """Compare canary vs production performance"""
        if (self.canary_metrics["total_requests"] < self.min_requests or 
            self.prod_metrics["total_requests"] < self.min_requests):
            return False, {"status": "insufficient_data"}
        
        canary_accuracy = (
            self.canary_metrics["correct_predictions"] / 
            self.canary_metrics["total_requests"]
        )
        prod_accuracy = (
            self.prod_metrics["correct_predictions"] / 
            self.prod_metrics["total_requests"]
        )
        
        canary_latency = self.canary_metrics["avg_latency"]
        prod_latency = self.prod_metrics["avg_latency"]
        
        # Canary should be at least as good as production
        is_better = (
            canary_accuracy >= prod_accuracy * self.performance_threshold and
            canary_latency <= prod_latency * 1.1  # Allow 10% latency increase
        )
        
        comparison = {
            "canary_accuracy": canary_accuracy,
            "prod_accuracy": prod_accuracy,
            "canary_latency": canary_latency,
            "prod_latency": prod_latency,
            "is_better": is_better
        }
        
        return is_better, comparison
    
    def should_rollback(self) -> bool:
        """Determine if canary should be rolled back"""
        is_better, comparison = self.compare_performance()
        if not is_better:
            logger.warning(f"Canary performance below threshold: {comparison}")
            return True
        return False
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get current metrics for both models"""
        return {
            "canary": self.canary_metrics,
            "production": self.prod_metrics
        } 